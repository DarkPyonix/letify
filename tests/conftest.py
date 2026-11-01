"""Shared fixtures and the few fakes the suite is allowed to use.

Everything that can run for real runs for real. The local provider starts a worker
subprocess behind the same framed protocol a remote runtime uses, so the protocol, the
pool, the store and the release rule are all exercised rather than stubbed.

A fake appears here only where the real thing needs something a test machine does not
have: the Colab CLI, the Elice HTTP API, the Modal adapter, a cloud bucket
and nvidia-smi. They are small objects rather than mocks, so a test still asserts on
what a caller observes instead of on which method was called.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Generated provider types are written by Launcher(); the suite must not write them into
# the repository it runs from. Tests of the generator enable it explicitly.
os.environ["LETIFY_STUBS"] = "0"

import letify
from letify.config.schema import ProviderConfig
from letify.declare.env import Env
from letify.declare.instance import Instance
from letify.providers.local import Local
from letify.runtime import telemetry

# -- the real launcher ---------------------------------------------------------


@pytest.fixture
def let(tmp_path: Path) -> letify.Launcher:
    # home=False keeps the developer's own accounts out of the test run, and an empty project
    # directory keeps out the .letify/config.toml of the repository the suite runs from.
    project = tmp_path / "empty-project" / ".letify"
    project.mkdir(parents=True)
    return letify.Launcher(project, home=False, announce=False)


@pytest.fixture
def cpu(let: letify.Launcher) -> letify.Instance:
    return let.providers.local.CPU


@pytest.fixture
def config_file(tmp_path: Path):
    """Write a project's .letify/config.toml and return the .letify directory."""

    def write(body: str, name: str = ".letify") -> Path:
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        (directory / "config.toml").write_text(body, encoding="utf-8")
        return directory

    return write


@pytest.fixture
def launcher_from(config_file):
    """Build a launcher over one written configuration file."""

    def build(body: str, **kwargs: Any) -> letify.Launcher:
        return letify.Launcher(config_file(body), home=False, announce=False, **kwargs)

    return build


@pytest.fixture
def isolated_home(monkeypatch, tmp_path: Path) -> Path:
    """Point Path.home, "~" expansion and the working directory at empty directories.

    The command line entry point reads ~/.letify and ./.letify, so a test that goes
    through it has to be moved off the developer's own files. HOME is set as well as
    Path.home, because Path.expanduser reads the environment rather than Path.home, and
    login expands the default key path "~/.ssh/id_letify". Without it a test would see
    the developer's own key and behave differently on a machine that has none.
    """
    home = tmp_path / "home"
    (home / ".letify").mkdir(parents=True)
    project = tmp_path / "project"
    (project / ".letify").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(project)
    return project


def provider_of(cls, alias: str = "p", **options: Any):
    """Build one provider directly from options, with no configuration file."""
    return cls(ProviderConfig(alias, cls.kind, dict(options), 0))


class LeasingLocal(Local):
    """A local provider whose session is treated as able to outlive this process.

    The real Local needs no lease, because a subprocess dies with its parent. This
    subclass exists so the boot sequence the spec describes, arming the lease included,
    can be exercised over a real worker.
    """

    needs_lease = True


class PreparingLocal(Local):
    """A local provider that installs the declared environment like a remote one does.

    The real Local skips installation because this machine already runs in its
    environment. This subclass exists so the environment cache path in Runtime can be
    exercised over a real worker instead of needing a remote machine.
    """

    remote_env = True


class NoBlake3Local(Local):
    """A local provider whose worker cannot import blake3, like a runtime venv without letify.

    The worker is the real one behind the real framed protocol. Hiding the module before
    the bootstrap runs makes it hash with blake2b, which the client then has to follow.
    """

    def open_channel(self, runtime: Any) -> Any:
        from letify.protocol.worker import BOOTSTRAP
        from letify.runtime.channel import PersistentChannel

        hide = "import sys; sys.modules['blake3'] = None\n"
        return PersistentChannel([sys.executable, "-u", "-c", hide + BOOTSTRAP], name=runtime.name)


@pytest.fixture
def uv_project(tmp_path: Path, monkeypatch) -> Path:
    """A tiny uv project that depends on this checkout of letify, locked with the real uv.

    The working directory moves into it, so a default ``Env()`` names its ``uv.lock``. The
    runtime's project root moves under the test's temporary directory, so a sync through
    PreparingLocal never writes into the developer's home.
    """
    import shutil

    from letify.runtime import bootstrap

    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover - every machine that runs this suite has uv
        pytest.skip("uv is not installed on this machine")
    project = tmp_path / "uv-project"
    project.mkdir()
    checkout = Path(letify.__file__).resolve().parent.parent
    (project / "pyproject.toml").write_text(
        "[project]\n"
        'name = "tiny"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["letify"]\n'
        "\n"
        "[tool.uv.sources]\n"
        f"letify = {{ path = {checkout.as_posix()!r} }}\n",
        encoding="utf-8",
    )
    locked = subprocess.run([uv, "lock", "--offline"], cwd=project, capture_output=True, text=True)
    if locked.returncode != 0:  # pragma: no cover - only a machine with a cold uv cache
        subprocess.run([uv, "lock"], cwd=project, capture_output=True, check=True)
    monkeypatch.chdir(project)
    monkeypatch.setattr(bootstrap, "DEFAULT_WORKSPACE_ROOT", str(tmp_path / "runtime-workspace"))
    # The automatic environment archive store lives under the user's own ~/.letify, so
    # without this an archive a previous run published is found and a test that expects a
    # sync sees "archive" instead. Spec "Materializing into a runtime".
    from letify import paths

    store = tmp_path / "storage"
    monkeypatch.setattr(paths, "storage_cache_directory", lambda name: store / name)
    return project


# -- subprocess ----------------------------------------------------------------


@dataclass
class FakeCompleted:
    """Stands in for subprocess.CompletedProcess where the command needs a live account."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class RunRecorder:
    """Answers subprocess.run with canned output and remembers what was asked.

    ``result`` may be a callable taking the command, for a module that runs more than
    one distinct command.
    """

    result: Any = field(default_factory=FakeCompleted)
    error: BaseException | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, command, **kwargs: Any) -> FakeCompleted:
        self.calls.append({"command": list(command), **kwargs})
        if self.error is not None:
            raise self.error
        if callable(self.result):
            return self.result(list(command))
        return self.result

    @property
    def command(self) -> list[str]:
        return self.calls[-1]["command"]

    @property
    def commands(self) -> list[list[str]]:
        return [call["command"] for call in self.calls]


@pytest.fixture
def patch_run(monkeypatch):
    """Replace subprocess.run inside one module and return the recorder."""

    def patch(module: Any, **kwargs: Any) -> RunRecorder:
        recorder = RunRecorder(**kwargs)
        monkeypatch.setattr(module.subprocess, "run", recorder)
        if module.__name__ == "letify.providers.colab":
            monkeypatch.setattr(
                module.subprocess,
                "Popen",
                lambda command, **options: FakeProcess(
                    list(command), ["LETIFY-KEEP-ALIVE ready\n"], **options
                ),
            )
        return recorder

    return patch


@pytest.fixture(autouse=True)
def no_uv_variable(monkeypatch):
    """`uv run pytest` sets UV, which would let the real uv answer where tests patch PATH."""
    monkeypatch.delenv("UV", raising=False)


@pytest.fixture(autouse=True)
def no_project_environment(monkeypatch):
    """Hide the virtual environment the suite runs in from the tool lookup.

    Spec "Installing external tools" links tools into the project environment, which for
    the suite would be the repository's own .venv. A test that needs one creates it.
    """
    from letify import install

    monkeypatch.setattr(install, "project_environment", lambda: None)
    # Automatic install would reach GitHub; a test that installs serves a release locally.
    monkeypatch.setenv("LETIFY_AUTO_INSTALL", "0")


@pytest.fixture
def patch_which(monkeypatch):
    """Decide what is on PATH for one module, without touching the real PATH."""

    def patch(module: Any, present: Any = True) -> None:
        def which(name: str) -> str | None:
            if callable(present):
                return present(name)
            if present is True:
                return f"/usr/bin/{name}"
            if present is False:
                return None
            return f"/usr/bin/{name}" if name in present else None

        monkeypatch.setattr(module.shutil, "which", which)

    return patch


@pytest.fixture
def no_module(monkeypatch):
    """Make an optional dependency look absent, so its import raises ImportError."""

    def hide(*names: str) -> None:
        for name in names:
            monkeypatch.setitem(sys.modules, name, None)

    return hide


# -- device inventory ----------------------------------------------------------


@pytest.fixture
def reserving(monkeypatch):
    """A provider with a declared device table and nothing else it needs to connect.

    Local rather than a remote kind, because reservation is bookkeeping over a declared
    inventory and has no transport in it.
    """

    def build(**table: dict[str, Any]):
        provider = provider_of(Local, "box", devices=dict(table))
        # Nothing on the developer's own machine may decide the answer, so no card is taken
        # unless a test says it is. patch_smi is how a test says so.
        monkeypatch.setattr(telemetry, "busy_indices", lambda **kwargs: ())
        # The local provider reads the machine's real cards otherwise, and a test must not
        # depend on what is plugged into the developer's laptop.
        monkeypatch.setattr(
            provider,
            "discover",
            lambda: {name: Instance(provider, gpu=name) for name in provider.inventory},
        )
        return provider

    return build


@pytest.fixture
def live(request):
    """Start a session through the pool, for a test whose subject is the runtime itself.

    Nothing public hands out a session, and these tests are not the surface a user writes
    against: they are about what a live runtime does. The pool is held for the rest of the
    test, which is what let.keep_alive() does, so the session is still there to be used.
    """

    def start(let, instance, env: Any = None, volumes: Any = ()) -> Any:
        let.pool.hold()
        request.addfinalizer(let.pool.unhold)
        runtime = let.pool.acquire(instance, env or Env(), volumes)
        let.pool.release(runtime)
        return runtime

    return start


@pytest.fixture
def one_card_cpu(monkeypatch) -> letify.Instance:
    """A remote CPU shape on a provider whose inventory holds exactly one of it."""
    provider = provider_of(Local, "box", devices={"cpu": {"count": 1}})
    monkeypatch.setattr(telemetry, "busy_indices", lambda **kwargs: ())
    return Instance(provider, gpu=None)._placed("remote")


@pytest.fixture
def patch_smi(monkeypatch):
    """Say which device indices another user is computing on, and who owns them."""

    def patch(busy: list[int] | None = None, owners: dict[int, tuple[str, ...]] | None = None):
        taken = set(busy or ())

        def busy_indices(owners_out: dict[int, tuple[str, ...]] | None = None, **kwargs):
            if owners_out is not None:
                owners_out.update(owners or {index: ("someone",) for index in taken})
            return tuple(sorted(taken))

        monkeypatch.setattr(telemetry, "busy_indices", busy_indices)

    return patch


# -- the lease renewal loop ----------------------------------------------------


class RecordedProvider:
    """A provider that answers the runtime state check with whatever it was told to."""

    def __init__(self) -> None:
        from letify.providers.base import RuntimeState

        self.state = RuntimeState()
        self.asked = 0

    def runtime_state(self, runtime: Any) -> Any:
        self.asked += 1
        return self.state


class RenewalRecorder:
    """Counts lease renewals, standing in for a runtime.

    The real worker's lease operation returns only the new deadline, so how many times
    the renewal loop ran cannot be observed through a live session. This records each
    renewal instead, and can be told to start failing to stand for a session that is
    gone.
    """

    name = "recorded-runtime"

    def __init__(self) -> None:
        self.renewals: list[float] = []
        self.fail_after: int | None = None
        #: Number of leading attempts that fail with a transient error, such as a renewal
        #: reply delayed behind a large transfer, before renewals resume.
        self.transient_for: int = 0
        self._attempts = 0
        #: What the provider answers when asked whether the runtime still exists.
        self.provider = RecordedProvider()
        #: Messages the lease declared the runtime lost with.
        self.lost: list[str] = []

    def lose(self, message: str) -> None:
        self.lost.append(message)

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> float:
        self._attempts += 1
        if self._attempts <= self.transient_for:
            raise letify.RuntimeFailure("recorded-runtime: the call exceeded 60s")
        if self.fail_after is not None and len(self.renewals) >= self.fail_after:
            raise letify.RuntimeLost("the session is gone")
        self.renewals.append(float(payload["grace"]))
        return 0.0


@pytest.fixture
def renewal_recorder() -> RenewalRecorder:
    return RenewalRecorder()


# -- the Elice HTTP API --------------------------------------------------------


@dataclass
class FakeResponse:
    """One canned answer: a status, and a body sent as JSON unless ``text`` is given."""

    status_code: int = 200
    body: Any = None
    text: str = ""


class FakeEliceServer:
    """The Elice Cloud API on loopback, answering what a test tells it to.

    A real HTTP server, so the standard library client the provider uses is the one under
    test. The API lives under ``/api`` as it does on the portal. Each request is recorded
    with its method, path below ``/api``, query parameters (None when there are none),
    JSON body and Authorization header.
    """

    def __init__(self) -> None:
        import http.server
        import threading

        self.requests: list[dict[str, Any]] = []
        self.responses: dict[tuple[str, str], FakeResponse] = {}
        self.default = FakeResponse(200, {})
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                owner._handle(self, "GET")

            def do_POST(self) -> None:
                owner._handle(self, "POST")

            def do_DELETE(self) -> None:
                owner._handle(self, "DELETE")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self._server.server_address[1]}/api"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def answer(self, method: str, path: str, response: FakeResponse) -> None:
        self.responses[(method, path)] = response

    @property
    def last(self) -> dict[str, Any]:
        return self.requests[-1]

    def _handle(self, handler: Any, method: str) -> None:
        import json
        import urllib.parse

        parsed = urllib.parse.urlsplit(handler.path)
        path = parsed.path.removeprefix("/api")
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
        self.requests.append(
            {
                "method": method,
                "path": path,
                "params": dict(urllib.parse.parse_qsl(parsed.query)) or None,
                "json": _json_or_none(body),
                "form": dict(urllib.parse.parse_qsl(body.decode(errors="replace"))) or None,
                "authorization": handler.headers.get("Authorization"),
                "org": handler.headers.get("x-elice-org-name-short"),
            }
        )
        response = self.responses.get((method, path), self.default)
        if response.text:
            payload, kind = response.text.encode(), "text/html"
        else:
            payload, kind = json.dumps(response.body).encode(), "application/json"
        handler.send_response(response.status_code)
        handler.send_header("Content-Type", kind)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)


def _json_or_none(body: bytes) -> Any:
    """A request body decoded as JSON, or None when it is empty or form encoded."""
    import json

    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


@pytest.fixture
def fake_elice():
    server = FakeEliceServer()
    yield server
    server.close()


#: The stand-in for Elice's ``eci`` binary. It answers from a JSON state file and appends
#: every call, with its arguments and the ECI_* variables it saw, to that file.
FAKE_ECI_SOURCE = r"""
import json, os, sys

path = os.environ["FAKE_ECI_STATE"]
with open(path, encoding="utf-8") as handle:
    state = json.load(handle)
argv = sys.argv[1:]
state["calls"].append(
    {"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith("ECI_")}}
)

def save():
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)

def done(body=None, code=0, err=""):
    save()
    if body is not None:
        print(json.dumps(body))
    if err:
        print(err, file=sys.stderr)
    sys.exit(code)

def flag(name):
    return argv[argv.index(name) + 1] if name in argv else None

words = [a for a in argv if not a.startswith("--")]
joined = " ".join(words)
for prefix, failure in state.get("fail", {}).items():
    if joined.startswith(prefix):
        done(code=failure.get("code", 1), err=failure.get("stderr", "failed"))
if os.environ.get("ECI_API_TOKEN") != state["token"]:
    done(code=1, err="Error: 401 Unauthorized: invalid token")

def vm(name):
    return next((v for v in state["vms"] if name in (v["name"], v["id"])), None)

if joined.startswith("config verify"):
    done(code=0)
if joined.startswith("zone list"):
    done(state["zones"])
if joined.startswith("instance-type list"):
    if "--activated" in argv:
        done(code=2, err="Error: No such option: --activated")
    done(state["instance_types"])
if joined.startswith("pricing list"):
    done(state["pricing"])
if joined.startswith("org info"):
    done(state["org"])
if joined.startswith("compute vm list"):
    done(state["vms"])
if joined.startswith("compute vm get"):
    found = vm(words[3])
    queued = state.get("transitions", {}).get(words[3])
    if found and queued:
        found["status"] = queued.pop(0)
    done(found) if found else done(code=1, err=f"Error: virtual machine {words[3]} not found")
if joined.startswith("compute vm launch"):
    number = len(state["vms"]) + 1
    record = {
        "id": f"vm-{number}",
        "name": flag("--name"),
        "status": "started",
        "instance_type": flag("--instance-type"),
        "pricing_type": flag("--price-type") or "ondemand",
        "public_ip": f"203.0.113.{number}",
    }
    state["vms"].append(record)
    done(record)
if joined.startswith("compute vm start"):
    vm(words[3])["status"] = "started"
    done({})
if joined.startswith("compute vm stop"):
    vm(words[3])["status"] = "idle"
    done({})
if joined.startswith("compute vm delete"):
    if "--cascade" not in argv or "-y" not in argv:
        done(code=1, err="aborted: stdin is not a TTY")
    state["vms"] = [v for v in state["vms"] if v["name"] != words[3]]
    done({"status": "deleted"})
done(code=2, err=f"fake eci does not know: {joined}")
"""


class FakeEci:
    """Elice's ``eci`` command on PATH, answering from a state a test sets.

    A real executable, so the subprocess call, the environment letify builds and the JSON
    it parses are all exercised. Only the service behind the command is faked.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.state_path = directory / "state.json"
        binary = directory / "eci"
        binary.write_text(f"#!{sys.executable}\n{FAKE_ECI_SOURCE}", encoding="utf-8")
        binary.chmod(0o755)
        self.write(
            {
                "token": "token-1",
                "zones": [{"id": "zone-1", "name": "central-01-a"}],
                "vms": [],
                "instance_types": [
                    {"id": "it-cpu", "name": "C-4", "cpu_vcore": 4, "devices": []},
                    {"id": "it-cpu-small", "name": "C-2", "cpu_vcore": 2, "devices": []},
                    {
                        "id": "it-a100",
                        "name": "G-A100-1",
                        "cpu_vcore": 16,
                        "devices": ["nvidia_a100_80gb_pcie"],
                        "activated": True,
                    },
                    {
                        "id": "it-a100x2",
                        "name": "G-A100-2",
                        "cpu_vcore": 32,
                        "devices": ["nvidia_a100_80gb_pcie", "nvidia_a100_80gb_pcie"],
                        "activated": True,
                    },
                    {
                        "id": "it-h100-retired",
                        "name": "G-NHHS-80-OLD",
                        "cpu_vcore": 24,
                        "devices": ["nvidia_h100_80gb_sxm"],
                        "activated": False,
                    },
                ],
                "pricing": [
                    {"name": "G-A100-1", "pricing_type": "ondemand", "price_per_hour": "2500"},
                    {"name": "G-A100-1", "pricing_type": "spot", "price_per_hour": "900"},
                ],
                "org": {"name_short": "lab", "resource_quota": {"compute": {"devices": 8}}},
                "fail": {},
                "calls": [],
            }
        )

    def read(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def set(self, **values: Any) -> None:
        state = self.read()
        state.update(values)
        self.write(state)

    def machine(self, name: str, **fields: Any) -> None:
        """Change one listed machine, as Elice would behind letify's back."""
        state = self.read()
        for record in state["vms"]:
            if record["name"] == name:
                record.update(fields)
        self.write(state)

    def remove(self, name: str) -> None:
        state = self.read()
        state["vms"] = [record for record in state["vms"] if record["name"] != name]
        self.write(state)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.read()["calls"]

    def commands(self) -> list[str]:
        """Each call's words, without flags or their values, joined by spaces."""
        switches = {"--wait", "--no-spec"}
        found = []
        for call in self.calls:
            words, skip = [], False
            for arg in call["argv"]:
                if skip:
                    skip = False
                elif arg.startswith("--"):
                    skip = arg not in switches
                else:
                    words.append(arg)
            found.append(" ".join(words))
        return found


@pytest.fixture
def fake_eci(tmp_path: Path, monkeypatch) -> FakeEci:
    directory = tmp_path / "fake-eci-bin"
    directory.mkdir()
    fake = FakeEci(directory)
    monkeypatch.setenv("FAKE_ECI_STATE", str(fake.state_path))
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ.get('PATH', '')}")
    return fake


@pytest.fixture
def fake_google():
    """Google's OAuth token endpoint and Colab's ``ccu-info`` on loopback.

    The same recording server as the Elice stand-in, because both are plain HTTP answered
    from a table. Paths are below ``/api``.
    """
    server = FakeEliceServer()
    yield server
    server.close()


# -- Modal ---------------------------------------------------------------------

#: The standard library stand-in for the Modal adapter, speaking the same JSON lines.
FAKE_MODAL_ADAPTER = Path(__file__).with_name("fake_modal_adapter.py")


class FakeModalAdapter:
    """Where the stand-in adapter records what it was asked, and how to make it misbehave."""

    def __init__(self, monkeypatch, state: Path):
        self.state = state
        self._monkeypatch = monkeypatch

    def requests(self, op: str | None = None) -> list[dict[str, Any]]:
        log = self.state / "requests.jsonl"
        if not log.is_file():
            return []
        found = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        return [r for r in found if op is None or r["op"] == op]

    def env(self) -> dict[str, str]:
        return json.loads((self.state / "env.json").read_text(encoding="utf-8"))

    def app_events(self) -> list[list[str]]:
        """``run_start`` and ``run_stop`` with the app name, in the order they happened."""
        log = self.state / "apps.jsonl"
        if not log.is_file():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    def sandbox_pids(self) -> list[int]:
        """The local process id behind each sandbox the stand-in started, in order."""
        log = self.state / "sandboxes.jsonl"
        if not log.is_file():
            return []
        return [json.loads(line)[1] for line in log.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def alive(pid: int) -> bool:
        """Whether a sandbox process still runs. A zombie waiting to be reaped has ended."""
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return False
        return stat.rsplit(")", 1)[1].split()[0] != "Z"

    def end_sandbox(self, sandbox: str, exit_code: int) -> None:
        """Make ``poll`` answer ``exit_code`` for a sandbox, as Modal does once it ended."""
        (self.state / f"ended-{sandbox}").write_text(str(exit_code), encoding="utf-8")

    def fail(self, *ops: str) -> None:
        self._monkeypatch.setenv("FAKE_MODAL_FAIL", ",".join(ops))

    def exit_on(self, *ops: str) -> None:
        self._monkeypatch.setenv("FAKE_MODAL_EXIT", ",".join(ops))

    def tunnel_to(self, host: str, port: int, *, tls: bool) -> None:
        """Answer ``tunnel`` with this address instead of the sandbox's own port."""
        self._monkeypatch.setenv(
            "FAKE_MODAL_TUNNEL", json.dumps({"host": host, "port": port, "tls": tls})
        )


class TlsProxy:
    """A local TLS-terminating proxy, as a Modal encrypted port is, in front of a plain port.

    One thread relays each connection with non-blocking sockets, so the proxy never uses its
    own TLS object from two threads.
    """

    def __init__(self, cert: Path, key: Path):
        import ssl

        self.cert = cert
        self._server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._server_context.load_cert_chain(str(cert), str(key))
        #: A client context that trusts the proxy's certificate for the name ``localhost``.
        self.context = ssl.create_default_context(cafile=str(cert))

    def start(self, target_port: int) -> int:
        """Listen on a free port, relaying each connection to ``target_port``, and return it."""
        import threading

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        threading.Thread(target=self._accept, args=(server, target_port), daemon=True).start()
        return server.getsockname()[1]

    def _accept(self, server: socket.socket, target_port: int) -> None:
        import threading

        while True:
            try:
                raw, _ = server.accept()
                tls = self._server_context.wrap_socket(raw, server_side=True)
                upstream = socket.create_connection(("127.0.0.1", target_port))
            except OSError:
                continue
            threading.Thread(target=_relay, args=(tls, upstream), daemon=True).start()


def _relay(tls: Any, upstream: socket.socket) -> None:
    import selectors
    import ssl

    selector = selectors.DefaultSelector()
    tls.setblocking(False)
    upstream.setblocking(False)
    selector.register(tls, selectors.EVENT_READ)
    selector.register(upstream, selectors.EVENT_READ)
    to_upstream = bytearray()
    to_tls = bytearray()
    try:
        while True:
            for key, _ in selector.select(0.01):
                sock = key.fileobj
                try:
                    data = sock.recv(1 << 16)
                except (ssl.SSLWantReadError, ssl.SSLWantWriteError, BlockingIOError):
                    continue
                if not data:
                    return
                (to_upstream if sock is tls else to_tls).extend(data)
            while tls.pending():
                to_upstream.extend(tls.recv(tls.pending()))
            for buffer, sock in ((to_upstream, upstream), (to_tls, tls)):
                while buffer:
                    try:
                        sent = sock.send(buffer[: 1 << 16])
                    except (ssl.SSLWantReadError, ssl.SSLWantWriteError, BlockingIOError):
                        break
                    del buffer[:sent]
    except OSError:
        return
    finally:
        for sock in (tls, upstream):
            with contextlib.suppress(OSError):
                sock.close()


@pytest.fixture
def tls_proxy(tmp_path: Path) -> TlsProxy:
    """A TLS proxy with a certificate made for this test only, never stored in the repository."""
    import shutil
    import subprocess

    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is needed to make a test certificate")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return TlsProxy(cert, key)


@pytest.fixture
def fake_modal(monkeypatch, tmp_path: Path) -> FakeModalAdapter:
    """Run the stand-in adapter wherever letify would run the real one through uv.

    Only the command is replaced. The process, the pipes, the JSON lines and the
    environment letify builds for the account are all real.
    """
    from letify import tools

    state = tmp_path / "fake-modal"
    monkeypatch.setenv("FAKE_MODAL_STATE", str(state))
    monkeypatch.setattr(tools, "find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(
        tools, "modal_adapter_command", lambda uv: [sys.executable, str(FAKE_MODAL_ADAPTER)]
    )
    from letify.providers import modal as modal_module

    # The stand-in sandbox is a local process, so its data port is one free on this machine.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        monkeypatch.setattr(modal_module, "DATA_PORT", probe.getsockname()[1])
    return FakeModalAdapter(monkeypatch, state)


# -- cloud object stores -------------------------------------------------------


class FakeGCSServer:
    """The parts of the Cloud Storage JSON API and Google's token endpoints letify uses.

    A real HTTP server on loopback, so the standard library client letify ships is the one
    under test, from this process and from a worker process alike. Objects live in
    ``objects`` keyed by object name; every request is appended to ``requests``.
    """

    #: Listing pages are this short so pagination is exercised with a handful of objects.
    PAGE_SIZE = 2

    def __init__(self, token: str = "token-1"):
        import http.server
        import threading

        self.token = token
        self.bucket = "study-bucket"
        self.objects: dict[str, bytes] = {}
        self.requests: list[dict[str, Any]] = []
        self.refresh_grants: list[dict[str, str]] = []
        self.exchanges: list[dict[str, str]] = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                owner._handle(self, "GET")

            def do_POST(self) -> None:
                owner._handle(self, "POST")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def accepts(self, authorization: str | None) -> bool:
        return authorization in (f"Bearer {self.token}", f"Bearer down-{self.token}")

    def _handle(self, handler: Any, method: str) -> None:
        import json
        import urllib.parse

        parsed = urllib.parse.urlsplit(handler.path)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
        authorization = handler.headers.get("Authorization")
        self.requests.append(
            {
                "method": method,
                "path": parsed.path,
                "query": query,
                "authorization": authorization,
            }
        )

        def answer(status: int, payload: bytes = b"", kind: str = "application/json") -> None:
            handler.send_response(status)
            handler.send_header("Content-Type", kind)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        if parsed.path == "/token" and method == "POST":
            form = dict(urllib.parse.parse_qsl(body.decode()))
            self.refresh_grants.append(form)
            reply = {"access_token": self.token, "expires_in": 3599, "token_type": "Bearer"}
            return answer(200, json.dumps(reply).encode())
        if parsed.path == "/v1/token" and method == "POST":
            form = dict(urllib.parse.parse_qsl(body.decode()))
            self.exchanges.append(form)
            reply = {"access_token": f"down-{form.get('subject_token')}", "expires_in": 3599}
            return answer(200, json.dumps(reply).encode())

        if not self.accepts(authorization):
            return answer(401, b'{"error": {"message": "unauthenticated"}}')

        listing = f"/storage/v1/b/{self.bucket}/o"
        upload = f"/upload/storage/v1/b/{self.bucket}/o"
        if method == "POST" and parsed.path == upload:
            self.objects[query["name"]] = body
            return answer(200, json.dumps({"name": query["name"]}).encode())
        if method == "GET" and parsed.path == listing:
            names = sorted(k for k in self.objects if k.startswith(query.get("prefix", "")))
            start = int(query.get("pageToken") or 0)
            page = names[start : start + self.PAGE_SIZE]
            reply: dict[str, Any] = {"items": [{"name": name} for name in page]}
            if start + self.PAGE_SIZE < len(names):
                reply["nextPageToken"] = str(start + self.PAGE_SIZE)
            return answer(200, json.dumps(reply).encode())
        if method == "GET" and parsed.path.startswith(listing + "/"):
            name = urllib.parse.unquote(parsed.path[len(listing) + 1 :])
            if name not in self.objects:
                return answer(404, b'{"error": {"message": "not found"}}')
            if query.get("alt") == "media":
                return answer(200, self.objects[name], "application/octet-stream")
            return answer(200, json.dumps({"name": name}).encode())
        return answer(400, b'{"error": {"message": "unexpected request"}}')

    def downloads(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["query"].get("alt") == "media"]


@pytest.fixture
def fake_gcs(monkeypatch):
    """A Cloud Storage endpoint on loopback, since a real bucket needs an account.

    ``GOOGLE_OAUTH_ACCESS_TOKEN`` carries the token it accepts, which is the first rule of
    the login lookup, so a test that is not about the lookup needs no login.
    """
    server = FakeGCSServer()
    monkeypatch.setenv("GOOGLE_OAUTH_ACCESS_TOKEN", server.token)
    yield server
    server.close()


class FakeDriveServer:
    """The parts of the Google Drive v3 API and Google's token endpoint letify uses.

    A real HTTP server on loopback, so the standard library client letify ships is the one
    under test, from this process and from a worker process alike. Files live in ``files``
    keyed by id; every request is appended to ``requests``, with its lower cased headers.
    A request without the quota project header is answered ``403 SERVICE_DISABLED``, as the
    real API answers it. A resumable session accepts its pieces without a credential, as
    the real one does, because the session URL authorizes its own upload.
    """

    def __init__(self, token: str = "drive-token-1", project: str = "drive-project"):
        import http.server
        import threading

        self.token = token
        self.project = project
        self.files: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.refresh_grants: list[dict[str, str]] = []
        self.fail_with: int | None = None
        #: When set, a piece of a resumable upload waits for this event before it is taken.
        self.hold_uploads: threading.Event | None = None
        #: When set, a DELETE waits for this event before it is answered.
        self.hold_deletes: threading.Event | None = None
        self._count = 0
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                owner._handle(self, "GET")

            def do_POST(self) -> None:
                owner._handle(self, "POST")

            def do_PUT(self) -> None:
                owner._handle(self, "PUT")

            def do_PATCH(self) -> None:
                owner._handle(self, "PATCH")

            def do_DELETE(self) -> None:
                owner._handle(self, "DELETE")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def login(self, alias: str, scopes: list[str] | None = None) -> Path:
        """Write the stored Google login of one account, pointing its token endpoint here."""
        import json

        from letify import paths

        directory = paths.account_directory(alias)
        directory.mkdir(parents=True, exist_ok=True)
        stored = {
            "client_id": "client-1",
            "client_secret": "secret-1",
            "refresh_token": "refresh-1",
            "token_uri": f"{self.endpoint}/token",
            "scopes": scopes or ["https://www.googleapis.com/auth/drive.file"],
        }
        path = directory / "token.json"
        path.write_text(json.dumps(stored))
        return path

    def blobs(self) -> dict[str, bytes]:
        """Content of every file that is not a folder and not a ref, by name."""
        return {
            f["name"]: f["content"]
            for f in self.files.values()
            if not f["folder"] and not f["name"].startswith("ref ")
        }

    def refs(self) -> dict[str, str]:
        return {
            f["name"][4:]: f["content"].decode()
            for f in self.files.values()
            if f["name"].startswith("ref ")
        }

    def downloads(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["query"].get("alt") == "media"]

    def uploads(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "PUT"]

    def _new_id(self) -> str:
        self._count += 1
        return f"id{self._count}"

    def _select(self, expression: str) -> list[dict[str, Any]]:
        import re

        found = [f for f in self.files.values() if not f["trashed"]]
        for match in re.finditer(r"name = '((?:[^'\\]|\\.)*)'", expression):
            wanted = re.sub(r"\\(.)", r"\1", match.group(1))
            found = [f for f in found if f["name"] == wanted]
        for match in re.finditer(r"'([^']+)' in parents", expression):
            found = [f for f in found if f["parent"] == match.group(1)]
        if "mimeType = 'application/vnd.google-apps.folder'" in expression:
            found = [f for f in found if f["folder"]]
        return sorted(found, key=lambda f: f["id"])

    def _handle(self, handler: Any, method: str) -> None:
        import hashlib
        import json
        import urllib.parse

        parsed = urllib.parse.urlsplit(handler.path)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
        headers = {k.lower(): v for k, v in handler.headers.items()}
        self.requests.append(
            {
                "method": method,
                "path": parsed.path,
                "query": query,
                "headers": headers,
                "length": len(body),
            }
        )

        def answer(
            status: int,
            payload: bytes = b"",
            extra: dict[str, str] | None = None,
            kind: str = "application/json",
        ) -> None:
            handler.send_response(status)
            handler.send_header("Content-Type", kind)
            handler.send_header("Content-Length", str(len(payload)))
            for key, value in (extra or {}).items():
                handler.send_header(key, value)
            handler.end_headers()
            handler.wfile.write(payload)

        def reply(value: Any, status: int = 200) -> None:
            answer(status, json.dumps(value).encode())

        if parsed.path == "/token" and method == "POST":
            self.refresh_grants.append(dict(urllib.parse.parse_qsl(body.decode())))
            return reply({"access_token": self.token, "expires_in": 3599})
        if self.fail_with is not None:
            return answer(self.fail_with, b'{"error": {"message": "injected"}}')
        if headers.get("x-goog-user-project") != self.project:
            return answer(403, b'{"error": {"status": "SERVICE_DISABLED"}}')

        session_prefix = "/upload/session/"
        if parsed.path.startswith(session_prefix) and method == "PUT":
            if self.hold_uploads is not None:
                self.hold_uploads.wait(timeout=20)
            session = self.sessions[parsed.path[len(session_prefix) :]]
            start = int(headers["content-range"].split(" ")[1].split("-")[0])
            if start != len(session["body"]):
                return answer(400, b'{"error": {"message": "out of order"}}')
            session["body"] += body
            if len(session["body"]) < session["size"]:
                return answer(308, extra={"Range": f"bytes=0-{len(session['body']) - 1}"})
            identifier = self._new_id()
            self.files[identifier] = {
                **session["meta"],
                "id": identifier,
                "content": session["body"],
                "folder": False,
                "trashed": False,
            }
            return reply({"id": identifier, "name": session["meta"]["name"]})

        if headers.get("authorization") != f"Bearer {self.token}":
            return answer(401, b'{"error": {"message": "unauthenticated"}}')

        if method == "GET" and parsed.path == "/drive/v3/files":
            matches = self._select(query.get("q", ""))
            start = int(query.get("pageToken") or 0)
            size = int(query.get("pageSize") or 2)
            page = matches[start : start + size]
            listed = []
            for f in page:
                entry = {"id": f["id"], "name": f["name"]}
                if "md5Checksum" in query.get("fields", "") and not f["folder"]:
                    entry["md5Checksum"] = hashlib.md5(f["content"]).hexdigest()
                listed.append(entry)
            result: dict[str, Any] = {"files": listed}
            if start + size < len(matches):
                result["nextPageToken"] = str(start + size)
            return reply(result)
        if method == "POST" and parsed.path == "/drive/v3/files":
            meta = json.loads(body)
            identifier = self._new_id()
            self.files[identifier] = {
                "id": identifier,
                "name": meta["name"],
                "content": b"",
                "trashed": False,
                "parent": (meta.get("parents") or ["root"])[0],
                "folder": meta.get("mimeType") == "application/vnd.google-apps.folder",
            }
            return reply({"id": identifier, "name": meta["name"]})
        if method == "POST" and parsed.path == "/upload/drive/v3/files":
            if query.get("uploadType") == "resumable":
                meta = json.loads(body)
                meta.setdefault("parent", (meta.get("parents") or ["root"])[0])
                number = str(len(self.sessions) + 1)
                self.sessions[number] = {
                    "meta": meta,
                    "body": b"",
                    "size": int(headers["x-upload-content-length"]),
                }
                return answer(200, extra={"Location": f"{self.endpoint}{session_prefix}{number}"})
            _, _, rest = body.partition(b"\r\n\r\n")
            boundary = headers["content-type"].split("boundary=")[1].encode()
            meta_text, _, tail = rest.partition(b"\r\n--" + boundary + b"\r\n")
            meta = json.loads(meta_text)
            content = tail.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--" + boundary, 1)[0]
            identifier = self._new_id()
            self.files[identifier] = {
                "id": identifier,
                "name": meta["name"],
                "content": content,
                "trashed": False,
                "parent": (meta.get("parents") or ["root"])[0],
                "folder": False,
            }
            return reply({"id": identifier, "name": meta["name"]})
        if method == "PATCH" and parsed.path.startswith("/upload/drive/v3/files/"):
            identifier = parsed.path.rsplit("/", 1)[-1]
            self.files[identifier]["content"] = body
            return reply({"id": identifier})
        if method == "DELETE" and parsed.path.startswith("/drive/v3/files/"):
            if self.hold_deletes is not None:
                self.hold_deletes.wait(timeout=20)
            if self.files.pop(parsed.path.rsplit("/", 1)[-1], None) is None:
                return answer(404, b'{"error": {"message": "not found"}}')
            return answer(204)
        if method == "GET" and parsed.path.startswith("/drive/v3/files/"):
            found = self.files.get(parsed.path.rsplit("/", 1)[-1])
            if found is None:
                return answer(404, b'{"error": {"message": "not found"}}')
            if query.get("alt") == "media":
                return answer(200, found["content"], kind="application/octet-stream")
            return reply({"id": found["id"], "name": found["name"]})
        return answer(400, b'{"error": {"message": "unexpected request"}}')


@pytest.fixture
def fake_drive(isolated_home):
    """A Google Drive endpoint on loopback, since a real Drive needs an account.

    The login of the account ``drive-login`` is written under the isolated home, with its
    token endpoint pointing at the same server.
    """
    server = FakeDriveServer()
    server.login("drive-login")
    yield server
    server.close()


# -- the Colab runtime proxy's Jupyter file API --------------------------------


class FakeJupyterServer:
    """The Jupyter contents API and ``/files`` handler a Colab runtime proxy exposes.

    A real HTTP server on loopback whose contents root is ``/`` of this machine, as it is on
    a Colab VM, so a test names absolute paths inside its temporary directory. Chunked
    uploads follow Jupyter's large file manager: chunk ``1`` creates, any other appends.
    """

    def __init__(self, token: str = "proxy-token-1"):
        import http.server
        import threading

        self.token = token
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                owner._handle(self, "GET")

            def do_PUT(self) -> None:
                owner._handle(self, "PUT")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handle(self, handler: Any, method: str) -> None:
        import json
        import urllib.parse

        parsed = urllib.parse.urlsplit(handler.path)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else b""
        with self._lock:
            self.requests.append(
                {
                    "method": method,
                    "path": urllib.parse.unquote(parsed.path),
                    "query": query,
                    "range": handler.headers.get("Range"),
                    "header_token": handler.headers.get("X-Colab-Runtime-Proxy-Token"),
                }
            )

        def answer(status: int, payload: bytes = b"", extra: dict[str, str] | None = None) -> None:
            handler.send_response(status)
            for key, value in (extra or {}).items():
                handler.send_header(key, value)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        if query.get("colab-runtime-proxy-token") != self.token or query.get("authuser") != "0":
            return answer(403, b'{"message": "forbidden"}')

        path = urllib.parse.unquote(parsed.path)
        if path.startswith("/api/contents/"):
            local = Path("/") / path[len("/api/contents/") :]
            if method == "PUT":
                model = json.loads(body)
                content = base64.b64decode(model["content"])
                chunk = model.get("chunk")
                mode = "ab" if chunk not in (None, 1) else "wb"
                with open(local, mode) as handle:
                    handle.write(content)
                return answer(200, json.dumps({"path": str(local), "type": "file"}).encode())
            if not local.is_file():
                return answer(404, b'{"message": "no such file"}')
            model = {"path": str(local), "type": "file", "size": local.stat().st_size}
            return answer(200, json.dumps(model).encode())
        if path.startswith("/files/") and method == "GET":
            local = Path("/") / path[len("/files/") :]
            if not local.is_file():
                return answer(404, b"no such file")
            data = local.read_bytes()
            wanted = handler.headers.get("Range")
            if wanted:
                first, _, last = wanted.removeprefix("bytes=").partition("-")
                start, end = int(first), min(int(last), len(data) - 1)
                piece = data[start : end + 1]
                return answer(206, piece, {"Content-Range": f"bytes {start}-{end}/{len(data)}"})
            return answer(200, data)
        return answer(400, b'{"message": "unexpected request"}')

    def puts(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "PUT"]

    def ranges(self) -> list[str]:
        return [r["range"] for r in self.requests if r["path"].startswith("/files/")]


@pytest.fixture
def fake_jupyter():
    server = FakeJupyterServer()
    yield server
    server.close()


def write_colab_session(alias: str, name: str, url: str, token: str) -> Path:
    """Record a session the way ``colab new`` does, in the account's CLI state file."""
    import json

    from letify.config.secrets import account_directory

    path = account_directory(alias) / ".config" / "colab-cli" / "sessions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: {"name": name, "token": token, "url": url, "endpoint": "e-1"}}
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


# -- a one-shot channel that really runs the driver ----------------------------


def local_one_shot_runner(record: list[str] | None = None):
    """A one-shot runner that executes the driver script in a fresh interpreter.

    This is what a provider such as ``colab exec`` offers: run a command, collect its
    output, keep nothing. Running it locally means the driver script, the markers and
    the decoder are all the real ones.
    """

    def run(source: str, timeout: float | None = None) -> str:
        if record is not None:
            record.append(source)
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=timeout or 120,
        )
        return result.stdout

    return run


# -- the connection pipeline ---------------------------------------------------


class StunServer:
    """A STUN binding responder over TCP on loopback.

    A public STUN server needs the network. This one answers with the address the request
    came from, which is what a real server reports, so the client parser, the bound port and
    the reuse options all run for real.
    """

    def __init__(self) -> None:
        import socket
        import threading

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.address = self.sock.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        import struct

        while True:
            try:
                conn, peer = self.sock.accept()
            except OSError:
                return
            with conn:
                header = conn.recv(20)
                if len(header) < 20:
                    continue
                _, _, cookie = struct.unpack("!HHI", header[:8])
                transaction = header[8:20]
                port = peer[1] ^ (cookie >> 16)
                ip = struct.unpack("!I", bytes(int(p) for p in peer[0].split(".")))[0] ^ cookie
                value = struct.pack("!BBHI", 0, 1, port, ip)
                attribute = struct.pack("!HH", 0x0020, len(value)) + value
                conn.sendall(
                    struct.pack("!HHI", 0x0101, len(attribute), cookie) + transaction + attribute
                )

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def stun_server():
    server = StunServer()
    yield server
    server.close()


class UdpStunServers:
    """Three STUN servers on loopback that report the address a client was seen at.

    That is a NAT that keeps ports, which two sidecars on this machine can really punch through.
    ``env`` is the value of ``LETIFY_LINK_STUN`` that makes the sidecars ask them.
    """

    def __init__(self, count: int = 3) -> None:
        self.socks = []
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", 0))
            self.socks.append(sock)
            threading.Thread(target=self.serve, args=(sock,), daemon=True).start()

    @property
    def env(self) -> str:
        return ",".join(f"{sock.getsockname()[0]}:{sock.getsockname()[1]}" for sock in self.socks)

    @staticmethod
    def serve(sock: socket.socket) -> None:
        cookie = 0x2112A442
        while True:
            try:
                data, peer = sock.recvfrom(1500)
            except OSError:
                return
            if len(data) < 20:
                continue
            port = peer[1] ^ (cookie >> 16)
            ip = struct.unpack("!I", socket.inet_aton(peer[0]))[0] ^ cookie
            value = struct.pack("!BBHI", 0, 1, port, ip)
            attribute = struct.pack("!HH", 0x0020, len(value)) + value
            sock.sendto(
                struct.pack("!HHI", 0x0101, len(attribute), cookie) + data[8:20] + attribute, peer
            )

    def close(self) -> None:
        for sock in self.socks:
            sock.close()


@pytest.fixture
def udp_stun():
    servers = UdpStunServers()
    yield servers
    servers.close()


class FakeLink:
    """A link with no connection behind it, carrying the probe result a test chose."""

    persistent = True

    def __init__(self, strategy: str, rank: int, result: Any = None, *, probe_error: bool = False):
        self.strategy = strategy
        self.rank = rank
        self.result = result
        self.probe_error = probe_error
        self.closed = False

    def probe_stream(self) -> Any:
        return self if self.result is not None or self.probe_error else None

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        return ["ssh", self.strategy, *([remote_command] if remote_command else [])]

    def close(self) -> None:
        self.closed = True


class FakeProbe:
    """Hands back the result a FakeLink carries, standing in for seconds of transfer."""

    def __init__(self) -> None:
        self.measured: list[str] = []

    def measure(self, stream: Any) -> Any:
        self.measured.append(stream.strategy)
        if stream.probe_error:
            raise OSError("the link dropped during the probe")
        return stream.result


class FakeStrategy:
    """A strategy that connects after a delay, fails, or is not applicable."""

    def __init__(
        self,
        name: str,
        rank: int,
        *,
        result: Any = None,
        delay: float = 0.0,
        error: str | None = None,
        unmet: str | None = None,
        probe_error: bool = False,
        probed: bool = True,
        cancellable: bool = False,
    ):
        import threading

        self.name = name
        self.probed = probed
        self.rank = rank
        self.result = result
        self.delay = delay
        self.error = error
        self.unmet = unmet
        self.probe_error = probe_error
        #: Whether the delay ends early when the pipeline sets the attempt's cancel event.
        self.cancellable = cancellable
        self.cancelled = False
        #: Set when an attempt returns or raises.
        self.ended = threading.Event()
        self.attempts = 0
        self.assumed = 0
        self.links: list[FakeLink] = []

    def needs(self, target: Any) -> str | None:
        return self.unmet

    def _link(self) -> FakeLink:
        link = FakeLink(self.name, self.rank, self.result, probe_error=self.probe_error)
        self.links.append(link)
        return link

    def attempt(self, target: Any, cancel: Any = None) -> FakeLink:
        import time

        from letify.transport import nat

        self.attempts += 1
        try:
            if self.cancellable and cancel is not None:
                if cancel.wait(self.delay):
                    self.cancelled = True
                    raise nat.Cancelled(f"{self.name} was cancelled")
            else:
                time.sleep(self.delay)
            if self.error:
                raise OSError(self.error)
            return self._link()
        finally:
            self.ended.set()

    def assume(self, target: Any) -> FakeLink:
        self.assumed += 1
        return self._link()


class LoopbackRendezvous:
    """A rendezvous whose remote side is a thread on this machine, running the real remote half."""

    def __init__(self, **overrides: Any):
        self.requests: list[dict[str, Any]] = []
        self.overrides = overrides

    def unavailable(self) -> str | None:
        return None

    def exchange(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        import threading

        from letify.transport import nat

        self.requests.append(request)
        answer, continuation = nat.begin({**request, **self.overrides})
        threading.Thread(target=continuation, daemon=True).start()
        return answer


class CannedRendezvous:
    """A rendezvous that answers every request with the same reply, for strategies whose
    remote half needs a live tool such as ssh or tailcat."""

    lead_seconds = 0.0

    def __init__(self, answer: dict[str, Any] | None = None, unavailable: str | None = None):
        self.answer = answer or {}
        self.requests: list[dict[str, Any]] = []
        self._unavailable = unavailable

    def unavailable(self) -> str | None:
        return self._unavailable

    def exchange(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.requests.append(request)
        return dict(self.answer)

    def tailcat_endpoint(self, ssh_port: int, timeout: float) -> tuple[str, int]:
        answer = self.exchange({"kind": "tailcat", "ssh_port": ssh_port}, timeout)
        return answer["address"], ssh_port


class FakeProcess:
    """A Popen stand-in whose output is the lines a real tool prints."""

    def __init__(self, command: list[str], lines: list[str], **kwargs: Any):
        import io

        self.command = command
        self.kwargs = kwargs
        self.stdout = io.StringIO("".join(lines))
        self.stderr = io.StringIO("".join(lines))
        self.stdin = io.StringIO()
        self.written: list[bytes] = []
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def poll(self) -> int | None:
        return None


@pytest.fixture
def patch_popen(monkeypatch):
    """Replace subprocess.Popen inside one module with processes that print given lines."""

    def patch(module: Any, lines: list[str]) -> list[FakeProcess]:
        started: list[FakeProcess] = []

        def popen(command: list[str], **kwargs: Any) -> FakeProcess:
            process = FakeProcess(list(command), lines, **kwargs)
            started.append(process)
            return process

        monkeypatch.setattr(module.subprocess, "Popen", popen)
        return started

    return patch


# -- a Kaggle session, and the cloud that hands out its URL ---------------------


def kaggle_test_cookie(exp_iso: str = "2099-01-01T00:00:00Z") -> str:
    """A cookie of the shape the provider requires, with a far-future expiry."""

    def part(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    header = part({"alg": "none", "typ": "JWT"})
    client_token = f"{header}.{part({'sub': 'irack000', 'exp': exp_iso})}."
    jar = {
        "ka_sessionid": "sid",
        "XSRF-TOKEN": "xtok",
        "build-hash": "bh",
        "CLIENT-TOKEN": client_token,
    }
    return "; ".join(f"{name}={value}" for name, value in jar.items())


class _Reply:
    """A minimal stand-in for the object ``urlopen`` returns, with the JSON body."""

    def __init__(self, payload: dict[str, Any]):
        self._data = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeKaggleCloud:
    """Stands in for every live Kaggle service the token chain reaches.

    A loopback HTTP server answers the Jupyter proxy REST, with the session token in the URL
    path as the real routed URL carries it. Everything else, the internal Kaggle API, the
    Firebase exchange and Firestore, is answered by a fake ``urlopen`` the provider's one HTTP
    seam is pointed at. A live account and a GPU are the only things replaced: the token
    chain, the channel and the worker are the real code.

    ``end_session()`` makes every later proxy request answer 404, as an ended session does.
    """

    RUN_ID = 350000001
    NOTEBOOK_ID = 134000001

    def __init__(self, token: str = "kaggle-session-token"):
        import http.server
        import re
        import threading
        import uuid

        self.token = token
        self.requests: list[dict[str, Any]] = []
        self.kernels: set[str] = set()
        self.cloud_calls: list[tuple[str, dict[str, Any]]] = []
        self.cancelled: list[int] = []
        #: Notebook ids deleted through the cookie's ``DeleteKernel`` call.
        self.deleted: list[int] = []
        self.ended = False
        #: Endpoint name to (HTTP status, message) for internal calls the fake refuses.
        self.refuse: dict[str, tuple[int, str]] = {}
        #: A notebook id deleted on kaggle.com. Starting a session on it is refused the way
        #: Kaggle refuses one, while every other notebook of the account still works.
        self.deleted_notebook: int | None = None
        self._lock = threading.Lock()
        self._uuid = uuid
        self._route = re.compile(r"^/k/(?P<run>[^/]+)/(?P<token>[^/]+)/proxy(?P<rest>.*)$")
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return None

            def do_GET(self) -> None:
                owner._handle(self, "GET")

            def do_PUT(self) -> None:
                owner._handle(self, "PUT")

            def do_POST(self) -> None:
                owner._handle(self, "POST")

            def do_DELETE(self) -> None:
                owner._handle(self, "DELETE")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self.proxy_host = f"http://127.0.0.1:{self.port}"
        #: The routed URL the Firestore document hands back: token in the path, no query.
        self.url = f"{self.proxy_host}/k/{self.RUN_ID}/{token}/proxy"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def end_session(self) -> None:
        self.ended = True

    # -- the fake urlopen the provider's HTTP seam is pointed at --

    def urlopen(self, request: Any, timeout: float | None = None) -> Any:
        import urllib.request as real_urllib

        from letify.providers import kaggle as kaggle_module

        url = request.full_url
        if url.startswith(self.proxy_host):
            return real_urllib.urlopen(request, timeout=timeout)
        if url.startswith(kaggle_module.KAGGLE_INTERNAL):
            return self._cloud(url[len(kaggle_module.KAGGLE_INTERNAL) :], request)
        if url.startswith(kaggle_module.IDENTITY_TOOLKIT):
            return _Reply({"idToken": "fake-id-token", "refreshToken": "fake-refresh"})
        if url.startswith(kaggle_module.FIRESTORE_BASE):
            return _Reply({"fields": {"jupyterUrl": {"stringValue": self.url}}})
        raise AssertionError(f"unexpected URL in the token chain: {url}")

    def _cloud(self, path: str, request: Any) -> _Reply:
        body = json.loads(request.data) if request.data else {}
        self.cloud_calls.append((path, body))
        refused = self.refuse.get(path.rsplit("/", 1)[-1])
        deleted = self.deleted_notebook
        if refused is None and deleted is not None and body.get("kernelId") == deleted:
            refused = (403, "Permission 'kernels.update' was denied")
        if refused is not None:
            import io
            import urllib.error

            status, message = refused
            payload = io.BytesIO(json.dumps({"message": message}).encode())
            raise urllib.error.HTTPError(request.full_url, status, "refused", {}, payload)
        if path.endswith("CreateKernelWithSettings"):
            return _Reply({"id": self.NOTEBOOK_ID, "currentUrlSlug": "letify-runtime"})
        if path.endswith("GetOrCreateKernelSession"):
            return _Reply({"draft": {"sequence": 1}})
        if path.endswith("CommitAndRun"):
            return _Reply({"kernelRunId": self.RUN_ID})
        if path.endswith("GetFirebaseConfig"):
            return _Reply({"apiKey": "fake-api-key"})
        if path.endswith("GetFirebaseAuthToken"):
            return _Reply({"authToken": "fake-custom-token"})
        if path.endswith("UpdateUserKernelFirestoreAuth"):
            return _Reply({"sessionId": f"webtier-{self.RUN_ID}"})
        if path.endswith("CancelKernelSession"):
            self.cancelled.append(int(body.get("kernelSessionId")))
            return _Reply({})
        if path.endswith("DeleteKernel"):
            self.deleted.append(int(body.get("kernelId")))
            return _Reply({})
        if path.endswith("GetAcceleratorQuotaStatistics"):
            return _Reply(
                {
                    "quotaRefreshTime": "2026-09-19T00:00:00Z",
                    "gpuQuota": {"totalTimeAllowed": "108000s", "timeUsed": "11700s"},
                    "tpuQuota": {"totalTimeAllowed": "72000s", "timeUsed": "0s"},
                }
            )
        if path.endswith("GetCurrentUser"):
            return _Reply({"displayName": "IRACK"})
        raise AssertionError(f"unexpected internal call: {path}")

    # -- the loopback Jupyter proxy --

    def made(self, method: str, prefix: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == method and r["path"].startswith(prefix)]

    def _handle(self, handler: Any, method: str) -> None:
        import urllib.parse

        parsed = urllib.parse.urlsplit(handler.path)
        route = self._route.match(urllib.parse.unquote(parsed.path))
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length else b""
        path = route.group("rest") if route else parsed.path
        path_token = route.group("token") if route else None
        with self._lock:
            self.requests.append(
                {
                    "method": method,
                    "path": path,
                    "path_token": path_token,
                    "authorization": handler.headers.get("Authorization"),
                }
            )

        def answer(status: int, payload: bytes = b"", extra: dict[str, str] | None = None) -> None:
            handler.send_response(status)
            for key, value in (extra or {}).items():
                handler.send_header(key, value)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        if self.ended:
            return answer(404, b'{"message": "session not found"}')
        if path_token != self.token:
            return answer(404, b'{"message": "not found"}')
        if path == "/api/status" and method == "GET":
            return answer(200, b'{"started": "2026-09-14T00:00:00Z"}')
        if path == "/api/kernels" and method == "POST":
            kernel = self._uuid.uuid4().hex
            self.kernels.add(kernel)
            return answer(201, json.dumps({"id": kernel, "name": "python3"}).encode())
        if path.startswith("/api/kernels/"):
            kernel = path.rsplit("/", 1)[-1]
            if kernel not in self.kernels:
                return answer(404, b'{"message": "no such kernel"}')
            if method == "DELETE":
                self.kernels.discard(kernel)
                return answer(204)
            return answer(200, json.dumps({"id": kernel}).encode())
        if path.startswith("/api/contents/"):
            local = Path("/") / path[len("/api/contents/") :]
            if method == "PUT":
                model = json.loads(raw)
                content = base64.b64decode(model["content"])
                mode = "ab" if model.get("chunk") not in (None, 1) else "wb"
                with open(local, mode) as handle:
                    handle.write(content)
                return answer(200, json.dumps({"path": str(local), "type": "file"}).encode())
            if not local.is_file():
                return answer(404, b'{"message": "no such file"}')
            model = {"path": str(local), "type": "file", "size": local.stat().st_size}
            return answer(200, json.dumps(model).encode())
        return answer(400, b'{"message": "unexpected request"}')


@pytest.fixture
def fake_kaggle(isolated_home, monkeypatch, tmp_path: Path):
    """A Kaggle account whose whole token chain and session are faked for alias ``kaggle_a``.

    The fake adapter in ``tests/fake_kaggle_adapter.py`` keeps the real adapter's contract and
    runs the worker in a local interpreter, so the driver programs are the real ones. The
    account also gets an API token, since login requires one: a test of the no-token path
    removes ``api_token`` itself after this fixture runs.
    """
    from letify.config.secrets import write_secret
    from letify.providers import kaggle as kaggle_module

    write_secret("kaggle_a", "cookie", kaggle_test_cookie())
    write_secret("kaggle_a", "access_token", "KGAT_" + "a" * 32)
    write_secret("kaggle_a", "username", "irack000")
    cloud = FakeKaggleCloud()
    monkeypatch.setattr(kaggle_module, "urlopen", cloud.urlopen)
    monkeypatch.setattr(kaggle_module, "JUPYTER_PROXY_HOST", cloud.proxy_host)
    monkeypatch.setattr(kaggle_module, "SESSION_START_TIMEOUT", 1.0)
    # The real gap between sessions is two minutes, which a test must not sit through. Its
    # own tests set it back. Spec "Kaggle", not making sessions faster than Kaggle allows.
    monkeypatch.setattr(kaggle_module, "MIN_SESSION_INTERVAL_S", 0.0)
    script = Path(__file__).with_name("fake_kaggle_adapter.py")
    monkeypatch.setattr(kaggle_module, "adapter_command", lambda: [sys.executable, str(script)])
    cloud.log = tmp_path / "kaggle-adapter.jsonl"
    monkeypatch.setenv("FAKE_KAGGLE_LOG", str(cloud.log))
    yield cloud
    cloud.close()


class FakeKaggleCLI:
    """Stands in for the official ``kaggle`` CLI, by replacing ``subprocess.run`` in the
    Kaggle provider module.

    Records every command it was asked to run. ``kernels delete`` and ``quota`` are the two
    commands the provider uses; both answer success unless ``fail`` names the subcommand.
    """

    def __init__(self):
        self.calls: list[list[str]] = []
        self.environments: list[dict[str, str]] = []
        self.fail: set[str] = set()
        #: Standard output a ``kernels list --csv`` call answers with. Kaggle names a
        #: notebook itself, so the titles are its own and the record is what tells
        #: letify's notebooks from the user's.
        self.kernels_csv = (
            "ref,title,author,lastRunTime,totalVotes\n"
            "irack000/letify-runtime,notebook8843ff70eb,irack000,2026-10-10 05:00:00,0\n"
            "irack000/my-analysis,My analysis,irack000,2026-10-09 11:00:00,3\n"
            "irack000/letify-runtime-2,notebook77c1a2,irack000,2026-10-10 06:00:00,0\n"
        )
        #: Standard output a ``quota`` call answers with, as JSON text.
        self.quota_json = json.dumps(
            [
                {
                    "resource": "GPU",
                    "used": "0.00h",
                    "remaining": "60.00h",
                    "total": "60.00h",
                    "refreshAt": "2026-10-10T00:00:00",
                },
                {
                    "resource": "TPU",
                    "used": "0.00h",
                    "remaining": "20.00h",
                    "total": "20.00h",
                    "refreshAt": "2026-10-10T00:00:00",
                },
            ]
        )

    def run(self, command: list[str], **kwargs: Any) -> FakeCompleted:
        self.calls.append(list(command))
        environment = dict(kwargs.get("env") or {})
        self.environments.append(environment)
        if not environment.get("KAGGLE_API_TOKEN"):
            return FakeCompleted(returncode=1, stdout="", stderr="Authentication required")
        for marker in self.fail:
            if marker in command:
                return FakeCompleted(returncode=1, stdout="", stderr="refused")
        if "quota" in command:
            return FakeCompleted(returncode=0, stdout=self.quota_json, stderr="")
        if "list" in command:
            return FakeCompleted(returncode=0, stdout=self.kernels_csv, stderr="")
        return FakeCompleted(returncode=0, stdout="{}", stderr="")


@pytest.fixture
def fake_kaggle_cli(monkeypatch):
    """Intercept every call the Kaggle provider makes to the official CLI."""
    from letify.providers import kaggle as kaggle_module

    cli = FakeKaggleCLI()
    monkeypatch.setattr(kaggle_module.tools, "find_uv", lambda: "uv")
    import subprocess as subprocess_module

    monkeypatch.setattr(subprocess_module, "run", cli.run)
    return cli


@pytest.fixture
def kaggle_api_token(isolated_home):
    """Write a Kaggle API token for alias ``kaggle_a``, as ``letify login kaggle`` would."""
    from letify.config.secrets import write_secret

    write_secret("kaggle_a", "username", "irack000")
    return write_secret("kaggle_a", "access_token", "KGAT_" + "a" * 32)


class FakeColabCli:
    """A Colab CLI that holds assignments, for ``patch_run`` as the ``result`` callable.

    ``sessions`` prints one ``[<name>] <endpoint>`` row per assignment, ``?`` for one with
    no local state. ``new`` can add an assignment and then fail, as a read timeout does.
    The unassign script is recognized by its ``python`` executable and removes the endpoint
    named by its last argument.
    """

    def __init__(self, assignments=(), new_adds=None, new_fails=False, list_fails_after=None):
        self.assignments = list(assignments)
        self.new_adds = new_adds
        self.new_fails = new_fails
        self.list_fails_after = list_fails_after
        self.listings = 0
        self.unassigned: list[str] = []

    def __call__(self, command):
        if "python" in command and "-c" in command and "new" not in command:
            self.unassigned.append(command[-1])
            self.assignments = [a for a in self.assignments if a[1] != command[-1]]
            return FakeCompleted()
        if "sessions" in command:
            self.listings += 1
            if self.list_fails_after is not None and self.listings > self.list_fails_after:
                return FakeCompleted(returncode=1, stderr="Read timed out")
            rows = [f"[{n}] {e} | Hardware: A100 | Variant: GPU\n" for n, e in self.assignments]
            return FakeCompleted(
                stdout="".join(rows) or "[colab] No active sessions found on server.\n"
            )
        if "new" in command:
            if self.new_adds:
                self.assignments.append(("?", self.new_adds))
            if self.new_fails:
                return FakeCompleted(returncode=1, stderr="ReadTimeout: Read timed out")
        return FakeCompleted()


@pytest.fixture(scope="session")
def sidecar_binary(tmp_path_factory) -> Path:
    from letify.transport import linkd

    core = Path(__file__).resolve().parent.parent / "letify-core"
    named = os.environ.get(linkd.ENV_BINARY)
    if named:
        return Path(named)
    if shutil.which("cargo") is None:
        pytest.skip(
            "cargo is not installed and LETIFY_LINK_BIN is not set, so letify-link is not built"
        )
    # One build per worker; cargo serialises concurrent builds on its own lock.
    result = subprocess.run(
        ["cargo", "build", "-p", "letify-link", "--release", "--features", "letify-link/relay"],
        cwd=core,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"cargo build -p letify-link failed:\n{result.stderr[-2000:]}")
    binary = core / "target" / "release" / linkd.BINARY
    assert binary.is_file(), binary
    return binary


@pytest.fixture(autouse=True)
def quick_link_probe(monkeypatch):
    """A letify-link connect measures its session with a probe that takes a fraction of a second.

    The real probe is 30 round trips and 2 s each way after a warm-up (Spec "Choosing a link"),
    which every connect of every test would otherwise pay. The shape is unchanged.
    """
    from letify.transport import linkchannel
    from letify.transport.probe import Probe

    class Quick(Probe):
        def warmup_for(self, rtt_ms: float) -> float:
            return 0.05

    monkeypatch.setattr(linkchannel, "make_probe", lambda: Quick(round_trips=5, seconds=0.2))


# -- the signalling service ----------------------------------------------------------------


class _FakeWebSocket:
    """The service side of one WebSocket: unmasks client frames, writes unmasked frames."""

    def __init__(self, connection: Any, reader: Any, fragment: bool) -> None:
        self._connection = connection
        self._reader = reader
        self._fragment = fragment
        self._lock = threading.Lock()
        self.pongs = 0

    def _write(self, data: bytes) -> None:
        with self._lock:
            try:
                self._connection.sendall(data)
            except OSError:
                pass

    @staticmethod
    def _frame(opcode: int, payload: bytes, fin: bool = True) -> bytes:
        import struct

        head = bytes([(0x80 if fin else 0) | opcode])
        if len(payload) < 126:
            return head + bytes([len(payload)]) + payload
        return head + bytes([126]) + struct.pack(">H", len(payload)) + payload

    def push(self, stored: dict[str, Any]) -> None:
        data = json.dumps(
            {k: stored[k] for k in ("seq", "from", "at", "body")}, separators=(",", ":")
        ).encode()
        if self._fragment:
            half = len(data) // 2
            self._write(self._frame(0x1, data[:half], fin=False))
            self._write(self._frame(0x9, b"hi"))
            self._write(self._frame(0x0, data[half:]))
        else:
            self._write(self._frame(0x1, data))

    def send_text(self, text: str) -> None:
        self._write(self._frame(0x1, text.encode()))

    def close(self, code: int = 1000) -> None:
        import struct

        self._write(self._frame(0x8, struct.pack(">H", code)))

    def cut(self) -> None:
        import socket as _socket

        try:
            self._connection.shutdown(_socket.SHUT_RDWR)
        except OSError:
            pass

    def read_text(self) -> str | None:
        """The next text message of the client, or None once the socket ended."""
        import struct

        while True:
            try:
                head = self._reader.read(2)
                if len(head) < 2:
                    return None
                opcode, length = head[0] & 0x0F, head[1] & 0x7F
                if not head[1] & 0x80:
                    return None
                if length == 126:
                    length = struct.unpack(">H", self._reader.read(2))[0]
                elif length == 127:
                    length = struct.unpack(">Q", self._reader.read(8))[0]
                mask = self._reader.read(4)
                data = bytearray(self._reader.read(length))
            except (OSError, ValueError):
                return None
            for i in range(len(data)):
                data[i] ^= mask[i % 4]
            if opcode == 0x1:
                return bytes(data).decode()
            if opcode == 0x8:
                self._write(self._frame(0x8, bytes(data[:2])))
                return None
            if opcode == 0x9:
                self._write(self._frame(0xA, bytes(data)))
            elif opcode == 0xA:
                self.pongs += 1


class FakeSignalService:
    """An in-process signalling service with the contract of Spec "Signalling service".

    It verifies every request with a real sidecar (``verifier``), so it applies the same
    canonical-key and strict-signature rules as letify-link, and it keeps mailboxes in memory.
    Real Cloudflare is the only thing it stands in for. ``clock`` is the service clock,
    ``rewrite`` lets a test act as a malicious service by changing a stored message, and
    ``requests`` records (method, path without query, status) of every request.
    """

    MAX_BODY = 4096
    TTL_S = 60
    DEPTH = 32
    WINDOW_S = 60

    def __init__(self, verifier: Any) -> None:
        import http.server
        import time as _time

        self.verifier = verifier
        self.clock = _time.time
        self.rewrite: Any = None
        self.requests: list[tuple[str, str, int]] = []
        #: The byte length of every message body the service accepted.
        self.sizes: list[int] = []
        self._boxes: dict[str, list[dict[str, Any]]] = {}
        self._counters: dict[str, int] = {}
        self._nonces: dict[str, float] = {}
        self._cond = threading.Condition()
        #: Open WebSockets by node, how many were accepted, and the acknowledgements received.
        self._sockets: dict[str, list[_FakeWebSocket]] = {}
        self.ws_accepted = 0
        self.ws_acks: list[int] = []
        #: Send every text frame as two fragments with a ping between them.
        self.fragment_frames = False
        #: Text pings received, and whether the service answers them with the text pong.
        self.ws_pings = 0
        self.answer_pings = True
        #: Set by ``close``, so a WebSocket still in its handshake is cut once it registers.
        self._closed = False
        service = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                service._serve(self, "GET")

            def do_POST(self) -> None:
                service._serve(self, "POST")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        with self._cond:
            self._closed = True
        self.drop_sockets()
        self._server.shutdown()
        self._server.server_close()

    def drop_sockets(self) -> None:
        """Cut every open WebSocket without a close frame, as a network failure does."""
        with self._cond:
            sockets = [sock for socks in self._sockets.values() for sock in socks]
        for sock in sockets:
            sock.cut()

    def sockets_open(self, node: str) -> int:
        with self._cond:
            return len(self._sockets.get(node, []))

    def waiting(self, node: str) -> int:
        with self._cond:
            return len(self._boxes.get(node, []))

    def _reply(self, h: Any, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        # Recorded before the first reply byte, so a test that acts on the client's view of
        # the reply never reads the record ahead of the server thread.
        self.requests.append((h.command, h.path.split("?")[0], status))
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def _fail(self, h: Any, status: int, code: str) -> None:
        self._reply(h, status, {"ok": False, "error": code})

    def _serve(self, h: Any, method: str) -> None:
        import hashlib
        import re
        import urllib.parse

        url = urllib.parse.urlsplit(h.path)
        if url.path == "/health":
            return self._reply(h, 200, {"ok": True})
        match = re.fullmatch(r"/(send|recv|ws)/([^/]+)", url.path)
        if match is None:
            return self._fail(h, 404, "not_found")
        kind, target = match.groups()
        if method != ("POST" if kind == "send" else "GET"):
            return self._fail(h, 405, "method_not_allowed")
        length = int(h.headers.get("Content-Length") or 0)
        if length > self.MAX_BODY:
            return self._fail(h, 413, "too_large")
        body = h.rfile.read(length) if length else b""
        node = h.headers.get("X-Letify-Node")
        stamp = h.headers.get("X-Letify-Time")
        nonce = h.headers.get("X-Letify-Nonce")
        signature = h.headers.get("X-Letify-Signature")
        if None in (node, stamp, nonce, signature):
            return self._fail(h, 400, "bad_request")
        assert node and stamp and nonce and signature
        if not (
            re.fullmatch(r"[0-9]{1,12}", stamp)
            and re.fullmatch(r"[0-9a-f]{32}", nonce)
            and re.fullmatch(r"[0-9a-f]{128}", signature)
            and re.fullmatch(r"[0-9a-f]{64}", node)
        ):
            return self._fail(h, 400, "bad_request")
        if abs(int(stamp) - self.clock()) > self.WINDOW_S:
            return self._fail(h, 401, "stale")
        text = f"{method}\n{h.path}\n{stamp}\n{nonce}\n{hashlib.sha256(body).hexdigest()}"
        try:
            valid = self.verifier.verify(
                node, "letify-signal v1", text.encode(), bytes.fromhex(signature)
            )
        except Exception:
            return self._fail(h, 400, "bad_key")
        if not valid:
            return self._fail(h, 401, "bad_signature")
        with self._cond:
            now = self.clock()
            self._nonces = {n: t for n, t in self._nonces.items() if now - t <= 2 * self.WINDOW_S}
            if f"{node}:{nonce}" in self._nonces:
                return self._fail(h, 401, "replay")
            self._nonces[f"{node}:{nonce}"] = now
        if kind == "send":
            return self._send(h, node, target, body)
        if target != node:
            return self._fail(h, 403, "forbidden")
        if kind == "ws":
            if (h.headers.get("Upgrade") or "").lower() != "websocket":
                return self._fail(h, 426, "upgrade_required")
            return self._websocket(h, node)
        query = urllib.parse.parse_qs(url.query)
        after = int(query.get("after", ["0"])[0])
        wait = min(float(query.get("wait", ["25"])[0]), 25.0)
        self._recv(h, node, after, wait)

    def _websocket(self, h: Any, node: str) -> None:
        import base64 as b64
        import hashlib

        key = h.headers.get("Sec-WebSocket-Key") or ""
        guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        accept = b64.b64encode(hashlib.sha1((key + guid).encode()).digest()).decode()
        h.close_connection = True
        sock = _FakeWebSocket(h.connection, h.rfile, self.fragment_frames)
        # Everything the client can observe is recorded before the upgrade reply is written.
        # The lock is held across the reply and the replay so a concurrent send cannot push a
        # frame ahead of the 101.
        with self._cond:
            self.requests.append((h.command, h.path.split("?")[0], 101))
            self.ws_accepted += 1
            self._sockets.setdefault(node, []).append(sock)
            h.wfile.write(
                (
                    "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                    f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode()
            )
            h.wfile.flush()
            now = self.clock()
            for stored in self._boxes.get(node, []):
                if stored["at"] + self.TTL_S > now:
                    sock.push(stored)
            closed = self._closed
        if closed:
            # A handshake that raced ``close`` ends like every other socket: cut.
            sock.cut()
            return
        try:
            while True:
                text = sock.read_text()
                if text is None:
                    break
                if text == "ping":
                    with self._cond:
                        self.ws_pings += 1
                    if self.answer_pings:
                        sock.send_text("pong")
                    continue
                try:
                    ack = int(json.loads(text)["ack"])
                except (ValueError, KeyError, TypeError):
                    sock.close(1008)
                    break
                with self._cond:
                    self.ws_acks.append(ack)
                    box = self._boxes.setdefault(node, [])
                    box[:] = [m for m in box if m["seq"] > ack]
        finally:
            with self._cond:
                if sock in self._sockets.get(node, []):
                    self._sockets[node].remove(sock)

    def _send(self, h: Any, sender: str, to: str, body: bytes) -> None:
        import base64 as b64

        if not body:
            return self._fail(h, 400, "bad_request")
        with self._cond:
            box = self._boxes.setdefault(to, [])
            now = self.clock()
            box[:] = [m for m in box if m["at"] + self.TTL_S > now]
            if len(box) >= self.DEPTH:
                return self._fail(h, 429, "mailbox_full")
            seq = self._counters.get(to, 0) + 1
            self._counters[to] = seq
            stored = self.rewrite(body) if self.rewrite else body
            box.append(
                {"seq": seq, "from": sender, "at": int(now), "body": b64.b64encode(stored).decode()}
            )
            self.sizes.append(len(body))
            for sock in list(self._sockets.get(to, [])):
                sock.push(box[-1])
            self._cond.notify_all()
        self._reply(h, 202, {"ok": True, "seq": seq})

    def _recv(self, h: Any, node: str, after: int, wait: float) -> None:
        import time as _time

        deadline = _time.monotonic() + wait
        with self._cond:
            while True:
                box = self._boxes.setdefault(node, [])
                box[:] = [m for m in box if m["seq"] > after]
                now = self.clock()
                box[:] = [m for m in box if m["at"] + self.TTL_S > now]
                if box or _time.monotonic() >= deadline:
                    messages = [dict(m) for m in box]
                    break
                self._cond.wait(timeout=max(0.0, deadline - _time.monotonic()))
            cursor = self._counters.get(node, 0)
        self._reply(h, 200, {"messages": messages, "cursor": cursor})


@pytest.fixture
def signal_service(sidecar_binary, tmp_path: Path):
    from letify.transport import linkd

    verifier = linkd.LinkDaemon(tmp_path / "service.key", binary=sidecar_binary)
    service = FakeSignalService(verifier)
    try:
        yield service
    finally:
        service.close()
        verifier.close()


# -- Ports a test picks for itself ---------------------------------------------------

#: Each xdist worker draws from its own slice below the ephemeral range, so two workers
#: never pick the same port and the kernel never hands one to an outgoing connection. A
#: punch binds with SO_REUSEPORT, so a port that two punches hold is shared by both.
_PORT_SLICE_START = 20000
_PORT_SLICE_SIZE = 1000
_port_cursor: dict[str, int] = {}


def free_port(worker: str | None = None) -> int:
    """A port nothing holds, from the slice of this xdist worker (or of ``worker``)."""
    name = worker or os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    index = int(name[2:]) if name.startswith("gw") and name[2:].isdigit() else 0
    base = _PORT_SLICE_START + (index % 12) * _PORT_SLICE_SIZE
    for _ in range(_PORT_SLICE_SIZE):
        offset = _port_cursor.get(name, 0)
        _port_cursor[name] = (offset + 1) % _PORT_SLICE_SIZE
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("0.0.0.0", base + offset))  # no reuse options: fails if anything holds it
        except OSError:
            continue
        finally:
            probe.close()
        return base + offset
    raise RuntimeError("no free port in the slice of " + name)  # pragma: no cover


def free_ports(count: int) -> list[int]:
    return [free_port() for _ in range(count)]


def refusing_port() -> socket.socket:
    """A port that answers a connection with a reset: bound by the caller, never listening."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    return holder
