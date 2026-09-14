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
import json
import os
import subprocess
import sys
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
    """Point Path.home and the working directory at empty directories.

    The command line entry point reads ~/.letify and ./.letify, so a test that goes
    through it has to be moved off the developer's own files.
    """
    home = tmp_path / "home"
    (home / ".letify").mkdir(parents=True)
    project = tmp_path / "project"
    (project / ".letify").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
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

    prepares_env = True


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
        return recorder

    return patch


@pytest.fixture(autouse=True)
def no_uv_variable(monkeypatch):
    """`uv run pytest` sets UV, which would let the real uv answer where tests patch PATH."""
    monkeypatch.delenv("UV", raising=False)


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

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> float:
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

    def fail(self, *ops: str) -> None:
        self._monkeypatch.setenv("FAKE_MODAL_FAIL", ",".join(ops))

    def exit_on(self, *ops: str) -> None:
        self._monkeypatch.setenv("FAKE_MODAL_EXIT", ",".join(ops))


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
    ):
        self.name = name
        self.probed = probed
        self.rank = rank
        self.result = result
        self.delay = delay
        self.error = error
        self.unmet = unmet
        self.probe_error = probe_error
        self.attempts = 0
        self.assumed = 0
        self.links: list[FakeLink] = []

    def needs(self, target: Any) -> str | None:
        return self.unmet

    def _link(self) -> FakeLink:
        link = FakeLink(self.name, self.rank, self.result, probe_error=self.probe_error)
        self.links.append(link)
        return link

    def attempt(self, target: Any) -> FakeLink:
        import time

        self.attempts += 1
        time.sleep(self.delay)
        if self.error:
            raise OSError(self.error)
        return self._link()

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
