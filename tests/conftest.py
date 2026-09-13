"""Shared fixtures and the few fakes the suite is allowed to use.

Everything that can run for real runs for real. The local provider starts a worker
subprocess behind the same framed protocol a remote runtime uses, so the protocol, the
pool, the store and the release rule are all exercised rather than stubbed.

A fake appears here only where the real thing needs something a test machine does not
have: the Colab CLI, the Elice HTTP API, Modal's client, a cloud bucket
and nvidia-smi. They are small objects rather than mocks, so a test still asserts on
what a caller observes instead of on which method was called.
"""

from __future__ import annotations

import base64
import os
import pickle
import subprocess
import sys
import types
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
def let() -> letify.Launcher:
    # home=False keeps the developer's own accounts out of the test run.
    return letify.Launcher(home=False, announce=False)


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
    """Say which device indices another process is computing on."""

    def patch(busy: list[int] | None = None) -> None:
        taken = set(busy or ())
        monkeypatch.setattr(telemetry, "busy_indices", lambda **kwargs: tuple(sorted(taken)))

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


class FakeResponse:
    """One HTTP response, with a body that may refuse to decode."""

    def __init__(self, status_code: int = 200, body: Any = None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("the body is not JSON")
        return self._body


class FakeHttpx:
    """The parts of httpx the Elice provider uses, with no network behind them."""

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.responses: dict[tuple[str, str], FakeResponse] = {}
        self.default = FakeResponse(200, {})

    def answer(self, method: str, path: str, response: FakeResponse) -> None:
        self.responses[(method, path)] = response

    def Client(self, **kwargs: Any) -> FakeHttpxClient:
        self.clients.append(kwargs)
        return FakeHttpxClient(self)

    @property
    def last(self) -> dict[str, Any]:
        return self.requests[-1]


class FakeHttpxClient:
    def __init__(self, owner: FakeHttpx):
        self.owner = owner

    def __enter__(self) -> FakeHttpxClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def request(self, method: str, path: str, **kwargs: Any) -> FakeResponse:
        self.owner.requests.append({"method": method, "path": path, **kwargs})
        return self.owner.responses.get((method, path), self.owner.default)


@pytest.fixture
def fake_httpx(monkeypatch):
    fake = FakeHttpx()
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return fake


# -- Modal ---------------------------------------------------------------------


class FakeSandbox:
    """A Modal sandbox stand-in whose replies are framed by the real protocol.

    Only the plumbing is fake. The bytes on the pipe are produced and read by the same
    codec a real sandbox would use.
    """

    def __init__(
        self,
        outcomes: list[dict[str, Any]] | None = None,
        *,
        logs: list[str] | None = None,
    ):
        self.written: list[bytes] = []
        self.terminated = False
        self.terminate_error: BaseException | None = None
        self.write_error: BaseException | None = None
        self._outcomes = list(outcomes or [])
        self._logs = list(logs or [])
        self._ready = True
        self.stdin = self

    # stdin
    def write(self, payload: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.written.append(payload)

    def drain(self) -> None:
        return None

    def terminate(self) -> None:
        if self.terminate_error is not None:
            raise self.terminate_error
        self.terminated = True

    @property
    def text(self) -> str:
        return b"".join(self.written).decode()

    @property
    def stdout(self):
        from letify.protocol import REPLY
        from letify.protocol.worker import READY

        lines: list[str] = []
        if self._ready:
            lines.append(READY + "\n")
            self._ready = False
        lines.extend(self._logs)
        self._logs = []
        if self._outcomes:
            outcome = self._outcomes.pop(0)
            blob = base64.b64encode(pickle.dumps(outcome, protocol=5)).decode()
            lines.append(REPLY + blob + "\n")
        return iter(lines)


class FakeModalVolume:
    """An in-memory stand-in for a Modal volume."""

    def __init__(self, name: str):
        self.name = name
        self.files: dict[str, bytes] = {}

    def listdir(self, path: str, recursive: bool = False):
        prefix = path.rstrip("/")
        found = [key for key in self.files if key == prefix or key.startswith(prefix + "/")]
        if not found:
            raise FileNotFoundError(path)
        return [types.SimpleNamespace(path=key) for key in found]

    def read_file(self, path: str):
        yield self.files[path]

    def batch_upload(self, force: bool = False):
        return _FakeBatch(self)


class _FakeBatch:
    def __init__(self, volume: FakeModalVolume):
        self.volume = volume

    def __enter__(self) -> _FakeBatch:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def put_file(self, stream: Any, path: str) -> None:
        self.volume.files[path] = stream.read()


class FakeModal:
    """The parts of the Modal client letify touches, recording what it was asked for."""

    def __init__(self, sandbox: FakeSandbox | None = None):
        self.sandbox = sandbox or FakeSandbox()
        self.created: dict[str, Any] = {}
        self.looked_up: list[str] = []
        self.volumes: dict[str, FakeModalVolume] = {}
        owner = self

        class App:
            @staticmethod
            def lookup(name: str, create_if_missing: bool = False) -> str:
                owner.looked_up.append(name)
                return f"app:{name}"

        class Image:
            def __init__(self) -> None:
                self.packages: tuple[str, ...] = ()

            @staticmethod
            def debian_slim() -> Any:
                return Image()

            def pip_install(self, *packages: str) -> Any:
                self.packages = packages
                return self

        class Sandbox:
            @staticmethod
            def create(*args: str, **kwargs: Any) -> FakeSandbox:
                owner.created = {"args": list(args), **kwargs}
                return owner.sandbox

        class Volume:
            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> FakeModalVolume:
                return owner.volumes.setdefault(name, FakeModalVolume(name))

        self.App = App
        self.Image = Image
        self.Sandbox = Sandbox
        self.Volume = Volume


@pytest.fixture
def fake_modal(monkeypatch):
    def install(sandbox: FakeSandbox | None = None) -> FakeModal:
        fake = FakeModal(sandbox)
        monkeypatch.setitem(sys.modules, "modal", fake)
        return fake

    return install


# -- cloud object stores -------------------------------------------------------


class FakeGCSBlob:
    def __init__(self, store: dict[str, bytes], key: str):
        self._store = store
        self.name = key

    def exists(self) -> bool:
        return self.name in self._store

    def upload_from_string(self, payload: Any) -> None:
        self._store[self.name] = payload if isinstance(payload, bytes) else str(payload).encode()

    def download_as_bytes(self) -> bytes:
        return self._store[self.name]

    def download_as_text(self) -> str:
        return self._store[self.name].decode()


class FakeGCSBucket:
    def __init__(self, name: str, store: dict[str, bytes]):
        self.name = name
        self._store = store

    def blob(self, key: str) -> FakeGCSBlob:
        return FakeGCSBlob(self._store, key)


class FakeGCSClient:
    def __init__(self, store: dict[str, bytes]):
        self._store = store

    def bucket(self, name: str) -> FakeGCSBucket:
        return FakeGCSBucket(name, self._store)

    def list_blobs(self, bucket: FakeGCSBucket, prefix: str = ""):
        return [
            types.SimpleNamespace(name=key) for key in sorted(self._store) if key.startswith(prefix)
        ]


@pytest.fixture
def fake_gcs(monkeypatch):
    """Install a google.cloud.storage stand-in, since a real bucket needs an account."""
    store: dict[str, bytes] = {}
    storage = types.SimpleNamespace(Client=lambda: FakeGCSClient(store))
    cloud = types.ModuleType("google.cloud")
    cloud.storage = storage  # type: ignore[attr-defined]
    google = types.ModuleType("google")
    google.cloud = cloud  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)
    return store


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
