"""Modal, reached through the Modal adapter in its own uv environment.

The letify process never imports ``modal``. ``Adapter`` starts ``modal_adapter.py``
through uv as one account and exchanges JSON lines with it, as spec "Modal adapter"
describes. This module owns that client, the provider and the sandbox channel. It does
not own the calls into Modal, which live in the adapter file.

Storage outlives a container because a Modal volume is mounted from outside it, so
this provider is persistent and the loop is shipped. CUDA call forwarding is not
offered: Modal exposes a function call into a container, not a device to forward
calls at.

A sandbox is used rather than a function call, because letify needs a process that
stays alive. Without one there is no object table for a handle to point at and no
blob table to keep a large argument from travelling twice.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import threading
import weakref
from collections.abc import Mapping
from typing import IO, TYPE_CHECKING, Any

from ..config import ProviderConfig
from ..declare.instance import Host, Instance
from ..errors import ProviderUnavailable, RuntimeFailure, UnsupportedMode
from ..protocol import wire
from ..runtime.channel import Connection, FramedChannel
from .base import Provider

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: GPU names Modal accepts, with the memory each one carries.
GPUS = {
    "T4": {"vram_gb": 16},
    "L4": {"vram_gb": 24},
    "A10": {"vram_gb": 24},
    "A100": {"vram_gb": 40},
    "A100_80GB": {"vram_gb": 80},
    "L40S": {"vram_gb": 48},
    "H100": {"vram_gb": 80},
    "H200": {"vram_gb": 141},
    "B200": {"vram_gb": 180},
    "RTX_PRO_6000": {"vram_gb": 96},
}

#: Modal spells some of these with a hyphen or a suffix.
WIRE_NAMES = {
    "A100_80GB": "A100-80GB",
    "RTX_PRO_6000": "RTX-PRO-6000",
}

#: Packages the sandbox image installs for the worker.
WORKER_PACKAGES = ("cloudpickle", "blake3")


class VolumePathMissing(RuntimeFailure):
    """The adapter answered that a volume path does not exist."""


def _close_process(process: subprocess.Popen[bytes], stderr: IO[bytes]) -> None:
    """Close the adapter's input so it exits, and reap it."""
    try:
        if process.stdin is not None:
            process.stdin.close()
    except OSError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    for stream in (process.stdout, stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


class Adapter:
    """One Modal adapter process, and the JSON lines exchanged with it.

    The process starts on the first request, so building a provider or a backend starts
    nothing. Requests are serialized by a lock, because the adapter answers one at a time.
    """

    def __init__(self, command: list[str] | None, env: dict[str, str] | None, *, name: str):
        self._command = command
        self._env = env
        self.alias = name
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: IO[bytes] | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._closed = False

    @classmethod
    def for_account(cls, alias: str) -> Adapter:
        """An adapter acting as ``alias``, started through uv when first asked."""
        return cls(None, None, name=alias)

    def _start(self) -> subprocess.Popen[bytes]:
        from .. import tools

        if self._process is not None:
            return self._process
        if self._closed:
            raise RuntimeFailure(f"{self.alias}: the Modal adapter was already closed")
        command = self._command
        env = self._env
        if command is None:
            uv = tools.find_uv()
            if uv is None:
                raise ProviderUnavailable("modal", tools.missing_uv_message())
            command = tools.modal_adapter_command(uv)
            env = tools.modal_environment(self.alias)
        # A file rather than a pipe, so a chatty adapter cannot fill a pipe nobody reads and
        # block. It lives as long as the process, so a context manager cannot hold it.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=env,
            )
        except OSError as exc:
            raise RuntimeFailure(
                f"{self.alias}: the Modal adapter could not start: {exc}", command=" ".join(command)
            ) from exc
        weakref.finalize(self, _close_process, self._process, self._stderr)
        return self._process

    def _stderr_tail(self) -> str:
        if self._stderr is None:
            return ""
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode(errors="replace")[-2000:]
        except (OSError, ValueError):
            return ""

    def _broken(self, what: str) -> RuntimeFailure:
        stderr = self._stderr_tail()
        detail = f"\n--- adapter stderr ---\n{stderr}" if stderr.strip() else ""
        return RuntimeFailure(f"{self.alias}: the Modal adapter {what}.{detail}", stderr=stderr)

    def request(self, op: str, **fields: Any) -> Any:
        """Send one request and return its value, raising what the reply's kind means."""
        with self._lock:
            process = self._start()
            assert process.stdin is not None and process.stdout is not None
            self._next_id += 1
            request_id = self._next_id
            line = json.dumps({"id": request_id, "op": op, **fields}) + "\n"
            try:
                process.stdin.write(line.encode())
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise self._broken(f"stopped before it took {op!r}") from exc
            try:
                raw = process.stdout.readline()
            except (OSError, ValueError) as exc:
                raise self._broken(f"stopped while answering {op!r}") from exc
            if not raw:
                process.wait()
                raise self._broken(f"exited with {process.returncode} while answering {op!r}")
        try:
            reply = json.loads(raw)
        except ValueError as exc:
            raise self._broken(
                f"answered {op!r} with a line that is not JSON: {raw[:200]!r}"
            ) from exc
        if not isinstance(reply, dict) or reply.get("id") != request_id:
            raise self._broken(f"answered {op!r} out of turn: {raw[:200]!r}")
        if reply.get("ok"):
            return reply.get("value")
        kind = reply.get("kind")
        error = str(reply.get("error") or "no message")
        if kind == "unavailable":
            raise ProviderUnavailable("modal", error)
        if kind == "not_found":
            raise VolumePathMissing(f"{self.alias}: {error} does not exist")
        raise RuntimeFailure(f"{self.alias}: Modal refused {op!r}: {error}")

    def close(self) -> None:
        """Close the adapter's input, which makes it terminate its sandboxes and exit."""
        with self._lock:
            self._closed = True
            if self._process is not None and self._stderr is not None:
                _close_process(self._process, self._stderr)


class Modal(Provider):
    """One Modal workspace."""

    kind = "modal"
    default_persistence = "persistent"
    has_fast_path = False

    #: Modal bills in dollars and exposes no workspace balance through its SDK, so the
    #: figure has to come from a configured command or from the dashboard.
    usage_unit = "USD"
    usage_source = "the Modal SDK exposes no workspace balance"

    #: A sandbox keeps a process alive, so handles and blob reuse work.
    persistent_channel = True

    #: A sandbox loses its disk, so the root is a Modal volume mounted at this path.
    default_workspace = "/letify"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._sandboxes: dict[str, str] = {}
        self._adapter: Adapter | None = None

    def adapter(self) -> Adapter:
        """The adapter acting as this account, started on its first request."""
        if self._adapter is None:
            self._adapter = Adapter.for_account(self.alias)
        return self._adapter

    def discover(self) -> Mapping[str, Instance]:
        """Return Modal's published GPU list, plus a CPU instance with no GPU.

        The list is static, so no call to Modal is made here. Asking for a GPU the
        workspace cannot get fails when a runtime starts, not now.
        """
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None)}
        table.update(
            {
                name: Instance(self, gpu=name, vram_gb=spec.get("vram_gb"))
                for name, spec in GPUS.items()
            }
        )
        return table

    def store_backend(self) -> str:
        return "modal"

    def wire_name(self, instance: Instance) -> str:
        """Translate an instance into the string Modal's API expects."""
        gpu = instance.gpu or ""
        return WIRE_NAMES.get(gpu, gpu)

    def check_mode(self, instance: Instance) -> None:
        if instance.placement is Host.local:
            raise UnsupportedMode(
                "Modal cannot serve host='local'. It exposes function calls into a "
                "container, so there is no device to forward CUDA calls to. This is a "
                "limit of the service, not a speed judgement. Use host='remote'."
            )

    def open_channel(self, runtime: Runtime) -> Channel:
        """A Modal sandbox running one Python that reads framed requests."""
        from ..protocol.worker import BOOTSTRAP

        adapter = self.adapter()
        app = str(self.config.option("app", "letify"))
        root = self.workspace_root
        # The workspace root is a Modal volume, so it outlives each sandbox. A mount path
        # has to be absolute inside the sandbox.
        volumes = {root: f"{app}-workspace"} if root.startswith("/") else {}
        # Not `python3 -`: that reads standard input to the end before running anything, so
        # the requests that follow the source would be compiled as source too.
        created = adapter.request(
            "create",
            app=app,
            volumes=volumes,
            args=["python3", "-u", "-c", BOOTSTRAP],
            packages=list(WORKER_PACKAGES),
            gpu=self.wire_name(runtime.instance) or None,
            timeout=int(self.config.option("timeout", 3600)),
        )
        sandbox = str(created["sandbox"])
        self._sandboxes[runtime.name] = sandbox
        return SandboxChannel(adapter, sandbox, name=runtime.name)

    def stop(self, runtime: Runtime) -> None:
        sandbox = self._sandboxes.pop(runtime.name, None)
        if sandbox is None:
            return
        try:
            self.adapter().request("terminate", sandbox=sandbox)
        except (RuntimeFailure, ProviderUnavailable):
            # Terminating is best effort. A sandbox that is already gone is fine.
            pass


class SandboxChannel(FramedChannel):
    """Carries the worker's frames over a sandbox's pipes, through the adapter.

    The frames are the ones that run over SSH, as spec "Modal adapter" describes. Only the
    plumbing differs: Modal returns a sandbox's output as text, so the worker writes each
    frame as a base64 line, a write is a ``write`` request, and each line is read with a
    ``read_until`` request.
    """

    text_frames = True

    def __init__(self, adapter: Adapter, sandbox: str, *, name: str):
        self.adapter = adapter
        self.sandbox = sandbox
        self.name = name
        self._connection = None

    def start(self) -> None:
        if self._connection is not None:
            return
        self._raw = bytearray()
        self._connection = Connection(
            self.name,
            self._write,
            wire.chunks_readinto(self._read_chunks),
            self._emit,
            death_detail=lambda: bytes(self._raw[-2000:]).decode("utf-8", "replace"),
        )
        self._send_worker()
        self._await_ready()

    def close(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.sender.frame(wire.SHUTDOWN, 0)
        except Exception:
            pass

    def _write(self, view: memoryview) -> int:
        data = base64.b64encode(view).decode("ascii")
        self.adapter.request("write", sandbox=self.sandbox, data=data)
        return view.nbytes

    def _read_chunks(self) -> list[bytes]:
        """The next frame line, decoded. A line that is not base64 is output from before the
        worker started, such as the interpreter's own error, and is kept for the failure."""
        import binascii

        read = self.adapter.request("read_until", sandbox=self.sandbox, prefixes=[""])
        chunks: list[bytes] = []
        for line in read.get("lines") or []:
            try:
                chunks.append(base64.b64decode(line.strip(), validate=True))
            except (binascii.Error, ValueError):
                self._raw += line.encode("utf-8", "replace")
        if not chunks and not read.get("eof"):
            # Only unframed output came back, so read again rather than report an end.
            return self._read_chunks()
        return chunks

    def _startup_failure(self, expired: bool, cause: Exception) -> Exception:
        # Spec "Modal adapter": a sandbox that ends without replying is a ProtocolError.
        return cause


__all__ = [
    "GPUS",
    "WIRE_NAMES",
    "WORKER_PACKAGES",
    "Adapter",
    "Modal",
    "SandboxChannel",
    "VolumePathMissing",
]
