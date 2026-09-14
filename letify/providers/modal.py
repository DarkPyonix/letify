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
import contextlib
import json
import secrets
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Mapping
from typing import IO, TYPE_CHECKING, Any

from ..config import ProviderConfig
from ..declare.instance import Host, Instance
from ..errors import LetifyError, ProviderUnavailable, RuntimeFailure, UnsupportedMode
from ..protocol import wire
from ..runtime.channel import Connection, FramedChannel
from .base import Provider
from .usage import Usage

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: The credit the Starter plan includes each month, in USD, used when an entry declares no
#: ``monthly_credit``.
STARTER_MONTHLY_CREDIT = 30.0

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

#: The port the worker listens on inside a sandbox for the data channel, exposed with
#: Modal ``encrypted_ports``.
DATA_PORT = 8765

#: Seconds the worker waits for the data connection after it answers ``listen``.
LISTEN_WAIT = 60

#: Seconds the channel allows for connecting through the tunnel and for the hello after it.
DATA_CONNECT_TIMEOUT = 30

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

    #: Modal bills in dollars and publishes the month's spend, not a balance, so the balance
    #: is the monthly credit minus the month's metered cost.
    usage_unit = "USD"
    usage_source = "Modal billing summary for this month, against the monthly credit"

    def report_usage(self) -> Usage:
        """This month's metered cost against ``monthly_credit``, through a one-off adapter.

        The adapter is its own process and is closed here, so asking for usage leaves no
        process behind and touches no sandbox.
        """
        declared = self.config.option("monthly_credit")
        limit = float(declared) if isinstance(declared, (int, float)) else STARTER_MONTHLY_CREDIT
        adapter = Adapter.for_account(self.alias)
        try:
            summary = adapter.request("billing_summary")
            used = float(summary["metered_cost"])
            end = summary.get("end")
        except (LetifyError, ValueError, KeyError, TypeError) as exc:
            return Usage(
                alias=self.alias,
                kind=self.kind,
                unit=self.usage_unit,
                source=self.usage_source,
                limit=limit,
                as_of=time.time(),
                note=f"billing summary could not be read: {exc}",
            )
        finally:
            adapter.close()
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            remaining=max(limit - used, 0.0),
            limit=limit,
            used=used,
            resets_at=float(end) if isinstance(end, (int, float)) else None,
            as_of=time.time(),
        )

    #: A sandbox keeps a process alive, so handles and blob reuse work.
    persistent_channel = True

    #: A sandbox loses its disk, so the root is a Modal volume mounted at this path.
    default_workspace = "/letify"

    #: A cold import from the volume reads thousands of small files over the network, so the
    #: project ``.venv`` is built on the sandbox's own disk.
    env_root = "/root/.letify-env"

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
                "container, so there is no device to forward PyTorch operators to. This is a "
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
        data_port = DATA_PORT if self.config.option("data_channel", True) is not False else None
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
            ports=[] if data_port is None else [data_port],
        )
        sandbox = str(created["sandbox"])
        self._sandboxes[runtime.name] = sandbox
        return SandboxChannel(adapter, sandbox, name=runtime.name, data_port=data_port)

    def stop(self, runtime: Runtime) -> None:
        sandbox = self._sandboxes.pop(runtime.name, None)
        if sandbox is None:
            return
        try:
            self.adapter().request("terminate", sandbox=sandbox)
        except (RuntimeFailure, ProviderUnavailable):
            # Terminating is best effort. A sandbox that is already gone is fine.
            pass


#: The most plaintext one TLS write encrypts, so the socket order lock is never held long.
TLS_WRITE = 1 << 20


class TlsStream:
    """TLS over a connected socket, for one reading thread and any number of writing threads.

    Spec "Modal data channel": the TLS object works on two memory buffers. ``_state`` guards
    the TLS object and both buffers, ``_wire`` orders encrypted bytes on the socket, and no
    socket call is made while ``_state`` is held, so a write blocked on a full socket never
    stops the reading thread from decrypting.
    """

    def __init__(self, sock: socket.socket, tls: Any, incoming: Any, outgoing: Any):
        self._sock = sock
        self._tls = tls
        self._incoming = incoming
        self._outgoing = outgoing
        self._state = threading.Lock()
        self._wire = threading.Lock()

    @classmethod
    def client(cls, sock: socket.socket, context: ssl.SSLContext, host: str) -> TlsStream:
        """Complete a client handshake for ``host`` over ``sock`` and return the stream."""
        incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        tls = context.wrap_bio(incoming, outgoing, server_hostname=host)
        stream = cls(sock, tls, incoming, outgoing)
        stream._handshake()
        return stream

    def _handshake(self) -> None:
        while True:
            with self._state:
                try:
                    self._tls.do_handshake()
                    done = True
                except ssl.SSLWantReadError:
                    done = False
                data = self._outgoing.read()
            if data:
                with self._wire:
                    self._sock.sendall(data)
            if done:
                return
            if not self._fill():
                raise ConnectionResetError("the TLS peer closed the connection in the handshake")

    def _fill(self) -> bool:
        """Move received bytes into the TLS object's input. False at end of stream."""
        data = self._sock.recv(1 << 20)
        if not data:
            return False
        with self._state:
            self._incoming.write(data)
        return True

    def send(self, data: Any) -> int:
        """Encrypt at most ``TLS_WRITE`` bytes of ``data``, send them, and return the count."""
        piece = memoryview(data).cast("B")[:TLS_WRITE]
        with self._wire:
            with self._state:
                count = self._tls.write(piece)
                # Also carries any bytes the reading thread's TLS work left behind, in order.
                encrypted = self._outgoing.read()
            self._sock.sendall(encrypted)
        return count

    def recv_into(self, view: memoryview) -> int:
        """Fill ``view`` with decrypted bytes and return how many, 0 at end of stream."""
        while True:
            count = 0
            closed = False
            with self._state:
                while count < view.nbytes:
                    try:
                        got = self._tls.read(view.nbytes - count, view[count:])
                    except ssl.SSLWantReadError:
                        break
                    except (ssl.SSLZeroReturnError, ssl.SSLEOFError):
                        closed = True
                        break
                    if not got:
                        closed = True
                        break
                    count += got
                pending = self._outgoing.pending
            if pending:
                self._flush()
            if count:
                return count
            if closed or not self._fill():
                return 0

    def _flush(self) -> None:
        """Send bytes the TLS object produced while reading, unless a writer is sending now.

        A writer holding ``_wire`` sends them with its own bytes, so the reading thread never
        waits for a blocked write.
        """
        if not self._wire.acquire(blocking=False):
            return
        try:
            with self._state:
                data = self._outgoing.read()
            if data:
                self._sock.sendall(data)
        finally:
            self._wire.release()


class SandboxChannel(FramedChannel):
    """Carries the worker's frames over a sandbox's pipes, through the adapter.

    The frames are the ones that run over SSH, as spec "Modal adapter" describes. Only the
    plumbing differs: Modal returns a sandbox's output as text, so the worker writes each
    frame as a base64 line, a write is a ``write`` request, and each line is read with a
    ``read_until`` request.

    With ``data_port`` set, the frames move to a TCP connection through the sandbox's
    encrypted port after each hello, as spec "Modal data channel" describes, and standard
    input and output carry only the worker source and the ``listen`` request.
    """

    text_frames = True

    def __init__(self, adapter: Adapter, sandbox: str, *, name: str, data_port: int | None = None):
        self.adapter = adapter
        self.sandbox = sandbox
        self.name = name
        self.data_port = data_port
        #: The context a TLS tunnel is verified with. None uses the system's trusted roots.
        self.tls_context: ssl.SSLContext | None = None
        self._connection = None
        self._stdio: Connection | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._connection is not None:
            return
        self._raw = bytearray()
        self._stdio = Connection(
            self.name,
            self._write,
            wire.chunks_readinto(self._read_chunks),
            self._emit,
            death_detail=self._raw_text,
        )
        self._connection = self._stdio
        self._send_worker()
        self._await_ready()

    def _raw_text(self) -> str:
        return bytes(self._raw[-2000:]).decode("utf-8", "replace")

    def _send_worker(self) -> None:
        # After a reexec the data connection is gone, so the source goes back over stdio.
        if self._connection is not self._stdio:
            self._close_socket()
            self._connection = self._stdio
        super()._send_worker()

    def _await_ready(self) -> None:
        super()._await_ready()
        if self.data_port is None:
            return
        try:
            self._connection = self._open_data(self.data_port)
        except (LetifyError, OSError, ValueError, KeyError, TypeError) as exc:
            self._close_socket()
            print(
                f"letify: {self.name}: the data channel did not open ({exc}); "
                "frames stay on standard input and output",
                file=sys.stderr,
                flush=True,
            )

    def _open_data(self, port: int) -> Connection:
        """Ask the worker to listen, connect through the tunnel, and wait for its hello."""
        stdio = self._stdio
        assert stdio is not None
        token = secrets.token_hex(32)
        stdio.request(
            {"op": "listen", "port": port, "token": token, "wait": LISTEN_WAIT},
            timeout=DATA_CONNECT_TIMEOUT + LISTEN_WAIT,
            kill=self._kill,
        )
        tunnel = self.adapter.request("tunnel", sandbox=self.sandbox, port=port)
        host = str(tunnel["host"])
        raw = socket.create_connection((host, int(tunnel["port"])), timeout=DATA_CONNECT_TIMEOUT)
        self._socket = raw
        try:
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            stream: Any = raw
            if tunnel.get("tls"):
                context = self.tls_context or ssl.create_default_context()
                stream = TlsStream.client(raw, context, host)
            hello = b"LETIFY-DATA " + token.encode("ascii") + b"\n"
            view = memoryview(hello)
            while view:
                view = view[stream.send(view) :]
            connection = Connection(
                self.name, stream.send, stream.recv_into, self._emit, death_detail=self._raw_text
            )
            # The connect timeout still applies to the socket, so a missing hello fails here.
            connection.wait_hello()
        except BaseException:
            self._close_socket()
            raise
        raw.settimeout(None)
        return connection

    def _close_socket(self) -> None:
        sock, self._socket = self._socket, None
        if sock is None:
            return
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            sock.close()

    def _kill(self) -> None:
        # A blocked read on the data connection returns once the socket is shut down.
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)

    def close(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.sender.frame(wire.SHUTDOWN, 0)
        except Exception:
            pass
        self._close_socket()

    #: The most bytes one write request carries. Modal refuses a sandbox stdin write that
    #: would buffer more than 2 MiB, so a larger frame goes out in several requests.
    WRITE_LIMIT = 1 << 20

    def _write(self, view: memoryview) -> int:
        piece = view[: self.WRITE_LIMIT]
        data = base64.b64encode(piece).decode("ascii")
        self.adapter.request("write", sandbox=self.sandbox, data=data)
        return piece.nbytes

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
    "TlsStream",
    "VolumePathMissing",
]
