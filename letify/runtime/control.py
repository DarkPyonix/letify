"""Run ids, the run registry and the control socket of a process that owns running calls.

Owns the id a call gets when it is sent, the record ``~/.letify/runs/<id>.json`` that makes
it findable from the command line, the Unix socket a command reaches the owning process on,
and the client side that ``letify runs``, ``letify interrupt`` and ``letify exec`` use. It
does not own the worker ops those commands become, which ``letify.protocol.worker`` owns, or
how a request travels to the runtime, which ``letify.runtime.channel`` owns.

Spec "Controlling a running call". The request is one JSON object per line on the socket and
the answer is one JSON object per line, so a tool that is not letify can drive it. There is
no control socket on Windows, and a run there has no record.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import secrets
import socket
import stat
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .. import paths
from ..errors import LetifyError, RunNotFound, UnsupportedMode
from .channel import INTERRUPT_HOOK

#: Characters of a run id: 12 lowercase hexadecimal digits.
RUN_ID_LENGTH = 12

#: The longest Unix socket path used. The kernel limit is 107 bytes on Linux and 103 on
#: macOS, so this leaves room under both.
SOCKET_LIMIT = 100

#: The largest request line the socket accepts, in bytes.
MAX_REQUEST = 16 << 20

#: Seconds an interrupt sent on Ctrl-C waits for the worker's answer.
INTERRUPT_WAIT_S = 10.0

#: Seconds ``exec`` waits for the answer when the command names none.
EXEC_WAIT_S = 60.0


def available() -> bool:
    """Whether this platform has the control socket."""
    return hasattr(socket, "AF_UNIX") and os.name != "nt"


def runs_directory() -> Path:
    """``~/.letify/runs``, whether or not it exists yet."""
    return paths.letify_root() / "runs"


def socket_path(pid: int) -> Path:
    """Where the process ``pid`` listens: ``runs/<pid>.sock``, or a shorter path if too long.

    A Unix socket path is limited to about 107 bytes. When the preferred path exceeds
    ``SOCKET_LIMIT`` the socket goes in the SSH control directory, which letify already
    keeps short for the same reason.
    """
    preferred = runs_directory() / f"{pid}.sock"
    if len(os.fsencode(preferred)) <= SOCKET_LIMIT:
        return preferred
    return paths.ssh_control_directory() / f"run-{pid}.sock"


def _private_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        directory.chmod(0o700)


# -- the owning process -----------------------------------------------------------


class Run:
    """One call that has been sent: its id, its record and how a command reaches its runtime."""

    def __init__(self, runtime: Any, function: str) -> None:
        self.id = secrets.token_hex(RUN_ID_LENGTH // 2)
        self.function = function
        self.runtime = runtime
        self.started_at = time.time()
        self._record: Path | None = None

    @property
    def persistent(self) -> bool:
        return bool(getattr(self.runtime, "persistent_channel", False))

    @property
    def provider(self) -> str:
        return str(getattr(getattr(self.runtime, "provider", None), "alias", ""))

    def record(self, socket_file: Path | None) -> dict[str, Any]:
        return {
            "run": self.id,
            "function": self.function,
            "runtime": str(getattr(self.runtime, "name", "")),
            "provider": self.provider,
            "started_at": self.started_at,
            "pid": os.getpid(),
            "socket": str(socket_file) if socket_file is not None else None,
        }

    def write(self, socket_file: Path | None) -> None:
        """Write the record, owner readable only, before the call is sent."""
        directory = runs_directory()
        _private_directory(directory)
        final = directory / f"{self.id}.json"
        partial = directory / f".{self.id}.json.partial"
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.record(socket_file), handle)
        os.replace(partial, final)
        self._record = final

    def end(self) -> None:
        """Remove the record. The call has returned or raised."""
        with _LOCK:
            _RUNS.pop(self.id, None)
        if self._record is not None:
            with contextlib.suppress(OSError):
                self._record.unlink()
            self._record = None

    # -- reaching the runtime --------------------------------------------------

    def forward(self, payload: dict[str, Any], wait: float) -> tuple[bool, Any]:
        """Send ``payload`` on the call's channel and wait up to ``wait`` seconds.

        Returns ``(True, value)``, or ``(False, None)`` when the answer did not come in time.
        The request itself is not cancelled, since a request with a timeout would kill the
        worker when it expired. A failure to send raises.
        """
        outcome: dict[str, Any] = {}
        done = threading.Event()

        def send() -> None:
            try:
                outcome["value"] = self.runtime.request(payload)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                done.set()

        threading.Thread(target=send, name=f"letify-run-{self.id}", daemon=True).start()
        if not done.wait(wait):
            return False, None
        if "error" in outcome:
            raise outcome["error"]
        return True, outcome["value"]

    def on_interrupt(self) -> None:
        """The first Ctrl-C reached the thread waiting for this call: forward it."""
        print(
            f"letify: interrupting {self.id}; press Ctrl-C again to stop waiting",
            file=sys.stderr,
            flush=True,
        )
        if not self.persistent:
            return
        try:
            self.forward({"op": "interrupt"}, INTERRUPT_WAIT_S)
        except Exception as exc:
            print(f"letify: could not interrupt {self.id}: {exc}", file=sys.stderr, flush=True)

    @contextlib.contextmanager
    def interruptible(self) -> Iterator[None]:
        """Make a KeyboardInterrupt in the waiting thread interrupt the body first."""
        token = INTERRUPT_HOOK.set(self.on_interrupt if self.persistent else None)
        try:
            yield
        finally:
            INTERRUPT_HOOK.reset(token)


_RUNS: dict[str, Run] = {}
_LOCK = threading.Lock()
_SERVER: _Server | None = None
_WARNED = False


def begin(runtime: Any, function: str) -> Run:
    """Give a call that is about to be sent its run id and write its record.

    A record that cannot be written, for example in a read-only home, is reported once on
    standard error. The call still runs, and Ctrl-C still reaches it, but no command can
    find it.
    """
    global _WARNED
    run = Run(runtime, function)
    with _LOCK:
        _RUNS[run.id] = run
    if not available():
        return run
    try:
        server = _server()
        run.write(server.path)
    except OSError as exc:
        if not _WARNED:
            _WARNED = True
            print(
                f"letify: runs cannot be controlled from the command line: {exc}",
                file=sys.stderr,
            )
    return run


def _server() -> _Server:
    global _SERVER
    with _LOCK:
        # Rebuilt when the process forked, or when its socket is gone or the state root moved.
        if _SERVER is None or not _SERVER.current():
            if _SERVER is not None:
                _SERVER.drop()
            _SERVER = _Server()
        return _SERVER


class _Server:
    """The control socket of this process, answering on a daemon thread."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.path = socket_path(self.pid)
        _private_directory(self.path.parent)
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.socket.bind(str(self.path))
            self.path.chmod(0o600)
            self.socket.listen(8)
        except OSError:
            self.socket.close()
            raise
        threading.Thread(target=self._accept, name="letify-control", daemon=True).start()
        atexit.register(self.close)

    def current(self) -> bool:
        """Whether this is still the socket a record should name."""
        return self.pid == os.getpid() and self.path == socket_path(self.pid) and self.path.exists()

    def drop(self) -> None:
        """Stop listening. The records are left alone, which belong to their runs."""
        atexit.unregister(self.close)
        with contextlib.suppress(OSError):
            # Wakes the accept thread, which a close alone leaves blocked on Linux.
            self.socket.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self.socket.close()
        if self.pid == os.getpid():
            with contextlib.suppress(OSError):
                self.path.unlink()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            # Wakes the accept thread, which a close alone leaves blocked on Linux.
            self.socket.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self.socket.close()
        with contextlib.suppress(OSError):
            self.path.unlink()
        # A record still there belongs to a call the process is leaving.
        for run in list(_RUNS.values()):
            run.end()

    def _accept(self) -> None:
        while True:
            try:
                connection, _ = self.socket.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        with connection:
            try:
                line = _read_line(connection)
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("the request is not a JSON object")
                answer = handle(request)
            except (ValueError, OSError) as exc:
                answer = {"ok": False, "type": "UsageError", "error": f"bad request: {exc}"}
            with contextlib.suppress(OSError):
                connection.sendall(json.dumps(answer).encode("utf-8") + b"\n")


def _read_line(connection: socket.socket) -> bytes:
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_REQUEST:
            raise ValueError("the request is too large")
    return bytes(data)


def handle(request: dict[str, Any]) -> dict[str, Any]:
    """Answer one control request: ``{"cmd": "interrupt" or "exec", "run": id, ...}``."""
    with _LOCK:
        run = _RUNS.get(str(request.get("run")))
    if run is None:
        return {"ok": False, "type": "RunNotFound", "error": f"no run {request.get('run')}"}
    command = request.get("cmd")
    if command not in ("interrupt", "exec"):
        return {"ok": False, "type": "UsageError", "error": f"unknown command {command!r}"}
    if not run.persistent:
        return {
            "ok": False,
            "type": "UnsupportedMode",
            "error": (
                f"{command} needs a persistent channel, and {run.provider or 'this provider'} "
                f"runs {run.function} on a one-shot channel"
            ),
        }
    try:
        if command == "interrupt":
            done, value = run.forward({"op": "interrupt"}, INTERRUPT_WAIT_S * 3)
            if not done:
                return {"ok": False, "type": "Timeout", "error": "the runtime did not answer"}
            return {"ok": True, **value}
        wait = float(request.get("timeout") or EXEC_WAIT_S)
        done, value = run.forward({"op": "inspect_exec", "source": request["source"]}, wait)
        if not done:
            return {
                "ok": False,
                "type": "Timeout",
                "error": f"no answer within {wait:g} s, and the source keeps running",
            }
        return {"ok": bool(value.get("ran")), **value}
    except (LetifyError, KeyError, TypeError) as exc:
        return {"ok": False, "type": type(exc).__name__, "error": str(exc)}


# -- the command line -------------------------------------------------------------


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def list_runs() -> list[dict[str, Any]]:
    """The live runs of this machine, oldest first. A record whose owner is gone is removed."""
    if not available():
        return []
    directory = runs_directory()
    rows: list[dict[str, Any]] = []
    try:
        names = sorted(directory.glob("*.json"))
    except OSError:
        return []
    for record in names:
        try:
            row = json.loads(record.read_text(encoding="utf-8"))
            pid = int(row["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _alive(pid):
            with contextlib.suppress(OSError):
                record.unlink()
            continue
        rows.append(row)
    rows.sort(key=lambda row: row.get("started_at", 0))
    return rows


def ask(run: str, command: dict[str, Any], *, wait: float = EXEC_WAIT_S + 30) -> dict[str, Any]:
    """Send ``command`` to the process that owns ``run`` and return its answer.

    Raises ``RunNotFound`` when no live run has that id, and ``UnsupportedMode`` on a
    platform with no control socket. Any other failure comes back as ``{"ok": False, ...}``.
    """
    if not available():
        raise UnsupportedMode("run control needs a Unix socket, which Windows does not have")
    row = next((row for row in list_runs() if row.get("run") == run), None)
    if row is None:
        raise RunNotFound(f"no run {run} is live on this machine")
    if not row.get("socket"):
        return {"ok": False, "type": "RuntimeFailure", "error": f"run {run} has no control socket"}
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(wait)
    try:
        client.connect(row["socket"])
        client.sendall(json.dumps({**command, "run": run}).encode("utf-8") + b"\n")
        answer = json.loads(_read_line(client))
    except (OSError, ValueError) as exc:
        return {"ok": False, "type": "RuntimeFailure", "error": f"run {run}: {exc}"}
    finally:
        client.close()
    if answer.get("type") == "RunNotFound":
        raise RunNotFound(f"no run {run} is live on this machine")
    return answer  # type: ignore[no-any-return]
