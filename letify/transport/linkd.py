"""The Python side of the letify-link sidecar: find it, start it, speak its control protocol.

Owns locating the ``letify-link`` executable, starting one per node, the framed JSON control
channel, and the sockets that carry a stream. It does not own the rendezvous (the caller
supplies an object that delivers a message to a peer), the candidate addresses, or any use of
the streams: a stream is a connected ``socket.socket`` and the caller reads and writes it.
Spec "Sidecar interface".
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from ..errors import LetifyError, RuntimeFailure

#: Where the wheel puts letify-core's binaries, beside letify-quic.
LIB_DIR = Path(__file__).resolve().parent.parent / "remoting" / "lib"
#: What cargo names the sidecar on each platform.
BINARY = "letify-link.exe" if sys.platform == "win32" else "letify-link"
#: Said when the wheel for this platform carries no binary, so the reason is one string.
MISSING = "letify-link is not in this wheel"
#: The environment variable that names an executable to use before any other.
ENV_BINARY = "LETIFY_LINK_BIN"
#: The longest control message the sidecar accepts.
MAX_MESSAGE = 1 << 20


class LinkMissing(LetifyError):
    """The wheel has no letify-link. letify does not fall back to local execution."""


class LinkError(LetifyError):
    """The sidecar refused a request or the control channel failed. ``kind`` is the sidecar's."""

    def __init__(self, message: str, kind: str = "usage"):
        self.kind = kind
        super().__init__(message)


class LinkInfrastructureError(LinkError, RuntimeFailure):
    """A failure a new attempt may cure: no path answered, the handshake was refused."""

    def __init__(self, message: str):
        RuntimeFailure.__init__(self, message)
        self.kind = "infrastructure"


class LinkClosed(LinkError):
    """The session or the control channel was closed."""

    def __init__(self, message: str = "closed"):
        super().__init__(message, "closed")


def binary_path() -> Path | None:
    """The sidecar to run: ``LETIFY_LINK_BIN``, then the wheel's copy, then one on PATH."""
    override = os.environ.get(ENV_BINARY)
    if override:
        return Path(override)
    bundled = LIB_DIR / BINARY
    if bundled.is_file():
        return bundled
    found = shutil.which("letify-link")
    return Path(found) if found else None


def require_binary() -> Path:
    """``binary_path`` or ``LinkMissing`` with the reason the spec fixes."""
    found = binary_path()
    if found is None or not found.is_file():
        raise LinkMissing(MISSING)
    return found


class Rendezvous(Protocol):
    """What the caller supplies: deliver ``message`` to node ``to`` through a bootstrap."""

    def send(self, to: str, message: bytes) -> None: ...


def _connect(
    address: str, timeout: float | None = 10.0, secret: str | None = None
) -> socket.socket:
    """Connect to a sidecar socket. A ``tcp:`` stream socket needs its ``secret`` first."""
    kind, _, rest = address.partition(":")
    if kind == "unix":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(rest)
    elif kind == "tcp":
        host, _, port = rest.rpartition(":")
        sock = socket.create_connection((host, int(port)), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if secret:
            sock.sendall(bytes.fromhex(secret))
    else:
        raise LinkError(f"unknown socket address {address!r}")
    sock.settimeout(None)
    return sock


def _authenticate(sock: socket.socket, token: str, timeout: float = 10.0) -> None:
    """Send the ``auth`` message the loopback control channel requires and check the reply."""
    body = json.dumps({"op": "auth", "token": token, "id": 0}).encode()
    sock.settimeout(timeout)
    try:
        sock.sendall(struct.pack(">I", len(body)) + body)
        head = b""
        while len(head) < 4:
            chunk = sock.recv(4 - len(head))
            if not chunk:
                raise LinkError("the sidecar closed the control channel at the token")
            head += chunk
        (length,) = struct.unpack(">I", head)
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise LinkError("the sidecar closed the control channel at the token")
            data += chunk
    except OSError as error:
        raise LinkError(f"the control token was not accepted: {error}") from error
    if not json.loads(data).get("ok"):
        raise LinkError("the sidecar refused the control token")
    sock.settimeout(None)


class LinkSession:
    """One encrypted session to a peer. Streams are sockets; paths are not visible."""

    def __init__(self, daemon: LinkDaemon, number: int, peer: str):
        self._daemon = daemon
        self.number = number
        self.peer = peer
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._streams: queue.Queue[dict[str, Any]] = queue.Queue()
        self.closed = threading.Event()
        self.close_reason = ""

    def open_stream(self, priority: str = "interactive") -> socket.socket:
        """Open a stream to the peer and return the local socket that carries it.

        ``priority`` is ``"interactive"`` (control frames and small messages, served first) or
        ``"bulk"`` (transfers, which share what interactive streams leave).
        """
        reply = self._daemon.request(
            {"op": "open_stream", "session": self.number, "priority": priority}
        )
        return _connect(reply["socket"], secret=reply.get("secret"))

    def accept_stream(self, timeout: float | None = None) -> socket.socket:
        """Wait for a stream the peer opened and return its local socket."""
        try:
            event = self._streams.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("no incoming stream") from None
        if event.get("event") == "session_closed":
            self._streams.put(event)
            raise LinkClosed(f"session closed: {event.get('reason', '')}")
        return _connect(event["socket"], secret=event.get("secret"))

    def paths(self) -> list[dict[str, Any]]:
        """What the session measured on each live path."""
        return self._daemon.request({"op": "paths", "session": self.number})["paths"]

    def wait_event(self, name: str, timeout: float | None = None) -> dict[str, Any]:
        """The next event called ``name`` (``path_added``, ``path_probe``); others are skipped."""
        while True:
            try:
                event = self.events.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(f"no {name} event") from None
            if event["event"] == name:
                return event

    def close(self) -> None:
        """End the session. The peer's streams end as after a clean close."""
        if self.closed.is_set():
            return
        with contextlib.suppress(LinkError, OSError):
            self._daemon.request({"op": "close", "session": self.number})
        self._mark_closed("closed locally")

    def _mark_closed(self, reason: str) -> None:
        if not self.closed.is_set():
            self.close_reason = reason
            self.closed.set()
            self._streams.put({"event": "session_closed", "reason": reason})

    def _dispatch(self, event: dict[str, Any]) -> None:
        if event["event"] == "stream_incoming":
            self._streams.put(event)
        elif event["event"] == "session_closed":
            self._mark_closed(event.get("reason", ""))
        else:
            self.events.put(event)


class LinkListener:
    """Accepts sessions that peers in ``allow`` start."""

    def __init__(self, daemon: LinkDaemon, number: int, address: str | None):
        self._daemon = daemon
        self.number = number
        self.address = address
        self._sessions: queue.Queue[LinkSession] = queue.Queue()

    def accept_session(self, timeout: float | None = None) -> LinkSession:
        """Wait for a peer to start a session."""
        try:
            return self._sessions.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("no incoming session") from None

    def close(self) -> None:
        with contextlib.suppress(LinkError, OSError):
            self._daemon.request({"op": "close", "listener": self.number})


class LinkDaemon:
    """One running sidecar. Closing it, or the death of this process, ends every session."""

    def __init__(
        self,
        key_path: str | os.PathLike[str],
        *,
        binary: str | os.PathLike[str] | None = None,
        rendezvous: Rendezvous | None = None,
        start_timeout: float = 20.0,
        loopback: bool = False,
    ):
        self._binary = Path(binary) if binary else require_binary()
        self.rendezvous = rendezvous
        self._dir = Path(tempfile.mkdtemp(prefix="ll"))
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._sessions: dict[int, LinkSession] = {}
        self._listeners: dict[int, LinkListener] = {}
        self._backlog: dict[int, list[dict[str, Any]]] = {}
        self._outbox: queue.Queue[tuple[str, bytes] | None] = queue.Queue()
        self._closed = False
        control = "tcp" if loopback or sys.platform == "win32" else str(self._dir / "c")
        self._stderr_path = self._dir / "stderr"
        try:
            with open(self._stderr_path, "wb") as stderr:
                self._proc = subprocess.Popen(
                    [str(self._binary), "serve", "--control", control, "--key", str(key_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                )
        except OSError as error:
            self._cleanup()
            raise LinkMissing(f"{MISSING}: cannot run {self._binary}: {error}") from error
        ready = self._read_ready(start_timeout)
        self.node_id: str = ready["node_id"]
        try:
            self.control_address: str = ready["control"]
            self._sock = _connect(self.control_address)
            if ready.get("token"):
                _authenticate(self._sock, ready["token"])
        except (OSError, LinkError) as error:
            self.close()
            raise LinkError(f"cannot reach the sidecar control channel: {error}") from error
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()

    # -- start up ---------------------------------------------------------------------------

    def _read_ready(self, timeout: float) -> dict[str, Any]:
        lines: queue.Queue[str] = queue.Queue()

        def first() -> None:
            assert self._proc.stdout is not None
            lines.put(self._proc.stdout.readline().decode("utf-8", "replace"))

        threading.Thread(target=first, daemon=True).start()
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise LinkError(f"the sidecar printed no ready line in {timeout:.0f} s") from None
        try:
            ready = json.loads(line)
            if ready.get("event") != "ready":
                raise ValueError(line)
        except ValueError:
            detail = self._stderr_text()
            self.close()
            raise LinkError(f"the sidecar did not start: {detail or line.strip()}") from None
        return ready

    def _stderr_text(self) -> str:
        try:
            return self._stderr_path.read_text(errors="replace").strip()[-500:]
        except OSError:
            return ""

    def _drain_stdout(self) -> None:
        # Keeps the pipe empty so the sidecar never blocks or takes SIGPIPE on a print.
        assert self._proc.stdout is not None
        with contextlib.suppress(OSError, ValueError):
            while self._proc.stdout.read(4096):
                pass

    # -- the control channel ----------------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        chunks = []
        while n:
            data = self._sock.recv(min(n, 1 << 16))
            if not data:
                raise EOFError
            chunks.append(data)
            n -= len(data)
        return b"".join(chunks)

    def _read_loop(self) -> None:
        try:
            while True:
                (length,) = struct.unpack(">I", self._recv_exact(4))
                message = json.loads(self._recv_exact(length))
                if "event" in message:
                    self._on_event(message)
                else:
                    with self._lock:
                        waiter = self._pending.pop(message.get("id"), None)
                    if waiter is not None:
                        waiter.put(message)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            self._fail_all()

    def _fail_all(self) -> None:
        with self._lock:
            waiters = list(self._pending.values())
            self._pending.clear()
            sessions = list(self._sessions.values())
        for waiter in waiters:
            waiter.put({"ok": False, "kind": "closed", "error": "the sidecar is gone"})
        for session in sessions:
            session._mark_closed("the sidecar is gone")
        self._outbox.put(None)

    def _on_event(self, event: dict[str, Any]) -> None:
        name = event["event"]
        if name == "rendezvous_out":
            self._outbox.put((event["to"], bytes.fromhex(event["message"])))
        elif name == "session_incoming":
            with self._lock:
                listener = self._listeners.get(event["listener"])
            if listener is None:
                return
            session = self._attach(event["session"], event["peer"])
            listener._sessions.put(session)
        elif "session" in event:
            with self._lock:
                session = self._sessions.get(event["session"])
                if session is None:
                    # The event can overtake the reply that names the session.
                    self._backlog.setdefault(event["session"], []).append(event)
                    return
            session._dispatch(event)

    def _attach(self, number: int, peer: str) -> LinkSession:
        session = LinkSession(self, number, peer)
        with self._lock:
            self._sessions[number] = session
            backlog = self._backlog.pop(number, [])
        for event in backlog:
            session._dispatch(event)
        return session

    def _send_loop(self) -> None:
        # Rendezvous messages leave on their own thread: a bootstrap may take seconds.
        while (item := self._outbox.get()) is not None:
            to, message = item
            if self.rendezvous is None:
                continue
            with contextlib.suppress(Exception):
                self.rendezvous.send(to, message)

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the reply fields, or raise the error the sidecar gave."""
        number = next(self._ids)
        waiter: queue.Queue[dict[str, Any]] = queue.Queue(1)
        with self._lock:
            if self._closed:
                raise LinkClosed("the daemon is closed")
            self._pending[number] = waiter
        body = json.dumps({**message, "id": number}).encode()
        if len(body) > MAX_MESSAGE:
            raise LinkError("control message over 1 MiB")
        try:
            self._sock.sendall(struct.pack(">I", len(body)) + body)
        except OSError as error:
            raise LinkClosed(f"the control channel is closed: {error}") from error
        reply = waiter.get()
        if reply.get("ok"):
            return reply
        kind = reply.get("kind", "usage")
        text = reply.get("error", "request failed")
        if kind == "infrastructure":
            raise LinkInfrastructureError(text)
        if kind == "closed":
            raise LinkClosed(text)
        raise LinkError(text, kind)

    # -- the API ----------------------------------------------------------------------------

    def deliver(self, from_node: str, message: bytes) -> None:
        """Hand the sidecar a rendezvous message that a bootstrap received from ``from_node``."""
        self.request({"op": "rendezvous_message", "from": from_node, "message": message.hex()})

    def listen(self, allow: Iterable[str], bind: str | None = None) -> LinkListener:
        """Accept sessions from the nodes in ``allow``, on a TCP address when ``bind`` is given."""
        request: dict[str, Any] = {"op": "listen", "allow": list(allow)}
        if bind:
            request["bind"] = bind
        reply = self.request(request)
        listener = LinkListener(self, reply["listener"], reply.get("addr"))
        with self._lock:
            self._listeners[listener.number] = listener
        return listener

    def open_session(
        self,
        peer_node_id: str,
        candidates: Iterable[Any],
        rendezvous: Rendezvous | None = None,
        *,
        config: dict[str, int] | None = None,
    ) -> LinkSession:
        """Start a session to ``peer_node_id`` over ``candidates``.

        A candidate is an address string ``host:port`` or a dict with ``name`` and ``tcp`` (or
        ``punched``). ``rendezvous``, when given, replaces the daemon's for later messages.
        """
        if rendezvous is not None:
            self.rendezvous = rendezvous
        wire = []
        for i, candidate in enumerate(candidates):
            if isinstance(candidate, str):
                candidate = {"name": f"tcp{i}", "tcp": candidate}
            wire.append(candidate)
        request: dict[str, Any] = {"op": "open_session", "peer": peer_node_id, "candidates": wire}
        if config:
            request["config"] = config
        reply = self.request(request)
        return self._attach(reply["session"], reply["peer"])

    def punch(
        self,
        peer_node_id: str,
        role: str,
        *,
        sockets: int = 4,
        listener: LinkListener | None = None,
        public_ip: str | None = None,
        lead_ms: int | None = None,
        window_ms: int | None = None,
    ) -> list[str]:
        """Run the TCP punch plan through the rendezvous.

        The initiator gets the names of its punched sockets, to pass as ``punched`` candidates.
        The responder hands its sockets to ``listener`` and gets an empty list.
        """
        request: dict[str, Any] = {
            "op": "punch",
            "peer": peer_node_id,
            "role": role,
            "sockets": sockets,
        }
        if listener is not None:
            request["listener"] = listener.number
        for key, value in (
            ("public_ip", public_ip),
            ("lead_ms", lead_ms),
            ("window_ms", window_ms),
        ):
            if value is not None:
                request[key] = value
        return self.request(request).get("punched", [])

    @property
    def pid(self) -> int:
        return self._proc.pid

    def close(self) -> None:
        """Close the control channel; the sidecar exits and so does every session."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        sock = getattr(self, "_sock", None)
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                sock.close()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._cleanup()

    def _cleanup(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self) -> LinkDaemon:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "BINARY",
    "ENV_BINARY",
    "MISSING",
    "LinkClosed",
    "LinkDaemon",
    "LinkError",
    "LinkInfrastructureError",
    "LinkListener",
    "LinkMissing",
    "LinkSession",
    "Rendezvous",
    "binary_path",
    "require_binary",
]
