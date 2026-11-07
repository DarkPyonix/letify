"""A letify-link bootstrap carried over the signalling service's mailbox.

Owns talking to the signalling service (signed requests on kept HTTP connections, the short
long poll of a connect, the WebSocket watch of a standing wait), the signed envelope that
wraps every bootstrap line, ``SignalBootstrap`` (the ``Bootstrap`` of ``linkchannel.py`` over
that mailbox) and ``SignalAgent`` (the program on the address-less machine that serves a paired
node's bootstrap). It does not own the service (``letify-signal/``), signing (the sidecar's
``sign`` and ``verify``), the shim, the punch, pairing or the process that keeps an agent
running. Spec "Signalling service" and "Sidecar interface" item 9. Standard library only.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.client
import json
import queue
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from . import linkd, linkwebsocket

#: The service the owner operates.
DEFAULT_URL = "https://letify.darkpyonix.dev"
#: The signature domain of a request, and of an envelope.
REQUEST_DOMAIN = "letify-signal v1"
LINE_DOMAIN = "letify-signal line v1"
#: The largest message body the service takes, in bytes.
MAX_MESSAGE = 4096
#: Seconds an envelope's time may differ from the receiver's clock, in either direction.
WINDOW_S = 60
#: Seconds one long poll waits.
POLL_S = 25.0
#: Seconds a pump of the agent waits between checks of its stop flag.
AGENT_POLL_S = 2.0
#: Bounds of the timeouts the agent takes from a config line.
MAX_WAIT_S = 600
MAX_RESUME_S = 3600
DEFAULT_WAIT_S = 120
DEFAULT_RESUME_S = 30
#: Pings sent before the shim config.
PINGS = 3
#: Pings sent and not timed: the first pays the one-time TCP and TLS set-up of both connections.
WARM_PINGS = 1
#: The ops of a bootstrap line that the agent passes to the shim.
SHIM_OPS = ("rendezvous", "punch", "handover")
#: Bounds of the WebSocket keepalive, in seconds. See ``linkwebsocket``.
WS_PING_S = linkwebsocket.PING_S
WS_PONG_S = linkwebsocket.PONG_S
#: Idle HTTP connections a client keeps. Spec "Signalling service" item 1, "Connections".
MAX_IDLE = 4
#: Transient failures of the long poll a bootstrap rides out before it gives up.
POLL_RETRIES = 3

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX128 = re.compile(r"[0-9a-f]{128}")
_FIELDS = {"v", "from", "to", "sid", "n", "t", "line", "sig"}


class SignalError(OSError):
    """The signalling service did not answer, or refused a request."""


class Refused(Exception):
    """An envelope that must be dropped, with the reason."""


class _Daemon(Protocol):
    node_id: str

    def sign(self, domain: str, message: bytes) -> bytes: ...

    def verify(self, node: str, domain: str, message: bytes, signature: bytes) -> bool: ...


@dataclass(frozen=True)
class Incoming:
    """One message of a mailbox. ``sender`` is what the service says, not yet verified."""

    seq: int
    sender: str
    at: int
    body: bytes


def _user_agent() -> str:
    """The User-Agent of every request. Spec "Signalling service" item 1, "User-Agent".

    Cloudflare's Browser Integrity Check refuses the urllib default with error 1010.
    """
    from .. import __version__

    return f"letify/{__version__}"


def _proxied(url: str) -> bool:
    """Whether the environment sends a request for ``url`` through a proxy."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return False
    return parts.scheme in urllib.request.getproxies() and not urllib.request.proxy_bypass(host)


#: Failures of a reused connection that mean the service had already closed it.
_STALE = (
    http.client.RemoteDisconnected,
    http.client.CannotSendRequest,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


def _opener(url: str) -> urllib.request.OpenerDirector:
    """An opener that never sends a loopback request through a proxy."""
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class SignalClient:
    """Signed requests to the signalling service for the node of ``daemon``."""

    def __init__(
        self,
        daemon: _Daemon,
        base_url: str = DEFAULT_URL,
        *,
        clock: Callable[[], float] = time.time,
        nonce: Callable[[], str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.daemon = daemon
        self.base_url = base_url.rstrip("/")
        self._clock = clock
        self._nonce = nonce or (lambda: secrets.token_hex(16))
        self._timeout = timeout
        self._opener = _opener(self.base_url)
        parts = urllib.parse.urlsplit(self.base_url)
        self._scheme = parts.scheme
        self._netloc = parts.netloc
        self._proxied = _proxied(self.base_url)
        self._idle: list[http.client.HTTPConnection] = []
        self._idle_lock = threading.Lock()

    def close(self) -> None:
        """Close the idle connections. A later request opens a new one."""
        with self._idle_lock:
            idle, self._idle = self._idle, []
        for connection in idle:
            connection.close()

    @property
    def node_id(self) -> str:
        return self.daemon.node_id

    def _headers(
        self, method: str, target: str, body: bytes, signer_id: str | None
    ) -> dict[str, str]:
        stamp = str(int(self._clock()))
        nonce = self._nonce()
        text = f"{method}\n{target}\n{stamp}\n{nonce}\n{hashlib.sha256(body).hexdigest()}"
        signature = self.daemon.sign(REQUEST_DOMAIN, text.encode())
        return {
            "User-Agent": _user_agent(),
            "X-Letify-Node": signer_id or self.daemon.node_id,
            "X-Letify-Time": stamp,
            "X-Letify-Nonce": nonce,
            "X-Letify-Signature": signature.hex(),
        }

    def _take(self) -> tuple[http.client.HTTPConnection, bool]:
        """An idle connection and True, or a new one and False."""
        with self._idle_lock:
            if self._idle:
                return self._idle.pop(), True
        factory = (
            http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        )
        return factory(self._netloc), False

    def _give_back(self, connection: http.client.HTTPConnection) -> None:
        with self._idle_lock:
            if len(self._idle) < MAX_IDLE:
                self._idle.append(connection)
                return
        connection.close()

    def _exchange(
        self, method: str, target: str, body: bytes, timeout: float, signer_id: str | None
    ) -> tuple[int, Any, bytes]:
        """One request on a kept connection: status, headers and the whole reply body."""
        for _ in range(2):
            connection, reused = self._take()
            headers = self._headers(method, target, body, signer_id)
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            try:
                connection.request(
                    method, target, body=body if method == "POST" else None, headers=headers
                )
                response = connection.getresponse()
                data = response.read()
            except _STALE:
                connection.close()
                if reused:
                    continue
                raise
            except BaseException:
                connection.close()
                raise
            if response.will_close:
                connection.close()
            else:
                self._give_back(connection)
            return response.status, response.headers, data
        raise AssertionError("unreachable")  # pragma: no cover

    def _request(
        self,
        method: str,
        target: str,
        body: bytes,
        *,
        timeout: float,
        signer_id: str | None = None,
    ) -> dict[str, Any]:
        if self._proxied:
            return self._request_urllib(method, target, body, timeout, signer_id)
        try:
            status, headers, data = self._exchange(method, target, body, timeout, signer_id)
            if status >= 400:
                code = "unknown"
                with contextlib.suppress(ValueError, OSError, AttributeError):
                    code = str(json.loads(data).get("error", code))
                raise self._refused(status, code, headers.get("Retry-After"))
            return json.loads(data)  # type: ignore[no-any-return]
        except SignalError:
            raise
        except (http.client.HTTPException, TimeoutError, OSError, ValueError) as error:
            reason = getattr(error, "reason", error)
            raise SignalError(f"the signalling service did not answer: {reason}") from error

    @staticmethod
    def _refused(status: int, code: str, retry: str | None) -> SignalError:
        return SignalError(
            f"the signalling service refused the request: {code} (HTTP {status}"
            + (f", retry in {retry} s)" if retry else ")")
        )

    def _request_urllib(
        self, method: str, target: str, body: bytes, timeout: float, signer_id: str | None
    ) -> dict[str, Any]:
        """The request through ``urllib``, with a new connection, for a proxied service."""
        headers = self._headers(method, target, body, signer_id)
        request = urllib.request.Request(
            self.base_url + target, data=body if method == "POST" else None, method=method
        )
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return json.loads(response.read())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as error:
            code = "unknown"
            with contextlib.suppress(ValueError, OSError):
                code = str(json.loads(error.read()).get("error", code))
            raise self._refused(error.code, code, error.headers.get("Retry-After")) from error
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            reason = getattr(error, "reason", error)
            raise SignalError(f"the signalling service did not answer: {reason}") from error

    def send(self, to: str, body: bytes) -> int:
        """Put ``body`` in the mailbox of ``to`` and return its sequence number."""
        if len(body) > MAX_MESSAGE:
            raise SignalError(
                f"a message of {len(body)} bytes is over the {MAX_MESSAGE} byte cap of the "
                "signalling service"
            )
        return self.post_unchecked(to, body)

    def post_unchecked(self, to: str, body: bytes, signer_id: str | None = None) -> int:
        """``send`` without the size check, to test what the service itself refuses."""
        reply = self._request(
            "POST", f"/send/{to}", body, timeout=self._timeout, signer_id=signer_id
        )
        return int(reply["seq"])

    def recv_from(
        self, node: str, after: int, wait_s: float = POLL_S
    ) -> tuple[list[Incoming], int]:
        """Read the mailbox of ``node`` as this node, which the service allows only for itself."""
        wait = max(0, min(int(wait_s), int(POLL_S)))
        target = f"/recv/{node}?after={int(after)}&wait={wait}"
        reply = self._request("GET", target, b"", timeout=self._timeout + wait)
        messages = [
            Incoming(int(m["seq"]), str(m["from"]), int(m["at"]), base64.b64decode(m["body"]))
            for m in reply["messages"]
        ]
        return messages, int(reply["cursor"])

    def recv(self, after: int = 0, wait_s: float = POLL_S) -> tuple[list[Incoming], int]:
        """Acknowledge up to ``after``, then wait up to ``wait_s`` for a message above it."""
        return self.recv_from(self.daemon.node_id, after, wait_s)

    def watch(self) -> SignalWatch:
        """A standing wait on this node's mailbox over ``/ws/<node>``, for an idle receiver."""
        return SignalWatch(self)


class SignalWatch:
    """The standing wait of one node on its own mailbox, over a WebSocket.

    A pending long poll keeps the service's Durable Object awake, while an idle WebSocket lets
    it hibernate, so a receiver that waits all day uses this and not ``SignalClient.recv``. The
    service sends the messages it holds when a socket opens and each new one as it arrives, and
    deletes up to the ``seq`` the client acknowledges. ``read`` therefore keeps the highest
    ``seq`` it returned, skips what the service sends again after a reconnect, and acknowledges
    the messages of the previous call at the start of the next one, once the caller has handled
    them. ``read`` opens the socket when there is none, so the first call after a drop is the
    reconnect; a failure raises SignalError and the caller decides how long to wait.
    """

    def __init__(self, client: SignalClient) -> None:
        self._client = client
        self._ws: linkwebsocket.WebSocket | None = None
        self._last = 0
        self._acked = 0
        self._closed = False

    @property
    def node_id(self) -> str:
        return self._client.node_id

    def _open(self) -> linkwebsocket.WebSocket:
        client = self._client
        target = f"/ws/{client.node_id}"
        headers = client._headers("GET", target, b"", None)
        parts = urllib.parse.urlsplit(client.base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        url = f"{scheme}://{parts.netloc}{target}"
        try:
            return linkwebsocket.WebSocket.connect(
                url, headers, timeout=client._timeout, ping_s=WS_PING_S, pong_s=WS_PONG_S
            )
        except linkwebsocket.HandshakeRefused as error:
            retry = error.headers.get("retry-after")
            raise SignalError(
                f"the signalling service refused the request: {error.error_code()} "
                f"(HTTP {error.status}" + (f", retry in {retry} s)" if retry else ")")
            ) from error
        except (linkwebsocket.WebSocketError, OSError, ValueError) as error:
            raise SignalError(f"the signalling service did not answer: {error}") from error

    def _drop(self, error: Exception) -> SignalError:
        ws, self._ws = self._ws, None
        if ws is not None:
            ws.close()
        return SignalError(f"the signalling WebSocket ended: {error}")

    def read(self, wait_s: float) -> list[Incoming]:
        """Messages that arrived, waiting up to ``wait_s`` seconds when none is there yet.

        Sends no request while it waits. Returns early with every message already received
        once one arrives. Raises SignalError when the socket cannot be opened or ended.
        """
        if self._closed:
            return []
        if self._ws is None:
            self._ws = self._open()
        ws = self._ws
        found: list[Incoming] = []
        deadline = time.monotonic() + max(0.0, wait_s)
        try:
            if self._acked < self._last:
                ws.send_text(json.dumps({"ack": self._last}))
                self._acked = self._last
            while True:
                left = 0.0 if found else deadline - time.monotonic()
                text = ws.recv_text(max(0.0, left))
                if text is None:
                    break
                message = self._decode(text)
                if message is not None and message.seq > self._last:
                    self._last = message.seq
                    found.append(message)
        except linkwebsocket.WebSocketError as error:
            if self._closed:
                return []
            failure = self._drop(error)
            if not found:
                raise failure from error
        return found

    @staticmethod
    def _decode(text: str) -> Incoming | None:
        try:
            m = json.loads(text)
            body = base64.b64decode(m["body"])
            return Incoming(int(m["seq"]), str(m["from"]), int(m["at"]), body)
        except (ValueError, KeyError, TypeError):
            return None

    def close(self) -> None:
        """End the watch. A read in another thread returns an empty list."""
        self._closed = True
        ws = self._ws
        if ws is not None:
            ws.close()


# -- the envelope -------------------------------------------------------------------------


@dataclass(frozen=True)
class Opened:
    """A verified envelope."""

    sender: str
    sid: str
    n: int
    t: int
    line: str


def _line_message(sender: str, to: str, sid: str, n: int, t: int, line: str) -> bytes:
    return f"{sender}\n{to}\n{sid}\n{n}\n{t}\n{line}".encode()


def seal(daemon: _Daemon, to: str, sid: str, n: int, line: str, *, t: int | None = None) -> bytes:
    """The envelope for ``line``, signed by ``daemon``'s node. Raises SignalError over the cap."""
    stamp = int(time.time()) if t is None else t
    signature = daemon.sign(LINE_DOMAIN, _line_message(daemon.node_id, to, sid, n, stamp, line))
    body = json.dumps(
        {
            "v": 1,
            "from": daemon.node_id,
            "to": to,
            "sid": sid,
            "n": n,
            "t": stamp,
            "line": line,
            "sig": signature.hex(),
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if len(body) > MAX_MESSAGE:
        raise SignalError(
            f"a line of {len(line.encode())} bytes makes an envelope of {len(body)} bytes, over "
            f"the {MAX_MESSAGE} byte cap of the signalling service; lines are not split"
        )
    return body


def open_envelope(
    daemon: _Daemon,
    body: bytes,
    *,
    peer: str,
    sid: str | None,
    last_n: int,
    now: float | None = None,
) -> Opened:
    """Verify an envelope for ``daemon``'s node, or raise Refused with the reason.

    ``peer`` is the node the envelope must come from, ``sid`` the bootstrap it must belong to
    (None accepts any), and ``last_n`` the highest ``n`` already taken from that peer and sid.
    The cheap checks run first, because the signature costs a call to the sidecar.
    """
    try:
        envelope = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise Refused("the message is not JSON") from None
    if not isinstance(envelope, dict) or set(envelope) != _FIELDS or envelope["v"] != 1:
        raise Refused("the envelope has other fields than the format")
    sender, to, env_sid, n, t = (envelope[k] for k in ("from", "to", "sid", "n", "t"))
    line, sig = envelope["line"], envelope["sig"]
    ok = (
        isinstance(sender, str) and _HEX64.fullmatch(sender)
        and isinstance(to, str) and _HEX64.fullmatch(to)
        and isinstance(env_sid, str) and _HEX32.fullmatch(env_sid)
        and type(n) is int and n >= 1
        and type(t) is int
        and isinstance(line, str)
        and isinstance(sig, str) and _HEX128.fullmatch(sig)
    )  # fmt: skip
    if not ok:
        raise Refused("the envelope has other fields than the format")
    if sender != peer:
        raise Refused(f"the envelope is not from the expected node ({sender[:12]})")
    if to != daemon.node_id:
        raise Refused("the envelope is addressed to another node")
    if sid is not None and env_sid != sid:
        raise Refused("the envelope belongs to another bootstrap")
    if abs(t - (time.time() if now is None else now)) > WINDOW_S:
        raise Refused("the envelope is stale")
    if n <= last_n:
        raise Refused(f"the envelope is replayed (n {n}, already took {last_n})")
    message = _line_message(sender, to, env_sid, n, t, line)
    if not daemon.verify(sender, LINE_DOMAIN, message, bytes.fromhex(sig)):
        raise Refused("the envelope signature does not verify")
    return Opened(sender, env_sid, n, t, line)


def lean_config(config: dict[str, Any], *, version: str | None = None) -> dict[str, Any]:
    """The config line a signalled bootstrap sends: no path, command or source of the machine."""
    if version is None:
        import letify

        version = letify.__version__
    return {
        "op": "config",
        "version": version,
        "wait_s": int(config.get("wait_s", DEFAULT_WAIT_S)),
        "resume_s": int(config.get("resume_s", DEFAULT_RESUME_S)),
    }


# -- the local side -----------------------------------------------------------------------


class SignalBootstrap:
    """A ``Bootstrap`` whose lines travel as signed envelopes in the mailbox of the peer.

    ``peer`` is the node id of the agent. The lines the shim prints come back in the mailbox of
    this node, and an envelope that is not from ``peer``, not for this bootstrap, stale,
    replayed or wrongly signed is dropped and counted in ``stderr_text``.
    """

    rtt_s: float | None = None

    def __init__(
        self,
        client: SignalClient,
        peer: str,
        *,
        sid: str | None = None,
        version: str | None = None,
        ping_timeout_s: float = 30.0,
    ) -> None:
        if version is None:
            import letify

            version = letify.__version__
        self._client = client
        self._peer = peer
        self._sid = sid or secrets.token_hex(16)
        self._version = version
        self._ping_timeout_s = ping_timeout_s
        self._out_n = 0
        self._in_n = 0
        self._sent_config = False
        self._send_lock = threading.Lock()
        self._inbound: queue.Queue[bytes | None] = queue.Queue()
        self._pongs: dict[int, threading.Event] = {}
        self._notes: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- receiving --------------------------------------------------------------------------

    def listen_only(self) -> None:
        """Start reading the mailbox without pinging the agent."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def start(self) -> None:
        """Start reading, ping the agent to measure the round trip on warm connections, or raise.

        Spec "Signalling service" item 8. The first ping warms the connections and is not timed.
        """
        self.listen_only()
        samples: list[float] = []
        for number in range(1, PINGS + 1):
            arrived = threading.Event()
            self._pongs[number] = arrived
            began = time.monotonic()
            self._send_json({"op": "ping", "id": number})
            if not arrived.wait(self._ping_timeout_s):
                self.close()
                raise SignalError(
                    f"the agent {self._peer[:12]} did not answer in {self._ping_timeout_s:.0f} s"
                    + (f": {self.stderr_text()}" if self._notes else "")
                )
            if number > WARM_PINGS:
                samples.append(time.monotonic() - began)
        self.rtt_s = max(samples)

    def _note(self, text: str) -> None:
        self._notes.append(text)
        del self._notes[:-20]

    def _loop(self) -> None:
        after = 0
        failures = 0
        while not self._stop.is_set():
            try:
                messages, _ = self._client.recv(after, wait_s=POLL_S)
            except (SignalError, linkd.LinkError) as error:
                failures += 1
                if self._stop.is_set():
                    return
                if failures >= POLL_RETRIES:
                    self._note(str(error))
                    self._inbound.put(None)
                    return
                self._stop.wait(1.0)
                continue
            failures = 0
            if self._stop.is_set():
                return
            for message in messages:
                after = max(after, message.seq)
                self._take(message)

    def _take(self, message: Incoming) -> None:
        try:
            opened = open_envelope(
                self._client.daemon,
                message.body,
                peer=self._peer,
                sid=self._sid,
                last_n=self._in_n,
            )
        except Refused as error:
            self._note(str(error))
            return
        self._in_n = opened.n
        try:
            event = json.loads(opened.line)
        except ValueError:
            self._note("a line of the agent is not JSON")
            return
        if not isinstance(event, dict):
            return
        if event.get("event") == "pong":
            arrived = self._pongs.get(event.get("id"))  # type: ignore[arg-type]
            if arrived is not None:
                arrived.set()
            return
        if event.get("event") == "ready" and event.get("version") != self._version:
            event = {
                "event": "error",
                "error": (
                    f"letify version mismatch: client {self._version}, agent {event.get('version')}"
                ),
            }
            self._inbound.put(json.dumps(event).encode())
            return
        self._inbound.put(opened.line.encode())

    # -- the Bootstrap protocol -------------------------------------------------------------

    def _send_json(self, payload: dict[str, Any]) -> None:
        with self._send_lock:
            self._out_n += 1
            body = seal(
                self._client.daemon,
                self._peer,
                self._sid,
                self._out_n,
                json.dumps(payload, separators=(",", ":")),
            )
            self._client.send(self._peer, body)

    def send_line(self, line: bytes | str) -> None:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        text = text.rstrip("\n")
        if not self._sent_config:
            self._sent_config = True
            with contextlib.suppress(ValueError):
                config = json.loads(text)
                if isinstance(config, dict) and "op" not in config:
                    text = json.dumps(lean_config(config, version=self._version))
        try:
            with self._send_lock:
                self._out_n += 1
                body = seal(self._client.daemon, self._peer, self._sid, self._out_n, text)
                self._client.send(self._peer, body)
        except SignalError as error:
            if "cap" in str(error):
                raise
            self._note(str(error))
            self._inbound.put(None)

    def recv_line(self) -> bytes | None:
        """The next line the shim printed, blocking, or None once the stream has ended."""
        return self._inbound.get()

    def stderr_text(self) -> str:
        return "; ".join(self._notes[-5:])

    def close(self) -> None:
        """Stop reading the mailbox. A long poll in flight ends by itself."""
        self._stop.set()
        self._inbound.put(None)


# -- the agent ----------------------------------------------------------------------------


class _Session:
    """The bootstrap a paired node is running through the agent."""

    def __init__(self, sender: str, sid: str) -> None:
        self.sender = sender
        self.sid = sid
        self.last_n = 0
        self.out_n = 0
        self.proc: subprocess.Popen[bytes] | None = None
        self.lock = threading.Lock()


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    if type(value) is not int:
        return default
    return max(low, min(high, value))


class SignalAgent:
    """The program on the address-less machine that serves the bootstrap of a paired node.

    It reads the mailbox of its node, and for a node in ``paired`` that sends a valid config
    line it runs ``linkshim.py`` with a configuration built from its own arguments. Nothing a
    line says names a program, a path or a source. Spec "Signalling service" item 6.
    """

    def __init__(
        self,
        client: SignalClient,
        *,
        paired: Iterable[str],
        binary: str,
        python: str,
        bootstrap: str,
        relay: str,
        version: str | None = None,
        bind: str = "0.0.0.0:0",
        worker_command: str | None = None,
        relay_command: str | None = None,
        ssh_target: str | None = None,
    ) -> None:
        if version is None:
            import letify

            version = letify.__version__
        self._client = client
        self._paired = set(paired)
        self._own = {
            "binary": binary,
            "python": python,
            "bootstrap": bootstrap,
            "relay": relay,
            "bind": bind,
        }
        if worker_command:
            self._own["worker_command"] = worker_command
        if relay_command:
            self._own["relay_command"] = relay_command
        if ssh_target:
            # The agent's own choice; no line from a paired node can name the target.
            self._own["ssh_target"] = ssh_target
        self._version = version
        self._stop = threading.Event()
        self._session: _Session | None = None
        self._handed_over = threading.Event()
        #: Bootstraps started, and the reasons messages were dropped (the last 50).
        self.started = 0
        self.dropped: list[str] = []

    def _drop(self, reason: str) -> None:
        self.dropped.append(reason)
        del self.dropped[:-50]

    def shim_config(self, sender: str, line: dict[str, Any]) -> dict[str, Any]:
        """The shim configuration for ``sender``'s config ``line``: the agent's own, plus two
        timeouts. Every other field of the line is ignored."""
        config: dict[str, Any] = dict(self._own)
        config["allow"] = sender
        config["wait_s"] = _clamp(line.get("wait_s"), 1, MAX_WAIT_S, DEFAULT_WAIT_S)
        config["resume_s"] = _clamp(line.get("resume_s"), 1, MAX_RESUME_S, DEFAULT_RESUME_S)
        return config

    def shim_command(self, sender: str, command: dict[str, Any]) -> dict[str, Any] | None:
        """The line to give the shim for ``command``, or None for an op the agent does not pass.

        The node the command names is always the sender, whatever the line says.
        """
        op = command.get("op")
        if op == "rendezvous":
            return {"op": "rendezvous", "from": sender, "message": command.get("message", "")}
        if op == "punch":
            allowed = (
                "sockets",
                "udp",
                "observers",
                "public_ip",
                "lead_ms",
                "window_ms",
                "answer_margin_ms",
            )
            return {
                "op": "punch",
                "peer": sender,
                **{k: command[k] for k in allowed if k in command},
            }
        if op == "handover":
            return {"op": "handover"}
        return None

    # -- serving ----------------------------------------------------------------------------

    def stop(self) -> None:
        """End ``serve`` and the shim it started, unless that has been handed over."""
        self._stop.set()
        session = self._session
        if session is not None and session.proc is not None and not self._handed_over.is_set():
            with contextlib.suppress(OSError):
                session.proc.kill()

    def serve(self, timeout: float = 300.0) -> None:
        """Serve one bootstrap: return after the hand-over, after ``stop`` or at the timeout."""
        deadline = time.monotonic() + timeout
        after = 0
        failures = 0
        while not self._stop.is_set() and not self._handed_over.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                messages, _ = self._client.recv(after, wait_s=min(AGENT_POLL_S, remaining))
            except SignalError as error:
                failures += 1
                self._drop(str(error))
                if failures >= POLL_RETRIES * 5:
                    break
                self._stop.wait(1.0)
                continue
            failures = 0
            for message in messages:
                after = max(after, message.seq)
                self._handle(message)
        session = self._session
        if session is not None and session.proc is not None and not self._handed_over.is_set():
            with contextlib.suppress(OSError):
                session.proc.kill()

    @property
    def alive(self) -> bool:
        """Whether this agent's shim is running."""
        session = self._session
        return session is not None and session.proc is not None and session.proc.poll() is None

    def kill(self) -> None:
        """End the shim even after the hand-over, which ``stop`` leaves to finish."""
        self._stop.set()
        session = self._session
        if session is not None and session.proc is not None:
            with contextlib.suppress(OSError):
                session.proc.kill()

    def handle(self, message: Incoming) -> None:
        """Take one message of the mailbox, for a caller that owns the read loop."""
        self._handle(message)

    def _handle(self, message: Incoming) -> None:
        if message.sender not in self._paired:
            self._drop(f"a message from {message.sender[:12]} which is not paired")
            return
        session = self._session
        if session is not None and session.sender != message.sender:
            self._drop("a second bootstrap while one is running")
            return
        try:
            opened = open_envelope(
                self._client.daemon,
                message.body,
                peer=message.sender,
                sid=session.sid if session else None,
                last_n=session.last_n if session else 0,
            )
        except Refused as error:
            self._drop(str(error))
            return
        if session is None:
            session = self._session = _Session(opened.sender, opened.sid)
        session.last_n = opened.n
        try:
            line = json.loads(opened.line)
        except ValueError:
            self._drop("a line that is not JSON")
            return
        if not isinstance(line, dict):
            self._drop("a line that is not an object")
            return
        op = line.get("op")
        if op == "ping":
            self._reply(session, {"event": "pong", "id": line.get("id")})
        elif op == "config":
            self._start_shim(session, line)
        elif op in SHIM_OPS:
            command = self.shim_command(session.sender, line)
            if session.proc is None or command is None:
                self._drop(f"the op {op!r} before the config line")
                return
            self._to_shim(session, command)
        else:
            self._drop(f"the op {op!r}, which the agent does not pass")

    def _reply(self, session: _Session, event: dict[str, Any]) -> None:
        with session.lock:
            session.out_n += 1
            try:
                body = seal(
                    self._client.daemon,
                    session.sender,
                    session.sid,
                    session.out_n,
                    json.dumps(event, separators=(",", ":")),
                )
                self._client.send(session.sender, body)
            except SignalError as error:
                self._drop(str(error))

    def _to_shim(self, session: _Session, command: dict[str, Any]) -> None:
        assert session.proc is not None and session.proc.stdin is not None
        with contextlib.suppress(OSError, ValueError):
            session.proc.stdin.write(json.dumps(command).encode() + b"\n")
            session.proc.stdin.flush()

    def _start_shim(self, session: _Session, line: dict[str, Any]) -> None:
        if session.proc is not None:
            self._drop("a second config line")
            return
        if line.get("version") != self._version:
            self._reply(
                session,
                {
                    "event": "error",
                    "error": (
                        f"letify version mismatch: client {line.get('version')}, "
                        f"agent {self._version}"
                    ),
                },
            )
            return
        from . import linkchannel

        config = self.shim_config(session.sender, line)
        session.proc = subprocess.Popen(
            [config["python"], "-u", "-c", linkchannel.shim_source()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.started += 1
        self._to_shim(session, config)
        threading.Thread(target=self._forward, args=(session,), daemon=True).start()

    def _forward(self, session: _Session) -> None:
        """Move the lines the shim prints into the mailbox of the paired node."""
        proc = session.proc
        assert proc is not None and proc.stdout is not None
        ready = False
        for raw in iter(proc.stdout.readline, b""):
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("event") == "ready":
                ready = True
                event["version"] = self._version
            self._reply(session, event)
            if event.get("event") == "handed_over":
                self._handed_over.set()
                break
        if not self._handed_over.is_set():
            stderr = b""
            with contextlib.suppress(OSError, ValueError):
                assert proc.stderr is not None
                stderr = proc.stderr.read()
            if not self._stop.is_set():
                note = "the shim exited" + ("" if ready else " before it was ready")
                self._reply(
                    session,
                    {
                        "event": "error",
                        "error": f"{note}: {stderr.decode('utf-8', 'replace')[-400:]}",
                    },
                )
        threading.Thread(target=proc.wait, daemon=True).start()


__all__ = [
    "DEFAULT_URL",
    "MAX_MESSAGE",
    "Incoming",
    "Opened",
    "Refused",
    "SignalAgent",
    "SignalBootstrap",
    "SignalClient",
    "SignalError",
    "SignalWatch",
    "lean_config",
    "open_envelope",
    "seal",
]
