"""A minimal RFC 6455 WebSocket client on the standard library.

Owns the opening handshake (``GET`` with an upgrade, ``Sec-WebSocket-Accept`` check), frame
reading and writing with client masking, text messages (fragmented ones are put together),
ping and pong, the close handshake, and a keepalive (the text ``ping``, which the service
answers with the text ``pong``) that treats a missing answer as a dead connection. It does not
own signing the upgrade request, the meaning of a message, reconnecting, proxies or
permessage-deflate (``linksignal.py`` builds the first on top of it, and the others are not
offered). Binary messages are read and dropped. Spec "Signalling service" item 1.
Standard library only.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import socket
import ssl
import struct
import time
import urllib.parse
from collections.abc import Mapping

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
#: The largest message accepted, in bytes. The signalling service sends well under 8 KiB.
MAX_MESSAGE = 1 << 20
#: The largest HTTP reply head accepted during the handshake, in bytes.
MAX_HEAD = 16384
#: Seconds of silence before a ping, and seconds a ping may stay unanswered.
PING_S = 30.0
PONG_S = 10.0
#: The text keepalive. The signalling service registers it as a Hibernation auto-response,
#: which the runtime answers without waking the object. A ``pong`` is never returned as a message.
PING_TEXT = "ping"
PONG_TEXT = "pong"

_CONTINUATION, _TEXT, _BINARY, _CLOSE, _PING, _PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(OSError):
    """The connection failed, ended, or sent something the protocol does not allow."""


class HandshakeRefused(WebSocketError):
    """The server answered the upgrade request with something other than 101."""

    def __init__(self, status: int, body: bytes, headers: Mapping[str, str]) -> None:
        super().__init__(f"the server refused the WebSocket upgrade with HTTP {status}")
        self.status = status
        self.body = body
        self.headers = headers

    def error_code(self) -> str:
        """The ``error`` field of a JSON reply body, or ``unknown``."""
        with contextlib.suppress(ValueError, AttributeError):
            return str(json.loads(self.body).get("error", "unknown"))
        return "unknown"


class WebSocket:
    """One open connection. Use from one thread; ``close`` may be called from another."""

    def __init__(self, sock: socket.socket, *, ping_s: float = PING_S, pong_s: float = PONG_S):
        self._sock = sock
        self._buffer = bytearray()
        self._fragments = bytearray()
        self._text = False
        self._in_message = False
        self._ping_s = ping_s
        self._pong_s = pong_s
        self._heard = time.monotonic()
        self._pong_due: float | None = None
        self.closed = False

    # -- opening ----------------------------------------------------------------------------

    @classmethod
    def connect(
        cls,
        url: str,
        headers: Mapping[str, str] | None = None,
        *,
        timeout: float = 10.0,
        ping_s: float = PING_S,
        pong_s: float = PONG_S,
    ) -> WebSocket:
        """Open ``url`` (``ws``, ``wss``, ``http`` or ``https``) with extra request headers."""
        parts = urllib.parse.urlsplit(url)
        secure = parts.scheme in ("wss", "https")
        if parts.scheme not in ("ws", "wss", "http", "https") or not parts.hostname:
            raise WebSocketError(f"not a WebSocket URL: {url!r}")
        port = parts.port or (443 if secure else 80)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        sock = socket.create_connection((parts.hostname, port), timeout=timeout)
        try:
            if secure:
                sock = ssl.create_default_context().wrap_socket(
                    sock, server_hostname=parts.hostname
                )
            sock.settimeout(timeout)
            key = base64.b64encode(os.urandom(16)).decode()
            host = parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"
            lines = [
                f"GET {target} HTTP/1.1",
                f"Host: {host}",
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Sec-WebSocket-Key: {key}",
                "Sec-WebSocket-Version: 13",
            ]
            lines += [f"{name}: {value}" for name, value in (headers or {}).items()]
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
            leftover = cls._handshake(sock, key)
        except BaseException:
            with contextlib.suppress(OSError):
                sock.close()
            raise
        opened = cls(sock, ping_s=ping_s, pong_s=pong_s)
        opened._buffer += leftover
        return opened

    @staticmethod
    def _handshake(sock: socket.socket, key: str) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            if len(data) > MAX_HEAD:
                raise WebSocketError("the handshake reply head is too long")
            chunk = sock.recv(4096)
            if not chunk:
                raise WebSocketError("the server closed the connection during the handshake")
            data += chunk
        head, _, rest = bytes(data).partition(b"\r\n\r\n")
        first, *rows = head.decode("latin-1").split("\r\n")
        fields = first.split(" ", 2)
        if len(fields) < 2 or not fields[1].isdigit():
            raise WebSocketError(f"the handshake reply is not HTTP: {first[:60]!r}")
        status = int(fields[1])
        reply = {
            name.strip().lower(): value.strip()
            for name, _, value in (row.partition(":") for row in rows)
        }
        if status != 101:
            body = bytearray(rest)
            want = min(int(reply.get("content-length", "0") or 0), MAX_HEAD)
            while len(body) < want:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                body += chunk
            raise HandshakeRefused(status, bytes(body), reply)
        expected = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        if reply.get("sec-websocket-accept") != expected:
            raise WebSocketError("the Sec-WebSocket-Accept of the reply is wrong")
        if "websocket" not in reply.get("upgrade", "").lower():
            raise WebSocketError("the reply does not upgrade to websocket")
        return rest

    # -- writing ----------------------------------------------------------------------------

    @staticmethod
    def _masked(opcode: int, payload: bytes) -> bytes:
        """A final frame with ``payload``, masked as a client must."""
        size = len(payload)
        head = bytearray([0x80 | opcode])
        if size < 126:
            head.append(0x80 | size)
        elif size < 1 << 16:
            head += bytes([0x80 | 126]) + struct.pack(">H", size)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", size)
        mask = os.urandom(4)
        return bytes(head) + mask + bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))

    def _send_frame(self, opcode: int, payload: bytes = b"") -> None:
        try:
            self._sock.sendall(self._masked(opcode, payload))
        except OSError as error:
            self.closed = True
            raise WebSocketError(f"the connection broke while writing: {error}") from error

    def send_text(self, text: str) -> None:
        self._send_frame(_TEXT, text.encode())

    def close(self, code: int = 1000) -> None:
        """Send a close frame if the connection is open, then drop the socket."""
        if not self.closed:
            self.closed = True
            with contextlib.suppress(OSError):
                self._sock.settimeout(1.0)
                frame_code = struct.pack(">H", code)
                self._sock.sendall(self._masked(_CLOSE, frame_code))
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._sock.close()

    # -- reading ----------------------------------------------------------------------------

    def recv_text(self, timeout: float) -> str | None:
        """The next text message, or None when ``timeout`` seconds pass without one.

        Answers pings, sends the text ``ping`` after ``ping_s`` of silence, and raises
        WebSocketError when the peer closes, the pong does not come, or the connection breaks.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            frame = self._parse()
            if frame is not None:
                text = self._take(*frame)
                if text is not None:
                    return text
                continue
            now = time.monotonic()
            if self.closed:
                raise WebSocketError("the connection is closed")
            wake = deadline
            if self._pong_due is not None:
                if now >= self._pong_due:
                    self.close(1001)
                    raise WebSocketError(f"no pong in {self._pong_s:.0f} s")
                wake = min(wake, self._pong_due)
            else:
                if now - self._heard >= self._ping_s:
                    self._pong_due = now + self._pong_s
                    self.send_text(PING_TEXT)
                    continue
                wake = min(wake, self._heard + self._ping_s)
            if now >= deadline:
                return None
            self._fill(max(wake - now, 0.001))

    def _fill(self, wait: float) -> None:
        try:
            self._sock.settimeout(wait)
            chunk = self._sock.recv(65536)
        except (TimeoutError, ssl.SSLWantReadError):
            return
        except OSError as error:
            self.closed = True
            raise WebSocketError(f"the connection broke while reading: {error}") from error
        if not chunk:
            self.closed = True
            raise WebSocketError("the server closed the connection")
        self._buffer += chunk

    def _parse(self) -> tuple[bool, int, bytes] | None:
        """One complete frame from the buffer as (fin, opcode, payload), or None."""
        buf = self._buffer
        if len(buf) < 2:
            return None
        first, second = buf[0], buf[1]
        if first & 0x70:
            self._fail(1002, "a frame uses a reserved bit")
        if second & 0x80:
            self._fail(1002, "the server sent a masked frame")
        size, at = second & 0x7F, 2
        if size == 126:
            if len(buf) < 4:
                return None
            size, at = struct.unpack(">H", bytes(buf[2:4]))[0], 4
        elif size == 127:
            if len(buf) < 10:
                return None
            size, at = struct.unpack(">Q", bytes(buf[2:10]))[0], 10
        opcode = first & 0x0F
        if size > MAX_MESSAGE or (opcode >= 0x8 and (size > 125 or not first & 0x80)):
            self._fail(1009 if size > MAX_MESSAGE else 1002, "a frame is too large or invalid")
        if len(buf) < at + size:
            return None
        payload = bytes(buf[at : at + size])
        del buf[: at + size]
        return bool(first & 0x80), opcode, payload

    def _fail(self, code: int, reason: str) -> None:
        self.close(code)
        raise WebSocketError(reason)

    def _take(self, fin: bool, opcode: int, payload: bytes) -> str | None:
        """Handle one frame. Returns the text of a completed text message, else None."""
        self._heard = time.monotonic()
        self._pong_due = None
        if opcode == _PING:
            self._send_frame(_PONG, payload)
            return None
        if opcode == _PONG:
            return None
        if opcode == _CLOSE:
            if not self.closed:
                self.closed = True
                with contextlib.suppress(OSError):
                    self._sock.sendall(self._masked(_CLOSE, payload[:2]))
            code = struct.unpack(">H", payload[:2])[0] if len(payload) >= 2 else 1005
            self.close()
            raise WebSocketError(f"the server closed the WebSocket (code {code})")
        if opcode in (_TEXT, _BINARY):
            if self._in_message:
                self._fail(1002, "a new message starts inside a fragmented one")
            self._text = opcode == _TEXT
            self._fragments = bytearray(payload)
        elif opcode == _CONTINUATION:
            if not self._in_message:
                self._fail(1002, "a continuation frame without a message")
            self._fragments += payload
        else:
            self._fail(1002, f"an unknown opcode {opcode}")
        if len(self._fragments) > MAX_MESSAGE:
            self._fail(1009, "a message is too large")
        self._in_message = not fin
        if not fin:
            return None
        data, self._fragments = bytes(self._fragments), bytearray()
        if not self._text:
            return None
        try:
            text = data.decode()
        except UnicodeDecodeError:
            self._fail(1007, "a text message is not UTF-8")
            return None
        return None if text == PONG_TEXT else text
