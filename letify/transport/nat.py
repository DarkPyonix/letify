"""NAT traversal and the remote half of every strategy, in the standard library only.

This module is sent as source to the remote machine, which may not have letify, so it
imports nothing outside the standard library and nothing relative. It owns STUN over
TCP, the simultaneous open, the probe responder, the byte splice, and ``begin``, which
answers one rendezvous request. It does not own choosing a strategy or reaching the
remote side.
"""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

MAGIC_COOKIE = 0x2112A442
#: A STUN server reached over TCP 443, which restrictive networks usually still allow.
DEFAULT_STUN = ("stun.nextcloud.com", 443)
HELLO = b"LETIFY-PUNCH1"
TOKEN_BYTES = 16
CHUNK = 64 * 1024
ANSWER_MARKER = "LETIFY-ANSWER "


class Cancelled(Exception):
    """An attempt was abandoned because another strategy was already chosen."""


# -- sockets ---------------------------------------------------------------------


def reusable_socket(port: int = 0, host: str = "0.0.0.0") -> socket.socket:
    """A TCP socket bound so a listener and a connector can share its port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    return sock


def recv_exact(stream, size: int) -> bytes:
    """Read exactly ``size`` bytes, raising EOFError when the peer closes first."""
    parts = []
    while size:
        chunk = stream.recv(size)
        if not chunk:
            raise EOFError("the peer closed the connection")
        parts.append(chunk)
        size -= len(chunk)
    return b"".join(parts)


# -- STUN ------------------------------------------------------------------------


def stun_request(transaction: bytes) -> bytes:
    return struct.pack("!HHI", 0x0001, 0, MAGIC_COOKIE) + transaction


def parse_stun_response(data: bytes, transaction: bytes) -> tuple[str, int]:
    """The mapped address in a binding response, preferring XOR-MAPPED-ADDRESS."""
    kind, length, cookie = struct.unpack("!HHI", data[:8])
    if data[8:20] != transaction:
        raise ValueError("the STUN reply is for another transaction")
    if kind != 0x0101:
        raise ValueError(f"the STUN reply is not a binding success: {kind:#06x}")
    body = data[20 : 20 + length]
    plain = None
    while len(body) >= 4:
        attribute, size = struct.unpack("!HH", body[:4])
        value = body[4 : 4 + size]
        body = body[4 + size + (-size % 4) :]
        if attribute in (0x0020, 0x0001) and len(value) >= 8 and value[1] == 1:
            port, address = struct.unpack("!HI", value[2:8])
            if attribute == 0x0020:
                port ^= cookie >> 16
                address ^= cookie
                return socket.inet_ntoa(struct.pack("!I", address)), port
            plain = (socket.inet_ntoa(struct.pack("!I", address)), port)
    if plain is None:
        raise ValueError("the STUN reply carries no mapped address")
    return plain


def stun_mapping(
    port: int, server: tuple[str, int] = DEFAULT_STUN, *, timeout: float = 5.0
) -> tuple[str, int]:
    """The public address and port a NAT gives connections from local ``port``."""
    transaction = secrets.token_bytes(12)
    sock = reusable_socket(port)
    try:
        sock.settimeout(timeout)
        sock.connect(tuple(server))
        sock.sendall(stun_request(transaction))
        header = recv_exact(sock, 20)
        length = struct.unpack("!H", header[2:4])[0]
        return parse_stun_response(header + recv_exact(sock, length), transaction)
    finally:
        sock.close()


def default_route_interface(route_table: Path = Path("/proc/net/route")) -> str | None:
    """The name of the interface the default route leaves through."""
    if route_table.is_file():
        for line in route_table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 1 and fields[1] == "00000000":
                return fields[0]
        return None
    try:  # pragma: no cover - macOS and Windows, not the test machine
        if sys.platform == "darwin":
            out = subprocess.run(
                ["route", "-n", "get", "default"], capture_output=True, text=True, timeout=10
            ).stdout
            found = re.search(r"interface:\s*(\S+)", out)
            return found.group(1) if found else None
        if sys.platform == "win32":
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-NetRoute -DestinationPrefix 0.0.0.0/0 | Sort-Object RouteMetric"
                    " | Select-Object -First 1).InterfaceAlias",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            return out or None
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        return None
    return None  # pragma: no cover


# -- the simultaneous open ---------------------------------------------------------


def punch(
    port: int,
    peer: tuple[str, int],
    token: bytes,
    *,
    initiator: bool,
    start_at: float,
    window: float = 15.0,
    cancel: threading.Event | None = None,
) -> socket.socket:
    """Connect to ``peer`` from ``port`` while listening on it, and agree on one connection.

    The initiator takes the first connection that completes and writes the hello with the
    token. The other side keeps the connection that the hello arrives on. Setting
    ``cancel`` ends the wait for ``start_at`` and the dialing loop with ``Cancelled``.
    """
    delay = start_at - time.time()
    if delay > 0:
        if cancel is None:
            time.sleep(delay)
        elif cancel.wait(delay):
            raise Cancelled(f"the punch to {peer[0]}:{peer[1]} was cancelled before it began")
    deadline = max(time.time(), start_at) + window
    listener = reusable_socket(port)
    listener.listen(8)
    listener.setblocking(False)
    connector: socket.socket | None = None
    retry_at = 0.0
    candidates: list[socket.socket] = []
    expected = HELLO + token

    def close_all(keep: socket.socket | None) -> None:
        for sock in [listener, connector, *candidates]:
            if sock is not None and sock is not keep:
                sock.close()

    while time.time() < deadline:
        if cancel is not None and cancel.is_set():
            close_all(None)
            raise Cancelled(f"the punch to {peer[0]}:{peer[1]} was cancelled")
        if connector is None and time.time() >= retry_at:
            connector = reusable_socket(port)
            connector.setblocking(False)
            connector.connect_ex(tuple(peer))
        readers = [listener, *candidates]
        writers = [connector] if connector is not None else []
        readable, writable, _ = select.select(readers, writers, [], 0.05)
        if listener in readable:
            try:
                accepted, _ = listener.accept()
                candidates.append(accepted)
            except OSError:
                pass
        if connector is not None and connector in writable:
            if connector.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0:
                candidates.append(connector)
            else:
                connector.close()
                retry_at = time.time() + 0.2
            connector = None
        if initiator and candidates:
            chosen = candidates[0]
            chosen.setblocking(True)
            chosen.sendall(expected)
            close_all(chosen)
            return chosen
        if not initiator:
            for sock in [s for s in candidates if s in readable]:
                try:
                    seen = sock.recv(len(expected), socket.MSG_PEEK)
                except OSError:
                    seen = b""
                if len(seen) < len(expected) and seen == expected[: len(seen)] and seen:
                    continue
                candidates.remove(sock)
                if seen == expected:
                    sock.setblocking(True)
                    recv_exact(sock, len(expected))
                    close_all(sock)
                    return sock
                sock.close()
    close_all(None)
    raise TimeoutError(f"no connection with {peer[0]}:{peer[1]} within {window:g} s")


# -- the probe responder and the splice ----------------------------------------------


def serve_probe(stream) -> bool:
    """Answer the probe until asked to bridge (True) or the peer leaves (False)."""
    while True:
        try:
            op = recv_exact(stream, 1)
        except (EOFError, OSError):
            return False
        if op == b"P":
            stream.sendall(b"P" + recv_exact(stream, 8))
        elif op == b"U":
            total = 0
            while True:
                size = struct.unpack("!I", recv_exact(stream, 4))[0]
                if size == 0:
                    break
                total += len(recv_exact(stream, size))
            stream.sendall(struct.pack("!Q", total))
        elif op == b"D":
            seconds = struct.unpack("!d", recv_exact(stream, 8))[0]
            block = b"\0" * CHUNK
            began = time.perf_counter()
            while time.perf_counter() - began < seconds:
                stream.sendall(struct.pack("!I", CHUNK) + block)
            stream.sendall(struct.pack("!I", 0))
        else:
            return op == b"B"


def splice(a: socket.socket, b: socket.socket) -> None:
    """Copy bytes both ways until either side closes."""

    def pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(CHUNK)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        for sock in (dst, src):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    other = threading.Thread(target=pump, args=(b, a), daemon=True)
    other.start()
    pump(a, b)
    other.join()
    a.close()
    b.close()


def serve_link(sock: socket.socket, target: tuple[str, int] = ("127.0.0.1", 22)) -> None:
    """Answer the probe on a punched connection, then splice it to the SSH server."""
    if serve_probe(sock):
        splice(sock, socket.create_connection(tuple(target)))
    else:
        sock.close()


# -- one rendezvous request ------------------------------------------------------------


#: Errors that mean the peer cannot be reached, an expected outcome of a punch.
_UNREACHABLE = frozenset(
    getattr(errno, name)
    for name in ("ENETUNREACH", "EHOSTUNREACH", "ECONNREFUSED", "ECONNRESET", "ETIMEDOUT")
    if hasattr(errno, name)
)


def _noop() -> None:
    return None


def begin(request: dict):
    """Answer one rendezvous request and return ``(answer, continuation)``.

    The answer goes back through the rendezvous at once. The continuation is what keeps
    running afterwards: the punch and the splice, or the Tailcat or SSH process.
    """
    kind = request.get("kind")
    if kind == "ping":
        return {"pong": True}, _noop
    if request.get("authorized_key"):
        _authorize(request["authorized_key"])
    if request.get("start_sshd"):
        _start_sshd()
    if kind == "tcp_punch":
        holder = reusable_socket(0)
        port = holder.getsockname()[1]
        mapping = stun_mapping(port, tuple(request.get("stun") or DEFAULT_STUN))
        holder.close()
        token = bytes.fromhex(request["token"])
        peer = tuple(request["mapping"])
        ssh = ("127.0.0.1", int(request.get("ssh_port", 22)))

        window = float(request.get("window", 15.0))

        def punch_and_serve() -> None:
            where = f"{peer[0]}:{peer[1]}"
            fallback = "the Tailcat link is used instead"
            try:
                sock = punch(
                    port,
                    peer,
                    token,
                    initiator=False,
                    start_at=float(request["start_at"]),
                    window=window,
                )
            except TimeoutError:
                # A punch that does not connect is the normal path to the Tailcat link.
                message = f"TCP punch with {where} did not connect within {window:g} s"
                print(f"letify agent: {message}; {fallback}", file=sys.stderr, flush=True)
                return
            except OSError as error:
                if error.errno in _UNREACHABLE:
                    message = f"TCP punch with {where} failed ({error.strerror or error})"
                    print(f"letify agent: {message}; {fallback}", file=sys.stderr, flush=True)
                    return
                raise
            serve_link(sock, ssh)

        return {"mapping": list(mapping)}, punch_and_serve
    if kind == "tailcat":
        process = subprocess.Popen(
            [request.get("binary", "tailcat"), "serve", str(request.get("ssh_port", 22))],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            found = re.search(r"address:\s*(tc\S+)", line)
            if found:
                return {"address": found.group(1)}, process.wait
        raise RuntimeError("tailcat serve exited without printing its address")
    if kind == "reverse_ssh":
        # The directory is under the account's workspace root, which may start with ~.
        key = Path(os.path.expanduser(request["key_directory"])) / "letify_reverse"
        key.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(request["private_key"])
        process = subprocess.Popen(
            [
                "ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-R", f"0:127.0.0.1:{int(request.get('ssh_port', 22))}",
                "-p", str(request["port"]), "-i", str(key),
                f"{request['user']}@{request['address']}",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )  # fmt: skip
        for line in process.stderr:
            found = re.search(r"Allocated port (\d+)", line)
            if found:
                return {"port": int(found.group(1))}, process.wait
        raise RuntimeError("ssh -R exited without allocating a port")
    raise ValueError(f"unknown rendezvous request {kind!r}")


def _authorize(public_key: str) -> None:
    path = Path.home() / ".ssh" / "authorized_keys"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = path.read_text() if path.is_file() else ""
    if public_key.strip() not in existing:
        with path.open("a") as handle:
            handle.write(public_key.strip() + "\n")
    path.chmod(0o600)


def _start_sshd() -> None:  # pragma: no cover - installs a system package on the remote VM
    """Install and start an SSH server, for a machine such as a Colab VM that has none."""
    if not Path("/usr/sbin/sshd").exists():
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "openssh-server"],
            check=True,
            env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
        )
    Path("/run/sshd").mkdir(parents=True, exist_ok=True)
    subprocess.run(["/usr/sbin/sshd"], check=False)


def run_detached(request_json: str, source: str) -> None:  # pragma: no cover - remote entry
    """Entry point for a command rendezvous: answer, then keep running after the command ends."""
    child = subprocess.Popen(
        [sys.executable, "-c", source + f"\n_child({request_json!r})\n"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    for line in child.stdout:
        if line.startswith(ANSWER_MARKER):
            print(line.strip(), flush=True)
            return
    raise SystemExit("the remote half exited without answering")


def _child(request_json: str) -> None:  # pragma: no cover - runs in the detached process
    try:
        answer, continuation = begin(json.loads(request_json))
    except Exception as exc:
        answer, continuation = {"error": f"{type(exc).__name__}: {exc}"}, _noop
    print(ANSWER_MARKER + json.dumps(answer), flush=True)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    continuation()


class PipeStream:
    """Standard input and output as a stream, for the probe responder over an SSH command."""

    def recv(self, size: int) -> bytes:
        return os.read(0, size)

    def sendall(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(1, view) :]
