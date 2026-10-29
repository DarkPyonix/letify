"""The letify-link sidecar and its Python binding, over loopback.

Spec "Sidecar interface". The tests start real sidecars built from letify-core, because the
protocol between Python and the executable is the thing under test. They are skipped, with the
reason, when cargo is missing and no binary is named by LETIFY_LINK_BIN.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from letify.transport import linkd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "letify-core"
MIB = 1 << 20


class Relay:
    """An in-memory rendezvous: a message for a node is handed to that node's daemon."""

    def __init__(self) -> None:
        self.daemons: dict[str, linkd.LinkDaemon] = {}
        self.sent: list[tuple[str, str]] = []

    def endpoint(self, sender: linkd.LinkDaemon) -> Relay.Endpoint:
        return Relay.Endpoint(self, sender)

    class Endpoint:
        def __init__(self, hub: Relay, sender: linkd.LinkDaemon) -> None:
            self.hub, self.sender = hub, sender

        def send(self, to: str, message: bytes) -> None:
            self.hub.sent.append((self.sender.node_id, to))
            self.hub.daemons[to].deliver(self.sender.node_id, message)


@pytest.fixture
def pair(sidecar_binary, tmp_path):
    hub = Relay()
    a = linkd.LinkDaemon(tmp_path / "a.key", binary=sidecar_binary)
    b = linkd.LinkDaemon(tmp_path / "b.key", binary=sidecar_binary)
    hub.daemons = {a.node_id: a, b.node_id: b}
    a.rendezvous = hub.endpoint(a)
    b.rendezvous = hub.endpoint(b)
    try:
        yield a, b, hub
    finally:
        a.close()
        b.close()


def _digest_exchange(sock, data: bytes) -> str:
    """Send ``data`` and read until the peer ends its side; return the digest of what came."""
    seen = hashlib.sha256()

    def send() -> None:
        sock.sendall(data)
        sock.shutdown(1)

    writer = threading.Thread(target=send)
    writer.start()
    while chunk := sock.recv(1 << 20):
        seen.update(chunk)
    writer.join()
    return seen.hexdigest()


def test_32_mib_both_ways_over_a_stream_arrives_intact(pair):
    a, b, _ = pair
    listener = b.listen([a.node_id], bind="127.0.0.1:0")
    session_a = a.open_session(b.node_id, [listener.address])
    session_b = listener.accept_session(timeout=10)
    assert session_a.peer == b.node_id and session_b.peer == a.node_id

    up, down = secrets.token_bytes(32 * MIB), secrets.token_bytes(32 * MIB)
    sock_a = session_a.open_stream()
    sock_b = session_b.accept_stream(timeout=10)
    got: dict[str, str] = {}
    other = threading.Thread(target=lambda: got.update(b=_digest_exchange(sock_b, down)))
    other.start()
    got["a"] = _digest_exchange(sock_a, up)
    other.join()
    assert got["b"] == hashlib.sha256(up).hexdigest()
    assert got["a"] == hashlib.sha256(down).hexdigest()
    assert len(session_a.paths()) == 1
    sock_a.close()
    sock_b.close()


def test_a_punch_through_the_python_rendezvous_gives_a_session(pair):
    a, b, hub = pair
    listener = b.listen([a.node_id])
    result: dict[str, list[str]] = {}
    responder = threading.Thread(
        target=lambda: result.update(
            b=b.punch(
                a.node_id, "responder", sockets=2, listener=listener, lead_ms=300, window_ms=10000
            )
        )
    )
    responder.start()
    names = a.punch(b.node_id, "initiator", sockets=2, lead_ms=300, window_ms=10000)
    responder.join()
    assert names and result["b"] == []
    # An offer went one way and an answer came back, both through the Python object.
    assert (a.node_id, b.node_id) in hub.sent and (b.node_id, a.node_id) in hub.sent

    session_a = a.open_session(b.node_id, [{"name": n, "punched": n} for n in names])
    session_b = listener.accept_session(timeout=10)
    sock_a = session_a.open_stream()
    sock_b = session_b.accept_stream(timeout=10)
    sock_a.sendall(b"hello")
    assert sock_b.recv(5) == b"hello"
    assert session_a.wait_event("path_added", timeout=10)["session"] == session_a.number


def test_a_peer_outside_the_allow_list_is_an_infrastructure_failure(pair):
    a, b, _ = pair
    listener = b.listen(["11" * 32], bind="127.0.0.1:0")
    with pytest.raises(linkd.LinkInfrastructureError):
        a.open_session(
            b.node_id,
            [listener.address],
            config={"handshake_timeout_ms": 1500, "start_timeout_ms": 3000},
        )


def test_a_closed_session_ends_for_the_peer(pair):
    a, b, _ = pair
    listener = b.listen([a.node_id], bind="127.0.0.1:0")
    session_a = a.open_session(b.node_id, [listener.address])
    session_b = listener.accept_session(timeout=10)
    session_a.close()
    with pytest.raises(linkd.LinkClosed):
        session_b.accept_stream(timeout=10)


def test_killing_the_python_side_ends_the_sidecar(sidecar_binary, tmp_path):
    script = (
        "import sys, time\n"
        "from letify.transport import linkd\n"
        "d = linkd.LinkDaemon(sys.argv[1], binary=sys.argv[2])\n"
        "print(d.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path / "k.key"), str(sidecar_binary)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        pid = int(child.stdout.readline())
        os.kill(pid, 0)
        child.send_signal(signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM)
        child.wait(timeout=10)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.1)
        pytest.fail("the sidecar outlived the Python process that started it")
    finally:
        child.kill()


def test_a_wheel_without_the_binary_says_so(monkeypatch, tmp_path):
    monkeypatch.delenv(linkd.ENV_BINARY, raising=False)
    monkeypatch.setattr(linkd, "LIB_DIR", tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(linkd.LinkMissing, match="letify-link is not in this wheel"):
        linkd.LinkDaemon(tmp_path / "k.key")


def test_the_environment_override_wins(monkeypatch, tmp_path):
    fake = tmp_path / "x"
    fake.write_text("")
    monkeypatch.setenv(linkd.ENV_BINARY, str(fake))
    assert linkd.binary_path() == fake
