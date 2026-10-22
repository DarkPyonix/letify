"""QUIC over a punched UDP pair.

Spec "QUIC over a punched UDP pair". The binary is a build product, so the tests that
need it are skipped where it is absent, and everything else is exercised against fakes.
"""

from __future__ import annotations

import pytest
from conftest import CannedRendezvous

from letify.transport import nat, quic
from letify.transport.strategies import QuicUDP, Target

# -- Spec: QUIC over a punched UDP pair, finding the binary ------------------------


def test_the_carrier_is_found_in_the_wheel_before_the_path(monkeypatch, tmp_path) -> None:
    bundled = tmp_path / "letify-quic"
    bundled.write_text("#!/bin/true\n", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    assert quic.carrier_path() == bundled


def test_no_carrier_anywhere_is_not_an_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    monkeypatch.setattr(quic.shutil, "which", lambda name: None)
    assert quic.carrier_path() is None


# -- Spec: QUIC over a punched UDP pair, the strategy ------------------------------


def test_quic_needs_the_carrier_and_a_rendezvous(monkeypatch, tmp_path) -> None:
    assert QuicUDP.rank == 5
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    monkeypatch.setattr(quic.shutil, "which", lambda name: None)
    target = Target(alias="lab", rendezvous=CannedRendezvous())
    assert "letify-quic is not in this wheel" in QuicUDP().needs(target)

    bundled = tmp_path / "letify-quic"
    bundled.write_text("", encoding="utf-8")
    bundled.chmod(0o755)
    assert QuicUDP().needs(Target(alias="lab")) == "no rendezvous"
    assert QuicUDP().needs(target) is None


def test_the_strategy_exchanges_both_endpoints_and_proxies_ssh(
    monkeypatch, tmp_path, patch_run
) -> None:
    # Spec "QUIC over a punched UDP pair": the rendezvous carries both punched endpoints
    # and the token, and the client side becomes an SSH ProxyCommand.
    bundled = tmp_path / "letify-quic"
    bundled.write_text("", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    monkeypatch.setattr(
        nat, "udp_stun_mapping", lambda port, server=None, **kw: ("198.51.100.1", port)
    )
    recorder = patch_run(__import__("letify.transport.strategies", fromlist=["x"]))
    rendezvous = CannedRendezvous({"mapping": ["203.0.113.9", 44444]})
    target = Target(alias="lab", rendezvous=rendezvous, user="root")

    link = QuicUDP().attempt(target)

    request = rendezvous.requests[0]
    assert request["kind"] == "quic"
    assert request["mapping"][0] == "198.51.100.1"
    assert len(bytes.fromhex(request["token"])) == nat.TOKEN_BYTES
    command = link.ssh_command("uname")
    proxy = next(part for part in command if part.startswith("ProxyCommand="))
    assert "letify-quic" in proxy and "connect" in proxy
    assert "--peer 203.0.113.9:44444" in proxy
    assert f"--token {request['token']}" in proxy
    assert recorder.command[-1] == "exit 0"


# -- Spec: QUIC over a punched UDP pair, the remote half --------------------------


def test_the_remote_half_reports_its_mapping_and_starts_the_carrier(
    patch_popen, monkeypatch
) -> None:
    monkeypatch.setattr(
        nat, "udp_stun_mapping", lambda port, server=None, **kw: ("203.0.113.9", port)
    )
    monkeypatch.setattr(nat, "punch_udp", lambda port, peer, **kw: None)
    started = patch_popen(nat, [])
    answer, _ = nat.begin(
        {
            "kind": "quic",
            "mapping": ["198.51.100.1", 41000],
            "token": "ab" * 16,
            "ssh_port": 22,
            "binary": "/tmp/letify-quic",
            "stun": ["stun.example", 443],
        }
    )
    assert "mapping" in answer
    command = started[0].command
    assert command[0] == "/tmp/letify-quic"
    assert command[1] == "serve"
    assert "--peer" in command and "198.51.100.1:41000" in command
    assert "--forward" in command and "22" in command


def test_the_remote_source_fetches_the_wheel_for_its_own_platform() -> None:
    # Spec "QUIC over a punched UDP pair": the remote gets the binary from the letify
    # wheel at this client's version, which PyPI verifies, rather than a pinned hash.
    from letify import install

    source = install.remote_quic_source("9.9.9")
    # The version reaches the remote as the call's argument; the requirement itself is
    # built there, so the source carries the pattern rather than the finished string.
    assert "pip" in source and 'f"letify=={version}"' in source
    assert "_remote_quic('9.9.9')" in source
    assert "letify/remoting/lib/letify-quic" in source
    assert "LETIFY-QUIC " in source
    compile(source, "<remote>", "exec")


# -- Spec: QUIC over a punched UDP pair, the binary itself ------------------------


@pytest.mark.skipif(quic.carrier_path() is None, reason="letify-quic is a build product")
def test_the_carrier_refuses_a_call_with_no_token() -> None:
    import subprocess

    done = subprocess.run(
        [str(quic.carrier_path()), "serve", "--bind", "0", "--peer", "127.0.0.1:1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 2
    assert "--token is required" in done.stderr


@pytest.mark.skipif(quic.carrier_path() is None, reason="letify-quic is a build product")
def test_the_carrier_carries_a_stream_between_two_of_itself(tmp_path) -> None:
    # Spec "QUIC over a punched UDP pair": a real QUIC v1 handshake, then the stream is
    # spliced to the forwarded port on one side and to stdio on the other.
    import secrets
    import socket
    import subprocess
    import threading

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    forward = listener.getsockname()[1]
    message = b"over a QUIC stream" * 64

    def answer() -> None:
        conn, _ = listener.accept()
        conn.sendall(message)
        conn.close()

    threading.Thread(target=answer, daemon=True).start()

    def free() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    token = secrets.token_bytes(16).hex()
    server_port, client_port = free(), free()
    carrier = str(quic.carrier_path())
    server = subprocess.Popen(
        [carrier, "serve", "--bind", str(server_port), "--peer",
         f"127.0.0.1:{client_port}", "--token", token, "--forward", str(forward)],
        stderr=subprocess.PIPE,
    )
    try:
        client = subprocess.run(
            [carrier, "connect", "--bind", str(client_port), "--peer",
             f"127.0.0.1:{server_port}", "--token", token],
            capture_output=True,
            timeout=60,
        )
        assert client.stdout.startswith(message[:64]), client.stderr[-400:]
    finally:
        server.kill()
        listener.close()


class RecordingRendezvous:
    """A command rendezvous whose remote is a list of canned outputs, one per program run."""

    @staticmethod
    def build(outputs: list[str]):
        from letify.transport.rendezvous import CommandRendezvous

        class Recording(CommandRendezvous):
            def __init__(self) -> None:
                self.sources: list[str] = []
                self.outputs = list(outputs)

            def unavailable(self) -> str | None:
                return None

            def run_python(self, source: str, timeout: float) -> str:
                self.sources.append(source)
                return self.outputs.pop(0)

        return Recording()


def test_a_command_rendezvous_prepares_the_quic_carrier_once_and_names_it(monkeypatch) -> None:
    # Spec "QUIC over a punched UDP pair": a runtime image ships no letify-quic, so the
    # rendezvous takes it out of the letify wheel on the remote before the first QUIC
    # request, and every QUIC request then carries the path.
    import letify

    answer = nat.ANSWER_MARKER + '{"mapping": ["203.0.113.9", 41000]}'
    rendezvous = RecordingRendezvous.build(
        ['LETIFY-QUIC "/tmp/letify-quic-9.9.9"', answer, answer]
    )
    monkeypatch.setattr(letify, "__version__", "9.9.9")
    request = {"kind": "quic", "mapping": ["198.51.100.1", 41000], "token": "ab" * 16}
    rendezvous.exchange(dict(request), 30.0)
    rendezvous.exchange(dict(request), 30.0)
    assert "_remote_quic('9.9.9')" in rendezvous.sources[0]
    assert all("_remote_quic" not in source for source in rendezvous.sources[1:])
    assert all('"binary": "/tmp/letify-quic-9.9.9"' in source for source in rendezvous.sources[1:])


def test_a_quic_carrier_the_remote_could_not_prepare_fails_the_request() -> None:
    rendezvous = RecordingRendezvous.build(["could not download the letify wheel"])
    with pytest.raises(OSError, match="letify-quic"):
        rendezvous.exchange({"kind": "quic", "mapping": ["198.51.100.1", 41000]}, 30.0)


# -- Spec: QUIC over a punched UDP pair, the UDP endpoints ------------------------


def _stun_reply(request: bytes, address: tuple[str, int]) -> bytes:
    import socket
    import struct

    cookie = struct.unpack("!I", request[4:8])[0]
    port = address[1] ^ (cookie >> 16)
    host = struct.unpack("!I", socket.inet_aton(address[0]))[0] ^ cookie
    value = struct.pack("!BBHI", 0, 1, port, host)
    attribute = struct.pack("!HH", 0x0020, len(value)) + value
    return struct.pack("!HHI", 0x0101, len(attribute), cookie) + request[8:20] + attribute


def test_the_udp_mapping_is_asked_over_udp_from_the_port_itself() -> None:
    # A NAT maps UDP and TCP separately, so the port QUIC binds is asked about over UDP.
    import socket
    import threading

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    seen = []

    def answer() -> None:
        request, source = server.recvfrom(2048)
        seen.append(source)
        server.sendto(_stun_reply(request, ("203.0.113.7", 50505)), source)

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    holder.close()
    mapping = nat.udp_stun_mapping(port, server.getsockname(), timeout=2.0)
    thread.join(2)
    server.close()
    assert mapping == ("203.0.113.7", 50505)
    assert seen[0][1] == port


def test_a_udp_punch_sends_from_the_port_to_the_peer() -> None:
    import socket

    peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    peer.bind(("127.0.0.1", 0))
    peer.settimeout(2.0)
    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    holder.close()
    nat.punch_udp(port, peer.getsockname(), count=3, interval=0.0)
    sources = {peer.recvfrom(64)[1][1] for _ in range(3)}
    peer.close()
    assert sources == {port}


def test_the_remote_half_punches_toward_the_peer_before_the_carrier_listens(
    patch_popen, monkeypatch
) -> None:
    # The carrier only listens, so without packets toward the peer a NAT in front of the
    # remote drops the client's first QUIC packets.
    order = []
    monkeypatch.setattr(
        nat, "udp_stun_mapping", lambda port, server=None, **kw: ("203.0.113.9", port)
    )
    monkeypatch.setattr(nat, "punch_udp", lambda port, peer, **kw: order.append(("punch", port, peer)))
    started = patch_popen(nat, [])
    answer, _ = nat.begin(
        {
            "kind": "quic",
            "mapping": ["198.51.100.1", 41000],
            "token": "ab" * 16,
            "ssh_port": 22,
            "binary": "/tmp/letify-quic",
        }
    )
    bound = int(started[0].command[started[0].command.index("--bind") + 1])
    assert order == [("punch", bound, ("198.51.100.1", 41000))]
    assert answer["mapping"] == ["203.0.113.9", bound]


def test_the_strategy_prepares_the_remote_carrier_before_asking_for_its_mapping(
    monkeypatch, tmp_path, patch_run
) -> None:
    # A mapping asked for before a slow wheel download may expire before the carrier binds.
    order = []
    bundled = tmp_path / "letify-quic"
    bundled.write_text("", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    monkeypatch.setattr(
        nat, "udp_stun_mapping", lambda port, server=None, **kw: order.append("stun") or ("198.51.100.1", port)
    )
    patch_run(__import__("letify.transport.strategies", fromlist=["x"]))
    rendezvous = CannedRendezvous({"mapping": ["203.0.113.9", 44444]})
    rendezvous.prepare_quic = lambda: order.append("prepare")
    QuicUDP().attempt(Target(alias="lab", rendezvous=rendezvous, user="root"))
    assert order == ["prepare", "stun"]
