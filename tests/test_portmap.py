"""Asking the NAT for a mapping: PCP, NAT-PMP and UPnP IGD.

Spec "Asking the NAT for a mapping". No device answers in the test environment, so each
protocol is exercised against a server that speaks its wire format on the loopback.
"""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from letify.transport import portmap


def udp_server(answer) -> tuple[int, threading.Thread, list[bytes]]:
    """A UDP socket that replies to one request with ``answer(request)``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    seen: list[bytes] = []

    def serve() -> None:
        sock.settimeout(5)
        try:
            data, who = sock.recvfrom(2048)
        except OSError:
            return
        seen.append(data)
        reply = answer(data)
        if reply is not None:
            sock.sendto(reply, who)
        sock.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port, thread, seen


# -- Spec: Asking the NAT for a mapping, NAT-PMP -----------------------------------


def test_nat_pmp_asks_for_a_tcp_mapping_and_reads_the_answer() -> None:
    def answer(request: bytes) -> bytes:
        version, opcode, _reserved, internal, suggested, lifetime = struct.unpack(
            "!BBHHHI", request
        )
        assert (version, opcode) == (0, 2), (version, opcode)
        assert internal == 41000
        assert suggested == 41000
        assert lifetime == portmap.LIFETIME
        # version, opcode + 128, result 0, epoch, internal, mapped, lifetime
        return struct.pack("!BBHIHHI", 0, 130, 0, 12, internal, 41005, lifetime)

    port, thread, _ = udp_server(answer)
    mapping = portmap.nat_pmp(("127.0.0.1", port), 41000)
    thread.join(5)
    assert mapping is not None
    assert mapping.port == 41005
    assert mapping.protocol == "nat_pmp"


def test_nat_pmp_refusing_the_request_is_not_a_mapping() -> None:
    def answer(request: bytes) -> bytes:
        return struct.pack("!BBHIHHI", 0, 130, 2, 12, 41000, 0, 0)  # result 2, not authorized

    port, thread, _ = udp_server(answer)
    assert portmap.nat_pmp(("127.0.0.1", port), 41000) is None
    thread.join(5)


def test_a_gateway_that_does_not_answer_gives_no_mapping() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    assert portmap.nat_pmp(("127.0.0.1", port), 41000, timeout=0.3) is None


# -- Spec: Asking the NAT for a mapping, PCP ---------------------------------------


def test_pcp_asks_for_a_tcp_mapping_and_reads_the_answer() -> None:
    def answer(request: bytes) -> bytes:
        version, opcode = request[0], request[1]
        assert version == 2 and opcode == 1, (version, opcode)
        nonce = request[24:36]
        protocol = request[36]
        internal = struct.unpack("!H", request[40:42])[0]
        assert protocol == 6 and internal == 41000
        # A PCP response header is 24 bytes: version, opcode, reserved, result,
        # lifetime, epoch and twelve reserved bytes. RFC 6887 section 7.2.
        head = struct.pack("!BBBBII", 2, 1 | 0x80, 0, 0, portmap.LIFETIME, 7) + bytes(12)
        body = (
            nonce
            + bytes([6, 0, 0, 0])
            + struct.pack("!HH", internal, 41007)
            + bytes(10)
            + b"\xff\xff"
            + socket.inet_aton("203.0.113.9")
        )
        return head + body

    port, thread, _ = udp_server(answer)
    mapping = portmap.pcp(("127.0.0.1", port), 41000)
    thread.join(5)
    assert mapping is not None
    assert mapping.port == 41007
    assert mapping.address == "203.0.113.9"
    assert mapping.protocol == "pcp"


def test_pcp_answering_with_an_error_is_not_a_mapping() -> None:
    def answer(request: bytes) -> bytes:
        # result 1 is UNSUPP_VERSION, which a NAT-PMP only device answers with.
        return struct.pack("!BBBBII", 2, 1 | 0x80, 0, 1, 0, 7) + bytes(12) + bytes(36)

    port, thread, _ = udp_server(answer)
    assert portmap.pcp(("127.0.0.1", port), 41000) is None
    thread.join(5)


# -- Spec: Asking the NAT for a mapping, the order ---------------------------------


def test_pcp_is_asked_before_nat_pmp_and_upnp(monkeypatch) -> None:
    asked: list[str] = []

    def answering(name: str, result):
        def call(*args, **kwargs):
            asked.append(name)
            return result

        return call

    monkeypatch.setattr(portmap, "default_gateway", lambda: "192.0.2.1")
    monkeypatch.setattr(portmap, "pcp", answering("pcp", None))
    monkeypatch.setattr(portmap, "nat_pmp", answering("nat_pmp", None))
    monkeypatch.setattr(portmap, "upnp", answering("upnp", None))
    assert portmap.request(41000) is None
    assert asked == ["pcp", "nat_pmp", "upnp"]


def test_the_first_protocol_that_answers_wins(monkeypatch) -> None:
    found = portmap.Mapping(address="203.0.113.5", port=41009, protocol="nat_pmp", lifetime=3600)
    monkeypatch.setattr(portmap, "default_gateway", lambda: "192.0.2.1")
    monkeypatch.setattr(portmap, "pcp", lambda *a, **k: None)
    monkeypatch.setattr(portmap, "nat_pmp", lambda *a, **k: found)
    monkeypatch.setattr(portmap, "upnp", lambda *a, **k: pytest.fail("upnp was asked"))
    assert portmap.request(41000) is found


def test_no_gateway_means_no_mapping(monkeypatch) -> None:
    monkeypatch.setattr(portmap, "default_gateway", lambda: None)
    monkeypatch.setattr(portmap, "upnp", lambda *a, **k: None)
    assert portmap.request(41000) is None


def test_the_default_gateway_is_read_from_the_route_table(tmp_path) -> None:
    table = tmp_path / "route"
    table.write_text(
        "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\n"
        "eth0\t00000000\t0101A8C0\t0003\t0\t0\t0\t00000000\n",
        encoding="utf-8",
    )
    assert portmap.default_gateway(table) == "192.168.1.1"


def test_a_route_table_with_no_default_route_has_no_gateway(tmp_path) -> None:
    table = tmp_path / "route"
    table.write_text(
        "Iface\tDestination\tGateway\tFlags\neth0\t0002A8C0\t00000000\t0001\n", encoding="utf-8"
    )
    assert portmap.default_gateway(table) is None


# -- Spec: Asking the NAT for a mapping, feeding Connect back ----------------------


def test_connect_back_uses_a_mapping_when_the_account_declares_no_endpoint(monkeypatch) -> None:
    # Spec "Asking the NAT for a mapping": a granted mapping is exactly the endpoint
    # Connect back needs, so the strategy becomes applicable without configuration.
    from conftest import CannedRendezvous

    from letify.transport.strategies import ConnectBack, Target

    target = Target(alias="lab", rendezvous=CannedRendezvous())
    assert "no connect_back" in ConnectBack().needs(target)

    monkeypatch.setattr(
        portmap,
        "request",
        lambda port, **kw: portmap.Mapping(
            address="203.0.113.5", port=port, protocol="pcp", lifetime=3600
        ),
    )
    assert ConnectBack(mapping=True).needs(target) is None


def test_asking_is_skipped_when_the_account_turns_it_off() -> None:
    from conftest import CannedRendezvous

    from letify.transport.strategies import ConnectBack, Target

    target = Target(alias="lab", rendezvous=CannedRendezvous())
    assert "the NAT offered no mapping" not in (ConnectBack(mapping=False).needs(target) or "")
    assert "no connect_back" in ConnectBack(mapping=False).needs(target)


def test_a_nat_that_offers_nothing_fails_the_attempt_not_the_check(monkeypatch) -> None:
    # Spec "Asking the NAT for a mapping": needs() is called for every strategy before the
    # race begins, so asking the NAT belongs in attempt(), where it runs beside the other
    # strategies instead of ahead of them.
    from conftest import CannedRendezvous

    from letify.transport.strategies import ConnectBack, Target

    monkeypatch.setattr(portmap, "request", lambda port, **kw: None)
    target = Target(alias="lab", rendezvous=CannedRendezvous())
    strategy = ConnectBack(mapping=True)
    assert strategy.needs(target) is None
    with pytest.raises(OSError, match="the NAT offered no mapping"):
        strategy.attempt(target)
