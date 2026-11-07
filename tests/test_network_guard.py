"""The suite must not reach a public network address.

Pins the autouse ``no_public_network`` fixture in conftest.py. A test that talks to a
public service passes on a machine that can reach it and fails, or leaks, elsewhere.
"""

from __future__ import annotations

import socket

import pytest
from conftest import PublicNetworkAccess

from letify.transport import nat


def test_the_suite_cannot_reach_a_public_address() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(PublicNetworkAccess, match=r"test_the_suite_cannot.*8\.8\.8\.8"):
            sock.connect(("8.8.8.8", 443))
    finally:
        sock.close()
    # The violation is also recorded, so code that swallows the exception still fails the test.
    from conftest import _public_network_attempts

    _public_network_attempts.clear()


def test_the_suite_cannot_ask_the_default_stun_server() -> None:
    from conftest import _public_network_attempts

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(PublicNetworkAccess, match=r"stun\.nextcloud\.com"):
            udp.sendto(b"x", nat.DEFAULT_STUN)
        with pytest.raises(PublicNetworkAccess, match=r"stun\.nextcloud\.com"):
            socket.getaddrinfo(*nat.DEFAULT_STUN)
    finally:
        udp.close()
    _public_network_attempts.clear()


def test_loopback_and_unix_sockets_stay_reachable(tmp_path) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname())
    client.close()
    server.close()
    assert socket.getaddrinfo("localhost", 80)
    path = str(tmp_path / "s.sock")
    unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix.bind(path)
    unix.listen(1)
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    peer.connect(path)
    peer.close()
    unix.close()
