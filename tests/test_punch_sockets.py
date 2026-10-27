"""Spec: Transport, Rendezvous: a punch of several sockets in one rendezvous."""

from __future__ import annotations

import socket
import threading
import time

import pytest
from conftest import LoopbackRendezvous

from letify.transport import nat
from letify.transport.link import PunchedLink
from letify.transport.strategies import PUNCH_SOCKETS, Target, TCPPunch


def free_ports(count: int) -> list[int]:
    holders = [nat.reusable_socket(0, "127.0.0.1") for _ in range(count)]
    ports = [holder.getsockname()[1] for holder in holders]
    for holder in holders:
        holder.close()
    return ports


def punch_both(mine, theirs, token, *, window=5.0, dead=()):
    """Run the user's side and the remote side of ``punch_many`` over loopback."""
    start_at = time.time() + 0.1
    results: dict[str, object] = {}

    def run(name, ports, peers, initiator):
        try:
            results[name] = nat.punch_many(
                ports,
                [("127.0.0.1", 9 if i in dead else port) for i, port in enumerate(peers)],
                token,
                initiator=initiator,
                start_at=start_at,
                window=window,
            )
        except Exception as error:
            results[name] = error

    threads = [
        threading.Thread(target=run, args=("user", mine, theirs, True)),
        threading.Thread(target=run, args=("remote", theirs, mine, False)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30.0)
    return results["user"], results["remote"]


def test_one_rendezvous_punches_several_sockets() -> None:
    user, remote = punch_both(free_ports(3), free_ports(3), b"t" * 16)
    assert len(user) == len(remote) == 3
    for sock in user + remote:
        sock.close()


def test_a_failed_pair_leaves_the_others() -> None:
    mine, theirs = free_ports(3), free_ports(3)
    # Both sides dial a dead port for pair 1, so that pair times out and the others stay.
    user, remote = punch_both(mine, theirs, b"t" * 16, window=1.5, dead={1})
    assert len(user) == 2
    for sock in user + remote:
        sock.close()


def test_a_punch_where_no_pair_connects_raises() -> None:
    with pytest.raises(TimeoutError):
        nat.punch_many(
            free_ports(2),
            [("127.0.0.1", 9), ("127.0.0.1", 9)],
            b"t" * 16,
            initiator=True,
            start_at=0,
            window=0.3,
        )


def test_tokens_are_per_pair() -> None:
    token = b"t" * 16
    tokens = {nat.pair_token(token, index) for index in range(8)}
    assert len(tokens) == 8
    assert all(len(single) == nat.TOKEN_BYTES + 1 for single in tokens)


def test_punched_sockets_have_keepalive() -> None:
    user, remote = punch_both(free_ports(1), free_ports(1), b"t" * 16)
    for sock in user + remote:
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        if hasattr(socket, "TCP_KEEPIDLE"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 20
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 10
        sock.close()


def test_the_remote_answers_no_more_mappings_than_the_cap(stun_server) -> None:
    answer, _ = nat.begin(
        {
            "kind": "tcp_punch",
            "stun": list(stun_server.address),
            "mappings": [["192.0.2.1", 9]] * 20,
            "count": 20,
            "token": "00" * 16,
            "start_at": time.time() + 60,
        }
    )
    assert len(answer["mappings"]) == nat.MAX_PUNCH_SOCKETS
    assert answer["mapping"] == answer["mappings"][0]


def test_a_tcp_punch_over_loopback_returns_every_socket_it_asked_for(stun_server) -> None:
    rendezvous = LoopbackRendezvous()
    rendezvous.lead_seconds = 0.3
    target = Target(alias="lab", rendezvous=rendezvous, stun=stun_server.address, user="root")
    link = TCPPunch().attempt(target)
    request = rendezvous.requests[0]
    assert request["count"] == PUNCH_SOCKETS
    assert len(request["mappings"]) == PUNCH_SOCKETS
    assert len(link._spares) == PUNCH_SOCKETS - 1
    assert link.note.startswith(
        f"tcp_punch connected {PUNCH_SOCKETS} of {PUNCH_SOCKETS} sockets in "
    )
    link.close()


def test_a_punched_link_hands_out_spares_before_redialing() -> None:
    first, first_far = socket.socketpair()
    spare, spare_far = socket.socketpair()
    redialed: list[int] = []

    def redial() -> socket.socket:
        redialed.append(1)
        return socket.socketpair()[0]

    link = PunchedLink(
        "tcp_punch", 2, first, lambda port, command: ["ssh", str(port)], redial, spares=[spare]
    )
    assert link._take() is first
    assert link._take() is spare
    assert not redialed
    link._take()
    assert redialed == [1]
    for sock in (first_far, spare_far):
        sock.close()
    link.close()


def test_closing_a_punched_link_closes_its_spares() -> None:
    first, _far = socket.socketpair()
    spare, spare_far = socket.socketpair()
    link = PunchedLink("tcp_punch", 2, first, lambda p, c: [], lambda: first, spares=[spare])
    link.close()
    assert spare_far.recv(1) == b""


def test_serving_several_links_returns_only_when_every_link_has_ended():
    # Spec "Rendezvous": the detached remote process exits when its continuation returns,
    # so the continuation must outlive the splice on every spare, not just the first one.
    first_near, first_far = socket.socketpair()
    spare_near, spare_far = socket.socketpair()
    finished = threading.Event()

    def serve() -> None:
        nat.serve_links([first_far, spare_far])
        finished.set()

    threading.Thread(target=serve, daemon=True).start()
    first_near.close()
    assert not finished.wait(0.5)
    spare_near.close()
    assert finished.wait(5)
