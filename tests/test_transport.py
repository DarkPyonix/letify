"""Tests for the connection pipeline.

Spec: Transport. STUN, the simultaneous open and the probe run over loopback sockets;
only choices that would take seconds of real transfer are given canned results.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
from conftest import FakeProbe, FakeStrategy, StunServer

import letify
from letify.transport import nat
from letify.transport.pipeline import (
    Fingerprint,
    LinkCache,
    LinkFloor,
    Pipeline,
    network_fingerprint,
)
from letify.transport.probe import Probe, ProbeResult

MIB = 1024 * 1024
#: No test here is about the floor itself, so it is opened wide; the floor's own tests pass
#: one explicitly.
NO_FLOOR = LinkFloor(min_bps=0.0, max_rtt_ms=float("inf"))


def result(up: float, down: float, rtt: float = 5.0) -> ProbeResult:
    return ProbeResult(rtt_ms=rtt, upload_bps=up * MIB, download_bps=down * MIB)


def fingerprint(ip: str | None = "203.0.113.7", interface: str | None = "eth0"):
    return lambda: Fingerprint(public_ip=ip, interface=interface)


def pipeline(strategies, *, cache=None, grace=0.2, fp=None, probe=None, floor=NO_FLOOR) -> Pipeline:
    return Pipeline(
        strategies,
        target=None,
        alias="lab",
        probe=probe or FakeProbe(),
        cache=cache,
        fingerprint=fp or fingerprint(),
        grace=grace,
        timeout=5.0,
        floor=floor,
    )


def free_port() -> int:
    holder = nat.reusable_socket(0)
    port = holder.getsockname()[1]
    holder.close()
    return port


# -- Spec: Transport, Rendezvous: STUN over TCP ---------------------------------


def test_the_public_mapping_is_read_from_a_stun_server_over_tcp(stun_server) -> None:
    port = free_port()
    ip, mapped = nat.stun_mapping(port, stun_server.address)
    assert ip == "127.0.0.1"
    assert mapped == port


def test_a_stun_reply_for_another_transaction_is_refused() -> None:
    request = nat.stun_request(b"a" * 12)
    assert request[:2] == b"\x00\x01"
    with pytest.raises(ValueError, match="transaction"):
        nat.parse_stun_response(b"\x01\x01\x00\x00\x21\x12\xa4\x42" + b"b" * 12, b"a" * 12)


# -- Spec: Transport, Rendezvous: the simultaneous open --------------------------


def test_two_sides_that_connect_at_an_agreed_time_keep_the_same_connection() -> None:
    port_a, port_b = free_port(), free_port()
    token = b"t" * 16
    start = time.time() + 0.3
    got: dict[str, socket.socket] = {}

    def side(name: str, port: int, peer: int, initiator: bool) -> None:
        got[name] = nat.punch(
            port, ("127.0.0.1", peer), token, initiator=initiator, start_at=start, window=5.0
        )

    threads = [
        threading.Thread(target=side, args=("user", port_a, port_b, True)),
        threading.Thread(target=side, args=("remote", port_b, port_a, False)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    message = b"over the punched connection"
    got["user"].sendall(message)
    assert nat.recv_exact(got["remote"], len(message)) == message
    for sock in got.values():
        sock.close()


def test_a_dial_the_kernel_refuses_at_once_is_not_taken_for_a_connection() -> None:
    """Spec "Transport, Rendezvous: the simultaneous open": a dial that fails before any
    packet is sent is closed and repeated, not chosen.

    When the peer's SYN reaches the listener first, the port pair is taken and the dial
    fails at once with EADDRNOTAVAIL. The socket still looks writable with no pending error,
    so a punch that trusts that would write its hello into a socket connected to nobody and
    die with a broken pipe while the peer's connection sits in the listener's queue.
    """
    peer_listener = nat.reusable_socket(0, "127.0.0.1")
    peer_listener.listen(4)
    peer_port = peer_listener.getsockname()[1]
    holder = nat.reusable_socket(0)
    port = holder.getsockname()[1]
    # The port pair (port, peer_port) is in use before the punch dials it.
    taken = nat.reusable_socket(port)
    taken.connect(("127.0.0.1", peer_port))
    taken_far, _ = peer_listener.accept()
    token = b"d" * 16
    arrived: dict[str, socket.socket] = {}

    def peer_completes_its_own_dial() -> None:
        # Over loopback the taken pair is also the peer's own pair seen from its side, so
        # the peer dials from a fresh port. What the punch sees is the same: its dial is
        # refused, and a connection then arrives on its listener.
        time.sleep(0.25)
        arrived["sock"] = socket.create_connection(("127.0.0.1", port))

    threading.Thread(target=peer_completes_its_own_dial, daemon=True).start()
    try:
        chosen = nat.punch(
            port, ("127.0.0.1", peer_port), token, initiator=True, start_at=0, window=5.0
        )
        assert nat.recv_exact(arrived["sock"], len(nat.HELLO) + 16) == nat.HELLO + token
        chosen.close()
    finally:
        for sock in (taken, taken_far, peer_listener, holder, *arrived.values()):
            sock.close()


def test_a_punch_that_meets_nobody_gives_up_at_the_end_of_its_window() -> None:
    with pytest.raises(TimeoutError):
        nat.punch(free_port(), ("127.0.0.1", 9), b"t" * 16, initiator=False, start_at=0, window=0.3)


def test_a_punch_waiting_for_its_start_time_ends_when_it_is_cancelled() -> None:
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    began = time.monotonic()
    with pytest.raises(nat.Cancelled):
        nat.punch(
            free_port(),
            ("127.0.0.1", 9),
            b"t" * 16,
            initiator=True,
            start_at=time.time() + 10,
            window=15.0,
            cancel=cancel,
        )
    assert time.monotonic() - began < 1.0


def test_a_punch_that_is_dialing_ends_when_it_is_cancelled() -> None:
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    began = time.monotonic()
    with pytest.raises(nat.Cancelled):
        nat.punch(
            free_port(),
            ("127.0.0.1", 9),
            b"t" * 16,
            initiator=False,
            start_at=0,
            window=15.0,
            cancel=cancel,
        )
    assert time.monotonic() - began < 1.0


def test_a_hello_with_the_wrong_token_is_not_the_connection() -> None:
    port = free_port()
    caught: list[BaseException] = []

    def remote() -> None:
        try:
            nat.punch(port, ("127.0.0.1", 9), b"r" * 16, initiator=False, start_at=0, window=1.0)
        except BaseException as exc:
            caught.append(exc)

    thread = threading.Thread(target=remote)
    thread.start()
    time.sleep(0.2)
    intruder = socket.create_connection(("127.0.0.1", port))
    intruder.sendall(nat.HELLO + b"x" * 16)
    thread.join(5)
    intruder.close()
    assert isinstance(caught[0], TimeoutError)


# -- Spec: Transport, Choosing a link: the probe ----------------------------------


def test_the_probe_measures_round_trips_and_both_directions_against_the_responder() -> None:
    user, remote = socket.socketpair()
    threading.Thread(target=nat.serve_probe, args=(remote,), daemon=True).start()
    measured = Probe(round_trips=30, seconds=0.05).measure(user)
    user.close()
    assert measured.rtt_ms > 0
    assert measured.upload_bps > 0
    assert measured.download_bps > 0


def test_the_probe_defaults_are_thirty_round_trips_and_two_seconds_each_way() -> None:
    probe = Probe()
    assert (probe.round_trips, probe.seconds) == (30, 2.0)


def test_a_responder_asked_to_bridge_stops_answering_the_probe() -> None:
    user, remote = socket.socketpair()
    user.sendall(b"B")
    assert nat.serve_probe(remote) is True
    user.close()
    assert nat.serve_probe(remote) is False


def test_a_probed_link_is_spliced_to_the_ssh_server_after_the_probe() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    user, remote = socket.socketpair()
    threading.Thread(
        target=nat.serve_link, args=(remote, server.getsockname()), daemon=True
    ).start()
    Probe(round_trips=2, seconds=0.01).measure(user)
    user.sendall(b"B" + b"SSH-2.0-letify\r\n")
    conn, _ = server.accept()
    assert nat.recv_exact(conn, 16) == b"SSH-2.0-letify\r\n"
    conn.sendall(b"SSH-2.0-OpenSSH\r\n")
    assert nat.recv_exact(user, 17) == b"SSH-2.0-OpenSSH\r\n"
    for sock in (conn, user, server):
        sock.close()


# -- Spec: Transport, Choosing a link ---------------------------------------------


def test_a_lone_applicable_strategy_is_used_without_a_race_or_the_cache(isolated_home) -> None:
    """Spec "Choosing a link": with nothing to compare against, a lone strategy is not
    raced and not cached. It is still probed and held to the floor, [Link floor]: there
    is no alternative to fall back to, but a slow link is still a failure, not one."""
    direct = FakeStrategy("direct_ssh", 1, result=result(10, 10))
    skipped = FakeStrategy("tcp_punch", 2, unmet="no rendezvous")
    probe = FakeProbe()
    cache = LinkCache("lab")
    link = pipeline([direct, skipped], cache=cache, probe=probe).connect()
    assert link.strategy == "direct_ssh"
    assert (direct.assumed, direct.attempts, skipped.attempts) == (1, 0, 0)
    assert probe.measured == ["direct_ssh"]
    assert cache.load() is None


def test_a_lone_strategy_below_the_floor_fails_the_connection(isolated_home) -> None:
    slow = FakeStrategy("direct_ssh", 1, result=result(1.9, 1.9, rtt=99.6))
    with pytest.raises(letify.ProviderUnavailable, match="below the floor"):
        pipeline([slow], floor=LinkFloor.default()).connect()


def test_a_lone_strategy_that_cannot_be_probed_is_not_held_to_the_floor(isolated_home) -> None:
    # The provider fallback, such as Colab's own path, carries no probe to measure.
    fallback = FakeStrategy("fallback", 4, probed=False)
    assert pipeline([fallback], floor=LinkFloor.default()).connect().strategy == "fallback"


def test_a_lone_strategy_whose_probe_fails_is_still_used(isolated_home) -> None:
    broken = FakeStrategy("direct_ssh", 1, probe_error=True)
    assert pipeline([broken], floor=LinkFloor.default()).connect().strategy == "direct_ssh"


def test_the_lowest_rank_wins_when_it_is_not_far_slower_than_the_fastest(isolated_home) -> None:
    punch = FakeStrategy("tcp_punch", 2, result=result(5, 5))
    tailcat = FakeStrategy("tailcat", 3, result=result(18, 18))
    link = pipeline([tailcat, punch]).connect()
    assert link.strategy == "tcp_punch"
    assert tailcat.links[0].closed


def test_a_strategy_below_a_quarter_of_the_fastest_in_either_direction_is_rejected(
    isolated_home,
) -> None:
    # Colab's outbound UDP limit is the case: Tailcat uploads fine and downloads slowly.
    punch = FakeStrategy("tcp_punch", 2, result=result(13, 14))
    tailcat = FakeStrategy("tailcat", 3, result=result(24, 3))
    fallback = FakeStrategy("fallback", 4)
    assert pipeline([punch, tailcat, fallback]).connect().strategy == "tcp_punch"

    slow_up = FakeStrategy("tcp_punch", 2, result=result(2, 30))
    fast = FakeStrategy("tailcat", 3, result=result(20, 20))
    assert pipeline([slow_up, fast]).connect().strategy == "tailcat"
    assert slow_up.links[0].closed


def test_strategies_race_so_a_slow_timeout_does_not_delay_the_others(isolated_home) -> None:
    hanging = FakeStrategy("direct_ssh", 1, delay=3.0, error="timed out")
    punch = FakeStrategy("tcp_punch", 2, result=result(10, 10))
    began = time.monotonic()
    link = pipeline([hanging, punch], grace=0.2).connect()
    assert link.strategy == "tcp_punch"
    assert time.monotonic() - began < 2.0


def test_a_lower_rank_that_connects_within_the_grace_period_is_waited_for(isolated_home) -> None:
    direct = FakeStrategy("direct_ssh", 1, delay=0.3, result=result(10, 10))
    punch = FakeStrategy("tcp_punch", 2, result=result(10, 10))
    assert pipeline([direct, punch], grace=1.0).connect().strategy == "direct_ssh"


def test_the_fallback_connecting_first_does_not_start_the_grace_period(isolated_home) -> None:
    # Colab: colab exec answers at once, while the punch waits on a rendezvous that takes
    # seconds. The punch has to be given its own attempt time, not the grace period.
    punch = FakeStrategy("tcp_punch", 2, delay=0.8, result=result(10, 10))
    fallback = FakeStrategy("fallback", 4, probed=False)
    cache = LinkCache("lab")
    assert pipeline([punch, fallback], grace=0.2, cache=cache).connect().strategy == "tcp_punch"
    assert cache.load().strategy == "tcp_punch"


def test_the_fallback_is_chosen_once_every_probed_strategy_has_failed(isolated_home) -> None:
    punch = FakeStrategy("tcp_punch", 2, delay=0.5, error="no mapping")
    fallback = FakeStrategy("fallback", 4, probed=False)
    began = time.monotonic()
    assert pipeline([punch, fallback], grace=0.1).connect().strategy == "fallback"
    assert time.monotonic() - began >= 0.5


def test_the_default_grace_period_is_two_seconds() -> None:
    assert Pipeline([], target=None, alias="lab").grace == 2.0


def test_a_strategy_that_connects_after_the_choice_is_closed(isolated_home) -> None:
    late = FakeStrategy("direct_ssh", 1, delay=0.6, result=result(10, 10))
    punch = FakeStrategy("tcp_punch", 2, result=result(10, 10))
    assert pipeline([late, punch], grace=0.1).connect().strategy == "tcp_punch"
    time.sleep(0.8)
    assert late.links and late.links[0].closed


def test_an_attempt_still_running_when_the_choice_is_made_is_cancelled(isolated_home) -> None:
    direct = FakeStrategy("direct_ssh", 1, result=result(10, 10))
    punch = FakeStrategy("tcp_punch", 2, delay=20.0, result=result(10, 10), cancellable=True)
    began = time.monotonic()
    assert pipeline([direct, punch], grace=0.1).connect().strategy == "direct_ssh"
    assert punch.ended.wait(1.0)
    assert time.monotonic() - began < 2.0
    assert punch.cancelled
    assert punch.links == []


def test_a_cancelled_attempt_is_printed_as_cancelled_not_failed(isolated_home) -> None:
    lines: list[str] = []
    direct = FakeStrategy("direct_ssh", 1, result=result(10, 10))
    punch = FakeStrategy("tcp_punch", 2, delay=20.0, cancellable=True)
    chosen = Pipeline(
        [direct, punch],
        target=None,
        alias="lab",
        probe=FakeProbe(),
        fingerprint=fingerprint(),
        grace=0.1,
        timeout=5.0,
        say=lines.append,
    ).connect()
    assert chosen.strategy == "direct_ssh"
    assert punch.ended.wait(1.0)
    time.sleep(0.05)
    assert any("tcp_punch cancelled" in line for line in lines)
    assert not any("tcp_punch failed" in line for line in lines)


def test_a_lone_connected_strategy_is_chosen_even_when_its_probe_fails(isolated_home) -> None:
    broken = FakeStrategy("tcp_punch", 2, probe_error=True)
    failed = FakeStrategy("tailcat", 3, error="no UDP")
    assert pipeline([broken, failed]).connect().strategy == "tcp_punch"


def test_a_probe_failure_rejects_a_strategy_when_another_connected(isolated_home) -> None:
    broken = FakeStrategy("tcp_punch", 2, probe_error=True)
    tailcat = FakeStrategy("tailcat", 3, result=result(1, 1))
    assert pipeline([broken, tailcat]).connect().strategy == "tailcat"
    assert broken.links[0].closed


def test_an_unprobed_link_is_chosen_only_when_no_probed_link_remains(isolated_home) -> None:
    fallback = FakeStrategy("fallback", 4)
    punch = FakeStrategy("tcp_punch", 2, result=result(1, 1))
    assert pipeline([fallback, punch]).connect().strategy == "tcp_punch"

    cache = LinkCache("lab")
    fallback = FakeStrategy("fallback", 4)
    failed = FakeStrategy("tcp_punch", 2, error="refused")
    assert pipeline([fallback, failed], cache=cache).connect().strategy == "fallback"
    assert cache.load() is None


def test_when_nothing_connects_every_strategy_is_named_with_its_reason(isolated_home) -> None:
    strategies = [
        FakeStrategy("tcp_punch", 2, error="connection refused"),
        FakeStrategy("tailcat", 3, error="no UDP"),
        FakeStrategy("reverse_ssh", 4, unmet="no reverse_ssh entry"),
    ]
    with pytest.raises(letify.ProviderUnavailable) as caught:
        pipeline(strategies).connect()
    message = str(caught.value)
    for piece in ("tcp_punch: connection refused", "tailcat: no UDP", "no reverse_ssh entry"):
        assert piece in message


# -- Spec: Transport, Link cache ----------------------------------------------------


def test_the_winner_is_written_to_the_account_directory(isolated_home) -> None:
    cache = LinkCache("lab")
    punch = FakeStrategy("tcp_punch", 2, result=result(13, 14, rtt=180.5))
    tailcat = FakeStrategy("tailcat", 3, result=result(1, 1))
    pipeline([punch, tailcat], cache=cache).connect()
    written = json.loads(cache.path.read_text())
    assert cache.path.parent.name == "lab"
    assert cache.path.name == "link.json"
    assert written["strategy"] == "tcp_punch"
    assert written["fingerprint"] == {"public_ip": "203.0.113.7", "interface": "eth0"}
    assert written["probe"]["rtt_ms"] == 180.5
    assert written["probe"]["upload_bps"] == 13 * MIB


def test_the_cached_strategy_is_tried_alone_and_kept_at_half_its_throughput(
    isolated_home,
) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(30, 30))
    tailcat = FakeStrategy("tailcat", 3, result=result(10, 10))
    assert pipeline([punch, tailcat], cache=cache).connect().strategy == "tailcat"
    assert punch.attempts == 0


def test_a_cached_strategy_below_half_its_throughput_runs_the_full_race(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(30, 30))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 9))
    assert pipeline([punch, tailcat], cache=cache).connect().strategy == "tcp_punch"
    assert cache.load().strategy == "tcp_punch"


def test_a_changed_network_runs_the_full_race(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(20, 20))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 20))
    link = pipeline([punch, tailcat], cache=cache, fp=fingerprint(interface="wlan0")).connect()
    assert link.strategy == "tcp_punch"
    assert tailcat.attempts == 1
    assert cache.load().fingerprint == Fingerprint("203.0.113.7", "wlan0")


def test_an_unknown_public_address_matches_no_cache_entry(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint(None, "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(20, 20))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 20))
    chosen = pipeline([punch, tailcat], cache=cache, fp=fingerprint(ip=None)).connect()
    assert chosen.strategy == "tcp_punch"


def test_a_cache_file_that_does_not_parse_is_ignored(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.path.parent.mkdir(parents=True)
    cache.path.write_text("{not json")
    assert cache.load() is None


# -- Spec: Transport, Link floor ---------------------------------------------------


def test_a_relayed_strategy_below_the_floor_is_refused_even_though_it_connected(
    isolated_home,
) -> None:
    """Spec "Link floor": a strategy below the default floor is a failure, not a fallback,
    even when it is the only probed strategy left. This is the reported case: a Tailcat
    relay measures 1.9 MiB/s, 99.6 ms against a direct LAN link that measures 51 MiB/s."""
    relay = FakeStrategy("tailcat", 3, result=result(1.9, 1.9, rtt=99.6))
    broken = FakeStrategy("tcp_punch", 2, error="no mapping")
    floor = LinkFloor.default()
    with pytest.raises(letify.ProviderUnavailable, match="below the floor"):
        pipeline([broken, relay], floor=floor).connect()


def test_a_relayed_strategy_is_refused_when_a_faster_one_also_connected(isolated_home) -> None:
    direct = FakeStrategy("direct_ssh", 1, result=result(51, 51, rtt=0.2))
    relay = FakeStrategy("tailcat", 3, result=result(1.9, 1.9, rtt=99.6))
    floor = LinkFloor.default()
    chosen = pipeline([direct, relay], floor=floor).connect()
    assert chosen.strategy == "direct_ssh"
    assert relay.links[0].closed


def test_a_round_trip_above_the_floor_is_refused_even_with_fast_throughput(isolated_home) -> None:
    slow_round_trip = FakeStrategy("tailcat", 3, result=result(20, 20, rtt=500.0))
    floor = LinkFloor(min_bps=0.0, max_rtt_ms=300.0)
    broken = FakeStrategy("tcp_punch", 2, error="unreachable")
    expected = "round trip 500.0 ms above the floor"
    with pytest.raises(letify.ProviderUnavailable, match=expected):
        pipeline([broken, slow_round_trip], floor=floor).connect()


def test_the_default_floor_is_ten_mib_per_second_and_three_hundred_ms() -> None:
    floor = LinkFloor.default()
    assert floor.min_bps == 10.0 * MIB
    assert floor.max_rtt_ms == 300.0


def test_a_cached_strategy_below_the_floor_is_rejected_and_the_race_runs(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(1.9, 1.9, rtt=99.6), Fingerprint("203.0.113.7", "eth0"))
    direct = FakeStrategy("direct_ssh", 1, result=result(51, 51, rtt=0.2))
    relay = FakeStrategy("tailcat", 3, result=result(1.9, 1.9, rtt=99.6))
    chosen = pipeline([direct, relay], cache=cache, floor=LinkFloor.default()).connect()
    assert chosen.strategy == "direct_ssh"
    assert cache.load().strategy == "direct_ssh"


# -- Spec: Transport, Link cache: staleness and format ------------------------------


def test_a_stale_cache_entry_runs_the_full_race_even_on_the_same_network(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.save(
        "tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"), cached_at=time.time() - 7200
    )
    punch = FakeStrategy("tcp_punch", 2, result=result(30, 30))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 20))
    chosen = pipeline([punch, tailcat], cache=cache).connect()
    assert chosen.strategy == "tcp_punch"
    assert tailcat.attempts == 1


def test_a_fresh_cache_entry_is_not_raced_again(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(30, 30))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 20))
    assert pipeline([punch, tailcat], cache=cache).connect().strategy == "tailcat"
    assert punch.attempts == 0


def test_a_cache_file_in_an_older_format_is_ignored_and_remeasured(isolated_home) -> None:
    cache = LinkCache("lab")
    cache.path.parent.mkdir(parents=True)
    cache.path.write_text(
        json.dumps(
            {
                "strategy": "tailcat",
                "probe": result(20, 20).to_dict(),
                "fingerprint": {"public_ip": "203.0.113.7", "interface": "eth0"},
            }
        )
    )
    assert cache.load() is None


def test_the_fingerprint_names_the_default_route_interface_on_linux(tmp_path) -> None:
    table = tmp_path / "route"
    table.write_text(
        "Iface\tDestination\tGateway\ndocker0\t000011AC\t00000000\nenp3s0\t00000000\t0100A8C0\n"
    )
    assert nat.default_route_interface(route_table=table) == "enp3s0"


def test_the_fingerprint_reads_the_public_address_from_stun(stun_server: StunServer) -> None:
    assert network_fingerprint(stun_server.address).public_ip == "127.0.0.1"


def test_a_fingerprint_with_no_stun_answer_has_no_public_address() -> None:
    assert network_fingerprint(("127.0.0.1", 9), timeout=0.5).public_ip is None


# Spec: Transport, Link floor.
@pytest.mark.parametrize("up,down,rtt", [(2.6, 0.1, 177.4), (8.8, 10.7, 175.0)])
def test_floor_rejection_cannot_select_an_unprobed_fallback(isolated_home, up, down, rtt):
    slow = FakeStrategy("tailcat", 3, result=result(up, down, rtt))
    fallback = FakeStrategy("fallback", 4, probed=False)
    with pytest.raises(letify.ProviderUnavailable) as caught:
        pipeline([slow, fallback], floor=LinkFloor.default()).connect()
    message = str(caught.value)
    assert "tailcat: below the floor" in message
    assert f"up {up:.1f} MiB/s" in message
    assert "10.0 MiB/s" in message
    assert "min_mib_per_s" in message
    assert "max_rtt_ms" in message
    assert slow.links[0].closed
    assert fallback.links[0].closed


@pytest.mark.parametrize("race_fails", [False, True])
def test_a_cached_floor_rejection_cannot_escape_to_an_unprobed_fallback(
    isolated_home, monkeypatch, race_fails
):
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    slow = FakeStrategy("tailcat", 3, result=result(2.6, 0.1, 177.4))
    fallback = FakeStrategy("fallback", 4, probed=False)
    attempt = slow.attempt

    def reconnect(target, cancel=None):
        if race_fails and slow.attempts:
            raise OSError("connection lost after cached probe")
        return attempt(target, cancel=cancel)

    monkeypatch.setattr(slow, "attempt", reconnect)
    with pytest.raises(letify.ProviderUnavailable, match="tailcat: below the floor"):
        pipeline([slow, fallback], cache=cache, floor=LinkFloor.default()).connect()
    assert all(link.closed for link in slow.links + fallback.links)
    assert cache.load().probe == result(20, 20)


def test_floor_rejections_name_both_account_overrides():
    violations = LinkFloor.default().violations(result(2.6, 0.1, 500))
    assert any("up 2.6 MiB/s" in text for text in violations)
    assert any("down 0.1 MiB/s" in text for text in violations)
    assert any("round trip 500.0 ms" in text for text in violations)
    assert "min_mib_per_s" in ", ".join(violations)
    assert "max_rtt_ms" in ", ".join(violations)


# Spec: Transport, Link floor.
def test_a_cached_floor_rejection_cannot_accept_a_lone_failed_probe(isolated_home, monkeypatch):
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    slow = FakeStrategy("tailcat", 3, result=result(2.6, 0.1, 177.4))
    fallback = FakeStrategy("fallback", 4, probed=False, error="fallback unavailable")
    attempt = slow.attempt

    def reconnect(target, cancel=None):
        if slow.attempts:
            slow.probe_error = True
        return attempt(target, cancel=cancel)

    monkeypatch.setattr(slow, "attempt", reconnect)
    with pytest.raises(letify.ProviderUnavailable, match="tailcat: below the floor"):
        pipeline([slow, fallback], cache=cache, floor=LinkFloor.default()).connect()
    assert all(link.closed for link in slow.links)


# -- Spec: Transport, the simultaneous open: the window covers the rendezvous -------


def test_the_punch_window_is_longer_than_the_rendezvous_can_take() -> None:
    # Spec "the simultaneous open": the start time is fixed before the rendezvous, so the
    # remote reaches it while the answer is still travelling back. A window shorter than
    # the connect timeout can close before the user's side has begun.
    from letify.transport import strategies

    assert nat.DEFAULT_WINDOW > strategies.CONNECT_TIMEOUT


def test_a_punch_whose_window_has_already_closed_fails_at_once() -> None:
    # Spec "the simultaneous open": the remote reports when it stops, and the user's side
    # stops then too instead of dialing alone.
    began = time.perf_counter()
    with pytest.raises(TimeoutError) as raised:
        nat.punch(
            free_port(),
            ("127.0.0.1", 9),
            b"t" * 16,
            initiator=True,
            start_at=0,
            until=time.time() - 1.0,
        )
    assert time.perf_counter() - began < 2.0
    assert "window" in str(raised.value)


def test_the_remote_half_reports_when_it_stops_punching() -> None:
    # Spec "the simultaneous open": punch_until travels in the answer so both sides stop
    # together.
    answer, _ = nat.begin(
        {
            "kind": "tcp_punch",
            "mapping": ["203.0.113.7", 41000],
            "token": ("a" * 32),
            "start_at": time.time() + 1.0,
            "window": 45.0,
            "ssh_port": 22,
        }
    )
    assert "punch_until" in answer
    assert answer["punch_until"] >= time.time() + 45.0 - 1.0


def test_a_dial_under_way_is_replaced_at_a_fixed_cadence() -> None:
    # Spec "the simultaneous open": the kernel's own SYN backoff sends too few SYNs, so
    # the dial is reissued on a fresh socket while the window lasts.
    made: list[int] = []
    real = nat.reusable_socket

    def counting(port: int = 0, host: str = "0.0.0.0"):
        made.append(port)
        return real(port, host)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nat, "reusable_socket", counting)
        with pytest.raises(TimeoutError):
            # 198.51.100.0/24 is reserved for documentation, so the SYN goes unanswered
            # and the dial stays under way rather than failing fast.
            nat.punch(
                free_port(),
                ("198.51.100.7", 41000),
                b"t" * 16,
                initiator=True,
                start_at=0,
                window=2.5,
            )
    # One listener plus a dial reissued every 0.5 s over 2.5 s.
    assert len(made) >= 4, made
