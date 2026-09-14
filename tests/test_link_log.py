"""Tests for the log lines the connection pipeline prints.

Spec: Transport, Choosing a link (the log table) and Link cache (the cache's decisions).
Strategies and probes are the fakes from conftest, so delays and numbers are controlled.
"""

from __future__ import annotations

from conftest import FakeProbe, FakeStrategy

import letify
from letify.transport.pipeline import Fingerprint, LinkCache, Pipeline
from letify.transport.probe import ProbeResult

MIB = 1024 * 1024


def result(up: float, down: float, rtt: float = 5.0) -> ProbeResult:
    return ProbeResult(rtt_ms=rtt, upload_bps=up * MIB, download_bps=down * MIB)


def pipeline(strategies, *, cache=None, grace=0.2, interface="eth0", previous=None) -> Pipeline:
    return Pipeline(
        strategies,
        target=None,
        alias="lab",
        probe=FakeProbe(),
        cache=cache,
        fingerprint=lambda: Fingerprint("203.0.113.7", interface),
        grace=grace,
        timeout=5.0,
        previous=previous,
    )


def lines(capsys) -> list[str]:
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err.splitlines()


def one(found: list[str], *pieces: str) -> str:
    matching = [line for line in found if all(piece in line for piece in pieces)]
    assert matching, f"no line carries {pieces}: {found}"
    assert matching[0].startswith("letify: ")
    return matching[0]


# -- Spec: Transport, Choosing a link: race start, outcomes, probes, choice ----------


def test_the_race_start_names_attempted_and_skipped_strategies(isolated_home, capsys) -> None:
    direct = FakeStrategy("direct_ssh", 1, unmet="no address")
    punch = FakeStrategy("tcp_punch", 2, result=result(13, 14, rtt=180.5))
    tailcat = FakeStrategy("tailcat", 3, delay=0.05, result=result(24, 20))
    fallback = FakeStrategy("fallback", 4, probed=False)
    pipeline([direct, punch, tailcat, fallback]).connect()
    found = lines(capsys)
    start = one(found, "connecting to lab")
    assert "direct_ssh" in start and "no address" in start
    assert "tcp_punch" in start and "tailcat" in start
    assert "fallback held back" in start
    one(found, "lab", "tcp_punch connected in", " s")
    one(found, "lab", "tcp_punch", "180.5 ms", "13.0 MiB/s", "14.0 MiB/s")
    one(found, "lab", "chose tcp_punch", "lowest rank within 25% of the fastest")


def test_a_failed_and_a_timed_out_strategy_are_each_printed(isolated_home, capsys) -> None:
    hanging = FakeStrategy("direct_ssh", 1, delay=3.0, result=result(10, 10))
    broken = FakeStrategy("tailcat", 3, error="no UDP")
    punch = FakeStrategy("tcp_punch", 2, result=result(10, 10))
    pipeline([hanging, punch, broken], grace=0.2).connect()
    found = lines(capsys)
    one(found, "lab", "tailcat failed", "no UDP")
    one(found, "lab", "direct_ssh timed out")
    one(found, "lab", "chose tcp_punch", "only one connected")


def test_a_strategy_rejected_by_the_ratio_rule_is_printed_with_its_numbers(
    isolated_home, capsys
) -> None:
    punch = FakeStrategy("tcp_punch", 2, result=result(13, 14))
    tailcat = FakeStrategy("tailcat", 3, result=result(24, 3))
    pipeline([punch, tailcat]).connect()
    found = lines(capsys)
    rejected = one(found, "lab", "rejected tailcat")
    assert "3.0 MiB/s" in rejected and "14.0 MiB/s" in rejected
    one(found, "lab", "chose tcp_punch")


def test_a_lone_applicable_strategy_is_printed(isolated_home, capsys) -> None:
    direct = FakeStrategy("direct_ssh", 1)
    skipped = FakeStrategy("tcp_punch", 2, unmet="no rendezvous")
    pipeline([direct, skipped]).connect()
    found = lines(capsys)
    lone = one(found, "connecting to lab", "direct_ssh alone")
    assert "no rendezvous" in lone


def test_the_fallback_is_printed_as_a_fall_back(isolated_home, capsys) -> None:
    failed = FakeStrategy("tcp_punch", 2, delay=0.1, error="no mapping")
    fallback = FakeStrategy("fallback", 4, probed=False)
    pipeline([failed, fallback], grace=0.1).connect()
    found = lines(capsys)
    one(found, "lab", "chose fallback", "no probed strategy connected")
    one(found, "lab", "falling back to the provider's own path")


# -- Spec: Transport, Choosing a link: switches ---------------------------------------


def test_a_strategy_that_takes_over_from_the_first_connected_is_printed(
    isolated_home, capsys
) -> None:
    punch = FakeStrategy("tcp_punch", 2, delay=0.1, result=result(20, 20))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 20))
    pipeline([punch, tailcat], grace=1.0).connect()
    one(lines(capsys), "lab", "switching from tailcat to tcp_punch")


def test_a_link_connected_again_names_the_old_and_new_strategy(isolated_home, capsys) -> None:
    punch = FakeStrategy("tcp_punch", 2, result=result(20, 20))
    tailcat = FakeStrategy("tailcat", 3, error="no UDP")
    pipeline([punch, tailcat], previous="tailcat").connect()
    one(lines(capsys), "lab", "re-established over tcp_punch", "was tailcat")


# -- Spec: Transport, Link cache ------------------------------------------------------


def test_a_cached_strategy_accepted_is_printed_with_its_cached_throughput(
    isolated_home, capsys
) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(30, 30))
    tailcat = FakeStrategy("tailcat", 3, result=result(10, 10))
    pipeline([punch, tailcat], cache=cache).connect()
    found = lines(capsys)
    one(found, "lab", "cached tailcat alone", "20.0 MiB/s")
    one(found, "lab", "cached tailcat accepted")


def test_a_cached_strategy_below_half_is_rejected_and_the_cache_rewritten(
    isolated_home, capsys
) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(30, 30))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 9))
    pipeline([punch, tailcat], cache=cache).connect()
    found = lines(capsys)
    rejected = one(found, "lab", "cached tailcat rejected", "below 50%")
    assert "9.0 MiB/s" in rejected and "20.0 MiB/s" in rejected
    one(found, "lab", "cache rewritten", "tcp_punch")


def test_a_cached_strategy_that_fails_to_connect_is_rejected(isolated_home, capsys) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(30, 30))
    tailcat = FakeStrategy("tailcat", 3, error="no UDP")
    pipeline([punch, tailcat], cache=cache).connect()
    one(lines(capsys), "lab", "cached tailcat rejected", "failed to connect", "no UDP")


def test_a_changed_network_fingerprint_is_printed(isolated_home, capsys) -> None:
    cache = LinkCache("lab")
    cache.save("tailcat", result(20, 20), Fingerprint("203.0.113.7", "eth0"))
    punch = FakeStrategy("tcp_punch", 2, result=result(20, 20))
    tailcat = FakeStrategy("tailcat", 3, result=result(20, 20))
    pipeline([punch, tailcat], cache=cache, interface="wlan0").connect()
    one(lines(capsys), "lab", "cached tailcat rejected", "network fingerprint changed")


# -- Spec: Transport, Choosing a link: the Launcher's announce flag -------------------


def test_a_launcher_prints_the_connection_lines_by_default(config_file, capsys) -> None:
    body = '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
    letify.Launcher(config_file(body), home=False).provider("lab").connect()
    one(lines(capsys), "connecting to lab", "direct_ssh (gpu.example.edu:22) alone")


def test_a_quiet_launcher_prints_no_connection_lines(config_file, capsys) -> None:
    body = '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
    letify.Launcher(config_file(body), home=False, announce=False).provider("lab").connect()
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
