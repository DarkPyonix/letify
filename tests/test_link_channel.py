"""The worker channel and transfer streams over a letify-link session.

Spec "Sidecar interface" item 7 (Bootstrap and the worker channel). The "remote" is this
machine: the ssh hop is replaced by ``sh -c``, which is the only part not exercised. The
shim, the sidecars on both ends, the session, the streams and the worker are real. The tests
build the sidecar like tests/test_linkd.py and are skipped, with the reason, without cargo.
"""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import provider_of

from letify.errors import ProviderUnavailable, UnsupportedMode
from letify.providers.shell import Shell
from letify.runtime.channel import PersistentChannel
from letify.transport import linkchannel, linkd

MIB = 1 << 20
#: The probe the connect builds, taken before the autouse fixture swaps in a quick one.
REAL_MAKE_PROBE = linkchannel.make_probe


class LocalShell(Shell):
    """A shell account whose interpreter on the machine is the one running the tests."""

    remote_python = sys.executable

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        """The account's plain ssh command, run on this machine under ``sh -c``."""
        return self.__dict__["fake_ssh"].ssh_command(remote_command)

    def link(self, runtime: Any = None) -> Any:
        """An account with no address is reached through its pipeline, here the same hop."""
        if self.address is None:
            return self.__dict__["fake_ssh"]
        return super().link(runtime)


class FakeSsh:
    """Records the commands the bootstrap would send over ssh and runs them locally."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def ssh_command(self, remote: str | None = None) -> list[str]:
        self.commands.append(remote or "")
        return ["sh", "-c", remote or "true"]


FakeSshLink = FakeSsh


def machine(tmp_path: Path, fake: FakeSshLink, **options: Any) -> LocalShell:
    provider = provider_of(
        LocalShell, "lab", address="127.0.0.1", workspace=str(tmp_path / "ws"), **options
    )
    provider.__dict__["fake_ssh"] = fake
    return provider


def runtime(name: str = "r1") -> Any:
    return SimpleNamespace(name=name, external_id=None)


@pytest.fixture(autouse=True)
def local_sidecar(sidecar_binary, monkeypatch):
    """The local end of every session uses the built sidecar, as LETIFY_LINK_BIN says."""
    monkeypatch.setenv(linkd.ENV_BINARY, str(sidecar_binary))


@pytest.fixture
def ssh() -> FakeSshLink:
    return FakeSshLink()


@pytest.fixture
def linked(sidecar_binary, isolated_home, tmp_path, ssh):
    provider = machine(tmp_path, ssh, link="letify-link", link_binary=str(sidecar_binary))
    yield provider
    provider.close_link()


# The bodies are built by factories so they are nested functions, which cloudpickle ships by
# value. A module level function would travel by reference and the worker has no test module.


def worker_pid():
    def body():
        import os

        return os.getpid()

    return body


def digest():
    def body(data):
        import hashlib

        return hashlib.sha256(data).hexdigest()

    return body


def sleep_then_pid():
    def body(seconds):
        import os
        import time

        time.sleep(seconds)
        return os.getpid()

    return body


def test_a_call_runs_over_a_letify_link_worker_channel(linked, ssh) -> None:
    channel = linked.open_channel(runtime())
    try:
        assert isinstance(channel, linkchannel.LinkChannel)
        pid, _logs = channel.call(worker_pid(), (), {})
        assert pid != os.getpid()
        again, _logs = channel.call(worker_pid(), (), {})
        assert again == pid
        # ssh ran the shim once and nothing else: the data path is the session.
        assert len(ssh.commands) == 1
        assert "-u -c" in ssh.commands[0]
    finally:
        channel.close()


def test_a_large_argument_transfers_over_link_streams(linked, ssh) -> None:
    runtime_ = runtime()
    parts = [os.urandom(MIB) * 16 for _ in range(4)]
    channels = [linked.transfer_channel(runtime_) for _ in parts]
    results: list[Any] = [None] * len(parts)

    def send(i: int) -> None:
        results[i] = channels[i].call(digest(), (parts[i],), {})[0]

    try:
        assert all(isinstance(c, linkchannel.LinkChannel) for c in channels)
        threads = [threading.Thread(target=send, args=(i,)) for i in range(len(parts))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        assert results == [hashlib.sha256(p).hexdigest() for p in parts]
        assert sum(len(p) for p in parts) == 64 * MIB
        pids = {c.call(worker_pid(), (), {})[0] for c in channels}
        assert len(pids) == len(channels)
        assert len(ssh.commands) == 1
    finally:
        for channel in channels:
            channel.close()


def test_transfers_open_bulk_streams_and_the_worker_channel_an_interactive_one(
    linked, ssh, monkeypatch
) -> None:
    """Spec letify-link, Session item 7: transfers are bulk, the worker channel is not."""
    opened: list[str] = []
    real = linkd.LinkSession.open_stream

    def recording(self, priority: str = "interactive"):
        opened.append(priority)
        return real(self, priority)

    monkeypatch.setattr(linkd.LinkSession, "open_stream", recording)
    worker = linked.open_channel(runtime())
    transfer = linked.transfer_channel(runtime())
    try:
        worker.call(worker_pid(), (), {})
        transfer.call(worker_pid(), (), {})
    finally:
        worker.close()
        transfer.close()
    # The first bulk stream is the connect's own probe stream; the rest is the call.
    assert opened == ["bulk", "interactive", "bulk"]


class CuttableProxy:
    """A TCP forwarder whose live connections can be cut while it keeps accepting."""

    def __init__(self, target: str) -> None:
        host, _, port = target.rpartition(":")
        self.target = (host, int(port))
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(8)
        self.address = f"127.0.0.1:{self.server.getsockname()[1]}"
        self.pairs: list[tuple[socket.socket, socket.socket]] = []
        self.accepted = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            upstream = socket.create_connection(self.target)
            self.pairs.append((client, upstream))
            self.accepted += 1
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._copy, args=(a, b), daemon=True).start()

    @staticmethod
    def _copy(source: socket.socket, sink: socket.socket) -> None:
        try:
            while data := source.recv(65536):
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (source, sink):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def cut(self) -> None:
        pairs, self.pairs = self.pairs, []
        for client, upstream in pairs:
            for sock in (client, upstream):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self) -> None:
        self.cut()
        self.server.close()


def test_the_worker_channel_survives_its_tcp_path_dying(
    sidecar_binary,
    isolated_home,
    tmp_path,
    ssh,
    monkeypatch,
) -> None:
    proxies: list[CuttableProxy] = []
    original = linkd.LinkDaemon.open_session

    def through_proxy(self, peer, candidates, rendezvous=None, *, config=None):
        candidates = list(candidates)
        proxy = CuttableProxy(candidates[0])
        proxies.append(proxy)
        return original(self, peer, [proxy.address], rendezvous, config=config)

    monkeypatch.setattr(linkd.LinkDaemon, "open_session", through_proxy)
    provider = machine(tmp_path, ssh, link="letify-link", link_binary=str(sidecar_binary))
    channel = provider.open_channel(runtime())
    try:
        started = time.monotonic()
        outcome: list[Any] = []
        thread = threading.Thread(
            target=lambda: outcome.append(channel.call(sleep_then_pid(), (3.0,), {})[0])
        )
        thread.start()
        time.sleep(1.0)
        assert proxies[0].pairs, "the session has a TCP path to cut"
        proxies[0].cut()
        thread.join(60)
        assert outcome and outcome[0] != os.getpid()
        assert time.monotonic() - started >= 3.0
        assert proxies[0].accepted >= 2, "the sidecar dialed the path again"
    finally:
        channel.close()
        provider.close_link()
        for proxy in proxies:
            proxy.close()


def test_a_session_over_a_punched_pair_carries_a_call(sidecar_binary, isolated_home, tmp_path, ssh):
    provider = machine(
        tmp_path, ssh, link="letify-link", link_binary=str(sidecar_binary), link_punch=True
    )
    channel = provider.open_channel(runtime())
    try:
        assert channel.call(worker_pid(), (), {})[0] != os.getpid()
    finally:
        channel.close()
        provider.close_link()


@pytest.mark.parametrize("option", [{}, {"link": "ssh"}])
def test_link_option_absent_keeps_ssh(isolated_home, tmp_path, ssh, option) -> None:
    provider = machine(tmp_path, ssh, **option)
    assert provider.uses_letify_link is False
    assert type(provider.open_channel(runtime())) is PersistentChannel
    assert type(provider.transfer_channel(runtime())) is PersistentChannel
    assert provider.__dict__.get("_link_remotes") is None


def test_an_unknown_link_value_is_refused(isolated_home, tmp_path, ssh) -> None:
    provider = machine(tmp_path, ssh, link="wireguard")
    with pytest.raises(ProviderUnavailable, match="wireguard"):
        provider.open_channel(runtime())


def test_a_missing_sidecar_binary_on_the_remote_fails_the_connect_naming_it(
    sidecar_binary,
    isolated_home,
    tmp_path,
    ssh,
) -> None:
    missing = tmp_path / "nowhere" / "letify-link"
    provider = machine(tmp_path, ssh, link="letify-link", link_binary=str(missing))
    with pytest.raises(ProviderUnavailable) as raised:
        provider.open_channel(runtime())
    assert f"letify-link is not at {missing}" in str(raised.value)
    assert "lab" in str(raised.value)
    # Only the shim ran over ssh: no ssh data path took over.
    assert len(ssh.commands) == 1


def test_a_link_run_record_without_the_link_section_is_refused(
    isolated_home, tmp_path, ssh
) -> None:
    provider = machine(tmp_path, ssh, link="letify-link")
    with pytest.raises(UnsupportedMode, match="letify-link"):
        provider.attach_channel("r1", {"door": "/tmp/door"})


def measured(rtt_ms: float, up_mib: float, down_mib: float) -> Any:
    from letify.transport.probe import ProbeResult

    return ProbeResult(rtt_ms, up_mib * MIB, down_mib * MIB)


def floored_remote(closed: list[bool]) -> Any:
    return SimpleNamespace(alias="lab", close=lambda: closed.append(True))


def test_a_session_below_the_link_floor_fails_the_connect_naming_the_settings_and_both_rates():
    closed: list[bool] = []
    with pytest.raises(ProviderUnavailable) as raised:
        linkchannel.hold_to_floor(
            floored_remote(closed), measured(400.0, 1.0, 2.0), 300.0, 10.0 * MIB
        )
    text = str(raised.value)
    assert "round trip 400.0 ms above the floor 300 ms" in text
    assert "up 1.0 MiB/s below 100% of the floor 10.0 MiB/s" in text
    assert "down 2.0 MiB/s below 100% of the floor 10.0 MiB/s" in text
    assert "min_mib_per_s" in text and "max_rtt_ms" in text
    assert closed == [True]


def test_a_policed_session_names_the_pattern_and_reverse_ssh() -> None:
    with pytest.raises(ProviderUnavailable) as raised:
        linkchannel.hold_to_floor(floored_remote([]), measured(50.0, 30.0, 0.1), 300.0, 10.0 * MIB)
    assert "reverse_ssh" in str(raised.value)


def test_a_session_at_the_floor_is_accepted() -> None:
    closed: list[bool] = []
    linkchannel.hold_to_floor(floored_remote(closed), measured(50.0, 12.0, 11.0), 300.0, 10.0 * MIB)
    assert closed == []
    # No floor given, nothing to hold it to.
    linkchannel.hold_to_floor(floored_remote(closed), measured(900.0, 0.1, 0.1), None, None)
    assert closed == []


def test_a_connect_whose_probe_measures_below_the_floor_fails_and_closes(
    linked, monkeypatch
) -> None:
    monkeypatch.setattr(
        linkchannel, "measure_session", lambda remote, min_bps=None: measured(60.0, 0.5, 0.4)
    )
    with pytest.raises(ProviderUnavailable) as raised:
        linked.open_channel(runtime())
    text = str(raised.value)
    assert "up 0.5 MiB/s" in text and "down 0.4 MiB/s" in text
    assert "min_mib_per_s" in text


def test_a_connect_whose_probe_cannot_run_fails_with_the_reason(linked, monkeypatch) -> None:
    def broken(remote, min_bps=None):
        raise OSError("the probe responder answered out of order")

    monkeypatch.setattr(linkchannel, "measure_session", broken)
    with pytest.raises(ProviderUnavailable, match="cannot measure the session"):
        linked.open_channel(runtime())


def test_the_connect_measures_the_session_over_a_probe_stream_of_the_shim(linked) -> None:
    channel = linked.open_channel(runtime())
    try:
        remote = linked.__dict__["_link_remotes"][""]
        assert remote.measured is not None
        assert remote.rtt_ms == remote.measured.rtt_ms > 0
        assert remote.upload_bps == remote.measured.upload_bps > 0
        assert remote.download_bps == remote.measured.download_bps > 0
        assert set(remote.phases) == {"bootstrap", "binary", "shim", "punch", "session", "probe"}
        assert all(seconds >= 0 for seconds in remote.phases.values())
    finally:
        channel.close()


def test_the_transfer_decision_reads_the_bulk_probe_not_the_idle_probe() -> None:
    from letify.store import pathdata

    session = SessionStub([{"srtt_ms": 90.0, "probe_rate": 0.15 * MIB}])
    remote = linkchannel.LinkRemote(None, session, "lab")  # type: ignore[arg-type]
    assert remote.upload_bps == 0.15 * MIB
    remote.measured = measured(55.0, 27.8, 30.0)
    assert (remote.rtt_ms, remote.upload_bps, remote.download_bps) == (
        55.0,
        27.8 * MIB,
        30.0 * MIB,
    )
    provider = SimpleNamespace(
        link=lambda runtime: linkchannel.PlainSshLink(None, lambda: remote),
        transfer_streams=4,
        transfer_parallel_mib=0,
    )
    assert pathdata._measured_rtt(provider, None) == 55.0


def test_paths_carry_the_kind_of_their_subflow(linked) -> None:
    channel = linked.open_channel(runtime())
    try:
        remote = linked.__dict__["_link_remotes"][""]
        assert [p["kind"] for p in remote.session.paths()] == ["tcp"]
    finally:
        channel.close()


def test_the_connect_announces_each_decision_in_order(linked, capsys) -> None:
    channel = linked.open_channel(runtime())
    try:
        err = capsys.readouterr().err
        lines = [line.split("letify: ", 1)[1] for line in err.splitlines() if "letify: " in line]
        order = [
            "lab letify-link: binary ",
            "lab letify-link: bootstrapping over ssh",
            "lab letify-link: shim ready in ",
            "lab letify-link: path ",
            "lab letify-link: probe round trip ",
            "lab letify-link: chose the session",
            "lab letify-link: connected in ",
        ]
        at = -1
        for prefix in order:
            found = [i for i, line in enumerate(lines) if line.startswith(prefix) and i > at]
            assert found, f"no line starting {prefix!r} after {at} in {lines}"
            at = found[0]
        path = next(line for line in lines if line.startswith("lab letify-link: path "))
        assert " tcp direct rtt " in path
        done = next(line for line in lines if line.startswith("lab letify-link: connected in "))
        for phase in ("bootstrap", "binary", "shim", "punch", "session", "probe"):
            assert f"{phase} " in done
    finally:
        channel.close()


def test_the_connect_announces_nothing_when_announce_is_off(linked, capsys) -> None:
    linked.announce = False
    channel = linked.open_channel(runtime())
    try:
        assert capsys.readouterr().err == ""
    finally:
        channel.close()


def test_a_rejected_session_announces_the_rejection(linked, capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        linkchannel, "measure_session", lambda remote, min_bps=None: measured(60.0, 0.5, 0.4)
    )
    with pytest.raises(ProviderUnavailable):
        linked.open_channel(runtime())
    assert "lab letify-link: rejected, " in capsys.readouterr().err


def test_the_wheel_extraction_names_the_sidecar_binary() -> None:
    from letify import install

    source = install.remote_link_source("2.10.2")
    assert "letify/remoting/lib/letify-link" in source and "LETIFY-LINK" in source
    assert "letify-quic" not in source and "LETIFY-QUIC" not in source
    assert "LETIFY-QUIC" in install.remote_quic_source("2.10.2")


# -- Gap 1: the transfer decision reads the session ---------------------------------


class SessionStub:
    def __init__(self, paths: list[dict[str, Any]]) -> None:
        self._paths = paths
        self.closed = threading.Event()

    def paths(self) -> list[dict[str, Any]]:
        return self._paths


def test_the_session_reports_its_smallest_round_trip_and_largest_rate() -> None:
    session = SessionStub(
        [
            {"srtt_ms": 90.0, "probe_rate": 4.0 * MIB},
            {"srtt_ms": 60.0, "probe_rate": 9.0 * MIB},
            {"srtt_ms": None, "probe_rate": None},
        ]
    )
    remote = linkchannel.LinkRemote(None, session, "lab")  # type: ignore[arg-type]
    assert remote.rtt_ms == 60.0
    assert remote.upload_bps == 9.0 * MIB
    empty = linkchannel.LinkRemote(None, SessionStub([]), "lab")  # type: ignore[arg-type]
    assert empty.rtt_ms is None and empty.upload_bps is None


def with_session(provider: Any, paths: list[dict[str, Any]]) -> Any:
    remote = linkchannel.LinkRemote(None, SessionStub(paths), "lab")  # type: ignore[arg-type]
    provider.__dict__["_link_remotes"] = {"": remote}
    return runtime_of(provider)


def runtime_of(provider: Any) -> Any:
    return SimpleNamespace(name="r1", external_id=None, provider=provider)


def test_the_transfer_decision_on_a_link_account_reads_the_session_not_an_ssh_probe(
    isolated_home, tmp_path, ssh, monkeypatch
) -> None:
    from letify.store import pathdata
    from letify.transport import pipeline

    monkeypatch.setattr(
        pipeline.Pipeline, "connect", lambda self: pytest.fail("the strategy race ran")
    )
    provider = machine(tmp_path, ssh, link="letify-link")
    runtime_ = with_session(provider, [{"srtt_ms": 150.0, "probe_rate": 20.0 * MIB}])
    assert pathdata._measured_rtt(provider, runtime_) == 150.0
    assert pathdata._streams_for(runtime_, 512 * MIB) == pathdata.STREAMS


def test_a_link_stream_costs_nothing_even_when_the_account_sets_a_cost(
    isolated_home, tmp_path, ssh
) -> None:
    from letify.store import pathdata

    # At 500 MiB/s a cost of 10 s would keep one stream, if it were charged.
    paths = [{"srtt_ms": 150.0, "probe_rate": 500.0 * MIB}]
    linked_ = machine(tmp_path, ssh, link="letify-link", transfer_connection_cost_s=10)
    assert linked_.transfer_connection_cost_s == 0.0
    assert pathdata._streams_for(with_session(linked_, paths), 512 * MIB) == pathdata.STREAMS
    costly = machine(tmp_path, ssh, transfer_connection_cost_s=10)
    assert costly.transfer_connection_cost_s == 10.0


def test_a_link_account_reports_no_measurement_before_the_session_exists(
    isolated_home, tmp_path, ssh
) -> None:
    provider = machine(tmp_path, ssh, link="letify-link")
    link = provider.link(runtime())
    assert link.rtt_ms is None and link.upload_bps is None
    assert link.spare_connections() == 0


# -- Gap 2: a lean bootstrap -------------------------------------------------------


def test_a_link_account_bootstraps_over_plain_ssh_without_the_strategy_race(
    linked, ssh, monkeypatch
) -> None:
    from letify.transport import pipeline

    monkeypatch.setattr(
        pipeline.Pipeline, "connect", lambda self: pytest.fail("the strategy race ran")
    )
    channel = linked.open_channel(runtime())
    try:
        assert channel.call(worker_pid(), (), {})[0] != os.getpid()
        assert len(ssh.commands) == 1
    finally:
        channel.close()


def no_address_machine(tmp_path: Path, fake: FakeSshLink, sidecar_binary: Path) -> LocalShell:
    provider = provider_of(
        LocalShell,
        "lab",
        tailcat="100.64.0.9",
        tailcat_port=7001,
        workspace=str(tmp_path / "ws"),
        link="letify-link",
        link_binary=str(sidecar_binary),
    )
    provider.__dict__["fake_ssh"] = fake
    return provider


def test_an_account_with_no_address_starts_its_session_from_punched_candidates_only(
    sidecar_binary, isolated_home, tmp_path, ssh, udp_stun, monkeypatch
) -> None:
    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)
    seen: list[Any] = []
    original = linkd.LinkDaemon.open_session

    def spy(self, peer, candidates, rendezvous=None, *, config=None):
        seen.append(list(candidates))
        return original(self, peer, candidates, rendezvous, config=config)

    monkeypatch.setattr(linkd.LinkDaemon, "open_session", spy)
    provider = no_address_machine(tmp_path, ssh, sidecar_binary)
    channel = provider.open_channel(runtime())
    try:
        assert channel.call(worker_pid(), (), {})[0] != os.getpid()
        assert seen and all(isinstance(c, dict) and "punched" in c for c in seen[0])
        # The pipeline's link carried the shim and nothing else: no data path used it.
        assert len(ssh.commands) == 1
        remote = provider.__dict__["_link_remotes"][""]
        assert remote.endpoint == ""
        assert remote.record()["endpoint"] == ""
    finally:
        channel.close()
        provider.close_link()


def test_a_punched_connect_announces_both_nat_types(
    sidecar_binary, isolated_home, tmp_path, ssh, udp_stun, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)
    provider = no_address_machine(tmp_path, ssh, sidecar_binary)
    channel = provider.open_channel(runtime())
    try:
        lines = capsys.readouterr().err.splitlines()
        nat = [line for line in lines if "letify-link: NAT " in line]
        assert len(nat) == 1 and " here, " in nat[0] and nat[0].endswith(" there")
        assert any(" punched rtt " in line for line in lines)
    finally:
        channel.close()
        provider.close_link()


def test_an_account_with_no_address_names_the_policed_paths_when_none_is_usable(
    sidecar_binary, isolated_home, tmp_path, ssh, udp_stun, monkeypatch
) -> None:
    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)
    # A floor no loopback path reaches makes every punched path count as policed.
    monkeypatch.setattr(linkchannel, "POLICE", {"floor_bps": 1 << 50, "burst_ms": 200})
    provider = no_address_machine(tmp_path, ssh, sidecar_binary)
    with pytest.raises(ProviderUnavailable) as raised:
        provider.open_channel(runtime())
    message = str(raised.value)
    assert "lab has no address and no punched path connected" in message
    assert "policed" in message
    provider.close_link()


def test_an_account_with_no_address_that_cannot_classify_its_nat_says_so(
    sidecar_binary, isolated_home, tmp_path, ssh, monkeypatch
) -> None:
    monkeypatch.setenv("LETIFY_LINK_STUN", "127.0.0.1:9")
    provider = no_address_machine(tmp_path, ssh, sidecar_binary)
    with pytest.raises(ProviderUnavailable, match="address observers answered"):
        provider.open_channel(runtime())
    provider.close_link()


def test_a_run_of_an_account_with_no_address_cannot_be_attached_without_a_new_punch(
    isolated_home, tmp_path, ssh
) -> None:
    provider = no_address_machine(tmp_path, ssh, Path("unused"))
    record = {"door": "/tmp/door", "link": {"endpoint": "", "node_id": "n", "local_node": "m"}}
    with pytest.raises(UnsupportedMode, match="no address"):
        provider.attach_channel("r1", record)


# -- Gap 3: the resume window is the lease -------------------------------------------


def test_the_local_session_and_the_shim_use_the_lease_as_the_resume_window(
    linked, ssh, monkeypatch
) -> None:
    from letify.runtime import lease

    seen: list[Any] = []
    original = linkd.LinkDaemon.open_session

    def spy(self, peer, candidates, rendezvous=None, *, config=None):
        seen.append(config)
        return original(self, peer, candidates, rendezvous, config=config)

    monkeypatch.setattr(linkd.LinkDaemon, "open_session", spy)
    channel = linked.open_channel(runtime())
    try:
        assert seen[0]["resume_window_s"] == int(lease.GRACE)
        shim = load_shim()
        configured = shim.Shim({"resume_s": int(lease.GRACE), "allow": "n", "bind": "x"})
        assert configured.listen_request()["config"] == {"resume_window_s": int(lease.GRACE)}
        assert int(lease.GRACE) == linkchannel.RESUME_S
    finally:
        channel.close()


def load_shim() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("linkshim_under_test", linkchannel.shim_path())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shim_leaves_when_no_session_has_a_path_for_the_resume_window() -> None:
    shim = load_shim().Shim({"resume_s": 10, "wait_s": 60})
    shim.started = time.monotonic()
    shim.handed_over = True
    shim.on_event({"event": "session_incoming", "session": 1})
    shim.on_event({"event": "session_incoming", "session": 2})
    shim.on_event({"event": "path_added", "session": 1, "name": "a"})
    shim.on_event({"event": "path_added", "session": 2, "name": "b"})
    shim.on_event({"event": "path_lost", "session": 1, "name": "a"})
    shim.last_path = time.monotonic() - 100
    # Session 2 still has a path, so the shim stays.
    shim.check_idle()
    shim.on_event({"event": "path_lost", "session": 2, "name": "b"})
    shim.last_path = time.monotonic() - 100
    with pytest.raises(SystemExit):
        shim.check_idle()


# -- Gap 5: the worker's standard error -----------------------------------------------


def test_a_worker_that_dies_before_speaking_shows_the_tail_of_its_standard_error(
    linked, ssh, monkeypatch
) -> None:
    from letify.errors import RuntimeFailure
    from letify.protocol import worker

    noisy = (
        "import sys\nfor i in range(60):\n    sys.stderr.write('row-%03d\\n' % i)\nsys.exit(3)\n"
    )
    monkeypatch.setattr(worker, "BOOTSTRAP", noisy)
    channel = linked.open_channel(runtime())
    try:
        with pytest.raises(RuntimeFailure) as raised:
            channel.start()
        text = str(raised.value)
        assert "row-059" in text and "row-020" in text
        assert "row-019" not in text
    finally:
        channel.close()


def test_a_stream_with_a_bad_first_line_is_closed() -> None:
    shim = load_shim().Shim({"resume_s": 10, "wait_s": 60, "python": sys.executable})
    lines = (
        b"not json\n",
        b'{"kind":"nope","token":"ab"}\n',
        b'{"kind":"worker","token":"../x"}\n',
    )
    for line in lines:
        near, far = socket.socketpair()
        far.settimeout(10)
        thread = threading.Thread(target=shim.serve_socket, args=(near,), daemon=True)
        thread.start()
        far.sendall(line)
        assert far.recv(10) == b""
        thread.join(10)
        far.close()


# -- The path set settles before the probe (spec "Sidecar interface" item 8) ----------------


class EventSession:
    def __init__(self) -> None:
        import queue

        self.events: queue.Queue[dict[str, Any]] = queue.Queue()


def test_the_probe_waits_until_every_candidate_has_joined() -> None:
    session = EventSession()
    session.events.put({"event": "path_added", "name": "punch0"})
    late = threading.Timer(0.3, lambda: session.events.put({"event": "path_added", "name": "p1"}))
    late.start()
    began = time.monotonic()
    names = linkchannel.settle_paths(session, ["punch0", "p1"], 5.0)  # type: ignore[arg-type]
    assert names == ["punch0", "p1"]
    assert time.monotonic() - began >= 0.25


def test_a_candidate_that_was_lost_does_not_hold_the_probe() -> None:
    session = EventSession()
    session.events.put({"event": "path_added", "name": "punch0"})
    session.events.put({"event": "path_lost", "name": "punch1", "reason": "the subflow died"})
    began = time.monotonic()
    names = linkchannel.settle_paths(session, ["punch0", "punch1"], 5.0)  # type: ignore[arg-type]
    assert names == ["punch0"]
    assert time.monotonic() - began < 1.0


def test_the_wait_for_the_path_set_ends_at_its_bound() -> None:
    session = EventSession()
    session.events.put({"event": "path_added", "name": "punch0"})
    began = time.monotonic()
    names = linkchannel.settle_paths(session, ["punch0", "punch1"], 0.3)  # type: ignore[arg-type]
    assert names == ["punch0"]
    assert 0.25 <= time.monotonic() - began < 2.0


# -- A failed punch explains itself (spec "NAT traversal" item 8, "Sidecar interface" item 8) --


class _FailingDaemon:
    """A daemon whose punch fails with a sidecar-style error, to drive ``_punch`` alone."""

    node_id = "local"

    def __init__(self, errors: list[linkd.LinkError]) -> None:
        self.errors = errors

    def punch(self, *args: Any, **kwargs: Any) -> list[str]:
        raise self.errors.pop(0)


class _QuietLines:
    def settle(self, names: Any, timeout: float) -> None:
        return None


def _punch_failure(errors: list[linkd.LinkError], rtt_s: float | None) -> tuple[str, list[str]]:
    told: list[str] = []
    with pytest.raises(ProviderUnavailable) as raised:
        linkchannel._punch(
            "lab",
            _FailingDaemon(errors),  # type: ignore[arg-type]
            "remote",
            SimpleNamespace(rtt_s=rtt_s),  # type: ignore[arg-type]
            lambda message: None,
            _QuietLines(),  # type: ignore[arg-type]
            None,
            {"observers": "default"},
            told.append,
        )
    return str(raised.value), told


def _failure(report: str, nat: str | None, peer: str | None) -> linkd.LinkError:
    error = linkd.LinkInfrastructureError(f"punch failed: tcp: no connection; {report}")
    error.detail = {"nat": nat, "peer_nat": peer, "report": report}
    return error


def test_each_failed_attempt_prints_its_nat_line_and_its_report() -> None:
    here = "here: NAT endpoint_independent, offered tcp 4 (0 predicted) udp 1"
    first = f"{here}; there: NAT symmetric:2, offered tcp 4 udp 1"
    second = f"{here}; there: no answer"
    message, told = _punch_failure(
        [
            _failure(first, "endpoint_independent", "symmetric:2"),
            _failure(second, "endpoint_independent", None),
        ],
        rtt_s=1.5,
    )
    assert told[0] == "NAT endpoint_independent here, symmetric:2 there (attempt 1)"
    assert told[1] == f"punch attempt 1: {first}"
    assert told[2] == "NAT endpoint_independent here, unknown there (attempt 2)"
    assert told[3] == f"punch attempt 2: {second}"
    assert "attempt 1 (lead " in message and "attempt 2 (lead " in message
    assert message.count("here: NAT endpoint_independent") == 2


def test_a_punch_with_no_classification_prints_a_nat_line_saying_why() -> None:
    reason = "cannot classify the NAT: 2 of 3 address observers answered"
    error = linkd.LinkInfrastructureError(reason)
    error.detail = {
        "nat": None,
        "peer_nat": None,
        "nat_error": reason,
        "report": (
            "here: NAT unknown (2 of 3 address observers answered), nothing offered; "
            "there: no answer"
        ),
    }
    _message, told = _punch_failure([error, error], rtt_s=1.0)
    assert told[0].startswith("NAT unknown here (cannot classify the NAT: 2 of 3 address observers")
    assert told[0].endswith(", unknown there (attempt 1)")


def test_an_error_with_no_detail_still_prints_a_nat_line() -> None:
    error = linkd.LinkInfrastructureError("the control channel failed")
    _message, told = _punch_failure([error], rtt_s=None)
    assert told == ["NAT unknown here (the control channel failed), unknown there"]


def test_a_successful_punch_prints_the_report_after_the_nat_line() -> None:
    report = (
        "here: NAT endpoint_independent, offered tcp 4 (0 predicted) udp 1; "
        "there: NAT symmetric:2, offered tcp 4 udp 1"
    )
    daemon = SimpleNamespace(
        last_punch={
            "nat": "endpoint_independent",
            "peer_nat": "symmetric:2",
            "report": report,
        }
    )
    assert linkchannel.nat_lines(daemon) == [  # type: ignore[arg-type]
        "NAT endpoint_independent here, symmetric:2 there",
        f"punch {report}",
    ]


def test_an_account_that_cannot_classify_its_nat_prints_the_nat_line_before_failing(
    sidecar_binary, isolated_home, tmp_path, ssh, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("LETIFY_LINK_STUN", "127.0.0.1:9")
    provider = no_address_machine(tmp_path, ssh, sidecar_binary)
    with pytest.raises(ProviderUnavailable, match="address observers answered"):
        provider.open_channel(runtime())
    provider.close_link()
    nat = [line for line in capsys.readouterr().err.splitlines() if "letify-link: NAT " in line]
    assert nat and nat[0].split("letify-link: ")[1].startswith("NAT unknown here (cannot classify")


def test_the_session_probe_ends_early_at_twice_the_account_floor() -> None:
    """Spec "Choosing a link": the early stop is relative to the floor the account holds."""
    probe = REAL_MAKE_PROBE(10.0 * MIB)
    assert probe.early_bps == 20.0 * MIB
    assert probe.min_round_trips == 10
    assert (probe.round_trips, probe.seconds) == (30, 2.0)


def test_a_session_probe_with_no_floor_never_ends_a_direction_early() -> None:
    for floor in (None, 0.0):
        probe = REAL_MAKE_PROBE(floor)
        assert probe.early_bps is None
        assert probe.min_round_trips == 10
