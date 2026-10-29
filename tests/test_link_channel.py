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


class LocalShell(Shell):
    """A shell account whose interpreter on the machine is the one running the tests."""

    remote_python = sys.executable

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        """The account's plain ssh command, run on this machine under ``sh -c``."""
        return self.__dict__["fake_ssh"].ssh_command(remote_command)


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


def test_a_path_below_the_link_floor_fails_the_connect_naming_the_settings() -> None:
    probe = {"event": "path_probe", "throughput": 1.0 * MIB, "rtt_ms": 5.0}

    class Events:
        def __init__(self) -> None:
            self.items = [probe]

        def empty(self) -> bool:
            return not self.items

        def get_nowait(self) -> dict[str, Any]:
            return self.items.pop()

    closed: list[bool] = []
    remote = SimpleNamespace(
        alias="lab",
        rtt_ms=400.0,
        session=SimpleNamespace(events=Events()),
        close=lambda: closed.append(True),
    )
    with pytest.raises(ProviderUnavailable) as raised:
        linkchannel._hold_to_floor(remote, 300.0, 10.0 * MIB)  # type: ignore[arg-type]
    text = str(raised.value)
    assert "round trip 400.0 ms above the floor 300 ms" in text
    assert "1.0 MiB/s below the floor 10.0 MiB/s" in text
    assert "min_mib_per_s" in text and "max_rtt_ms" in text
    assert closed == [True]


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


def test_a_link_account_with_no_address_names_the_missing_rendezvous(
    isolated_home, tmp_path, ssh
) -> None:
    provider = provider_of(
        LocalShell, "lab", tailcat="100.64.0.9", tailcat_port=7001, link="letify-link"
    )
    provider.__dict__["fake_ssh"] = ssh
    with pytest.raises(ProviderUnavailable) as raised:
        provider.open_channel(runtime())
    assert "lab" in str(raised.value)
    assert "letify-link needs an address until the NAT rendezvous lands" in str(raised.value)
    assert ssh.commands == []


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
