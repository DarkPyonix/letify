"""letify-link over the command channels of Colab and Kaggle, with fake runtimes.

Spec "Sidecar interface" item 8 (Bootstrap over a command channel). The runtime is this
machine: the command channel (``colab exec``, the Kaggle bridge's ``eval``) is replaced by a
function that runs each program locally after an artificial delay, which is the only part not
exercised. The mailbox, the pump, the shim, both sidecars, the punch, the session and the worker
are real. The tests need the built sidecar like tests/test_link_channel.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import provider_of

from letify.errors import ProviderUnavailable, RuntimeFailure
from letify.providers import colab as colab_module
from letify.providers.colab import Colab
from letify.providers.kaggle import Kaggle
from letify.transport import linkchannel, linkd, linkmailbox


@pytest.fixture(autouse=True)
def local_sidecar(sidecar_binary, monkeypatch, udp_stun):
    """Every sidecar is the built one, and every NAT classification asks the local servers."""
    monkeypatch.setenv(linkd.ENV_BINARY, str(sidecar_binary))
    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)


class FakeCommands:
    """Runs a Python program on this machine after ``latency`` seconds, as a command channel.

    Half of the delay comes before the program runs and half after, so a call takes
    ``latency`` seconds in all. The wheel extraction is answered with ``binary`` instead of a
    download.
    """

    def __init__(self, latency: float, binary: Path) -> None:
        self.latency = latency
        self.binary = binary
        self.sources: list[str] = []
        self.lock = threading.Lock()

    def run(self, source: str, timeout: float | None = None) -> str:
        with self.lock:
            self.sources.append(source)
        time.sleep(self.latency / 2)
        if "LETIFY-LINK" in source and "pip" in source:
            out = "LETIFY-LINK " + json.dumps(str(self.binary)) + "\n"
        else:
            done = subprocess.run(
                [sys.executable, "-"],
                input=source,
                capture_output=True,
                text=True,
                timeout=timeout or 120,
            )
            out = done.stdout + done.stderr
        time.sleep(self.latency / 2)
        return out

    @property
    def extractions(self) -> int:
        return sum(1 for s in self.sources if "LETIFY-LINK" in s and "pip" in s)


class FakeBridge:
    """The Kaggle kernel bridge: ``eval`` runs the source and returns ``__letify_value__``."""

    def __init__(self, commands: FakeCommands, library_path: str = "") -> None:
        self.commands = commands
        self.library_path = library_path

    def request(self, message: dict[str, Any], **kwargs: Any) -> tuple[Any, Any]:
        assert message["op"] == "eval"
        source = message["source"]
        if "LD_LIBRARY_PATH" in source and "environ" in source:
            return (self.library_path, None)
        scope: dict[str, Any] = {}
        # Reuse the command fake for its delay and the wheel answer, in this process.
        with self.commands.lock:
            self.commands.sources.append(source)
        time.sleep(self.commands.latency / 2)
        if "LETIFY-LINK" in source and "pip" in source:
            value = "LETIFY-LINK " + json.dumps(str(self.commands.binary)) + "\n"
        else:
            exec(compile(source, "<kernel>", "exec"), scope)
            value = scope.get("__letify_value__")
        time.sleep(self.commands.latency / 2)
        return (value, None)


def runtime(name: str = "r1") -> Any:
    return SimpleNamespace(name=name, external_id=None)


# The bodies are built by factories so cloudpickle ships them by value.


def worker_facts():
    def body():
        import os

        return {"pid": os.getpid(), "ld": os.environ.get("LD_LIBRARY_PATH", "")}

    return body


def colab(tmp_path: Path, commands: FakeCommands, monkeypatch, **options: Any) -> Colab:
    provider = provider_of(
        Colab, "colab_a", link="letify-link", workspace=str(tmp_path / "ws"), **options
    )
    monkeypatch.setattr(
        Colab, "_exec", lambda self, session, source, timeout: commands.run(source, timeout)
    )
    monkeypatch.setattr(Colab, "remote_python", property(lambda self: sys.executable))
    return provider


def kaggle(tmp_path: Path, **options: Any) -> Kaggle:
    provider = provider_of(
        Kaggle, "kaggle_a", link="letify-link", workspace=str(tmp_path / "ws"), **options
    )
    return provider


def spy_open_session(monkeypatch) -> list[Any]:
    seen: list[Any] = []
    original = linkd.LinkDaemon.open_session

    def spy(self, peer, candidates, rendezvous=None, *, config=None):
        seen.append(list(candidates))
        return original(self, peer, candidates, rendezvous, config=config)

    monkeypatch.setattr(linkd.LinkDaemon, "open_session", spy)
    return seen


def spy_punch(monkeypatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    original = linkd.LinkDaemon.punch

    def spy(self, peer, role, **kwargs):
        seen.append(dict(kwargs))
        return original(self, peer, role, **kwargs)

    monkeypatch.setattr(linkd.LinkDaemon, "punch", spy)
    return seen


def test_a_call_runs_over_a_link_session_for_a_fake_colab_runtime(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    commands = FakeCommands(1.0, sidecar_binary)
    provider = colab(tmp_path, commands, monkeypatch)
    sessions = spy_open_session(monkeypatch)
    punches = spy_punch(monkeypatch)
    channel = provider.open_channel(runtime())
    try:
        facts = channel.call(worker_facts(), (), {})[0]
        assert facts["pid"] != os.getpid()
        # The session started from punched candidates only: the runtime has no inbound TCP.
        assert sessions and all(isinstance(c, dict) and "punched" in c for c in sessions[0])
        # The binary was taken out of the wheel through the command channel, once.
        assert commands.extractions == 1
        remote = provider.__dict__["_link_remotes"]["r1"]
        assert remote.endpoint == ""
        # Spec item 8: lead = max(2 s, 3 x the measured round trip). A 1.0 s channel gives
        # a lead near 3 s, never the sidecar's default of 2 s.
        assert 2900 <= punches[0]["lead_ms"] <= 4500
        assert punches[0]["window_ms"] == linkchannel.PUNCH_WINDOW_MS
        # The link of the runtime reports the session, and carries no ssh command.
        link = provider.link(runtime())
        assert link.rtt_ms is None or link.rtt_ms >= 0
        with pytest.raises(RuntimeFailure):
            link.ssh_command("true")
    finally:
        channel.close()
        provider.close_link(runtime())


def test_the_colab_driver_library_wrapper_applies_to_the_stream_worker(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    commands = FakeCommands(0.3, sidecar_binary)
    provider = colab(tmp_path, commands, monkeypatch, link_binary=str(sidecar_binary))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    channel = provider.open_channel(runtime())
    try:
        facts = channel.call(worker_facts(), (), {})[0]
        assert facts["ld"].startswith(colab_module.DRIVER_LIBRARY_PATH)
        assert commands.extractions == 0
    finally:
        channel.close()
        provider.close_link(runtime())


def test_a_call_runs_over_a_link_session_for_a_fake_kaggle_runtime(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    commands = FakeCommands(0.5, sidecar_binary)
    bridge = FakeBridge(commands, library_path="/kaggle-libs")
    provider = kaggle(tmp_path, link_binary=str(sidecar_binary))
    sessions = spy_open_session(monkeypatch)
    monkeypatch.setattr(Kaggle, "remote_python", property(lambda self: sys.executable))
    channel = provider.channel_over(bridge, name="r1")
    try:
        assert type(channel).__name__ == "LinkChannel"
        facts = channel.call(worker_facts(), (), {})[0]
        assert facts["pid"] != os.getpid()
        # The kernel's library path reaches the stream worker, as it reaches a worker over ssh.
        assert facts["ld"].startswith("/kaggle-libs")
        assert sessions and all(isinstance(c, dict) and "punched" in c for c in sessions[0])
        assert provider.link(runtime()) is not None
        # A transfer stream is one more stream of the same session: no second punch.
        assert type(provider.transfer_channel(runtime())).__name__ == "LinkChannel"
    finally:
        channel.close()
        provider.__dict__["_link_remotes"]["r1"].close()


def test_the_punch_lead_scales_with_the_measured_bootstrap_round_trip(
    sidecar_binary, isolated_home
) -> None:
    assert linkchannel.punch_lead_ms(0.1) == 2000
    assert linkchannel.punch_lead_ms(0.6) == 2000
    assert linkchannel.punch_lead_ms(1.0) == 3000
    assert linkchannel.punch_lead_ms(4.2) == 12600
    measured = {}
    for latency in (0.1, 1.2):
        commands = FakeCommands(latency, sidecar_binary)
        bootstrap = linkmailbox.MailboxBootstrap(commands.run, python=sys.executable)
        bootstrap.start()
        try:
            measured[latency] = bootstrap.rtt_s
        finally:
            bootstrap.close()
    assert measured[0.1] is not None and measured[0.1] < 0.6
    assert 1.2 <= measured[1.2] < 2.5
    assert linkchannel.punch_lead_ms(measured[0.1]) == 2000
    assert linkchannel.punch_lead_ms(measured[1.2]) >= 3600


def test_a_missing_binary_on_the_runtime_fails_the_connect_naming_it(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    missing = tmp_path / "nowhere" / "letify-link"
    commands = FakeCommands(0.2, sidecar_binary)
    provider = colab(tmp_path, commands, monkeypatch, link_binary=str(missing))
    with pytest.raises(ProviderUnavailable) as raised:
        provider.open_channel(runtime())
    assert f"letify-link is not at {missing}" in str(raised.value)
    assert "colab_a" in str(raised.value)
    provider.close_link(runtime())


def test_a_punch_with_no_path_retries_once_then_fails_naming_both_attempts(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    # A floor no loopback path reaches makes every punched path count as policed.
    monkeypatch.setattr(linkchannel, "POLICE", {"floor_bps": 1 << 50, "burst_ms": 200})
    commands = FakeCommands(0.3, sidecar_binary)
    provider = colab(tmp_path, commands, monkeypatch, link_binary=str(sidecar_binary))
    punches = spy_punch(monkeypatch)
    sent: list[str] = []
    original = linkmailbox.MailboxBootstrap.send_line

    def record(self, line):
        sent.append(line if isinstance(line, str) else line.decode())
        return original(self, line)

    monkeypatch.setattr(linkmailbox.MailboxBootstrap, "send_line", record)
    with pytest.raises(ProviderUnavailable) as raised:
        provider.open_channel(runtime())
    message = str(raised.value)
    assert "colab_a has no address and no punched path connected" in message
    assert "attempt 1" in message and "attempt 2" in message
    assert "attempt 3" not in message
    assert "policed" in message
    # Two attempts, the second with the window 3 times as long, and the shim was told twice.
    assert len(punches) == 2
    assert punches[1]["window_ms"] == 3 * punches[0]["window_ms"]
    assert sum(1 for line in sent if '"op": "punch"' in line) == 2
    provider.close_link(runtime())
