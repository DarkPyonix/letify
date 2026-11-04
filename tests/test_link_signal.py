"""A letify-link bootstrap carried over the signalling service's mailbox.

Spec "Signalling service" and "Sidecar interface" item 9. The service is the in-process fake of
tests/conftest.py, which applies the same contract and verifies signatures with a real sidecar.
The agent, the shim, both sidecars, the punch and the session are real; the "remote" is this
machine. The tests build the sidecar like tests/test_linkd.py and are skipped, with the reason,
without cargo.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

import letify
from letify.protocol.worker import BOOTSTRAP
from letify.runtime.channel import relay_program
from letify.transport import linkchannel, linkd, linksignal

# -- helpers ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def local_sidecar(sidecar_binary, monkeypatch):
    monkeypatch.setenv(linkd.ENV_BINARY, str(sidecar_binary))


@pytest.fixture
def daemons(sidecar_binary, tmp_path):
    made: list[linkd.LinkDaemon] = []

    def make(name: str) -> linkd.LinkDaemon:
        daemon = linkd.LinkDaemon(tmp_path / f"{name}.key", binary=sidecar_binary)
        made.append(daemon)
        return daemon

    yield make
    for daemon in made:
        daemon.close()


def worker_pid():
    def body():
        import os

        return os.getpid()

    return body


def drain(client: linksignal.SignalClient, count: int, timeout: float = 10.0):
    """Take ``count`` messages from the client's mailbox, polling until they are there."""
    got: list[linksignal.Incoming] = []
    after = 0
    deadline = time.monotonic() + timeout
    while len(got) < count and time.monotonic() < deadline:
        messages, _ = client.recv(after, wait_s=0.2)
        for message in messages:
            after = max(after, message.seq)
            got.append(message)
    return got


# -- the client ---------------------------------------------------------------------------


def test_a_signed_message_reaches_the_node_it_is_addressed_to(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, signal_service.url)
    receiver = linksignal.SignalClient(b, signal_service.url)
    assert sender.send(b.node_id, b"hello") == 1
    (message,) = drain(receiver, 1)
    assert (message.sender, message.body, message.seq) == (a.node_id, b"hello", 1)


def test_a_receive_with_an_after_cursor_deletes_what_it_acknowledges(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, signal_service.url)
    receiver = linksignal.SignalClient(b, signal_service.url)
    sender.send(b.node_id, b"one")
    sender.send(b.node_id, b"two")
    first, cursor = receiver.recv(0, wait_s=0)
    assert [m.body for m in first] == [b"one", b"two"] and cursor == 2
    rest, _ = receiver.recv(1, wait_s=0)
    assert [m.body for m in rest] == [b"two"]
    assert signal_service.waiting(b.node_id) == 1


def test_a_long_poll_returns_as_soon_as_a_message_arrives(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, signal_service.url)
    receiver = linksignal.SignalClient(b, signal_service.url)
    threading.Timer(0.3, lambda: sender.send(b.node_id, b"late")).start()
    began = time.monotonic()
    messages, _ = receiver.recv(0, wait_s=10)
    assert [m.body for m in messages] == [b"late"]
    assert time.monotonic() - began < 5


def test_a_message_over_4096_bytes_is_refused_before_it_is_sent(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    client = linksignal.SignalClient(a, signal_service.url)
    with pytest.raises(linksignal.SignalError, match="4096"):
        client.send(b.node_id, b"x" * 4097)
    assert signal_service.requests == []
    assert client.send(b.node_id, b"x" * 4096) == 1


def test_the_service_refuses_an_oversize_body_that_skips_the_client_check(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    client = linksignal.SignalClient(a, signal_service.url)
    with pytest.raises(linksignal.SignalError, match="too_large"):
        client.post_unchecked(b.node_id, b"x" * 5000)


def test_the_fake_service_has_no_versioned_paths(signal_service, daemons) -> None:
    b = daemons("b")
    for path in ("/v1/health", f"/v1/send/{b.node_id}"):
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(f"{signal_service.url}{path}", timeout=5)
        assert raised.value.code == 404


def test_an_unsigned_request_is_refused_by_the_service(signal_service, daemons) -> None:
    b = daemons("b")
    request = urllib.request.Request(
        f"{signal_service.url}/send/{b.node_id}", data=b"x", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=5)
    assert raised.value.code == 400
    assert signal_service.waiting(b.node_id) == 0


def test_a_request_signed_with_the_wrong_key_is_refused_by_the_service(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    client = linksignal.SignalClient(a, signal_service.url)
    # The headers name b as the signer, but a made the signature.
    with pytest.raises(linksignal.SignalError, match="bad_signature"):
        client.post_unchecked(b.node_id, b"x", signer_id=b.node_id)
    assert signal_service.waiting(b.node_id) == 0


def test_a_stale_request_is_refused_by_the_service(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    skewed = linksignal.SignalClient(a, signal_service.url, clock=lambda: time.time() - 120)
    with pytest.raises(linksignal.SignalError, match="stale"):
        skewed.send(b.node_id, b"x")
    near = linksignal.SignalClient(a, signal_service.url, clock=lambda: time.time() - 30)
    assert near.send(b.node_id, b"x") == 1


def test_a_replayed_nonce_is_refused_by_the_service(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    client = linksignal.SignalClient(a, signal_service.url, nonce=lambda: "ab" * 16)
    assert client.send(b.node_id, b"x") == 1
    with pytest.raises(linksignal.SignalError, match="replay"):
        client.send(b.node_id, b"x")
    assert signal_service.waiting(b.node_id) == 1


def test_nobody_but_the_node_can_read_its_mailbox(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    linksignal.SignalClient(a, signal_service.url).send(b.node_id, b"private")
    intruder = linksignal.SignalClient(a, signal_service.url)
    with pytest.raises(linksignal.SignalError, match="forbidden"):
        intruder.recv_from(b.node_id, 0, wait_s=0)


def test_a_service_that_is_down_fails_with_the_reason(daemons) -> None:
    a, b = daemons("a"), daemons("b")
    client = linksignal.SignalClient(a, "http://127.0.0.1:9", timeout=2)
    with pytest.raises(linksignal.SignalError, match="the signalling service"):
        client.send(b.node_id, b"x")


# -- the envelope -------------------------------------------------------------------------


def seal_for(daemon, to: str, sid: str, n: int, line: str, t: int | None = None) -> bytes:
    return linksignal.seal(daemon, to, sid, n, line, t=t)


def test_an_envelope_opens_for_its_receiver_and_for_nobody_else(daemons) -> None:
    a, b, c = daemons("a"), daemons("b"), daemons("c")
    sid = "ab" * 16
    body = seal_for(a, b.node_id, sid, 1, '{"op":"ping"}')
    opened = linksignal.open_envelope(b, body, peer=a.node_id, sid=sid, last_n=0)
    assert opened.line == '{"op":"ping"}' and opened.n == 1
    with pytest.raises(linksignal.Refused, match="addressed to another node"):
        linksignal.open_envelope(c, body, peer=a.node_id, sid=sid, last_n=0)
    with pytest.raises(linksignal.Refused, match="not from the expected node"):
        linksignal.open_envelope(b, body, peer=c.node_id, sid=sid, last_n=0)


def test_an_envelope_with_a_changed_line_or_a_bad_signature_is_refused(daemons) -> None:
    a, b = daemons("a"), daemons("b")
    sid = "ab" * 16
    body = json.loads(seal_for(a, b.node_id, sid, 1, '{"op":"ping"}'))
    body["line"] = '{"op":"handover"}'
    with pytest.raises(linksignal.Refused, match="signature"):
        linksignal.open_envelope(b, json.dumps(body).encode(), peer=a.node_id, sid=sid, last_n=0)
    unsigned = dict(body, line='{"op":"ping"}')
    del unsigned["sig"]
    with pytest.raises(linksignal.Refused, match="fields"):
        linksignal.open_envelope(
            b, json.dumps(unsigned).encode(), peer=a.node_id, sid=sid, last_n=0
        )
    with pytest.raises(linksignal.Refused, match="JSON"):
        linksignal.open_envelope(b, b"not json", peer=a.node_id, sid=sid, last_n=0)


def test_an_envelope_signed_by_another_node_than_it_names_is_refused(daemons) -> None:
    a, b, c = daemons("a"), daemons("b"), daemons("c")
    sid = "ab" * 16
    # c signs an envelope that says it is from a.
    forged = json.loads(seal_for(c, b.node_id, sid, 1, "{}"))
    forged["from"] = a.node_id
    with pytest.raises(linksignal.Refused, match="signature"):
        linksignal.open_envelope(b, json.dumps(forged).encode(), peer=a.node_id, sid=sid, last_n=0)


def test_a_stale_or_replayed_or_foreign_session_envelope_is_refused(daemons) -> None:
    a, b = daemons("a"), daemons("b")
    sid = "ab" * 16
    old = seal_for(a, b.node_id, sid, 1, "{}", t=int(time.time()) - 120)
    with pytest.raises(linksignal.Refused, match="stale"):
        linksignal.open_envelope(b, old, peer=a.node_id, sid=sid, last_n=0)
    fresh = seal_for(a, b.node_id, sid, 3, "{}")
    assert linksignal.open_envelope(b, fresh, peer=a.node_id, sid=sid, last_n=2).n == 3
    with pytest.raises(linksignal.Refused, match="replayed"):
        linksignal.open_envelope(b, fresh, peer=a.node_id, sid=sid, last_n=3)
    with pytest.raises(linksignal.Refused, match="another bootstrap"):
        linksignal.open_envelope(b, fresh, peer=a.node_id, sid="cd" * 16, last_n=0)


def test_a_line_that_does_not_fit_in_one_message_is_an_error_and_is_not_split(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    client = linksignal.SignalClient(a, signal_service.url)
    bootstrap = linksignal.SignalBootstrap(client, b.node_id)
    with pytest.raises(OSError, match="4096"):
        bootstrap.send_line(json.dumps({"op": "rendezvous", "message": "ab" * 2200}).encode())
    assert signal_service.requests == []


def test_the_config_line_is_reduced_to_a_version_and_two_timeouts() -> None:
    full = {
        "binary": "/x/letify-link",
        "allow": "ab" * 32,
        "bind": "0.0.0.0:0",
        "python": "python3",
        "bootstrap": "x" * 20000,
        "relay": "y" * 20000,
        "worker_command": "wrap python3",
        "relay_command": "wrap python3",
        "wait_s": 120,
        "resume_s": 30,
    }
    lean = linksignal.lean_config(full, version="9.9.9")
    assert lean == {"op": "config", "version": "9.9.9", "wait_s": 120, "resume_s": 30}


# -- the agent ----------------------------------------------------------------------------


def make_agent(client, paired, sidecar_binary, **over: Any) -> linksignal.SignalAgent:
    options: dict[str, Any] = {
        "binary": str(sidecar_binary),
        "python": sys.executable,
        "bootstrap": BOOTSTRAP,
        "relay": relay_program(),
    }
    options.update(over)
    return linksignal.SignalAgent(client, paired=paired, **options)


def test_the_agent_builds_the_shim_config_from_its_own_configuration_only(
    signal_service, daemons, sidecar_binary
) -> None:
    a, b = daemons("a"), daemons("b")
    agent = make_agent(
        linksignal.SignalClient(b, signal_service.url),
        [a.node_id],
        sidecar_binary,
        worker_command="own-wrapper python",
    )
    sent = {
        "op": "config",
        "version": letify.__version__,
        "wait_s": 10,
        "resume_s": 20,
        "binary": "/tmp/evil",
        "python": "/tmp/evil-python",
        "bootstrap": "import os; os.system('touch /tmp/pwned')",
        "relay": "import os",
        "worker_command": "touch /tmp/pwned",
        "relay_command": "touch /tmp/pwned",
        "allow": "00" * 32,
        "bind": "6.6.6.6:1",
    }
    config = agent.shim_config(a.node_id, sent)
    assert config["binary"] == str(sidecar_binary)
    assert config["python"] == sys.executable
    assert config["bootstrap"] == BOOTSTRAP and config["relay"] == relay_program()
    assert config["worker_command"] == "own-wrapper python"
    assert "relay_command" not in config
    assert config["allow"] == a.node_id
    assert config["bind"] == "0.0.0.0:0"
    assert (config["wait_s"], config["resume_s"]) == (10, 20)
    # The timeouts are clamped to the shim's own bounds.
    huge = agent.shim_config(a.node_id, dict(sent, wait_s=10**9, resume_s=-5))
    assert huge["wait_s"] <= linksignal.MAX_WAIT_S and huge["resume_s"] >= 1


def test_the_agent_passes_only_the_ops_a_bootstrap_needs_and_pins_the_peer(
    signal_service, daemons, sidecar_binary
) -> None:
    a, b = daemons("a"), daemons("b")
    agent = make_agent(linksignal.SignalClient(b, signal_service.url), [a.node_id], sidecar_binary)
    for op in ("exec", "shell", "config", "stdin", ""):
        assert agent.shim_command(a.node_id, {"op": op}) is None
    punch = agent.shim_command(
        a.node_id, {"op": "punch", "peer": "00" * 32, "sockets": 4, "command": "touch /tmp/x"}
    )
    assert punch == {"op": "punch", "peer": a.node_id, "sockets": 4}
    rendezvous = agent.shim_command(
        a.node_id, {"op": "rendezvous", "from": "00" * 32, "message": "ab"}
    )
    assert rendezvous == {"op": "rendezvous", "from": a.node_id, "message": "ab"}
    assert agent.shim_command(a.node_id, {"op": "handover"}) == {"op": "handover"}


def test_a_crafted_config_naming_commands_starts_the_agents_shim_and_runs_nothing_else(
    signal_service, daemons, sidecar_binary, tmp_path
) -> None:
    a, b = daemons("a"), daemons("b")
    marker = tmp_path / "pwned"
    agent = make_agent(linksignal.SignalClient(b, signal_service.url), [a.node_id], sidecar_binary)
    thread = threading.Thread(target=agent.serve, kwargs={"timeout": 30}, daemon=True)
    thread.start()
    client = linksignal.SignalClient(a, signal_service.url)
    sid = "ab" * 16
    crafted = {
        "op": "config",
        "version": letify.__version__,
        "wait_s": 5,
        "resume_s": 5,
        "binary": "/nonexistent/letify-link",
        "python": f"touch {marker} #",
        "worker_command": f"touch {marker}",
        "relay_command": f"touch {marker}",
        "bootstrap": f"open({str(marker)!r}, 'w')",
    }
    client.send(b.node_id, linksignal.seal(a, b.node_id, sid, 1, json.dumps(crafted)))
    ready = None
    after = 0
    deadline = time.monotonic() + 20
    while ready is None and time.monotonic() < deadline:
        messages, _ = client.recv(after, wait_s=0.5)
        for message in messages:
            after = max(after, message.seq)
            opened = linksignal.open_envelope(a, message.body, peer=b.node_id, sid=sid, last_n=0)
            event = json.loads(opened.line)
            if event.get("event") == "ready":
                ready = event
            assert event.get("event") != "error", event
    assert ready is not None, "the agent's shim never became ready"
    assert ready["version"] == letify.__version__
    client.send(b.node_id, linksignal.seal(a, b.node_id, sid, 2, '{"op":"handover"}'))
    agent.stop()
    thread.join(timeout=15)
    assert not marker.exists()


def test_a_message_from_a_node_that_is_not_paired_starts_nothing(
    signal_service, daemons, sidecar_binary
) -> None:
    a, b, stranger = daemons("a"), daemons("b"), daemons("stranger")
    agent = make_agent(linksignal.SignalClient(b, signal_service.url), [a.node_id], sidecar_binary)
    thread = threading.Thread(target=agent.serve, kwargs={"timeout": 3}, daemon=True)
    thread.start()
    client = linksignal.SignalClient(stranger, signal_service.url)
    config = json.dumps({"op": "config", "version": letify.__version__, "wait_s": 5, "resume_s": 5})
    client.send(b.node_id, linksignal.seal(stranger, b.node_id, "ab" * 16, 1, config))
    thread.join(timeout=10)
    assert agent.started == 0
    assert agent.dropped and "not paired" in agent.dropped[0]
    # The agent never answers a stranger.
    assert signal_service.waiting(stranger.node_id) == 0


# -- connect_over -------------------------------------------------------------------------


def start_agent(signal_service, daemons, sidecar_binary, local_id: str, **over: Any):
    agent_daemon = daemons("agent")
    agent = make_agent(
        linksignal.SignalClient(agent_daemon, signal_service.url),
        [local_id],
        sidecar_binary,
        **over,
    )
    thread = threading.Thread(target=agent.serve, kwargs={"timeout": 120}, daemon=True)
    thread.start()
    return agent_daemon, agent, thread


def connect_through_signal(signal_service, daemon, agent_node, tmp_path, **over: Any):
    client = linksignal.SignalClient(daemon, signal_service.url)
    return linkchannel.connect_over(
        "lab",
        prepare=lambda: "unused",
        start=lambda: linkchannel._started(linksignal.SignalBootstrap(client, agent_node, **over)),
        address=None,
        python=sys.executable,
        bootstrap=BOOTSTRAP,
        key_path=tmp_path / "local.key",
        punch=True,
        channel="the signalling service",
    )


def test_an_address_less_machine_is_reached_through_the_mailbox_with_no_ssh_and_no_tailcat(
    signal_service, daemons, sidecar_binary, tmp_path, udp_stun, monkeypatch
) -> None:
    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)
    local = linkd.LinkDaemon(tmp_path / "local.key", binary=sidecar_binary)
    try:
        agent_daemon, agent, thread = start_agent(
            signal_service, daemons, sidecar_binary, local.node_id
        )
        seen: list[Any] = []
        original = linkd.LinkDaemon.open_session

        def spy(self, peer, candidates, rendezvous=None, *, config=None):
            seen.append(list(candidates))
            return original(self, peer, candidates, rendezvous, config=config)

        monkeypatch.setattr(linkd.LinkDaemon, "open_session", spy)
        remote = connect_through_signal(signal_service, local, agent_daemon.node_id, tmp_path)
        channel = linkchannel.LinkChannel(remote, name="r1")
        try:
            assert channel.call(worker_pid(), (), {})[0] != os.getpid()
            # No address and no ssh: the session started from punched candidates only.
            assert seen and all(isinstance(c, dict) and "punched" in c for c in seen[0])
            assert remote.endpoint == ""
            # Every message the service carried fitted in the cap, and none embedded a source.
            assert signal_service.sizes and max(signal_service.sizes) <= 4096
            assert {method for method, _, _ in signal_service.requests} <= {"GET", "POST"}
        finally:
            channel.close()
            agent.stop()
            thread.join(timeout=15)
    finally:
        local.close()


def test_a_version_mismatch_is_refused_with_its_reason(
    signal_service, daemons, sidecar_binary, tmp_path, udp_stun, monkeypatch
) -> None:
    from letify.errors import ProviderUnavailable

    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)
    local = linkd.LinkDaemon(tmp_path / "local.key", binary=sidecar_binary)
    try:
        agent_daemon, agent, thread = start_agent(
            signal_service, daemons, sidecar_binary, local.node_id, version="0.0.1"
        )
        with pytest.raises(ProviderUnavailable, match="letify version mismatch"):
            connect_through_signal(signal_service, local, agent_daemon.node_id, tmp_path)
        agent.stop()
        thread.join(timeout=15)
    finally:
        local.close()


def test_an_agent_that_does_not_answer_fails_the_connect_with_the_reason(
    signal_service, daemons, tmp_path, sidecar_binary
) -> None:
    from letify.errors import ProviderUnavailable

    local = linkd.LinkDaemon(tmp_path / "local.key", binary=sidecar_binary)
    silent = daemons("silent")
    try:
        with pytest.raises(ProviderUnavailable, match="did not answer"):
            connect_through_signal(
                signal_service, local, silent.node_id, tmp_path, ping_timeout_s=1.0
            )
    finally:
        local.close()


def test_a_service_that_changes_a_line_cannot_make_the_bootstrap_accept_it(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, signal_service.url)
    sid = "ab" * 16
    bootstrap = linksignal.SignalBootstrap(
        linksignal.SignalClient(b, signal_service.url), a.node_id, sid=sid
    )

    # The service rewrites the line inside the first envelope it stores.
    def rewrite(body: bytes) -> bytes:
        envelope = json.loads(body)
        envelope["line"] = '{"event":"ready","node_id":"' + "00" * 32 + '","port":1}'
        return json.dumps(envelope).encode()

    signal_service.rewrite = rewrite
    sender.send(b.node_id, seal_for(a, b.node_id, sid, 1, '{"event":"ready","node_id":"x"}'))
    signal_service.rewrite = None
    sender.send(b.node_id, seal_for(a, b.node_id, sid, 2, '{"event":"punched"}'))
    bootstrap.listen_only()
    assert json.loads(bootstrap.recv_line() or b"{}") == {"event": "punched"}
    assert "signature" in bootstrap.stderr_text()
    bootstrap.close()


def test_a_stranger_cannot_inject_a_line_into_a_running_bootstrap(signal_service, daemons) -> None:
    a, b, stranger = daemons("a"), daemons("b"), daemons("stranger")
    sid = "ab" * 16
    bootstrap = linksignal.SignalBootstrap(
        linksignal.SignalClient(b, signal_service.url), a.node_id, sid=sid
    )
    linksignal.SignalClient(stranger, signal_service.url).send(
        b.node_id, seal_for(stranger, b.node_id, sid, 1, '{"event":"ready","node_id":"x"}')
    )
    linksignal.SignalClient(a, signal_service.url).send(
        b.node_id, seal_for(a, b.node_id, sid, 1, '{"event":"punched"}')
    )
    bootstrap.listen_only()
    assert json.loads(bootstrap.recv_line() or b"{}") == {"event": "punched"}
    assert "not from the expected node" in bootstrap.stderr_text()
    bootstrap.close()


# -- the live service (skipped unless the owner points the suite at it) -----------------------


@pytest.mark.skipif(
    not os.environ.get("LETIFY_SIGNAL_URL"), reason="LETIFY_SIGNAL_URL names no live service"
)
def test_live_a_signed_round_trip_through_the_deployed_service(daemons) -> None:
    url = os.environ["LETIFY_SIGNAL_URL"]
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, url)
    receiver = linksignal.SignalClient(b, url)
    sender.send(b.node_id, b"live check")
    (message,) = drain(receiver, 1)
    assert message.sender == a.node_id and message.body == b"live check"
    receiver.recv(message.seq, wait_s=0)


# -- the standing wait over the WebSocket -------------------------------------------------
# Spec "Signalling service" item 1, "Which wait uses which endpoint".


def read_until(watch, count: int, timeout: float = 10.0):
    """Read from a watch until ``count`` messages arrived, riding out a dropped socket."""
    got: list[linksignal.Incoming] = []
    deadline = time.monotonic() + timeout
    while len(got) < count and time.monotonic() < deadline:
        try:
            got.extend(watch.read(0.2))
        except linksignal.SignalError:
            time.sleep(0.05)
    return got


def test_a_message_reaches_a_watching_node_over_the_websocket(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, signal_service.url)
    watch = linksignal.SignalClient(b, signal_service.url).watch()
    try:
        sender.send(b.node_id, b"hello")
        (message,) = read_until(watch, 1)
        assert (message.seq, message.sender, message.body) == (1, a.node_id, b"hello")
    finally:
        watch.close()
    assert ("GET", f"/ws/{b.node_id}", 101) in signal_service.requests
    assert not [r for r in signal_service.requests if r[1].startswith("/recv/")]


def test_a_message_waiting_before_the_socket_opened_is_delivered(signal_service, daemons) -> None:
    a, b = daemons("a"), daemons("b")
    linksignal.SignalClient(a, signal_service.url).send(b.node_id, b"early")
    watch = linksignal.SignalClient(b, signal_service.url).watch()
    try:
        (message,) = read_until(watch, 1)
        assert message.body == b"early"
    finally:
        watch.close()


def test_a_watch_with_nothing_to_deliver_sends_no_request_while_it_waits(
    signal_service, daemons
) -> None:
    b = daemons("b")
    watch = linksignal.SignalClient(b, signal_service.url).watch()
    try:
        assert watch.read(0.1) == []
        before = len(signal_service.requests)
        for _ in range(4):
            assert watch.read(0.25) == []
        assert len(signal_service.requests) == before
    finally:
        watch.close()


def test_a_read_returns_when_its_wait_is_over(signal_service, daemons) -> None:
    watch = linksignal.SignalClient(daemons("b"), signal_service.url).watch()
    try:
        began = time.monotonic()
        assert watch.read(0.4) == []
        assert 0.3 <= time.monotonic() - began < 3
    finally:
        watch.close()


def test_a_handled_message_is_acknowledged_so_the_service_deletes_it(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, signal_service.url)
    watch = linksignal.SignalClient(b, signal_service.url).watch()
    try:
        sender.send(b.node_id, b"one")
        assert len(read_until(watch, 1)) == 1
        # Nothing is acknowledged until the caller comes back for more.
        assert signal_service.waiting(b.node_id) == 1
        watch.read(0.3)
        deadline = time.monotonic() + 5
        while signal_service.waiting(b.node_id) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert signal_service.waiting(b.node_id) == 0
        assert signal_service.ws_acks[-1] == 1
    finally:
        watch.close()


def test_a_dropped_socket_reconnects_and_delivers_each_message_exactly_once(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    sender = linksignal.SignalClient(a, signal_service.url)
    watch = linksignal.SignalClient(b, signal_service.url).watch()
    try:
        sender.send(b.node_id, b"1")
        sender.send(b.node_id, b"2")
        first = read_until(watch, 2)
        # The messages are not acknowledged yet, so the service sends them again.
        signal_service.drop_sockets()
        sender.send(b.node_id, b"3")
        rest = read_until(watch, 1)
        extra = read_until(watch, 1, timeout=0.6)
        bodies = [m.body for m in first + rest + extra]
        assert bodies == [b"1", b"2", b"3"]
        assert signal_service.ws_accepted == 2
    finally:
        watch.close()


def test_a_read_on_a_cut_socket_raises_instead_of_waiting_for_the_next_message(
    signal_service, daemons
) -> None:
    watch = linksignal.SignalClient(daemons("b"), signal_service.url).watch()
    try:
        watch.read(0.1)
        signal_service.drop_sockets()
        with pytest.raises(linksignal.SignalError):
            for _ in range(20):
                watch.read(0.3)
    finally:
        watch.close()


def test_fragmented_frames_with_a_ping_between_them_are_put_together(
    signal_service, daemons
) -> None:
    a, b = daemons("a"), daemons("b")
    signal_service.fragment_frames = True
    watch = linksignal.SignalClient(b, signal_service.url).watch()
    try:
        linksignal.SignalClient(a, signal_service.url).send(b.node_id, b"in pieces")
        (message,) = read_until(watch, 1)
        assert message.body == b"in pieces"
    finally:
        watch.close()


def test_an_upgrade_with_a_bad_signature_is_refused_and_named(signal_service, daemons) -> None:
    class Forger:
        def __init__(self, daemon) -> None:
            self.node_id = daemon.node_id
            self._daemon = daemon

        def sign(self, domain: str, message: bytes) -> bytes:
            return self._daemon.sign(domain, message + b"x")

        def verify(self, *args: Any) -> bool:
            return self._daemon.verify(*args)

    watch = linksignal.SignalClient(Forger(daemons("b")), signal_service.url).watch()
    try:
        with pytest.raises(linksignal.SignalError, match="bad_signature"):
            watch.read(0.2)
    finally:
        watch.close()
    assert ("GET", f"/ws/{watch.node_id}", 401) in signal_service.requests
    assert signal_service.ws_accepted == 0


def test_a_service_that_is_down_makes_a_read_raise(signal_service, daemons) -> None:
    watch = linksignal.SignalClient(daemons("b"), signal_service.url).watch()
    signal_service.close()
    try:
        with pytest.raises(linksignal.SignalError):
            watch.read(0.2)
    finally:
        watch.close()


def test_an_idle_watch_sends_the_text_ping_and_a_pong_is_liveness_not_a_message(
    signal_service, daemons, monkeypatch
) -> None:
    monkeypatch.setattr(linksignal, "WS_PING_S", 0.2)
    monkeypatch.setattr(linksignal, "WS_PONG_S", 1.0)
    watch = linksignal.SignalClient(daemons("b"), signal_service.url).watch()
    try:
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            assert watch.read(0.3) == []
        assert signal_service.ws_pings >= 3
        assert signal_service.ws_accepted == 1
        assert [r for r in signal_service.requests if r[1].startswith("/recv/")] == []
    finally:
        watch.close()


def test_a_watch_whose_ping_is_not_answered_drops_the_socket_after_the_pong_limit(
    signal_service, daemons, monkeypatch
) -> None:
    monkeypatch.setattr(linksignal, "WS_PING_S", 0.1)
    monkeypatch.setattr(linksignal, "WS_PONG_S", 0.4)
    signal_service.answer_pings = False
    watch = linksignal.SignalClient(daemons("b"), signal_service.url).watch()
    try:
        began = time.monotonic()
        with pytest.raises(linksignal.SignalError, match="no pong"):
            while time.monotonic() - began < 5:
                watch.read(0.3)
        assert time.monotonic() - began < 3
    finally:
        watch.close()


def test_the_service_records_a_request_before_any_reply_byte_is_written(
    signal_service, daemons
) -> None:
    # Spec "Signalling service": the fake is observed by tests that act on the client's view
    # of a reply, so the record must exist by the time the client can see the reply.
    class SlowRecord(list):
        def append(self, item: Any) -> None:
            time.sleep(0.3)
            super().append(item)

    signal_service.requests = SlowRecord()

    class Forger:
        def __init__(self, daemon) -> None:
            self.node_id = daemon.node_id
            self._daemon = daemon

        def sign(self, domain: str, message: bytes) -> bytes:
            return self._daemon.sign(domain, message + b"x")

        def verify(self, *args: Any) -> bool:
            return self._daemon.verify(*args)

    watch = linksignal.SignalClient(Forger(daemons("b")), signal_service.url).watch()
    try:
        with pytest.raises(linksignal.SignalError, match="bad_signature"):
            watch.read(2.0)
    finally:
        watch.close()
    assert ("GET", f"/ws/{watch.node_id}", 401) in signal_service.requests


# -- the User-Agent -----------------------------------------------------------------------
# Spec "Signalling service" item 1, "User-Agent".


def test_every_signalling_request_names_letify_as_its_user_agent(signal_service, daemons) -> None:
    import letify

    a, b = daemons("a"), daemons("b")
    linksignal.SignalClient(a, signal_service.url).send(b.node_id, b"hello")
    watch = linksignal.SignalClient(b, signal_service.url).watch()
    try:
        assert len(read_until(watch, 1)) == 1
    finally:
        watch.close()
    assert ("GET", f"/ws/{b.node_id}", 101) in signal_service.requests
    assert signal_service.agents
    assert set(signal_service.agents) == {f"letify/{letify.__version__}"}
