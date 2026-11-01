"""The persistent agent and node pairing over the signalling service.

Spec "Agent and pairing". The service is the in-process FakeSignalService of tests/conftest.py,
and the agent, the shim, both sidecars, the punch and the session are real. The tests build the
sidecar like tests/test_link_signal.py and are skipped, with the reason, without cargo.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from typing import Any

import pytest

import letify
from letify.errors import ProviderUnavailable
from letify.protocol.worker import BOOTSTRAP
from letify.runtime.channel import relay_program
from letify.transport import linkagent, linkchannel, linkd, linksignal, setup

# -- helpers ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def local_sidecar(sidecar_binary, monkeypatch):
    monkeypatch.setenv(linkd.ENV_BINARY, str(sidecar_binary))
    monkeypatch.delenv("LETIFY_SIGNAL_URL", raising=False)


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


class Running:
    """An AgentService on its own thread, with its log, and the helpers a test needs."""

    def __init__(self, service, daemon, home, thread, lines) -> None:
        self.service = service
        self.daemon = daemon
        self.home = home
        self.thread = thread
        self.lines = lines

    @property
    def node_id(self) -> str:
        return self.daemon.node_id

    def stop(self) -> None:
        self.service.stop()
        self.thread.join(timeout=20)


@pytest.fixture
def start_service(signal_service, daemons, sidecar_binary, tmp_path):
    running: list[Running] = []

    def start(**over: Any) -> Running:
        home = linkagent.AgentHome(tmp_path / "agent")
        daemon = daemons("agent")
        lines: list[str] = []
        options: dict[str, Any] = {
            "binary": str(sidecar_binary),
            "python": sys.executable,
            "bootstrap": BOOTSTRAP,
            "relay": relay_program(),
            "log": lines.append,
            "poll_s": 0.3,
        }
        options.update(over)
        service = linkagent.AgentService(
            linksignal.SignalClient(daemon, signal_service.url), home, **options
        )
        thread = threading.Thread(target=service.serve, daemon=True)
        thread.start()
        started = Running(service, daemon, home, thread, lines)
        running.append(started)
        return started

    yield start
    for started in running:
        started.stop()


def user_pair(signal_service, user, token: str, label: str = "laptop", timeout: float = 15.0):
    client = linksignal.SignalClient(user, signal_service.url)
    return linkagent.pair(client, token, label=label, timeout_s=timeout)


def wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# -- the token ----------------------------------------------------------------------------


def test_a_token_v2_carries_the_agent_the_secret_the_url_and_the_version() -> None:
    token = linkagent.encode_token(
        agent="ab" * 32, secret="cd" * 32, signal="https://x.example", version="1.2.3", exp=99
    )
    assert token.startswith("letify-agent2:")
    fields = linkagent.parse_token(token)
    assert (fields.agent, fields.secret, fields.signal) == (
        "ab" * 32,
        "cd" * 32,
        "https://x.example",
    )
    assert (fields.version, fields.exp) == ("1.2.3", 99)


def test_an_old_tailcat_token_is_refused_by_name() -> None:
    old = setup.encode_token({"tailcat": "tc1abc", "tailcat_port": 4000, "user": "u", "port": 22})
    with pytest.raises(linkagent.TokenError, match="Tailcat token"):
        linkagent.parse_token(old)


def test_text_that_is_not_a_token_is_refused_with_the_prefix_it_needs() -> None:
    with pytest.raises(linkagent.TokenError, match="letify-agent2:"):
        linkagent.parse_token("hello")
    with pytest.raises(linkagent.TokenError, match="not a token"):
        linkagent.parse_token("letify-agent2:@@@")
    bad = linkagent.encode_token(
        agent="zz", secret="cd" * 32, signal="https://x.example", version="1", exp=1
    )
    with pytest.raises(linkagent.TokenError, match="agent"):
        linkagent.parse_token(bad)


def test_a_plain_http_signal_url_is_accepted_only_for_a_loopback_host() -> None:
    def token(url: str) -> str:
        return linkagent.encode_token(
            agent="ab" * 32, secret="cd" * 32, signal=url, version="1", exp=1
        )

    assert linkagent.parse_token(token("http://127.0.0.1:8787")).signal == "http://127.0.0.1:8787"
    with pytest.raises(linkagent.TokenError, match="http"):
        linkagent.parse_token(token("http://signal.example"))


# -- the signal URL -----------------------------------------------------------------------


def test_the_signal_url_comes_from_the_flag_then_the_environment_then_the_file_then_the_default(
    tmp_path, monkeypatch
) -> None:
    home = linkagent.AgentHome(tmp_path / "agent")
    assert linkagent.signal_url(home) == "https://letify.darkpyonix.dev"
    home.path.mkdir(parents=True)
    (home.path / "config.toml").write_text('signal = "https://file.example"\n')
    assert linkagent.signal_url(home) == "https://file.example"
    monkeypatch.setenv("LETIFY_SIGNAL_URL", "https://env.example/")
    assert linkagent.signal_url(home) == "https://env.example"
    assert linkagent.signal_url(home, flag="https://flag.example") == "https://flag.example"


# -- the state ----------------------------------------------------------------------------


def test_the_agent_key_and_paired_list_are_private_to_the_user(
    start_service, signal_service, daemons, sidecar_binary, tmp_path
) -> None:
    home = linkagent.AgentHome(tmp_path / "state" / "agent")
    daemon = linkagent.open_daemon(home, binary=sidecar_binary)
    try:
        assert stat.S_IMODE(home.key_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(home.path.stat().st_mode) == 0o700
        home.paired.add("ab" * 32, "laptop")
        assert stat.S_IMODE(home.paired.path.stat().st_mode) == 0o600
        # The same key gives the same identity on the next start.
        node = daemon.node_id
    finally:
        daemon.close()
    again = linkagent.open_daemon(home, binary=sidecar_binary)
    try:
        assert again.node_id == node
    finally:
        again.close()


def test_a_key_file_that_others_can_read_is_refused(tmp_path, sidecar_binary) -> None:
    home = linkagent.AgentHome(tmp_path / "agent")
    linkagent.open_daemon(home, binary=sidecar_binary).close()
    home.key_path.chmod(0o644)
    with pytest.raises(linkd.LinkError, match="chmod 600"):
        linkagent.open_daemon(home, binary=sidecar_binary)


def test_the_paired_list_keeps_a_label_and_a_time_and_removes_by_prefix(tmp_path) -> None:
    store = linkagent.PairedStore(tmp_path / "paired.json")
    store.add("ab" * 32, "laptop\x1b[31m" + "x" * 100, at=5)
    data = json.loads((tmp_path / "paired.json").read_text())
    entry = data["nodes"]["ab" * 32]
    assert entry["at"] == 5 and len(entry["label"]) == 64 and "\x1b" not in entry["label"]
    with pytest.raises(linkagent.AgentError, match="at least 8"):
        store.remove("abab")
    assert store.remove("ababab" * 2) == "ab" * 32
    assert store.nodes() == {}
    with pytest.raises(linkagent.AgentError, match="no paired node"):
        store.remove("ab" * 8)


# -- pairing ------------------------------------------------------------------------------


def test_pairing_succeeds_with_the_right_secret(start_service, signal_service, daemons) -> None:
    agent = start_service()
    user = daemons("user")
    token = agent.service.new_token(signal_service.url)
    assert user_pair(signal_service, user, token, label="my laptop") == agent.node_id
    nodes = agent.home.paired.nodes()
    assert list(nodes) == [user.node_id] and nodes[user.node_id]["label"] == "my laptop"
    assert any(line.startswith(f"paired {user.node_id[:12]} (my laptop)") for line in agent.lines)


def test_a_wrong_secret_is_refused_and_pairs_nothing(
    start_service, signal_service, daemons
) -> None:
    agent = start_service()
    user = daemons("user")
    fields = linkagent.parse_token(agent.service.new_token(signal_service.url))
    wrong = linkagent.encode_token(
        agent=fields.agent, secret="00" * 32, signal=fields.signal, version=fields.version,
        exp=fields.exp,
    )  # fmt: skip
    with pytest.raises(linkagent.PairingRefused, match="wrong, used or expired"):
        user_pair(signal_service, user, wrong)
    assert agent.home.paired.nodes() == {}


def test_a_secret_works_once_and_a_second_node_with_it_is_refused(
    start_service, signal_service, daemons
) -> None:
    agent = start_service()
    first, second = daemons("first"), daemons("second")
    token = agent.service.new_token(signal_service.url)
    user_pair(signal_service, first, token)
    with pytest.raises(linkagent.PairingRefused):
        user_pair(signal_service, second, token)
    assert list(agent.home.paired.nodes()) == [first.node_id]


def test_a_pairing_that_is_replayed_is_dropped_and_a_fresh_one_with_the_used_secret_is_refused(
    start_service, signal_service, daemons
) -> None:
    agent = start_service()
    user = daemons("user")
    token = linkagent.parse_token(agent.service.new_token(signal_service.url))
    client = linksignal.SignalClient(user, signal_service.url)
    sid = "ef" * 16
    line = json.dumps(
        {
            "op": "pair",
            "proof": linkagent.proof(token.secret, user.node_id, token.agent),
            "label": "a",
        }
    )
    body = linksignal.seal(user, token.agent, sid, 1, line)
    client.send(token.agent, body)
    assert wait_for(lambda: user.node_id in agent.home.paired.nodes())
    client.send(token.agent, body)
    assert wait_for(lambda: any("replayed" in d for d in agent.service.dropped))
    # A new conversation with the same, now used, secret gets the refusal.
    with pytest.raises(linkagent.PairingRefused):
        linkagent.pair(client, linkagent.encode_token(**{
            "agent": token.agent, "secret": token.secret, "signal": token.signal,
            "version": token.version, "exp": token.exp,
        }), label="again", timeout_s=15)  # fmt: skip


def test_an_expired_secret_is_refused(start_service, signal_service, daemons) -> None:
    now = [time.time()]
    agent = start_service(clock=lambda: now[0], secret_ttl_s=900)
    user = daemons("user")
    token = agent.service.new_token(signal_service.url)
    now[0] += 901
    with pytest.raises(linkagent.PairingRefused, match="expired"):
        user_pair(signal_service, user, token)
    assert agent.home.paired.nodes() == {}


def test_the_user_side_refuses_a_token_past_its_expiry_before_sending_anything(
    signal_service, daemons
) -> None:
    user = daemons("user")
    token = linkagent.encode_token(
        agent="ab" * 32, secret="cd" * 32, signal=signal_service.url, version="1", exp=1
    )
    with pytest.raises(linkagent.TokenError, match="expired"):
        user_pair(signal_service, user, token)
    assert signal_service.requests == []


def test_five_wrong_proofs_discard_the_secret(start_service, signal_service, daemons) -> None:
    agent = start_service()
    token = agent.service.new_token(signal_service.url)
    fields = linkagent.parse_token(token)
    wrong = linkagent.encode_token(
        agent=fields.agent, secret="11" * 32, signal=fields.signal, version=fields.version,
        exp=fields.exp,
    )  # fmt: skip
    for number in range(5):
        with pytest.raises(linkagent.PairingRefused):
            user_pair(signal_service, daemons(f"guess{number}"), wrong)
    # The right secret no longer works, because it was discarded by the failures.
    with pytest.raises(linkagent.PairingRefused):
        user_pair(signal_service, daemons("late"), token)
    assert agent.home.paired.nodes() == {}


def test_the_agent_log_holds_no_secret_proof_or_full_node_id(
    start_service, signal_service, daemons
) -> None:
    agent = start_service()
    user = daemons("user")
    token = agent.service.new_token(signal_service.url)
    fields = linkagent.parse_token(token)
    user_pair(signal_service, user, token)
    text = "\n".join(agent.lines)
    assert fields.secret not in text and token not in text
    assert linkagent.proof(fields.secret, user.node_id, fields.agent) not in text
    assert user.node_id not in text and agent.node_id not in text


# -- who is served ------------------------------------------------------------------------


def test_an_unpaired_nodes_bootstrap_starts_no_shim_and_gets_no_answer(
    start_service, signal_service, daemons
) -> None:
    agent = start_service()
    stranger = daemons("stranger")
    bootstrap = linksignal.SignalBootstrap(
        linksignal.SignalClient(stranger, signal_service.url), agent.node_id, ping_timeout_s=1.5
    )
    with pytest.raises(linksignal.SignalError, match="did not answer"):
        bootstrap.start()
    assert agent.service.started == 0
    assert any("not paired" in d for d in agent.service.dropped)
    assert signal_service.waiting(stranger.node_id) == 0


def test_a_paired_node_over_the_conversation_cap_is_told_the_agent_is_busy(
    start_service, signal_service, daemons
) -> None:
    agent = start_service(max_conversations=0)
    user = daemons("user")
    user_pair(signal_service, user, agent.service.new_token(signal_service.url))
    bootstrap = linksignal.SignalBootstrap(
        linksignal.SignalClient(user, signal_service.url), agent.node_id
    )
    bootstrap.listen_only()
    bootstrap.send_line(json.dumps({"op": "config", "version": letify.__version__}))
    reply = json.loads(bootstrap.recv_line() or b"{}")
    bootstrap.close()
    assert reply["event"] == "error" and "busy" in reply["error"]
    assert agent.service.started == 0


def test_a_stopped_agent_leaves_its_loop_at_once(start_service) -> None:
    agent = start_service()
    began = time.monotonic()
    agent.stop()
    assert not agent.thread.is_alive() and time.monotonic() - began < 10


def test_a_service_that_fails_is_retried_with_a_backoff_and_logged_once(
    start_service, signal_service, daemons
) -> None:
    agent = start_service(backoff_s=(0.05, 0.1))
    signal_service.close()
    assert wait_for(lambda: any("signalling service" in line for line in agent.lines))
    time.sleep(0.5)
    assert sum("signalling service" in line for line in agent.lines) == 1
    assert agent.thread.is_alive()


# -- connect_over -------------------------------------------------------------------------


def connect(signal_service, local, agent_node, tmp_path, **over: Any):
    client = linksignal.SignalClient(local, signal_service.url)
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


def pid_of_worker():
    def body():
        import os

        return os.getpid()

    return body


def test_a_paired_node_connects_end_to_end_and_unpairing_stops_the_next_connect(
    start_service, signal_service, daemons, sidecar_binary, tmp_path, udp_stun, monkeypatch
) -> None:
    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)
    agent = start_service()
    local = linkd.LinkDaemon(tmp_path / "local.key", binary=sidecar_binary)
    try:
        user_pair(signal_service, local, agent.service.new_token(signal_service.url))
        remote = connect(signal_service, local, agent.node_id, tmp_path)
        channel = linkchannel.LinkChannel(remote, name="r1")
        try:
            assert channel.call(pid_of_worker(), (), {})[0] != os.getpid()
            assert agent.service.started == 1
            assert any(
                line.startswith(f"conversation {local.node_id[:12]} started")
                for line in agent.lines
            )
        finally:
            channel.close()
        agent.home.paired.remove(local.node_id)
        assert wait_for(lambda: any(line.startswith("unpaired") for line in agent.lines), 15)
        with pytest.raises(ProviderUnavailable, match="did not answer"):
            connect(signal_service, local, agent.node_id, tmp_path, ping_timeout_s=2.0)
        assert agent.service.started == 1
    finally:
        local.close()


# -- the command line and the login -------------------------------------------------------


def test_the_agent_commands_list_and_unpair_a_node(tmp_path, monkeypatch, capsys) -> None:
    from letify import cli

    monkeypatch.setenv("LETIFY_HOME", str(tmp_path / "home"))
    store = linkagent.AgentHome(tmp_path / "home" / "agent").paired
    store.add("ab" * 32, "laptop", at=1)
    assert cli.main(["agent", "paired"]) == 0
    assert "ab" * 32 in capsys.readouterr().out
    assert cli.main(["agent", "unpair", "abababab"]) == 0
    assert store.nodes() == {}
    assert cli.main(["agent", "unpair", "abababab"]) == 1
    assert "no paired node" in capsys.readouterr().err


def test_login_pairs_with_the_agent_and_records_its_node_and_url(
    start_service, signal_service, tmp_path, monkeypatch
) -> None:
    from letify.config import login

    monkeypatch.setenv("LETIFY_HOME", str(tmp_path / "home"))
    agent = start_service()
    token = agent.service.new_token(signal_service.url)
    answers = login.Answers(
        alias="box", kind="tunnel", values={"connect": token}, interactive=False
    )
    options = login.tunnel_account(answers)
    assert options == {"kind": "tunnel", "agent": agent.node_id, "signal": signal_service.url}
    key = tmp_path / "home" / "accounts" / "box" / "link.key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert len(agent.home.paired.nodes()) == 1


def test_login_with_a_tailcat_token_does_not_take_the_agent_path(tmp_path, monkeypatch) -> None:
    from letify.config import login

    monkeypatch.setenv("LETIFY_HOME", str(tmp_path / "home"))
    taken: list[str] = []
    monkeypatch.setattr(login, "link_tunnel_account", lambda answers, token: taken.append(token))
    monkeypatch.setattr(
        "letify.install.ensure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tailcat"))
    )
    old = setup.encode_token({"tailcat": "tc1abc", "tailcat_port": 4000})
    with pytest.raises(Exception, match="tailcat"):
        login.tunnel_account(
            login.Answers(alias="box", kind="tunnel", values={"connect": old}, interactive=False)
        )
    assert taken == []
    new = linkagent.encode_token(
        agent="ab" * 32, secret="cd" * 32, signal="https://x.example", version="1", exp=1
    )
    login.tunnel_account(
        login.Answers(alias="box", kind="tunnel", values={"connect": new}, interactive=False)
    )
    assert taken == [new]


def test_a_tailcat_token_given_to_the_agent_login_is_refused_by_name(tmp_path, monkeypatch) -> None:
    from letify.config import login

    monkeypatch.setenv("LETIFY_HOME", str(tmp_path / "home"))
    old = setup.encode_token({"tailcat": "tc1abc", "tailcat_port": 4000})
    with pytest.raises(login.LoginError, match="Tailcat token"):
        login.link_tunnel_account(
            login.Answers(alias="box", kind="tunnel", values={"connect": old}, interactive=False),
            old,
        )


# -- the standing wait ----------------------------------------------------------------------
# Spec "Signalling service" item 1, "Which wait uses which endpoint", and "Agent and pairing" item 5.


def recv_requests(signal_service, node: str) -> list[tuple[str, str, int]]:
    return [r for r in signal_service.requests if r[1] == f"/recv/{node}"]


def test_the_agent_receives_a_pairing_over_the_websocket_and_never_long_polls(
    start_service, signal_service, daemons
) -> None:
    agent, user = start_service(), daemons("user")
    token = agent.service.new_token(signal_service.url)
    assert user_pair(signal_service, user, token) == agent.node_id
    assert ("GET", f"/ws/{agent.node_id}", 101) in signal_service.requests
    assert recv_requests(signal_service, agent.node_id) == []


def test_an_idle_agent_sends_no_request_while_it_waits(start_service, signal_service) -> None:
    agent = start_service()
    assert wait_for(lambda: signal_service.sockets_open(agent.node_id) == 1)
    before = len(signal_service.requests)
    time.sleep(1.5)
    assert len(signal_service.requests) == before
    assert recv_requests(signal_service, agent.node_id) == []


def test_an_agent_whose_socket_dropped_reconnects_and_still_serves_a_pairing(
    start_service, signal_service, daemons
) -> None:
    agent, user = start_service(backoff_s=(0.05, 0.1)), daemons("user")
    assert wait_for(lambda: signal_service.sockets_open(agent.node_id) == 1)
    signal_service.drop_sockets()
    assert wait_for(lambda: signal_service.ws_accepted == 2)
    token = agent.service.new_token(signal_service.url)
    assert user_pair(signal_service, user, token) == agent.node_id
    assert recv_requests(signal_service, agent.node_id) == []
    assert agent.home.paired.nodes() == [user.node_id]
