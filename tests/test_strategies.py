"""Tests for the connection strategies, the rendezvous and the remote agent.

Spec: Transport, Connection strategies, Rendezvous, Reverse SSH and Colab. The TCP punch
and the agent run over loopback sockets; ssh, ssh-keygen and tailcat are faked because
they need a remote machine.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest
from conftest import CannedRendezvous, FakeCompleted, LoopbackRendezvous

import letify
from letify.cli import build_parser, main
from letify.transport import agent as agent_module
from letify.transport import nat, strategies
from letify.transport import rendezvous as rendezvous_module
from letify.transport.link import OneShotLink
from letify.transport.probe import Probe
from letify.transport.rendezvous import CommandRendezvous, TailcatRendezvous, remote_script
from letify.transport.strategies import (
    DirectSSH,
    ProviderFallback,
    ReverseSSH,
    TailcatUDP,
    Target,
    TCPPunch,
)


def fake_sshd() -> socket.socket:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    return server


# -- Spec: Transport, Connection strategies: forward SSH ----------------------------


def test_forward_ssh_needs_an_address() -> None:
    assert DirectSSH().needs(Target(alias="lab")) == "no address"
    target = Target(alias="lab", address="gpu.example.edu", direct_ssh=lambda c: ["ssh"])
    assert DirectSSH().needs(target) is None
    assert DirectSSH.rank == 1


def test_forward_ssh_is_attempted_with_one_command_that_does_nothing(patch_run) -> None:
    recorder = patch_run(strategies)
    target = Target(alias="lab", address="gpu", direct_ssh=lambda c: ["ssh", "gpu", c or ""])
    link = DirectSSH().attempt(target)
    assert recorder.command == ["ssh", "gpu", "exit 0"]
    assert link.ssh_command("uname") == ["ssh", "gpu", "uname"]


def test_forward_ssh_that_fails_says_what_ssh_said(patch_run) -> None:
    patch_run(strategies, result=FakeCompleted(returncode=255, stderr="connection refused"))
    target = Target(alias="lab", address="gpu", direct_ssh=lambda c: ["ssh", "gpu"])
    with pytest.raises(OSError, match="connection refused"):
        DirectSSH().attempt(target)


def test_forward_ssh_assumed_alone_runs_nothing(patch_run) -> None:
    recorder = patch_run(strategies)
    target = Target(alias="lab", address="gpu", direct_ssh=lambda c: ["ssh", "gpu"])
    DirectSSH().assume(target)
    assert recorder.calls == []


# -- Spec: Transport, Rendezvous: TCP hole punching ---------------------------------


def test_a_tcp_punch_needs_a_rendezvous() -> None:
    assert TCPPunch().needs(Target(alias="lab")) == "no rendezvous"
    blocked = CannedRendezvous(unavailable="tailcat is not on PATH")
    assert TCPPunch().needs(Target(alias="lab", rendezvous=blocked)) == "tailcat is not on PATH"
    assert TCPPunch.rank == 2


def test_a_tcp_punch_over_loopback_carries_the_probe_and_then_ssh(stun_server) -> None:
    sshd = fake_sshd()
    rendezvous = LoopbackRendezvous()
    rendezvous.lead_seconds = 0.3
    target = Target(
        alias="lab",
        rendezvous=rendezvous,
        stun=stun_server.address,
        ssh_port=sshd.getsockname()[1],
        user="root",
    )
    link = TCPPunch().attempt(target)
    request = rendezvous.requests[0]
    assert request["kind"] == "tcp_punch"
    assert len(bytes.fromhex(request["token"])) == 16

    measured = Probe(round_trips=3, seconds=0.02).measure(link.probe_stream())
    assert measured.download_bps > 0

    command = link.ssh_command("true")
    port = int(command[command.index("-p") + 1])
    assert port not in (0, 22, sshd.getsockname()[1])
    assert command[-2:] == ["root@127.0.0.1", "true"]
    client = socket.create_connection(("127.0.0.1", port))
    client.sendall(b"SSH-2.0-letify\r\n")
    conn, _ = sshd.accept()
    assert nat.recv_exact(conn, 16) == b"SSH-2.0-letify\r\n"
    for sock in (client, conn, sshd):
        sock.close()
    link.close()


# -- Spec: Transport, Rendezvous: Tailcat -------------------------------------------


def test_tailcat_needs_the_binary_on_path_and_a_rendezvous(patch_which) -> None:
    patch_which(strategies, present=False)
    target = Target(alias="lab", rendezvous=CannedRendezvous())
    assert "tailcat" in TailcatUDP().needs(target)
    patch_which(strategies, present=True)
    assert TailcatUDP().needs(target) is None
    assert TailcatUDP().needs(Target(alias="lab")) == "no rendezvous"
    assert TailcatUDP.rank == 3


def test_ssh_over_tailcat_uses_it_as_the_proxy_command(patch_which, patch_run) -> None:
    patch_which(strategies, present=True)
    recorder = patch_run(strategies)
    rendezvous = CannedRendezvous({"address": "tcQ3xY"})
    target = Target(alias="lab", rendezvous=rendezvous, user="root")
    link = TailcatUDP().attempt(target)
    command = link.ssh_command("uname")
    assert "ProxyCommand=tailcat tcQ3xY 22" in command
    assert command[-1] == "uname"
    assert recorder.command[-1] == "exit 0"
    assert rendezvous.requests == [{"kind": "tailcat", "ssh_port": 22}]


def test_the_remote_half_reads_the_address_tailcat_serve_prints(patch_popen) -> None:
    started = patch_popen(nat, ["starting\n", "Server listening with new address: tcAbC123\n"])
    answer, _ = nat.begin({"kind": "tailcat", "ssh_port": 22})
    assert answer == {"address": "tcAbC123"}
    assert started[0].command == ["tailcat", "serve", "22"]


def test_a_rendezvous_through_the_agent_needs_no_exchange_for_tailcat(patch_which) -> None:
    patch_which(rendezvous_module, present=True)
    rendezvous = TailcatRendezvous("tcAgent", 40123)
    assert rendezvous.unavailable() is None
    assert rendezvous.tailcat_endpoint(22, 5.0) == ("tcAgent", 40123)
    patch_which(rendezvous_module, present=False)
    assert "tailcat" in TailcatRendezvous("tcAgent", 40123).unavailable()


# -- Spec: Transport, Reverse SSH ---------------------------------------------------


def keygen(command: list[str]) -> FakeCompleted:
    from pathlib import Path

    path = Path(command[command.index("-f") + 1])
    path.write_text("PRIVATE KEY\n")
    path.with_suffix(".pub").write_text("ssh-ed25519 AAAAsession letify\n")
    return FakeCompleted()


def test_reverse_ssh_is_only_raced_when_the_account_sets_it() -> None:
    target = Target(alias="lab", rendezvous=CannedRendezvous())
    assert ReverseSSH().needs(target) == "no reverse_ssh entry"
    assert ReverseSSH.rank == 4


def test_a_reverse_session_key_is_restricted_marked_and_removed_on_close(
    isolated_home, patch_run
) -> None:
    from pathlib import Path

    authorized = Path.home() / ".ssh" / "authorized_keys"
    authorized.parent.mkdir()
    authorized.write_text(
        "ssh-ed25519 AAAAmine me@laptop\n"
        'restrict,port-forwarding,command="/bin/false" ssh-ed25519 AAAAold letify-session lab\n'
        'restrict,port-forwarding,command="/bin/false" ssh-ed25519 AAAAother letify-session x\n'
    )
    patch_run(strategies, result=keygen)
    rendezvous = CannedRendezvous({"port": 40022})
    target = Target(
        alias="lab",
        rendezvous=rendezvous,
        user="researcher",
        reverse_ssh={"address": "home.example.com", "port": 2222, "user": "me"},
    )
    link = ReverseSSH().attempt(target)
    lines = authorized.read_text().splitlines()
    assert "AAAAold" not in authorized.read_text()
    assert "AAAAother" in authorized.read_text()
    assert lines[-1] == (
        'restrict,port-forwarding,command="/bin/false" ssh-ed25519 AAAAsession letify-session lab'
    )
    request = rendezvous.requests[0]
    assert request["kind"] == "reverse_ssh"
    assert request["private_key"] == "PRIVATE KEY\n"
    assert (request["address"], request["port"], request["user"]) == (
        "home.example.com",
        2222,
        "me",
    )
    command = link.ssh_command("uname")
    assert command[command.index("-p") + 1] == "40022"
    assert command[-2:] == ["researcher@127.0.0.1", "uname"]

    link.close()
    assert "letify-session lab" not in authorized.read_text()
    assert "AAAAmine" in authorized.read_text()


def test_the_remote_half_of_a_reverse_forward_lets_the_server_choose_the_port(
    patch_popen, tmp_path
) -> None:
    started = patch_popen(nat, ["Allocated port 40022 for remote forward to 127.0.0.1:22\n"])
    answer, _ = nat.begin(
        {
            "kind": "reverse_ssh",
            "address": "home.example.com",
            "port": 2222,
            "user": "me",
            "private_key": "PRIVATE KEY\n",
            "key_directory": str(tmp_path),
        }
    )
    assert answer == {"port": 40022}
    assert "0:127.0.0.1:22" in started[0].command
    assert (tmp_path / "letify_reverse").stat().st_mode & 0o777 == 0o600


# -- Spec: Transport, Connection strategies: provider fallback ----------------------


def test_the_provider_fallback_is_last_and_moves_behind_reverse_ssh() -> None:
    assert ProviderFallback().rank == 4
    assert ProviderFallback(rank=5).rank == 5
    assert ProviderFallback().needs(Target(alias="lab")) == "no provider fallback"
    link = OneShotLink("fallback", 4, lambda source, timeout: "out")
    target = Target(alias="lab", fallback=lambda: link)
    assert ProviderFallback().attempt(target) is link
    assert link.persistent is False
    assert link.probe_stream() is None
    with pytest.raises(letify.RuntimeFailure):
        link.ssh_command("uname")


# -- Spec: Transport, Rendezvous: the provider layer ---------------------------------


class Canned(CommandRendezvous):
    def __init__(self, output: str):
        self.output = output
        self.sources: list[str] = []

    def run_python(self, source: str, timeout: float) -> str:
        self.sources.append(source)
        return self.output


def test_a_command_rendezvous_reads_the_answer_line_the_remote_half_prints() -> None:
    rendezvous = Canned('noise\nLETIFY-ANSWER {"mapping": ["198.51.100.4", 41000]}\n')
    assert rendezvous.exchange({"kind": "ping"}, 5.0) == {"mapping": ["198.51.100.4", 41000]}
    compile(rendezvous.sources[0], "<remote>", "exec")


def test_a_command_rendezvous_that_gets_an_error_raises_it() -> None:
    with pytest.raises(OSError, match="stun failed"):
        Canned('LETIFY-ANSWER {"error": "stun failed"}\n').exchange({"kind": "ping"}, 5.0)
    with pytest.raises(OSError, match="no answer"):
        Canned("Traceback\n").exchange({"kind": "ping"}, 5.0)


def test_the_remote_script_is_the_standard_library_module_and_the_request() -> None:
    script = remote_script({"kind": "ping"})
    compile(script, "<remote>", "exec")
    assert "def punch(" in script
    assert script.rstrip().splitlines()[-1].startswith("run_detached(")
    assert "from ." not in script
    assert "import letify" not in script


def test_a_ping_is_answered_without_a_continuation() -> None:
    answer, continuation = nat.begin({"kind": "ping"})
    assert answer == {"pong": True}
    assert continuation() is None


# -- Spec: Transport, Rendezvous: the remote agent ------------------------------------


def test_the_agent_answers_a_rendezvous_line_and_splices_ssh() -> None:
    sshd = fake_sshd()
    agent = agent_module.Agent(ssh=sshd.getsockname())
    port = agent.bind()
    assert port != 0
    threading.Thread(target=agent.serve_forever, daemon=True).start()

    with socket.create_connection(("127.0.0.1", port)) as client:
        client.sendall(b"LETIFY-RDV " + json.dumps({"kind": "ping"}).encode() + b"\n")
        assert json.loads(client.makefile("rb").readline()) == {"pong": True}

    with socket.create_connection(("127.0.0.1", port)) as client:
        client.sendall(b"SSH-2.0-letify\r\n")
        conn, _ = sshd.accept()
        assert nat.recv_exact(conn, 16) == b"SSH-2.0-letify\r\n"
        conn.close()
    agent.close()
    sshd.close()


def test_the_agent_reports_a_failed_request_as_an_error_line() -> None:
    agent = agent_module.Agent()
    port = agent.bind()
    threading.Thread(target=agent.serve_forever, daemon=True).start()
    with socket.create_connection(("127.0.0.1", port)) as client:
        client.sendall(b"LETIFY-RDV " + json.dumps({"kind": "nonsense"}).encode() + b"\n")
        assert "unknown rendezvous request" in json.loads(client.makefile("rb").readline())["error"]
    agent.close()


def test_the_client_side_sends_one_request_line_through_tailcat(patch_popen, patch_which) -> None:
    patch_which(rendezvous_module, present=True)
    started = patch_popen(rendezvous_module, ['{"mapping": ["198.51.100.4", 41000]}\n'])
    answer = TailcatRendezvous("tcAgent", 40123).exchange({"kind": "tcp_punch"}, 5.0)
    assert answer == {"mapping": ["198.51.100.4", 41000]}
    assert started[0].command == ["tailcat", "tcAgent", "40123"]


def test_the_agent_starts_tailcat_serve_on_its_own_port(patch_popen, patch_which) -> None:
    patch_which(agent_module, present=True)
    started = patch_popen(agent_module, ["Server listening with new address: tcHome\n"])
    agent = agent_module.Agent()
    port = agent.bind()
    assert agent.start_tailcat() == "tcHome"
    assert started[0].command == ["tailcat", "serve", str(port)]
    agent.close()


def test_client_shell_connect_is_a_three_level_command() -> None:
    args = build_parser().parse_args(["client", "shell", "connect"])
    assert (args.command, args.client_command, args.shell_command) == (
        "client",
        "shell",
        "connect",
    )


def test_client_shell_connect_prints_what_the_account_needs(
    isolated_home, patch_popen, patch_which, monkeypatch, capsys
) -> None:
    patch_which(agent_module, present=True)
    patch_popen(agent_module, ["Server listening with new address: tcHome\n"])
    monkeypatch.setattr(agent_module.Agent, "serve_forever", lambda self: None)
    assert main(["client", "shell", "connect"]) == 0
    out = capsys.readouterr().out
    assert 'tailcat = "tcHome"' in out
    assert "tailcat_port = " in out


def test_client_shell_connect_without_tailcat_says_so(isolated_home, patch_which, capsys) -> None:
    patch_which(agent_module, present=False)
    assert main(["client", "shell", "connect"]) == 1
    assert "tailcat" in capsys.readouterr().err
