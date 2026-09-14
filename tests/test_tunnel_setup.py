"""Setting up a Tunnel account with two commands.

Spec sections pinned here: "Rendezvous" (what ``letify client shell connect`` checks and
prints, and the token), "Logging in" (``letify login tunnel --connect``), "What each kind
asks for" (a tunnel account has no address) and "Connection strategies" (no forward SSH
without an address).

SSH, tailcat and the agent's ``tailcat serve`` need a second machine and a network, so
they are faked through the conftest fixtures. The SSH banner check runs against a real
loopback socket.
"""

from __future__ import annotations

import socket
import threading
import tomllib
from importlib import import_module
from pathlib import Path

import pytest
from conftest import FakeCompleted, FakeStrategy, provider_of

from letify.cli import main
from letify.config import login
from letify.providers import shell as shell_module
from letify.providers.tunnel import Tunnel
from letify.transport import agent as agent_module
from letify.transport import setup
from letify.transport import strategies as strategies_module
from letify.transport.pipeline import Pipeline
from letify.transport.strategies import DirectSSH, Target

link_module = import_module("letify.transport.link")

TOKEN_FIELDS = {"tailcat": "tcAbc123", "tailcat_port": 40123, "user": "researcher", "port": 2222}
PROXY = "ProxyCommand=tailcat tcAbc123 40123"


# -- helpers ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def home_variable(monkeypatch, tmp_path: Path) -> None:
    """Expand ``~`` in a key path to the same home ``isolated_home`` points Path.home at."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def banner_server(line: bytes = b"SSH-2.0-OpenSSH_9.6\r\n") -> int:
    """A loopback listener that writes ``line`` to every connection, and its port."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)

    def serve() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            conn.sendall(line)
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return listener.getsockname()[1]


def closed_port() -> int:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    holder.close()
    return port


def on_linux_amd64(monkeypatch) -> None:
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")


def home_config() -> dict:
    return tomllib.loads((Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8"))


def home_file_exists() -> bool:
    return (Path.home() / ".letify" / "config.toml").exists()


def with_key() -> None:
    ssh = Path.home() / ".ssh"
    ssh.mkdir(parents=True, exist_ok=True)
    (ssh / "id_letify").write_text("private", encoding="utf-8")
    (ssh / "id_letify.pub").write_text("ssh-ed25519 AAAA me@here", encoding="utf-8")


def remote(*, fails: str | None = None, stderr: str = "refused"):
    """Answer SSH over Tailcat: one A100 for nvidia-smi, success otherwise.

    ``fails`` names a piece of the remote command whose call exits 255.
    """

    def answer(command: list[str]) -> FakeCompleted:
        if fails and (fails in command[-1] or fails in command):
            return FakeCompleted(returncode=255, stderr=stderr)
        if "nvidia-smi" in command[-1]:
            return FakeCompleted(stdout="0, NVIDIA A100-SXM4-80GB, 81920 MiB\n")
        return FakeCompleted()

    return answer


def ssh_calls(recorder) -> list[list[str]]:
    return [call for call in recorder.commands if call and call[0] == "ssh"]


def tunnel_login(*extra: str) -> list[str]:
    return ["login", "tunnel", "home_box", "--connect", setup.encode_token(TOKEN_FIELDS), *extra]


# -- Spec: Rendezvous, the token ---------------------------------------------------


def test_a_token_decodes_to_the_fields_it_was_made_from() -> None:
    token = setup.encode_token(TOKEN_FIELDS)
    assert setup.decode_token(token) == TOKEN_FIELDS
    assert "=" not in token
    assert set(token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_a_token_with_surrounding_whitespace_still_decodes() -> None:
    assert setup.decode_token(f"  {setup.encode_token(TOKEN_FIELDS)}\n") == TOKEN_FIELDS


@pytest.mark.parametrize(
    "token",
    ["not a token", setup.encode_token({"user": "researcher"}), "e30"],  # e30 is {}
)
def test_a_token_that_is_not_what_connect_prints_is_refused(token: str) -> None:
    with pytest.raises(ValueError):
        setup.decode_token(token)


# -- Spec: Rendezvous, install instructions ----------------------------------------


@pytest.mark.parametrize(
    ("machine", "arch"),
    [("x86_64", "amd64"), ("amd64", "amd64"), ("aarch64", "arm64"), ("armv7l", "armv7")],
)
def test_linux_gets_a_curl_command_for_its_architecture(machine: str, arch: str) -> None:
    text = setup.tailcat_install_instructions("Linux", machine)
    version = setup.TAILCAT_VERSION
    url = (
        f"https://github.com/tailscale/tailcat/releases/download/v{version}/"
        f"tailcat_{version}_linux_{arch}.tar.gz"
    )
    assert f"mkdir -p ~/.local/bin && curl -L {url} | tar xz -C ~/.local/bin tailcat" in text
    assert "~/.local/bin" in text


def test_macos_gets_homebrew() -> None:
    assert "brew install tailcat" in setup.tailcat_install_instructions("Darwin", "arm64")


def test_windows_gets_the_release_zip() -> None:
    text = setup.tailcat_install_instructions("Windows", "AMD64")
    assert f"tailcat_{setup.TAILCAT_VERSION}_windows_amd64.zip" in text
    assert "tailcat.exe" in text


def test_an_unknown_platform_gets_the_releases_page() -> None:
    text = setup.tailcat_install_instructions("Plan9", "mips")
    assert "https://github.com/tailscale/tailcat/releases" in text


# -- Spec: Rendezvous, the SSH banner check ------------------------------------------


def test_an_ssh_server_is_recognized_by_its_banner() -> None:
    assert setup.ssh_answers(banner_server())


def test_a_port_that_answers_with_something_else_is_not_an_ssh_server() -> None:
    assert not setup.ssh_answers(banner_server(b"HTTP/1.1 400 Bad Request\r\n"))


def test_a_closed_port_is_not_an_ssh_server() -> None:
    assert not setup.ssh_answers(closed_port())


# -- Spec: Rendezvous, letify client shell connect -----------------------------------


@pytest.fixture
def no_agent(monkeypatch):
    """Fail the test if the agent is started."""

    def refuse(self) -> int:
        raise AssertionError("the agent was started")

    monkeypatch.setattr(agent_module.Agent, "bind", refuse)


def test_connect_without_tailcat_prints_how_to_install_it_and_starts_nothing(
    patch_which, monkeypatch, no_agent, capsys
) -> None:
    patch_which(setup, present=False)
    on_linux_amd64(monkeypatch)
    assert main(["client", "shell", "connect", "--ssh-port", str(banner_server())]) == 1
    err = capsys.readouterr().err
    assert "tailcat is not on PATH" in err
    assert f"tailcat_{setup.TAILCAT_VERSION}_linux_amd64.tar.gz" in err


def test_connect_without_an_ssh_server_prints_how_to_start_one_and_starts_nothing(
    patch_which, no_agent, capsys
) -> None:
    patch_which(setup, present=True)
    port = closed_port()
    assert main(["client", "shell", "connect", "--ssh-port", str(port)]) == 1
    err = capsys.readouterr().err
    assert f"No SSH server answers on port {port}" in err
    assert "apt-get install -y openssh-server" in err
    assert "mkdir -p /run/sshd" in err
    assert "/usr/sbin/sshd" in err


def test_connect_prints_one_login_command_carrying_the_token(
    patch_which, monkeypatch, capsys
) -> None:
    patch_which(setup, present=True)
    monkeypatch.setattr(agent_module.Agent, "start_tailcat", lambda self: "tcAbc123")
    monkeypatch.setattr(agent_module.Agent, "serve_forever", lambda self: None)
    monkeypatch.setattr(setup, "local_user", lambda: "researcher")
    port = banner_server()
    assert main(["client", "shell", "connect", "--ssh-port", str(port), "--name", "home_box"]) == 0

    out = capsys.readouterr().out
    commands = [line.strip() for line in out.splitlines() if "letify login tunnel" in line]
    assert len(commands) == 1
    prefix = "letify login tunnel home_box --connect "
    assert commands[0].startswith(prefix)
    fields = setup.decode_token(commands[0][len(prefix) :])
    assert fields["tailcat"] == "tcAbc123"
    assert fields["user"] == "researcher"
    assert fields["port"] == port
    assert isinstance(fields["tailcat_port"], int) and fields["tailcat_port"] > 0
    assert "tmux" in out and "nohup" in out
    assert "new address" in out


def test_without_a_name_the_alias_is_the_host_name_made_an_identifier() -> None:
    assert setup.default_alias("gpu-box.lab.example") == "gpu_box_lab_example"
    assert setup.default_alias("4090-rig") == "machine_4090_rig"


# -- Spec: Logging in, letify login tunnel ---------------------------------------------


def test_a_tunnel_login_without_tailcat_here_writes_nothing_and_says_how_to_install(
    isolated_home, patch_which, patch_run, monkeypatch, capsys
) -> None:
    patch_which(setup, present=False)
    on_linux_amd64(monkeypatch)
    recorder = patch_run(login, result=remote())
    assert main([*tunnel_login(), "--no-input"]) == 1
    err = capsys.readouterr().err
    assert "tunnel login failed at tailcat" in err
    assert f"tailcat_{setup.TAILCAT_VERSION}_linux_amd64.tar.gz" in err
    assert recorder.calls == []
    assert not home_file_exists()


def test_a_tunnel_login_writes_the_account_from_the_token_with_no_address(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(setup, present=True)
    with_key()
    patch_run(login, result=remote())
    assert main([*tunnel_login(), "--no-input"]) == 0

    entry = home_config()["home_box"]
    assert "address" not in entry
    assert entry["kind"] == "tunnel"
    assert entry["tailcat"] == "tcAbc123"
    assert entry["tailcat_port"] == 40123
    assert entry["user"] == "researcher"
    assert entry["port"] == 2222
    assert entry["key"] == "~/.ssh/id_letify"
    assert entry["devices"] == {"A100": {"indices": "0"}}


def test_every_ssh_command_of_a_tunnel_login_goes_through_tailcat(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(setup, present=True)
    with_key()
    recorder = patch_run(login, result=remote())
    assert main([*tunnel_login(), "--no-input"]) == 0

    calls = ssh_calls(recorder)
    # Install the key, confirm it, check the workspace, detect the GPUs.
    assert len(calls) == 4
    for call in calls:
        assert PROXY in call
        assert "researcher@tcAbc123" in call
        assert call[call.index("-p") + 1] == "2222"
    install, confirm, workspace, devices = calls
    assert "BatchMode=yes" not in install
    assert recorder.calls[0]["input"] == "ssh-ed25519 AAAA me@here"
    assert "BatchMode=yes" in confirm
    assert ".letify-probe" in workspace[-1]
    assert "nvidia-smi" in devices[-1]


def test_skip_key_install_only_confirms_the_key_over_tailcat(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(setup, present=True)
    recorder = patch_run(login, result=remote())
    assert main([*tunnel_login("--skip-key-install", "--key", "~/.ssh/other"), "--no-input"]) == 0
    calls = ssh_calls(recorder)
    assert all("BatchMode=yes" in call for call in calls)
    assert all(str(Path.home() / ".ssh" / "other") in call for call in calls)
    assert home_config()["home_box"]["key"] == "~/.ssh/other"


@pytest.mark.parametrize(
    ("fails", "step"),
    [
        ("StrictHostKeyChecking=accept-new", "key install"),
        ("echo letify", "key confirmation"),
        (".letify-probe", "workspace"),
    ],
)
def test_a_tunnel_login_that_fails_at_a_step_names_it_and_writes_nothing(
    isolated_home, patch_which, patch_run, capsys, fails: str, step: str
) -> None:
    patch_which(setup, present=True)
    with_key()
    if step == "key install":
        # Only the interactive install lacks BatchMode, so fail the first call alone.
        def answer(command: list[str]) -> FakeCompleted:
            if "BatchMode=yes" not in command:
                return FakeCompleted(returncode=255, stderr="Permission denied")
            return remote()(command)

        patch_run(login, result=answer)
    else:
        patch_run(login, result=remote(fails=fails, stderr="Permission denied"))
    assert main([*tunnel_login(), "--no-input"]) == 1
    err = capsys.readouterr().err
    assert f"tunnel login failed at {step}: " in err
    assert "Permission denied" in err
    assert not home_file_exists()


def test_a_tunnel_login_with_a_bad_token_names_the_token_step_and_runs_nothing(
    isolated_home, patch_which, patch_run, capsys
) -> None:
    patch_which(setup, present=True)
    recorder = patch_run(login, result=remote())
    assert main(["login", "tunnel", "home_box", "--connect", "garbage", "--no-input"]) == 1
    assert "tunnel login failed at token: " in capsys.readouterr().err
    assert recorder.calls == []
    assert not home_file_exists()


def test_a_tunnel_login_without_connect_asks_for_the_token(
    isolated_home, patch_which, patch_run, monkeypatch
) -> None:
    patch_which(setup, present=True)
    with_key()
    patch_run(login, result=remote())
    asked: list[str] = []
    token = setup.encode_token(TOKEN_FIELDS)

    def read(prompt: str) -> str:
        asked.append(prompt)
        return token if prompt.startswith("Token printed") else ""

    monkeypatch.setattr(login, "read_line", read)
    assert main(["login", "tunnel", "home_box"]) == 0
    assert "Token printed by 'letify client shell connect': " in asked
    assert home_config()["home_box"]["tailcat"] == "tcAbc123"


def test_a_tunnel_login_without_connect_and_without_input_is_refused(
    isolated_home, patch_which, patch_run, capsys
) -> None:
    patch_which(setup, present=True)
    patch_run(login, result=remote())
    assert main(["login", "tunnel", "home_box", "--no-input"]) == 1
    assert "--connect" in capsys.readouterr().err
    assert not home_file_exists()


# -- Spec: Connection strategies, an account with no address -----------------------------


def test_a_tunnel_account_naming_tailcat_needs_no_address() -> None:
    provider = provider_of(Tunnel, "home_box", **TOKEN_FIELDS)
    assert provider.address is None
    target = provider.target()
    assert target.address is None
    assert target.direct_ssh is None
    assert DirectSSH().needs(target) == "no address"


def test_letify_check_on_a_tunnel_account_runs_over_tailcat(
    isolated_home, patch_which, patch_run, patch_popen, capsys
) -> None:
    (Path.home() / ".letify" / "config.toml").write_text(
        "[home_box]\n"
        'kind = "tunnel"\n'
        'tailcat = "tcAbc123"\n'
        "tailcat_port = 40123\n"
        'user = "researcher"\n'
        'stun = "127.0.0.1:9"\n',
        encoding="utf-8",
    )
    (isolated_home / ".letify" / "config.toml").write_text("[home_box]\n", encoding="utf-8")
    patch_which(strategies_module, present=True)
    patch_run(strategies_module)
    patch_popen(link_module, [])
    answered = patch_run(
        shell_module, result=FakeCompleted(stdout="Linux box\nA100\nletify-workspace-ok\n")
    )
    assert main(["check", "home_box"]) == 0
    assert "workspace ~/.letify-runtime: writable" in capsys.readouterr().out
    assert PROXY in answered.command
    assert not any("@None" in part or part == "None" for part in answered.command)


# -- Spec: Connection strategies, the published SSH port ----------------------------------


def test_forward_ssh_dials_the_public_port_when_the_account_sets_one() -> None:
    provider = provider_of(
        Tunnel, "box", address="203.0.113.9", port=8022, public_port=30501, user="researcher"
    )
    command = provider.ssh_command("true")
    assert command[:3] == ["ssh", "-p", "30501"]
    assert command[-2:] == ["researcher@203.0.113.9", "true"]


def test_punch_and_tailcat_keep_the_internal_ssh_port() -> None:
    provider = provider_of(Tunnel, "box", address="203.0.113.9", port=8022, public_port=30501)
    target = provider.target()
    assert target.ssh_port == 8022
    assert target.direct_port == 30501


class LabelledDirect(FakeStrategy):
    """A fake forward SSH that names itself the way DirectSSH does."""

    def label(self, target: Target) -> str:
        return DirectSSH().label(target)


def test_the_connection_log_names_the_port_forward_ssh_dials(isolated_home, capsys) -> None:
    target = Target(alias="box", address="203.0.113.9", direct_port=30501)
    Pipeline([LabelledDirect("direct_ssh", 1)], target=target, alias="box").connect()
    err = capsys.readouterr().err
    assert "letify: connecting to box" in err
    assert "direct_ssh (203.0.113.9:30501)" in err


def test_connect_puts_the_public_address_and_port_in_the_token(
    patch_which, monkeypatch, capsys
) -> None:
    patch_which(setup, present=True)
    monkeypatch.setattr(agent_module.Agent, "start_tailcat", lambda self: "tcAbc123")
    monkeypatch.setattr(agent_module.Agent, "serve_forever", lambda self: None)
    monkeypatch.setattr(setup, "local_user", lambda: "researcher")
    port = banner_server()
    argv = ["client", "shell", "connect", "--ssh-port", str(port), "--name", "home_box"]
    argv += ["--public-address", "203.0.113.9", "--public-port", "30501"]
    assert main(argv) == 0
    line = next(x.strip() for x in capsys.readouterr().out.splitlines() if "--connect" in x)
    fields = setup.decode_token(line.split("--connect ", 1)[1])
    assert fields["address"] == "203.0.113.9"
    assert fields["public_port"] == 30501
    assert fields["port"] == port


def test_a_tunnel_login_writes_the_address_and_public_port_from_the_token(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(setup, present=True)
    with_key()
    patch_run(login, result=remote())
    token = setup.encode_token({**TOKEN_FIELDS, "address": "203.0.113.9", "public_port": 30501})
    assert main(["login", "tunnel", "home_box", "--connect", token, "--no-input"]) == 0
    entry = home_config()["home_box"]
    assert entry["address"] == "203.0.113.9"
    assert entry["public_port"] == 30501
    assert entry["port"] == 2222


def test_a_tunnel_login_takes_the_address_and_public_port_from_options(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(setup, present=True)
    with_key()
    patch_run(login, result=remote())
    extra = ["--address", "203.0.113.9", "--public-port", "30501", "--no-input"]
    assert main([*tunnel_login(), *extra]) == 0
    entry = home_config()["home_box"]
    assert entry["address"] == "203.0.113.9"
    assert entry["public_port"] == 30501
