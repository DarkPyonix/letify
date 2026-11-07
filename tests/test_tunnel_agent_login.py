"""The agent as the tunnel login, and the notice a Tailcat account gets.

Spec "Agent and pairing" items 13 and 14, "Logging in" (``letify login tunnel``) and
"Recording devices at login". The session itself is faked here; its real behaviour is
pinned in test_link_agent.py and test_link_ssh_stream.py.
"""

from __future__ import annotations

import time
import tomllib
from pathlib import Path

import pytest
from conftest import provider_of

from letify.cli import main
from letify.config import login
from letify.providers import shell as shell_module
from letify.providers import tunnel as tunnel_module
from letify.providers.tunnel import Tunnel
from letify.transport import linkagent, setup

AGENT = "ab" * 32
SIGNAL = "https://signal.example"


def agent_token() -> str:
    return linkagent.encode_token(
        agent=AGENT, secret="cd" * 32, signal=SIGNAL, version="1", exp=int(time.time()) + 900
    )


def agent_login(*extra: str) -> list[str]:
    return ["login", "tunnel", "box", "--connect", agent_token(), *extra]


def home_file() -> Path:
    return Path.home() / ".letify" / "config.toml"


def home_config() -> dict:
    return tomllib.loads(home_file().read_text(encoding="utf-8"))


def declare_agent_account() -> None:
    home_file().write_text(
        f'[box]\nkind = "tunnel"\nagent = "{AGENT}"\nsignal = "{SIGNAL}"\n'
        '[box.devices]\nOLD = { indices = "0" }\n',
        encoding="utf-8",
    )


@pytest.fixture
def no_ssh_at_login(monkeypatch) -> None:
    """An agent login must never start ssh, so any subprocess of the login is a failure."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"login ran a subprocess: {args[:1]}")

    monkeypatch.setattr(login.subprocess, "run", refuse)


# -- the login over the session ------------------------------------------------------------


def test_an_agent_login_records_the_devices_the_machine_reports_over_the_session(
    isolated_home, agent_session, no_ssh_at_login
) -> None:
    assert main([*agent_login(), "--no-input"]) == 0
    entry = home_config()["box"]
    assert entry["kind"] == "tunnel"
    assert entry["agent"] == AGENT
    assert "address" not in entry and "key" not in entry and "user" not in entry
    assert entry["devices"] == {"A100": {"indices": "0"}}


def test_an_agent_login_checks_the_workspace_over_the_session(
    isolated_home, agent_session, no_ssh_at_login
) -> None:
    assert main([*agent_login("--workspace", "/data/letify"), "--no-input"]) == 0
    assert any("/data/letify" in " ".join(argv) for argv in agent_session.commands)
    assert home_config()["box"]["workspace"] == "/data/letify"


def test_an_agent_login_that_cannot_write_its_workspace_fails_at_workspace_and_writes_nothing(
    isolated_home, agent_session, no_ssh_at_login, capsys
) -> None:
    agent_session.workspace_error = "Permission denied"
    assert main([*agent_login(), "--no-input"]) == 1
    assert "tunnel login failed at workspace: " in capsys.readouterr().err
    assert not home_file().exists()


def test_an_agent_login_whose_session_cannot_open_fails_at_session_and_writes_nothing(
    isolated_home, agent_session, no_ssh_at_login, capsys
) -> None:
    agent_session.open_error = "the link is slower than the floor"
    assert main([*agent_login(), "--no-input"]) == 1
    err = capsys.readouterr().err
    assert "tunnel login failed at session: " in err
    assert "slower than the floor" in err
    assert not home_file().exists()


def test_a_machine_with_no_gpu_still_logs_in_and_writes_no_devices_table(
    isolated_home, agent_session, no_ssh_at_login, capsys
) -> None:
    agent_session.gpus = ""
    assert main([*agent_login(), "--no-input"]) == 0
    assert "devices" not in home_config()["box"]
    assert "No devices table was written" in capsys.readouterr().out


def test_the_session_of_an_agent_login_is_closed_when_the_login_ends(
    isolated_home, agent_session, no_ssh_at_login
) -> None:
    assert main([*agent_login(), "--no-input"]) == 0
    assert agent_session.opened == 1
    assert agent_session.closed == 1


def test_detecting_devices_again_works_for_an_agent_account(
    isolated_home, agent_session, no_ssh_at_login
) -> None:
    declare_agent_account()
    agent_session.gpus = (
        "0, NVIDIA A100-SXM4-80GB, 81920 MiB\n1, NVIDIA A100-SXM4-80GB, 81920 MiB\n"
    )
    assert main(["login", "tunnel", "box", "--no-input", "--detect-devices"]) == 0
    devices = home_config()["box"]["devices"]
    assert "OLD" not in devices
    assert devices["A100"]["indices"] == "0-1"


def test_a_new_workspace_of_an_existing_agent_account_is_checked_over_the_session(
    isolated_home, agent_session, no_ssh_at_login
) -> None:
    declare_agent_account()
    agent_session.workspace_error = "Read-only file system"
    answers = login.Answers(
        alias="box", kind="tunnel", values={"workspace": "/ro"}, interactive=False
    )
    with pytest.raises(login.LoginError, match="Read-only file system"):
        login.log_in(answers)
    assert "workspace" not in home_config()["box"]


# -- the Tailcat notice ---------------------------------------------------------------------


@pytest.fixture
def tailcat_link(monkeypatch):
    """Stop the pipeline where a Tailcat account would race it, and reset the notice."""
    monkeypatch.setattr(tunnel_module, "_tailcat_noticed", False, raising=False)
    monkeypatch.setattr(shell_module.Shell, "link", lambda self, runtime=None: object())


def tailcat_account(alias: str = "box", **over):
    return provider_of(Tunnel, alias, tailcat="tcAbc123", tailcat_port=40123, **over)


def test_a_tailcat_account_is_told_once_that_tailcat_is_removed_in_3_0_and_how_to_migrate(
    tailcat_link, capsys
) -> None:
    tailcat_account().link()
    err = capsys.readouterr().err
    assert "box reaches its machine through Tailcat" in err
    assert "removed in 3.0" in err
    assert "letify agent serve" in err
    assert "letify login tunnel box --connect letify-agent2:" in err


def test_a_second_tailcat_account_or_link_in_one_process_prints_nothing_more(
    tailcat_link, capsys
) -> None:
    first = tailcat_account()
    first.link()
    capsys.readouterr()
    first.link()
    tailcat_account("other").link()
    assert capsys.readouterr().err == ""


def test_a_tailcat_account_with_announce_off_prints_no_notice(tailcat_link, capsys) -> None:
    provider = tailcat_account()
    provider.announce = False
    provider.link()
    assert capsys.readouterr().err == ""


def test_an_agent_account_prints_no_tailcat_notice(tailcat_link, capsys) -> None:
    provider_of(Tunnel, "box", agent=AGENT, signal=SIGNAL).link()
    assert capsys.readouterr().err == ""


def test_a_tailcat_login_says_that_tailcat_is_removed_in_3_0(
    isolated_home, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "letify.install.ensure", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    token = setup.encode_token({"tailcat": "tcAbc123", "tailcat_port": 40123})
    answers = login.Answers(
        alias="box", kind="tunnel", values={"connect": token}, interactive=False
    )
    with pytest.raises(RuntimeError, match="stop"):
        login.tunnel_account(answers)
    err = capsys.readouterr().err
    assert "removed in 3.0" in err
    assert "letify agent serve" in err


def test_client_shell_connect_ends_with_the_line_that_says_tailcat_is_removed_in_3_0(
    monkeypatch, capsys
) -> None:
    from letify.transport import agent as agent_module

    monkeypatch.setattr(setup, "tailcat_on_path", lambda binary: True)
    monkeypatch.setattr(setup, "ssh_answers", lambda port: True)
    monkeypatch.setattr(agent_module.Agent, "bind", lambda self: 4000)
    monkeypatch.setattr(agent_module.Agent, "start_tailcat", lambda self: "tcAbc123")
    monkeypatch.setattr(agent_module.Agent, "serve_forever", lambda self: None)
    monkeypatch.setattr(setup, "local_user", lambda: "researcher")
    assert main(["client", "shell", "connect", "--tailcat", "/x/tailcat", "--name", "b"]) == 0
    assert "Tailcat is removed in 3.0. Run 'letify agent serve'" in capsys.readouterr().out
