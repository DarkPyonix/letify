"""The agent as the tunnel login.

Spec "Agent and pairing" item 13, "Logging in" (``letify login tunnel``) and
"Recording devices at login". The session itself is faked here; its real behaviour is
pinned in test_link_agent.py and test_link_ssh_stream.py.
"""

from __future__ import annotations

import time
import tomllib
from pathlib import Path

import pytest

from letify.cli import main
from letify.config import login
from letify.transport import linkagent

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
