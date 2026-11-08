"""A Tunnel account that has an address, and the login that asks for a token.

Spec sections pinned here: "Logging in" (``letify login tunnel --connect``), "Rendezvous"
(a plain machine has no rendezvous), "What each kind asks for" and "Connection strategies"
(no forward SSH without an address, and the published SSH port).
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from conftest import FakeStrategy, provider_of

from letify.cli import default_alias, main
from letify.config import login
from letify.errors import ProviderUnavailable
from letify.providers.tunnel import Tunnel
from letify.transport.pipeline import Pipeline
from letify.transport.strategies import DirectSSH, Target

link_module = import_module("letify.transport.link")


@pytest.fixture(autouse=True)
def home_variable(monkeypatch, tmp_path: Path) -> None:
    """Expand ``~`` in a key path to the same home ``isolated_home`` points Path.home at."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def home_file_exists() -> bool:
    return (Path.home() / ".letify" / "config.toml").exists()


def test_without_a_name_the_alias_is_the_host_name_made_an_identifier() -> None:
    assert default_alias("gpu-box.lab.example") == "gpu_box_lab_example"
    assert default_alias("4090-rig") == "machine_4090_rig"


def test_a_tunnel_login_with_a_bad_token_names_the_token_step_and_writes_nothing(
    isolated_home, capsys
) -> None:
    assert main(["login", "tunnel", "home_box", "--connect", "garbage", "--no-input"]) == 1
    assert "tunnel login failed at token: " in capsys.readouterr().err
    assert not home_file_exists()


def test_a_tunnel_login_without_connect_asks_for_the_token(
    isolated_home, monkeypatch, capsys
) -> None:
    asked: list[str] = []

    def read(prompt: str) -> str:
        asked.append(prompt)
        return "garbage"

    monkeypatch.setattr(login, "read_line", read)
    assert main(["login", "tunnel", "home_box"]) == 1
    assert "Token printed by 'letify agent serve': " in asked
    assert "tunnel login failed at token: " in capsys.readouterr().err


def test_a_tunnel_login_without_connect_and_without_input_is_refused(
    isolated_home, capsys
) -> None:
    assert main(["login", "tunnel", "home_box", "--no-input"]) == 1
    assert "--connect" in capsys.readouterr().err
    assert not home_file_exists()


def test_a_tunnel_account_with_no_agent_and_no_address_is_refused(isolated_home) -> None:
    provider = provider_of(Tunnel, "home_box", user="researcher")
    with pytest.raises(ProviderUnavailable, match="no 'address' field"):
        _ = provider.address
    assert DirectSSH().needs(Target(alias="home_box")) == "no address"


# -- Spec: Connection strategies, the published SSH port ----------------------------------


def test_forward_ssh_dials_the_public_port_when_the_account_sets_one() -> None:
    provider = provider_of(
        Tunnel, "box", address="203.0.113.9", port=8022, public_port=30501, user="researcher"
    )
    command = provider.ssh_command("true")
    assert command[:3] == ["ssh", "-p", "30501"]
    assert command[-2:] == ["researcher@203.0.113.9", "true"]


def test_forward_ssh_pins_the_host_key_in_the_account_known_hosts(isolated_home) -> None:
    provider = provider_of(Tunnel, "box", address="203.0.113.9", public_port=30501)
    command = provider.ssh_command("true")
    assert "StrictHostKeyChecking=accept-new" in command
    assert "HostKeyAlias=letify-box" in command
    assert any(part.startswith("UserKnownHostsFile=") and "box" in part for part in command)


def test_punch_keeps_the_internal_ssh_port() -> None:
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
