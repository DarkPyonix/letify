"""Tailcat is removed.

Spec "Connection strategies" and "Agent and pairing" item 14: an account that names
``tailcat``, ``tailcat_port`` or ``tailcat_binary`` is refused by name with the migration,
``letify setup tailcat`` and ``letify client shell connect`` do not exist, and no strategy
list has a Tailcat entry.
"""

from __future__ import annotations

import base64
import json

import pytest
from conftest import provider_of

from letify import cli
from letify.errors import ProviderUnavailable
from letify.providers.colab import Colab
from letify.providers.elice import Elice
from letify.providers.kaggle import Kaggle
from letify.providers.shell import Shell
from letify.providers.tunnel import Tunnel


def old_token(**fields) -> str:
    """A token as the removed ``letify client shell connect`` printed it."""
    raw = json.dumps(fields, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize("cls", [Shell, Tunnel, Colab, Kaggle, Elice])
@pytest.mark.parametrize(
    "options",
    [
        {"tailcat": "tc1abc", "tailcat_port": 4000},
        {"tailcat": "tc1abc"},
        {"tailcat_port": 4000},
        {"tailcat_binary": "/usr/bin/tailcat"},
    ],
)
def test_an_account_that_names_a_tailcat_field_is_refused_by_name(
    isolated_home, cls, options
) -> None:
    field = next(iter(options))
    with pytest.raises(ProviderUnavailable) as raised:
        provider_of(cls, "box", **options)
    message = str(raised.value)
    assert "box" in message
    assert f"'{field}'" in message
    assert "Tailcat was removed in 3.0" in message
    assert "letify agent serve" in message
    assert "letify login tunnel box --connect letify-agent2:" in message


def test_an_account_without_a_tailcat_field_is_built(isolated_home) -> None:
    assert provider_of(Tunnel, "box", agent="ab" * 32, signal="https://x.example").alias == "box"


@pytest.mark.parametrize(
    "argv",
    [["setup", "tailcat"], ["client", "shell", "connect"], ["client"]],
)
def test_the_removed_commands_do_not_exist(argv, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(argv)
    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_letify_setup_still_installs_eci() -> None:
    args = cli.build_parser().parse_args(["setup", "eci"])
    assert args.tool == "eci"


def test_the_strategy_lists_have_no_tailcat_entry(isolated_home) -> None:
    from letify.transport import strategies

    assert not hasattr(strategies, "TailcatUDP")
    shell = provider_of(Shell, "lab", address="gpu.example.edu", port_mapping=False)
    assert [s.name for s in shell.strategies()] == ["direct_ssh", "tcp_punch", "fallback"]
    kaggle = provider_of(Kaggle, "k", link="ssh")
    assert [s.name for s in kaggle.strategies()] == ["tcp_punch", "fallback"]
    assert "tailcat" not in [s.name for s in provider_of(Colab, "c", link="ssh").strategies()]


def test_the_ranks_are_forward_connect_back_punch_fallback(isolated_home) -> None:
    shell = provider_of(
        Shell, "lab", address="gpu.example.edu", port_mapping=False, connect_back={"port": 20130}
    )
    assert [(s.name, s.rank) for s in shell.strategies()] == [
        ("direct_ssh", 1),
        ("connect_back", 2),
        ("tcp_punch", 3),
        ("fallback", 4),
    ]


def test_no_tool_named_tailcat_can_be_installed() -> None:
    from letify import install

    assert install.TOOLS == ("eci",)


def test_a_tailcat_token_is_refused_by_name_with_the_migration_at_login(
    tmp_path, monkeypatch
) -> None:
    from letify.config import login

    monkeypatch.setenv("LETIFY_HOME", str(tmp_path / "home"))
    token = old_token(tailcat="tc1abc", tailcat_port=4000, user="u", port=22)
    with pytest.raises(login.LoginError) as raised:
        login.tunnel_account(
            login.Answers(alias="box", kind="tunnel", values={"connect": token}, interactive=False)
        )
    assert "Tailcat was removed in 3.0" in str(raised.value)
    assert "letify agent serve" in str(raised.value)


def test_a_token_that_is_not_an_agent_token_is_refused_at_login(tmp_path, monkeypatch) -> None:
    from letify.config import login

    monkeypatch.setenv("LETIFY_HOME", str(tmp_path / "home"))
    with pytest.raises(login.LoginError, match="not a token"):
        login.tunnel_account(
            login.Answers(
                alias="box", kind="tunnel", values={"connect": "garbage"}, interactive=False
            )
        )
