"""The command line.

Only one subcommand has a spec entry: "Efficiency model" says ``letify efficiency``
exposes the formula on the command line. The others (``providers``, ``devices``,
``status``, ``check``, ``probe``) are pinned here only where they surface something a
spec section does settle, which is the provider table under "Provider model" and the
capability probe under "letify-core". ``status`` and ``check`` have no spec entry at all
and are left for the report rather than frozen into this suite.

Every test runs through ``main(argv)`` with an empty home directory and an empty project
directory, because the entry point reads ``~/.letify`` and ``./.letify``.
"""

from __future__ import annotations

import json

import pytest

from letify import __version__
from letify.cli import build_parser, main
from letify.runtime import telemetry

# -- Spec: Efficiency model ----------------------------------------------------


def test_the_efficiency_formula_is_on_the_command_line(capsys) -> None:
    # The same numbers the spec table gives for a 0.5 s step with three synchronizations
    # at a 150 ms round trip.
    assert main(["efficiency", "0.5", "3", "150"]) == 0
    assert "52.6% of a direct run" in capsys.readouterr().out


def test_the_efficiency_subcommand_needs_no_configuration(isolated_home, capsys) -> None:
    # It is arithmetic, so it answers before any account is declared.
    assert main(["efficiency", "4.0", "1", "150"]) == 0
    assert "96." in capsys.readouterr().out


# -- Spec: Provider model ------------------------------------------------------


def test_the_declared_providers_are_listed_with_their_channel(isolated_home, capsys) -> None:
    (isolated_home / ".letify").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\npersistent = true\n', encoding="utf-8"
    )
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "lab" in out
    assert "shell" in out
    assert "persistent" in out
    assert "channel=persistent" in out
    # The local machine needs no declaration, so it is always there.
    assert "local" in out


def test_a_provider_that_cannot_be_built_is_reported_without_hiding_the_rest(
    isolated_home, capsys
) -> None:
    (isolated_home / ".letify").write_text('[odd]\nkind = "vastai"\n', encoding="utf-8")
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "odd" in out
    assert "unavailable" in out
    assert "local" in out


def test_the_accelerators_every_provider_offers_are_listed(isolated_home, capsys) -> None:
    (isolated_home / ".letify").write_text(
        '[lab]\nkind = "shell"\naddress = "a"\ngpus = ["A100"]\n', encoding="utf-8"
    )
    assert main(["devices"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert table["lab"] == ["A100"]
    assert "CPU" in table["local"]


def test_a_configuration_file_can_be_named_explicitly(isolated_home, tmp_path, capsys) -> None:
    elsewhere = tmp_path / "other.letify"
    elsewhere.write_text(
        '[lab]\nkind = "shell"\naddress = "a"\ngpus = ["H100"]\n', encoding="utf-8"
    )
    assert main(["--config", str(elsewhere), "devices"]) == 0
    assert json.loads(capsys.readouterr().out)["lab"] == ["H100"]


# -- Spec: letify-core ---------------------------------------------------------


def test_the_capability_probe_is_on_the_command_line(isolated_home, capsys) -> None:
    # Whether forwarding can run here, and what it would cost.
    assert main(["probe"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["platform"]
    assert report["reason"]
    assert "usable" in report


# -- the parser ----------------------------------------------------------------


def test_the_version_is_reported_and_nothing_else_happens(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2


def test_every_subcommand_is_reachable_from_the_parser() -> None:
    # One place to see what the command line offers.
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert sorted(actions[0].choices) == [
        "check",
        "devices",
        "efficiency",
        "probe",
        "providers",
        "status",
        "usage",
        "utilization",
    ]


# -- Spec: Remaining usage -----------------------------------------------------


def test_the_usage_table_lists_every_declared_alias(isolated_home, capsys) -> None:
    # An account missing from the table reads as an account with nothing left on it, so a
    # provider that could not even be built is listed with its reason too.
    (isolated_home / ".letify").write_text('[odd]\nkind = "vastai"\n', encoding="utf-8")
    assert main(["usage"]) == 0
    out = capsys.readouterr().out
    # This machine bills nobody, which is a different answer from an unknown balance.
    assert "unmetered" in out
    assert "odd" in out
    assert "unavailable" in out


def test_an_unreported_balance_says_where_it_would_have_come_from(isolated_home, capsys) -> None:
    (isolated_home / ".letify").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n', encoding="utf-8"
    )
    assert main(["usage", "lab"]) == 0
    out = capsys.readouterr().out
    assert "not reported" in out
    assert "no account behind it" in out


def test_a_configured_command_is_what_the_table_prints(isolated_home, capsys) -> None:
    # How a user supplies a figure letify has no endpoint for.
    (isolated_home / ".letify").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
        'usage_command = "echo 12.5"\nusage_unit = "hours"\nusage_limit = 40.0\n',
        encoding="utf-8",
    )
    assert main(["usage", "lab", "--json"]) == 0
    row = json.loads(capsys.readouterr().out)[0]
    assert row["remaining"] == 12.5
    assert row["used"] == 27.5
    assert row["source"] == "usage_command"


# -- Spec: GPU utilization -----------------------------------------------------


def test_an_instance_with_no_live_session_is_listed_with_its_reason(isolated_home, capsys) -> None:
    # Starting a session to measure its load would cost money and change the answer.
    (isolated_home / ".letify").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\ngpus = ["A100"]\n',
        encoding="utf-8",
    )
    assert main(["utilization", "lab", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["accelerator"] == "A100"
    assert rows[0]["devices"] == []
    assert "no live session" in rows[0]["reason"]


def test_a_device_reading_is_printed_with_the_fields_the_card_reported(
    isolated_home, capsys, monkeypatch
) -> None:
    # A card that reports neither power nor temperature is printed without them rather
    # than with a zero.
    monkeypatch.setattr(
        telemetry,
        "read_smi",
        lambda: "0, NVIDIA RTX PRO 6000, 87, 40960, 98304, [N/A], [Not Supported]\n",
    )
    assert main(["utilization", "local"]) == 0
    out = capsys.readouterr().out
    assert "87% busy" in out
    assert "40.0/96.0 GiB" in out
    assert "W" not in out


def test_the_cpu_shape_is_not_asked_how_busy_its_accelerator_is(
    isolated_home, capsys, monkeypatch
) -> None:
    # Every provider registers a CPU shape, and it has no device to report.
    monkeypatch.setattr(telemetry, "read_smi", lambda: "")
    assert main(["utilization", "local", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert all(row["accelerator"] != "CPU" for row in rows)
