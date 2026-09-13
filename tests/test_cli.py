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
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import FakeCompleted

import letify
from letify import __version__
from letify.cli import build_parser, main
from letify.config import login
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
    (isolated_home / ".letify" / "config.toml").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\npersistent = true\n', encoding="utf-8"
    )
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "lab" in out
    assert "shell" in out
    assert "persistent" in out
    assert "channel" not in out
    # The local machine needs no declaration, so it is always there.
    assert "local" in out


def test_a_provider_that_cannot_be_built_is_reported_without_hiding_the_rest(
    isolated_home, capsys
) -> None:
    config = isolated_home / ".letify" / "config.toml"
    config.write_text('[odd]\nkind = "vastai"\n', encoding="utf-8")
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "odd" in out
    assert "unavailable" in out
    assert "local" in out


def test_the_accelerators_every_provider_offers_are_listed(isolated_home, capsys) -> None:
    (isolated_home / ".letify" / "config.toml").write_text(
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
        "client",
        "devices",
        "efficiency",
        "login",
        "logout",
        "probe",
        "providers",
        "status",
        "stubs",
        "usage",
        "utilization",
    ]


# -- Spec: Remaining usage -----------------------------------------------------


def test_the_usage_table_lists_every_declared_alias(isolated_home, capsys) -> None:
    # An account missing from the table reads as an account with nothing left on it, so a
    # provider that could not even be built is listed with its reason too.
    config = isolated_home / ".letify" / "config.toml"
    config.write_text('[odd]\nkind = "vastai"\n', encoding="utf-8")
    assert main(["usage"]) == 0
    out = capsys.readouterr().out
    # This machine bills nobody, which is a different answer from an unknown balance.
    assert "unmetered" in out
    assert "odd" in out
    assert "unavailable" in out


def test_an_unreported_balance_says_where_it_would_have_come_from(isolated_home, capsys) -> None:
    (isolated_home / ".letify" / "config.toml").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n', encoding="utf-8"
    )
    assert main(["usage", "lab"]) == 0
    out = capsys.readouterr().out
    assert "not reported" in out
    assert "no account behind it" in out


def test_a_configured_command_is_what_the_table_prints(isolated_home, capsys) -> None:
    # How a user supplies a figure letify has no endpoint for.
    (isolated_home / ".letify" / "config.toml").write_text(
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
    (isolated_home / ".letify" / "config.toml").write_text(
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


# -- Spec: Logging in ----------------------------------------------------------


def test_logging_in_writes_the_account_at_home_and_a_reference_in_the_project(
    isolated_home, patch_run, capsys
) -> None:
    # Two files answer two questions: what this machine can reach, and what this
    # repository needs. Only the second one is safe to commit.
    patch_run(login, result=FakeCompleted())
    code = main(
        [
            "login",
            "shell",
            "lab",
            "--address",
            "gpu.example.edu",
            "--user",
            "researcher",
            "--key",
            "~/.ssh/id_letify",
            "--no-input",
            "--skip-key-install",
        ]
    )
    assert code == 0
    home_file = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert "gpu.example.edu" in home_file
    assert "researcher" in home_file

    project_file = (isolated_home / ".letify" / "config.toml").read_text(encoding="utf-8")
    # Naming the alias is what makes the account available here, so the table is empty.
    assert tomllib.loads(project_file) == {"lab": {}}
    assert "gpu.example.edu" not in project_file
    assert "researcher" not in project_file


def test_an_account_already_set_up_gets_only_the_reference(isolated_home, capsys) -> None:
    # The common case in a second repository: the account was declared once already.
    (Path.home() / ".letify" / "config.toml").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n', encoding="utf-8"
    )
    assert main(["login", "shell", "lab", "--no-input"]) == 0
    out = capsys.readouterr().out
    assert "already" in out
    project = isolated_home / ".letify" / "config.toml"
    assert tomllib.loads(project.read_text(encoding="utf-8")) == {"lab": {}}


def test_declaring_an_account_with_nothing_to_connect_to_is_refused(isolated_home, capsys) -> None:
    # Refusing beats writing a half account that fails at the first call.
    assert main(["login", "shell", "lab", "--no-input"]) == 1
    assert not (Path.home() / ".letify" / "config.toml").exists()
    assert "address" in capsys.readouterr().err


def test_a_reserved_alias_is_refused_before_anything_is_written(isolated_home, capsys) -> None:
    assert main(["login", "shell", "any", "--address", "h", "--no-input"]) == 1
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_an_elice_token_goes_to_the_account_directory_and_not_the_config(
    isolated_home, capsys
) -> None:
    # A token in a file is a token in a backup, so the file gets the pointer only.
    code = main(
        [
            "login",
            "elice",
            "elice_a100",
            "--zone-id",
            "zone-1",
            "--machine-id",
            "machine-1",
            "--token",
            "secret-token",
            "--no-input",
        ]
    )
    assert code == 0
    home_file = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert "secret-token" not in home_file
    assert "keyring" not in home_file
    token_file = Path.home() / ".letify" / "accounts" / "elice_a100" / "access_token"
    assert token_file.read_text(encoding="utf-8").strip() == "secret-token"
    if sys.platform != "win32":
        assert token_file.stat().st_mode & 0o777 == 0o600


def test_logging_in_to_colab_runs_the_colab_login_inside_the_account_directory(
    isolated_home, patch_which, patch_run, capsys
) -> None:
    # The user never runs the Colab CLI themselves. letify runs it through uv, with the
    # account directory as its home, so the token lands in ~/.letify/accounts/<alias>/.
    from letify import tools

    patch_which(tools, present=True)
    recorder = patch_run(login)
    assert main(["login", "colab", "colab_a", "--account", "me@example.com", "--no-input"]) == 0
    call = recorder.calls[-1]
    assert call["command"][:3] == ["/usr/bin/uv", "tool", "run"]
    assert call["command"][-2:] == ["colab", "sessions"]
    assert call["env"]["HOME"] == str(Path.home() / ".letify" / "accounts" / "colab_a")
    home_file = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert "me@example.com" in home_file
    assert "token" not in home_file


def test_a_colab_login_that_fails_writes_nothing(
    isolated_home, patch_which, patch_run, capsys
) -> None:
    from letify import tools

    patch_which(tools, present=True)
    patch_run(login, result=FakeCompleted(returncode=1))
    assert main(["login", "colab", "colab_a", "--account", "me@example.com", "--no-input"]) == 1
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_logging_in_to_colab_without_uv_says_how_to_get_it(
    isolated_home, patch_which, capsys
) -> None:
    from letify import tools

    patch_which(tools, present=False)
    assert main(["login", "colab", "colab_a", "--account", "me@example.com", "--no-input"]) == 1
    assert "uv was not found" in capsys.readouterr().err


def test_logging_out_takes_the_account_and_its_directory(isolated_home, capsys) -> None:
    # The repository still needs the account, so the reference stays; this machine is what
    # stopped having it.
    main(
        [
            "login",
            "elice",
            "e",
            "--zone-id",
            "z",
            "--machine-id",
            "m",
            "--token",
            "t",
            "--no-input",
        ]
    )
    assert main(["logout", "e"]) == 0
    assert "[e]" not in (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert not (Path.home() / ".letify" / "accounts" / "e").exists()
    project = isolated_home / ".letify" / "config.toml"
    assert "e" in tomllib.loads(project.read_text(encoding="utf-8"))


def test_logging_out_of_an_account_this_machine_never_had_says_so(isolated_home, capsys) -> None:
    assert main(["logout", "nope"]) == 1
    assert "nope" in capsys.readouterr().err


# -- Spec: SSH authentication --------------------------------------------------


def test_a_key_is_generated_only_when_the_configured_one_is_missing(
    isolated_home, patch_run, tmp_path
) -> None:
    # Generating over an existing key would lock the user out of every other machine that
    # already trusts it.
    recorder = patch_run(login, result=FakeCompleted())
    existing = tmp_path / "id_there"
    existing.write_text("key", encoding="utf-8")
    login.ensure_key(str(existing))
    assert [c for c in recorder.commands if c and c[0] == "ssh-keygen"] == []

    missing = tmp_path / "id_absent"
    login.ensure_key(str(missing))
    generated = [c for c in recorder.commands if c and c[0] == "ssh-keygen"]
    assert len(generated) == 1
    assert "ed25519" in generated[0]


def test_the_password_is_typed_once_and_never_written_anywhere(
    isolated_home, patch_run, monkeypatch, tmp_path
) -> None:
    # It installs the key and is then dropped. Nothing about it reaches the file.
    key = tmp_path / "id_letify"
    key.write_text("private", encoding="utf-8")
    (tmp_path / "id_letify.pub").write_text("ssh-ed25519 AAAA me@here", encoding="utf-8")
    monkeypatch.setattr(login, "read_password", lambda prompt: "hunter2")
    recorder = patch_run(login, result=FakeCompleted())
    login.install_key(address="gpu.example.edu", user="researcher", port=22, key_path=str(key))
    sent = " ".join(" ".join(command) for command in recorder.commands)
    assert "hunter2" not in sent
    # The key travels on stdin rather than as an argument, so it is not visible to
    # anything that can list processes on either machine.
    assert recorder.calls[-1]["input"] == "ssh-ed25519 AAAA me@here"
    assert "ssh-ed25519" not in sent


def test_the_key_is_proven_to_work_before_the_alias_is_declared(
    isolated_home, patch_run, capsys
) -> None:
    # Failing here beats failing at the first call, which costs GPU time to find out.
    patch_run(login, result=FakeCompleted(returncode=255, stderr="Permission denied"))
    code = main(
        [
            "login",
            "shell",
            "lab",
            "--address",
            "gpu.example.edu",
            "--key",
            "~/.ssh/id_letify",
            "--no-input",
        ]
    )
    assert code == 1
    assert not (Path.home() / ".letify" / "config.toml").exists()
    assert "Permission denied" in capsys.readouterr().err


def test_a_password_only_machine_is_refused_on_windows_where_the_tool_does_not_exist(
    isolated_home, monkeypatch, capsys
) -> None:
    # sshpass has no Windows build, so promising it would be a lie.
    monkeypatch.setattr(login.sys, "platform", "win32")
    code = main(
        [
            "login",
            "shell",
            "lab",
            "--address",
            "gpu.example.edu",
            "--auth",
            "password",
            "--no-input",
        ]
    )
    assert code == 1
    assert "sshpass" in capsys.readouterr().err


def test_the_non_interactive_flag_refuses_to_prompt_rather_than_hanging(
    isolated_home, monkeypatch, capsys
) -> None:
    # A login inside a script must fail loudly instead of blocking on a terminal read.
    def refuse(prompt: str) -> str:
        raise AssertionError("prompted under --no-input")

    monkeypatch.setattr(login, "read_line", refuse)
    monkeypatch.setattr(login, "read_password", refuse)
    assert main(["login", "elice", "e", "--no-input"]) == 1


def test_a_missing_value_is_asked_for_when_prompting_is_allowed(monkeypatch) -> None:
    # The prompt is the interactive half of the same flow the flags drive.
    answers = login.Answers(alias="lab", kind="shell")
    monkeypatch.setattr(login, "read_line", lambda prompt: "typed.example.edu")
    assert login.ask(answers, "address", "Machine address: ") == "typed.example.edu"


def test_an_empty_answer_to_a_required_prompt_is_refused(monkeypatch) -> None:
    answers = login.Answers(alias="lab", kind="shell")
    monkeypatch.setattr(login, "read_line", lambda prompt: "")
    with pytest.raises(letify.ConfigError, match="nothing was entered"):
        login.ask(answers, "address", "Machine address: ")
    # An optional field is simply absent instead.
    assert login.ask(answers, "user", "SSH user: ", required=False) is None


def test_an_unknown_authentication_method_is_refused(isolated_home, capsys) -> None:
    answers = login.Answers(alias="lab", kind="shell", values={"auth": "kerberos"})
    with pytest.raises(letify.ConfigError, match="auth must be one of"):
        login.shell_account(answers)


def test_a_password_machine_keeps_the_password_in_the_account_directory_off_windows(
    isolated_home, monkeypatch
) -> None:
    # The opt-in path for a machine whose administrator forbids key authentication. The
    # password has to be stored for sshpass to feed it, which is exactly why it is not the
    # default.
    monkeypatch.setattr(login.sys, "platform", "linux")
    answers = login.Answers(
        alias="lab",
        kind="shell",
        values={"address": "gpu.example.edu", "auth": "password"},
        token="hunter2",
        interactive=False,
    )
    options = login.shell_account(answers)
    assert "password_keyring" not in options
    password_file = Path.home() / ".letify" / "accounts" / "lab" / "password"
    assert password_file.read_text(encoding="utf-8").strip() == "hunter2"


def test_a_password_machine_with_no_password_is_refused(isolated_home, monkeypatch) -> None:
    monkeypatch.setattr(login.sys, "platform", "linux")
    answers = login.Answers(
        alias="lab",
        kind="shell",
        values={"address": "gpu.example.edu", "auth": "password"},
        interactive=False,
    )
    with pytest.raises(letify.ConfigError, match="needs a password"):
        login.shell_account(answers)


def test_the_optional_fields_reach_the_file_when_they_are_given(isolated_home, patch_run) -> None:
    # A non-default port and a persistent disk are worth recording; a default port is not.
    patch_run(login, result=FakeCompleted())
    assert (
        main(
            [
                "login",
                "shell",
                "lab",
                "--address",
                "gpu.example.edu",
                "--port",
                "2222",
                "--persistent",
                "--no-input",
                "--skip-key-install",
            ]
        )
        == 0
    )
    home_file = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert "port = 2222" in home_file
    assert "persistent = true" in home_file


def test_an_elice_endpoint_is_recorded_only_when_it_is_not_the_default(isolated_home) -> None:
    assert (
        main(
            [
                "login",
                "elice",
                "e",
                "--zone-id",
                "z",
                "--machine-id",
                "m",
                "--token",
                "t",
                "--endpoint",
                "https://portal.example/api",
                "--no-input",
            ]
        )
        == 0
    )
    assert "portal.example" in (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")


def modal_sign_in(monkeypatch, *, returncode: int = 0, writes: bool = True) -> list[dict]:
    """Stand in for Modal's own sign in: record the call and write the token file it names."""
    calls: list[dict] = []

    def run(command, **kwargs):
        calls.append({"command": list(command), **kwargs})
        if writes:
            target = Path(kwargs["env"]["MODAL_CONFIG_PATH"])
            target.write_text('[lab-team]\ntoken_id = "ak-1"\nactive = true\n', encoding="utf-8")
        return FakeCompleted(returncode=returncode)

    monkeypatch.setattr(login.subprocess, "run", run)
    return calls


def test_logging_in_to_modal_runs_its_token_flow_into_the_account_directory(
    isolated_home, patch_which, monkeypatch, capsys
) -> None:
    # Spec "Logging in": modal is never needed on PATH or in the project's environment.
    # letify runs `modal token new` through uv, and the token lands in the account
    # directory because MODAL_CONFIG_PATH points there.
    from letify import tools

    patch_which(tools, present=True)
    calls = modal_sign_in(monkeypatch)
    assert main(["login", "modal", "modal_lab", "--workspace", "lab-team", "--no-input"]) == 0

    [call] = calls
    assert call["command"][:3] == ["/usr/bin/uv", "tool", "run"]
    assert "modal>=1.0,<2" in call["command"]
    assert call["command"][-5:] == ["modal", "token", "new", "--profile", "lab-team"]
    token = Path.home() / ".letify" / "accounts" / "modal_lab" / "modal.toml"
    assert call["env"]["MODAL_CONFIG_PATH"] == str(token)
    assert "MODAL_TOKEN_ID" not in call["env"]
    # Interactive: Modal prints a link and waits for the browser, so nothing is captured.
    assert "capture_output" not in call and "stdout" not in call

    home_file = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert 'workspace = "lab-team"' in home_file
    assert "token" not in home_file
    assert token.is_file()


def test_a_modal_login_without_a_workspace_uses_the_profile_modal_picks(
    isolated_home, patch_which, monkeypatch, capsys
) -> None:
    from letify import tools

    patch_which(tools, present=True)
    calls = modal_sign_in(monkeypatch)
    assert main(["login", "modal", "modal_lab", "--no-input"]) == 0
    assert calls[0]["command"][-3:] == ["modal", "token", "new"]
    assert "workspace" not in (Path.home() / ".letify" / "config.toml").read_text("utf-8")


def test_a_modal_login_that_fails_writes_nothing(
    isolated_home, patch_which, monkeypatch, capsys
) -> None:
    from letify import tools

    patch_which(tools, present=True)
    modal_sign_in(monkeypatch, returncode=1)
    assert main(["login", "modal", "modal_lab", "--no-input"]) == 1
    assert not (Path.home() / ".letify" / "config.toml").exists()
    assert not (Path.home() / ".letify" / "accounts" / "modal_lab" / "modal.toml").exists()


def test_a_modal_login_that_leaves_no_token_writes_nothing(
    isolated_home, patch_which, monkeypatch, capsys
) -> None:
    from letify import tools

    patch_which(tools, present=True)
    modal_sign_in(monkeypatch, writes=False)
    assert main(["login", "modal", "modal_lab", "--no-input"]) == 1
    assert "modal.toml" in capsys.readouterr().err
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_logging_in_to_modal_without_uv_says_how_to_get_it(
    isolated_home, patch_which, capsys
) -> None:
    from letify import tools

    patch_which(tools, present=False)
    assert main(["login", "modal", "modal_lab", "--no-input"]) == 1
    assert "uv was not found" in capsys.readouterr().err


def test_a_kind_with_no_login_says_which_kinds_have_one(isolated_home, capsys) -> None:
    assert main(["login", "vastai", "v", "--no-input"]) == 1
    assert "Known kinds" in capsys.readouterr().err


def test_an_alias_that_cannot_be_an_attribute_is_refused_with_the_fix(
    isolated_home, capsys
) -> None:
    # Providers are reached by attribute access, so a hyphen cannot work.
    assert main(["login", "modal", "lab-a", "--no-input"]) == 1
    assert "lab_a" in capsys.readouterr().err


def test_installing_a_key_whose_public_half_is_missing_says_so(tmp_path) -> None:
    key = tmp_path / "id_letify"
    key.write_text("private", encoding="utf-8")
    with pytest.raises(letify.ConfigError, match="no public key"):
        login.install_key(address="h", user=None, port=22, key_path=str(key))


def test_a_key_that_cannot_be_generated_is_reported_rather_than_assumed(
    patch_run, tmp_path
) -> None:
    patch_run(login, result=FakeCompleted(returncode=1, stderr="ssh-keygen: no space left"))
    with pytest.raises(letify.ConfigError, match="ssh-keygen failed"):
        login.ensure_key(str(tmp_path / "id_absent"))


def test_installing_a_key_that_the_machine_refuses_is_reported(patch_run, tmp_path) -> None:
    key = tmp_path / "id_letify"
    key.write_text("private", encoding="utf-8")
    (tmp_path / "id_letify.pub").write_text("ssh-ed25519 AAAA", encoding="utf-8")
    patch_run(login, result=FakeCompleted(returncode=5, stderr="Permission denied"))
    with pytest.raises(letify.ConfigError, match="Permission denied"):
        login.install_key(address="h", user="u", port=22, key_path=str(key))


def test_forgetting_an_account_with_no_directory_is_not_an_error(isolated_home) -> None:
    # There is nothing to do either way: this machine holds nothing for that account.
    assert login.forget_secret("never-logged-in") is False
