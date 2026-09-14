"""The Kaggle account: login, the token files and the environment the Kaggle CLI runs in.

Spec sections pinned here: "Logging in" (the kaggle paragraphs) and "What each kind asks for".
The Kaggle CLI needs a live account, so subprocess.run is answered by the recorder in
conftest.py and the test asserts on the command, the environment and the files written.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import FakeCompleted

from letify.cli import main
from letify.config import login

ACCESS_TOKEN = "KGAT_0123456789abcdef"
LEGACY = {"username": "researcher", "key": "0123456789abcdef0123456789abcdef"}


def home_config() -> dict:
    return tomllib.loads((Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8"))


def account(alias: str) -> Path:
    return Path.home() / ".letify" / "accounts" / alias


@pytest.fixture
def kaggle_cli(patch_which, patch_run):
    """uv on PATH, and the Kaggle CLI answered with ``result``."""
    from letify import tools

    patch_which(tools, present=True)

    def answer(result: FakeCompleted | None = None):
        return patch_run(login, result=result or FakeCompleted(stdout="[]\n"))

    return answer


# -- Spec: Logging in, kaggle ------------------------------------------------------


def test_an_access_token_is_kept_in_the_account_directory_and_checked_read_only(
    isolated_home, kaggle_cli, capsys
) -> None:
    recorder = kaggle_cli()
    code = main(["login", "kaggle", "kaggle_a", "--token", ACCESS_TOKEN, "--no-input"])
    assert code == 0

    token = account("kaggle_a") / "access_token"
    assert token.read_text(encoding="utf-8").strip() == ACCESS_TOKEN
    if sys.platform != "win32":
        assert token.stat().st_mode & 0o777 == 0o600

    call = recorder.calls[-1]
    assert call["command"][:3] == ["/usr/bin/uv", "tool", "run"]
    assert "kaggle" in call["command"]
    assert call["command"][-3:] == ["quota", "--format", "json"]
    assert ACCESS_TOKEN not in " ".join(call["command"])
    assert call["env"]["KAGGLE_API_TOKEN"] == str(token)
    assert call["env"]["KAGGLE_CONFIG_DIR"] == str(account("kaggle_a"))
    assert call["env"]["HOME"] == str(account("kaggle_a"))

    assert home_config()["kaggle_a"] == {"kind": "kaggle"}
    text = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert ACCESS_TOKEN not in text
    captured = capsys.readouterr()
    assert ACCESS_TOKEN not in captured.out + captured.err


def test_a_legacy_kaggle_json_given_as_text_is_written_as_kaggle_json(
    isolated_home, kaggle_cli
) -> None:
    recorder = kaggle_cli()
    code = main(["login", "kaggle", "kaggle_a", "--token", json.dumps(LEGACY), "--no-input"])
    assert code == 0
    written = account("kaggle_a") / "kaggle.json"
    assert json.loads(written.read_text(encoding="utf-8")) == LEGACY
    if sys.platform != "win32":
        assert written.stat().st_mode & 0o777 == 0o600
    assert not (account("kaggle_a") / "access_token").exists()
    assert "KAGGLE_API_TOKEN" not in recorder.calls[-1]["env"]


def test_a_legacy_kaggle_json_given_as_a_path_is_read_from_that_file(
    isolated_home, kaggle_cli, tmp_path
) -> None:
    kaggle_cli()
    downloaded = tmp_path / "Downloads" / "kaggle.json"
    downloaded.parent.mkdir()
    downloaded.write_text(json.dumps(LEGACY), encoding="utf-8")
    assert main(["login", "kaggle", "kaggle_a", "--token", str(downloaded), "--no-input"]) == 0
    written = account("kaggle_a") / "kaggle.json"
    assert json.loads(written.read_text(encoding="utf-8")) == LEGACY


def test_a_kaggle_json_without_a_key_is_refused_with_nothing_written(
    isolated_home, kaggle_cli, capsys
) -> None:
    recorder = kaggle_cli()
    body = json.dumps({"username": "researcher"})
    assert main(["login", "kaggle", "kaggle_a", "--token", body, "--no-input"]) == 1
    assert "key" in capsys.readouterr().err
    assert recorder.calls == []
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_credentials_of_another_account_in_the_environment_are_not_passed_on(
    isolated_home, kaggle_cli, monkeypatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "someone-else")
    monkeypatch.setenv("KAGGLE_KEY", "their-key")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "their-token")
    recorder = kaggle_cli()
    assert main(["login", "kaggle", "kaggle_a", "--token", json.dumps(LEGACY), "--no-input"]) == 0
    env = recorder.calls[-1]["env"]
    assert "KAGGLE_USERNAME" not in env
    assert "KAGGLE_KEY" not in env
    assert "KAGGLE_API_TOKEN" not in env


def test_a_token_the_kaggle_cli_rejects_writes_nothing_and_hides_the_token(
    isolated_home, kaggle_cli, capsys
) -> None:
    kaggle_cli(FakeCompleted(returncode=1, stderr=f"401 Unauthorized for token {ACCESS_TOKEN}"))
    assert main(["login", "kaggle", "kaggle_a", "--token", ACCESS_TOKEN, "--no-input"]) == 1
    err = capsys.readouterr().err
    assert "exited 1" in err
    assert "401 Unauthorized" in err
    assert ACCESS_TOKEN not in err
    assert "***" in err
    assert not (account("kaggle_a") / "access_token").exists()
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_no_input_without_a_token_refuses(isolated_home, kaggle_cli, capsys) -> None:
    recorder = kaggle_cli()
    assert main(["login", "kaggle", "kaggle_a", "--no-input"]) == 1
    assert "--token" in capsys.readouterr().err
    assert recorder.calls == []


def test_with_a_terminal_the_token_is_asked_for_with_hidden_input(
    isolated_home, kaggle_cli, monkeypatch
) -> None:
    kaggle_cli()
    prompts: list[str] = []

    def hidden(prompt: str) -> str:
        prompts.append(prompt)
        return ACCESS_TOKEN

    monkeypatch.setattr(login, "read_password", hidden)
    monkeypatch.setattr(login, "read_line", lambda prompt: pytest.fail("asked in the clear"))
    assert main(["login", "kaggle", "kaggle_a"]) == 0
    assert prompts == ["Kaggle API token, or the path to kaggle.json: "]
    assert (account("kaggle_a") / "access_token").is_file()


def test_logging_in_to_kaggle_without_uv_says_how_to_get_it(
    isolated_home, patch_which, capsys
) -> None:
    from letify import tools

    patch_which(tools, present=False)
    assert main(["login", "kaggle", "kaggle_a", "--token", ACCESS_TOKEN, "--no-input"]) == 1
    assert "uv was not found" in capsys.readouterr().err


def test_the_session_url_is_kept_owner_only_and_never_in_the_config(
    isolated_home, kaggle_cli, capsys
) -> None:
    kaggle_cli()
    url = "https://kkb-production.jupyter-proxy.kaggle.net/k/1/abc/proxy?token=session-secret"
    code = main(
        ["login", "kaggle", "kaggle_a", "--token", ACCESS_TOKEN, "--connect", url, "--no-input"]
    )
    assert code == 0
    stored = account("kaggle_a") / "jupyter_url"
    assert stored.read_text(encoding="utf-8").strip() == url
    if sys.platform != "win32":
        assert stored.stat().st_mode & 0o777 == 0o600
    assert "session-secret" not in (Path.home() / ".letify" / "config.toml").read_text(
        encoding="utf-8"
    )
    captured = capsys.readouterr()
    assert "session-secret" not in captured.out + captured.err


def test_a_session_url_that_is_not_http_is_refused(isolated_home, kaggle_cli, capsys) -> None:
    kaggle_cli()
    code = main(
        ["login", "kaggle", "kaggle_a", "--token", ACCESS_TOKEN, "--connect", "kkb", "--no-input"]
    )
    assert code == 1
    assert "http" in capsys.readouterr().err
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_a_new_session_url_on_a_declared_account_replaces_only_the_url(
    isolated_home, kaggle_cli
) -> None:
    recorder = kaggle_cli()
    assert main(["login", "kaggle", "kaggle_a", "--token", ACCESS_TOKEN, "--no-input"]) == 0
    calls = len(recorder.calls)
    url = "https://kkb-production.jupyter-proxy.kaggle.net/k/2/def/proxy?token=next"
    assert main(["login", "kaggle", "kaggle_a", "--connect", url, "--no-input"]) == 0
    assert (account("kaggle_a") / "jupyter_url").read_text(encoding="utf-8").strip() == url
    assert (account("kaggle_a") / "access_token").read_text(encoding="utf-8") == ACCESS_TOKEN
    assert len(recorder.calls) == calls


def test_a_kaggle_login_records_the_workspace_without_a_remote_check(
    isolated_home, kaggle_cli
) -> None:
    kaggle_cli()
    code = main(
        [
            "login",
            "kaggle",
            "kaggle_a",
            "--token",
            ACCESS_TOKEN,
            "--workspace",
            "/kaggle/working/letify",
            "--no-input",
        ]
    )
    assert code == 0
    assert home_config()["kaggle_a"]["workspace"] == "/kaggle/working/letify"
