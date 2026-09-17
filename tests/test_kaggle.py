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
    fake_kaggle, kaggle_cli, capsys
) -> None:
    # The URL has to name a live session now, because a --connect login reads its GPUs.
    kaggle_cli()
    url = fake_kaggle.url
    code = main(
        ["login", "kaggle", "kaggle_b", "--token", ACCESS_TOKEN, "--connect", url, "--no-input"]
    )
    assert code == 0
    stored = account("kaggle_b") / "jupyter_url"
    assert stored.read_text(encoding="utf-8").strip() == url
    if sys.platform != "win32":
        assert stored.stat().st_mode & 0o777 == 0o600
    assert fake_kaggle.token not in (Path.home() / ".letify" / "config.toml").read_text(
        encoding="utf-8"
    )
    captured = capsys.readouterr()
    assert fake_kaggle.token not in captured.out + captured.err


def test_a_session_url_that_is_not_http_is_refused(isolated_home, kaggle_cli, capsys) -> None:
    kaggle_cli()
    code = main(
        ["login", "kaggle", "kaggle_a", "--token", ACCESS_TOKEN, "--connect", "kkb", "--no-input"]
    )
    assert code == 1
    assert "http" in capsys.readouterr().err
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_a_new_session_url_on_a_declared_account_replaces_only_the_url(
    fake_kaggle, kaggle_cli
) -> None:
    recorder = kaggle_cli()
    assert main(["login", "kaggle", "kaggle_b", "--token", ACCESS_TOKEN, "--no-input"]) == 0
    calls = len(recorder.calls)
    url = fake_kaggle.url
    assert main(["login", "kaggle", "kaggle_b", "--connect", url, "--no-input"]) == 0
    assert (account("kaggle_b") / "jupyter_url").read_text(encoding="utf-8").strip() == url
    assert (account("kaggle_b") / "access_token").read_text(encoding="utf-8") == ACCESS_TOKEN
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


# -- Spec: Remaining usage, Kaggle -------------------------------------------------

QUOTA = """Warning: Looks like you're using an outdated API Version
[
  {"resource": "GPU", "used": "3.25h", "remaining": "26.75h", "total": "30.00h",
   "refreshAt": "2026-09-19T00:00:00+00:00"},
  {"resource": "TPU", "used": "0.00h", "remaining": "20.00h", "total": "20.00h",
   "refreshAt": "2026-09-19T00:00:00+00:00"}
]
"""


def kaggle_provider():
    from conftest import provider_of

    from letify.providers import Kaggle

    return provider_of(Kaggle, "kaggle_a")


def test_a_kaggle_account_is_a_known_provider_kind() -> None:
    from letify.providers import KINDS, Kaggle

    assert KINDS["kaggle"] is Kaggle


def test_kaggle_usage_reads_the_weekly_gpu_quota(isolated_home, patch_which, patch_run) -> None:
    from letify import tools
    from letify.providers import kaggle as kaggle_module

    patch_which(tools, present=True)
    recorder = patch_run(kaggle_module, result=FakeCompleted(stdout=QUOTA))
    usage = kaggle_provider().usage()

    assert usage.unit == "GPU hours"
    assert usage.used == 3.25
    assert usage.remaining == 26.75
    assert usage.limit == 30.0
    assert "2026-09-19T00:00:00+00:00" in (usage.note or "")
    assert "TPU 0 h used, 20 h left of 20" in (usage.note or "")
    call = recorder.calls[-1]
    assert call["command"][-3:] == ["quota", "--format", "json"]
    assert call["env"]["KAGGLE_CONFIG_DIR"] == str(account("kaggle_a"))


def test_a_failed_quota_call_is_an_infrastructure_error_without_the_token(
    isolated_home, patch_which, patch_run
) -> None:
    import letify
    from letify import tools
    from letify.config.secrets import write_secret
    from letify.providers import kaggle as kaggle_module

    write_secret("kaggle_a", "access_token", ACCESS_TOKEN)
    patch_which(tools, present=True)
    patch_run(kaggle_module, result=FakeCompleted(returncode=1, stderr=f"401 {ACCESS_TOKEN}"))
    with pytest.raises(letify.RuntimeFailure) as caught:
        kaggle_provider().usage()
    assert ACCESS_TOKEN not in str(caught.value)
    assert "***" in str(caught.value)


def test_quota_output_with_no_gpu_row_is_refused(isolated_home, patch_which, patch_run) -> None:
    import letify
    from letify import tools
    from letify.providers import kaggle as kaggle_module

    patch_which(tools, present=True)
    patch_run(kaggle_module, result=FakeCompleted(stdout="No quota information available\n"))
    with pytest.raises(letify.RuntimeFailure, match="GPU"):
        kaggle_provider().usage()


def test_every_kaggle_call_names_this_accounts_token_file_in_kaggle_api_token(
    isolated_home, patch_which, patch_run, monkeypatch
) -> None:
    # The Kaggle CLI does not read access_token from KAGGLE_CONFIG_DIR, so the variable is what
    # authenticates. It holds the file's path, which keeps the token out of the environment.
    from letify import tools
    from letify.config.secrets import write_secret
    from letify.providers import kaggle as kaggle_module

    monkeypatch.setenv("KAGGLE_API_TOKEN", "another-accounts-token")
    path = write_secret("kaggle_a", "access_token", ACCESS_TOKEN)
    patch_which(tools, present=True)
    recorder = patch_run(kaggle_module, result=FakeCompleted(stdout=QUOTA))
    kaggle_provider().usage()
    env = recorder.calls[-1]["env"]
    assert env["KAGGLE_API_TOKEN"] == str(path)
    assert ACCESS_TOKEN not in env.values()


# -- Spec: Placements a provider cannot serve --------------------------------------


def test_declaring_host_local_on_kaggle_fails_at_decoration(let) -> None:
    import letify

    device = kaggle_provider().P100
    with pytest.raises(letify.UnsupportedMode, match="host='remote'"):

        @let.function(device=device)
        def body() -> None: ...

    with pytest.raises(letify.UnsupportedMode):
        let.function(device=device, host=letify.local)(lambda: None)
    declared = let.function(device=device, host=letify.remote)(lambda: None)
    assert declared.device.placement == "remote"


def test_an_any_request_resolving_to_kaggle_with_host_local_is_refused() -> None:
    import letify

    provider = kaggle_provider()
    with pytest.raises(letify.UnsupportedMode):
        provider.check_mode(provider.T4._placed("local"))
    assert provider.check_mode(provider.T4._placed("remote")) is None


def test_the_generated_types_mark_kaggle_accelerators_remote_only(isolated_home) -> None:
    import letify
    from letify import stubs

    (isolated_home / ".letify" / "config.toml").write_text('[kg]\nkind = "kaggle"\n')
    text = stubs.render(letify.Launcher(announce=False))
    body = text.split("class Kg(")[1].split("\nclass ")[0]
    assert "    P100: letify.declare.instance.RemoteOnlyInstance" in body
    assert ": Instance" not in body


TYPED = """\
import letify
from letify.declare.instance import Instance, RemoteOnlyInstance

let = letify.Launcher()
kaggle: RemoteOnlyInstance = RemoteOnlyInstance(None)  # type: ignore[arg-type]
lab: Instance = Instance(None)  # type: ignore[arg-type]


@let.function(device=kaggle, host=letify.remote)
def fine() -> None: ...


@let.function(device=lab)
def also_fine() -> None: ...


@let.function(device=kaggle, host=letify.local)
def wrong() -> None: ...


@let.function(device=kaggle)
def wrong_by_default() -> None: ...
"""


def test_a_type_checker_rejects_host_local_on_a_remote_only_instance(tmp_path) -> None:
    import os
    import shutil
    import subprocess

    import letify

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed, so no type checker can be run")
    snippet = tmp_path / "declare.py"
    snippet.write_text(TYPED, encoding="utf-8")
    checkout = str(Path(letify.__file__).resolve().parent.parent)
    command = [uv, "tool", "run", "--offline", "mypy", "--no-incremental", str(snippet)]
    checked = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "MYPYPATH": checkout},
        timeout=300,
    )
    if checked.returncode not in (0, 1) or "declare.py" not in checked.stdout:
        pytest.skip(f"mypy could not be run offline: {checked.stderr.strip()[-300:]}")
    # mypy also reports on letify's own modules it follows into; only the snippet is asserted on.
    errors = [line for line in checked.stdout.splitlines() if line.startswith("declare.py:")]
    lines = sorted({int(line.split(":")[1]) for line in errors if ": error:" in line})
    assert lines == [17, 21], checked.stdout


# -- Spec: Kaggle Jupyter Server session -------------------------------------------


def session_channel(fake_kaggle, name: str = "letify-t4-1"):
    provider = kaggle_provider()
    runtime = type("R", (), {"name": name})()
    return provider, runtime, provider.open_channel(runtime)


def test_a_registered_session_carries_one_worker_for_the_whole_runtime(fake_kaggle) -> None:
    """Spec "Kaggle Jupyter Server session": one worker in one cell, not one per program.

    A one-shot channel starts a fresh process per request, so nothing it builds survives.
    The worker the session keeps is what gives Kaggle an object table, a blob table that
    holds a large argument across calls, and project data detection, all of which the spec
    grants to a persistent channel and to nothing else.
    """
    provider, _runtime, channel = session_channel(fake_kaggle)
    assert channel.persistent is True
    assert provider.persistent_channel is True

    # State the first request leaves behind has to be there for the second one.
    channel.request({"op": "exec", "source": "LETIFY_KEPT = 6 * 7\n"})
    value, _logs = channel.request({"op": "eval", "source": "__letify_value__ = LETIFY_KEPT\n"})
    assert value == 42

    # One kernel for the runtime, however many requests crossed it.
    assert len(fake_kaggle.made("POST", "/api/kernels")) == 1


def test_a_program_runs_on_the_registered_session_in_the_kernel_letify_created(
    fake_kaggle,
) -> None:
    _provider, _runtime, channel = session_channel(fake_kaggle)
    value, _logs = channel.request({"op": "eval", "source": "__letify_value__ = 6 * 7"})
    assert value == 42

    created = fake_kaggle.made("POST", "/api/kernels")
    assert len(created) == 1
    runs = [json.loads(line) for line in fake_kaggle.log.read_text().splitlines()]
    assert {run["kernel"] for run in runs} == fake_kaggle.kernels
    assert all(fake_kaggle.token not in " ".join(run["argv"]) for run in runs)


def test_every_rest_request_carries_the_session_token_both_ways(fake_kaggle) -> None:
    session_channel(fake_kaggle)
    token = fake_kaggle.token
    assert fake_kaggle.requests
    assert all(r["query"].get("token") == token for r in fake_kaggle.requests)
    assert all(r["authorization"] == f"token {token}" for r in fake_kaggle.requests)


def test_an_account_with_no_registered_session_refuses_the_call(isolated_home) -> None:
    from letify.errors import ConfigError

    provider = kaggle_provider()
    with pytest.raises(ConfigError) as caught:
        provider.open_channel(type("R", (), {"name": "n", "instance": provider.CPU})())
    message = str(caught.value)
    assert "kaggle_a" in message
    assert "--connect" in message


def test_an_ended_session_at_start_says_how_to_register_a_new_one(fake_kaggle) -> None:
    import letify
    from letify.providers.kaggle import KaggleSessionEnded

    fake_kaggle.end_session()
    with pytest.raises(KaggleSessionEnded) as caught:
        session_channel(fake_kaggle)
    message = str(caught.value)
    assert isinstance(caught.value, letify.RuntimeLost)
    assert "Run, Kaggle Jupyter Server" in message
    assert "letify login kaggle kaggle_a --connect" in message
    assert "20 minutes" in message and "12 hour" in message
    assert fake_kaggle.token not in message
    assert "/k/123/proxy" not in message


def test_a_session_that_ends_under_a_running_worker_is_a_lost_runtime(fake_kaggle) -> None:
    """Spec "Kaggle Jupyter Server session": the bridge exiting is the worker dying.

    There is no "between programs" on a persistent channel. A session Kaggle ends takes the
    cell and the worker with it, so every blocked read ends as a lost channel does, and the
    failure names the runtime rather than the call that happened to be in flight.
    """
    import letify

    _provider, _runtime, channel = session_channel(fake_kaggle)
    channel.request({"op": "exec", "source": "LETIFY_ALIVE = True\n"})

    fake_kaggle.end_session()
    channel._kill()
    with pytest.raises((letify.RuntimeLost, letify.RuntimeFailure)):
        channel.request({"op": "eval", "source": "__letify_value__ = LETIFY_ALIVE\n"})


def test_a_program_that_raises_on_a_live_session_carries_its_own_traceback(fake_kaggle) -> None:
    """The worker reports the user's error, and the session is untouched by it.

    A one-shot adapter could only say that the program exited non zero. A worker answers
    with the exception itself, so user code failing is a RemoteError with its traceback
    rather than an infrastructure failure, and the next request still works.
    """
    import letify
    from letify.providers.kaggle import KaggleSessionEnded

    _provider, _runtime, channel = session_channel(fake_kaggle)
    with pytest.raises(letify.RemoteError) as caught:
        channel.request({"op": "exec", "source": "1 / 0"})
    assert not isinstance(caught.value, KaggleSessionEnded)
    assert "ZeroDivisionError" in str(caught.value)

    # The worker survived the user's error, which is the point of reporting it this way.
    value, _logs = channel.request({"op": "eval", "source": "__letify_value__ = 6 * 7\n"})
    assert value == 42


def test_files_move_as_worker_requests_rather_than_through_the_contents_api(
    fake_kaggle, tmp_path
) -> None:
    """Spec "Kaggle Jupyter Server session": the three file ops are worker requests.

    The Jupyter contents API was the substitute a one-shot channel needed, because a
    channel with no worker has nobody to ask. A worker serves them itself, so the bytes
    travel as frames and no contents request is made at all.
    """
    _provider, _runtime, channel = session_channel(fake_kaggle)
    target = tmp_path / "session" / "weights.bin"
    payload = bytes(range(256)) * 4

    value, _ = channel.request({"op": "put_file", "path": str(target), "payload": payload})
    assert value == {"path": str(target), "size": len(payload)}
    assert target.read_bytes() == payload

    back, _ = channel.request({"op": "get_file", "path": str(target)})
    assert bytes(back["payload"]) == payload
    assert not fake_kaggle.made("PUT", "/api/contents/")


def test_stopping_deletes_the_kernel_and_leaves_the_session_running(fake_kaggle) -> None:
    provider, runtime, _channel = session_channel(fake_kaggle)
    assert len(fake_kaggle.kernels) == 1
    provider.stop(runtime)
    assert fake_kaggle.kernels == set()
    assert len(fake_kaggle.made("DELETE", "/api/kernels/")) == 1
    assert fake_kaggle.made("GET", "/api/status")


# -- Spec: Recording devices at login for Kaggle -----------------------------------


def fake_smi(tmp_path, monkeypatch, output: str | None) -> None:
    """Put a fake nvidia-smi, or none, on the PATH the session programs see."""
    folder = tmp_path / "session-bin"
    folder.mkdir()
    if output is not None:
        tool = folder / "nvidia-smi"
        tool.write_text(f"#!/bin/sh\nprintf '{output}'\n", encoding="utf-8")
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(folder))


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
def test_a_connect_login_records_a_count_per_accelerator(
    fake_kaggle, kaggle_cli, tmp_path, monkeypatch
) -> None:
    kaggle_cli()
    fake_smi(tmp_path, monkeypatch, "0, Tesla T4, 15360 MiB\\n1, Tesla T4, 15360 MiB\\n")
    code = main(
        [
            "login",
            "kaggle",
            "kaggle_b",
            "--token",
            ACCESS_TOKEN,
            "--connect",
            fake_kaggle.url,
            "--no-input",
        ]
    )
    assert code == 0
    assert home_config()["kaggle_b"]["devices"] == {"T4": {"count": 2}}
    assert fake_kaggle.kernels == set()


def test_a_connect_login_to_a_cpu_session_writes_no_devices_table(
    fake_kaggle, kaggle_cli, tmp_path, monkeypatch, capsys
) -> None:
    kaggle_cli()
    fake_smi(tmp_path, monkeypatch, None)
    code = main(
        [
            "login",
            "kaggle",
            "kaggle_b",
            "--token",
            ACCESS_TOKEN,
            "--connect",
            fake_kaggle.url,
            "--no-input",
        ]
    )
    assert code == 0
    assert "devices" not in home_config()["kaggle_b"]
    assert "nvidia-smi" in capsys.readouterr().out


def test_a_connect_login_to_an_ended_session_writes_nothing(
    fake_kaggle, kaggle_cli, capsys
) -> None:
    kaggle_cli()
    fake_kaggle.end_session()
    code = main(
        [
            "login",
            "kaggle",
            "kaggle_b",
            "--token",
            ACCESS_TOKEN,
            "--connect",
            fake_kaggle.url,
            "--no-input",
        ]
    )
    assert code == 1
    assert "letify login kaggle kaggle_b --connect" in capsys.readouterr().err
    assert not (account("kaggle_b") / "access_token").exists()
    assert not (account("kaggle_b") / "jupyter_url").exists()


def test_kaggle_never_opts_out_of_preparing_the_runtime(fake_kaggle) -> None:
    from conftest import provider_of

    from letify.providers import Kaggle

    provider = provider_of(Kaggle, "kaggle_a")
    assert provider.prepares_workspace is True
    assert provider.remote_env is True
