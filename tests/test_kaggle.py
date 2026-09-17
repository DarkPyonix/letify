"""The Kaggle account: login, the token files and the environment the Kaggle CLI runs in.

Spec sections pinned here: "Logging in" (the kaggle paragraphs) and "What each kind asks for".
The Kaggle CLI needs a live account, so subprocess.run is answered by the recorder in
conftest.py and the test asserts on the command, the environment and the files written.
"""

from __future__ import annotations

import base64
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


COOKIE_EXP = "2099-01-01T00:00:00Z"


def make_client_token(exp_iso: str) -> str:
    def part(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    header = part({"alg": "none", "typ": "JWT"})
    return f"{header}.{part({'sub': 'irack000', 'exp': exp_iso})}."


def make_cookie(exp_iso: str = COOKIE_EXP, drop: list[str] | None = None) -> str:
    jar = {
        "ka_sessionid": "sid",
        "XSRF-TOKEN": "xtok",
        "__Host-KAGGLEID": "kid",
        "build-hash": "bh",
        "CLIENT-TOKEN": make_client_token(exp_iso),
    }
    for name in drop or []:
        jar.pop(name, None)
    return "; ".join(f"{name}={value}" for name, value in jar.items())


@pytest.fixture
def accept_cookie(monkeypatch):
    """Answer the online cookie check with a display name, and record the cookies checked."""
    seen: list[str] = []

    def fake(cookie: str) -> str:
        seen.append(cookie)
        return "\ubb38\ucc44\uc6b4 (IRACK)"

    monkeypatch.setattr("letify.providers.kaggle.verify_cookie", fake)
    return seen


def test_a_cookie_is_kept_owner_only_and_never_in_the_config(
    isolated_home, accept_cookie, capsys
) -> None:
    cookie = make_cookie()
    assert main(["login", "kaggle", "kaggle_a", "--cookie", cookie, "--no-input"]) == 0

    stored = account("kaggle_a") / "cookie"
    assert stored.read_text(encoding="utf-8").strip() == cookie
    if sys.platform != "win32":
        assert stored.stat().st_mode & 0o777 == 0o600
    assert home_config()["kaggle_a"] == {"kind": "kaggle"}
    text = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert "CLIENT-TOKEN" not in text
    out = capsys.readouterr()
    assert "ka_sessionid" not in out.out + out.err
    assert accept_cookie == [cookie]


def test_an_expired_cookie_is_refused_with_nothing_written(
    isolated_home, accept_cookie, capsys
) -> None:
    expired = make_cookie("2000-01-01T00:00:00Z")
    assert main(["login", "kaggle", "kaggle_a", "--cookie", expired, "--no-input"]) == 1
    assert "expired" in capsys.readouterr().err
    assert not (Path.home() / ".letify" / "config.toml").exists()
    assert not (account("kaggle_a") / "cookie").exists()
    assert accept_cookie == []  # an expired cookie is never sent to be checked


def test_a_cookie_missing_a_required_name_is_refused(isolated_home, accept_cookie, capsys) -> None:
    partial = make_cookie(drop=["ka_sessionid"])
    assert main(["login", "kaggle", "kaggle_a", "--cookie", partial, "--no-input"]) == 1
    assert "missing" in capsys.readouterr().err
    assert not (account("kaggle_a") / "cookie").exists()


def test_a_cookie_the_account_check_rejects_writes_nothing(
    isolated_home, monkeypatch, capsys
) -> None:
    def reject(cookie: str) -> str:
        raise ValueError("the Kaggle cookie was refused; log in to kaggle.com for a fresh one")

    monkeypatch.setattr("letify.providers.kaggle.verify_cookie", reject)
    assert main(["login", "kaggle", "kaggle_a", "--cookie", make_cookie(), "--no-input"]) == 1
    assert "refused" in capsys.readouterr().err
    assert not (account("kaggle_a") / "cookie").exists()
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_no_input_without_a_cookie_refuses(isolated_home, accept_cookie, capsys) -> None:
    assert main(["login", "kaggle", "kaggle_a", "--no-input"]) == 1
    assert "--cookie" in capsys.readouterr().err


def test_with_a_terminal_the_cookie_is_asked_for_hidden(
    isolated_home, accept_cookie, monkeypatch
) -> None:
    cookie = make_cookie()
    prompts: list[str] = []

    def hidden(prompt: str) -> str:
        prompts.append(prompt)
        return cookie

    monkeypatch.setattr(login, "read_password", hidden)
    monkeypatch.setattr(login, "read_line", lambda prompt: pytest.fail("asked in the clear"))
    assert main(["login", "kaggle", "kaggle_a"]) == 0
    assert prompts == [login.KAGGLE_COOKIE_PROMPT]
    assert (account("kaggle_a") / "cookie").is_file()


def test_a_cookie_can_be_read_from_a_file(isolated_home, accept_cookie, tmp_path) -> None:
    cookie = make_cookie()
    path = tmp_path / "kaggle_cookie.txt"
    path.write_text(cookie, encoding="utf-8")
    assert main(["login", "kaggle", "kaggle_a", "--cookie", str(path), "--no-input"]) == 0
    assert (account("kaggle_a") / "cookie").read_text(encoding="utf-8").strip() == cookie


def test_a_kaggle_login_records_the_workspace(isolated_home, accept_cookie) -> None:
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", make_cookie(),
         "--workspace", "/kaggle/working/letify", "--no-input"]
    ) == 0
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


def test_the_account_note_reports_days_left_on_the_cookie(isolated_home) -> None:
    """Spec "Kaggle account": the listing shows how many days the cookie has left."""
    from letify.config.secrets import write_secret

    write_secret("kaggle_a", "cookie", make_cookie())  # far-future expiry
    assert "cookie expires in" in (kaggle_provider().account_note() or "")


def test_the_account_note_flags_a_missing_cookie(isolated_home) -> None:
    assert "no cookie" in (kaggle_provider().account_note() or "")


def test_the_account_note_flags_an_expired_cookie(isolated_home) -> None:
    from letify.config.secrets import write_secret

    write_secret("kaggle_a", "cookie", make_cookie("2000-01-01T00:00:00Z"))
    assert "EXPIRED" in (kaggle_provider().account_note() or "")


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


def test_a_session_that_does_not_route_is_not_reported_as_having_ended(fake_kaggle) -> None:
    """Spec "Kaggle Jupyter Server session": the message does not state that the session ended.

    Kaggle's proxy answers 404 for an ended session, for a URL that no longer routes to a
    live one, and for a session id that never existed. A status read that is not 200
    therefore cannot tell those apart, so a message asserting the session ended states as
    fact something letify has not established. Here the server is running throughout: only
    the registered URL points somewhere it does not serve.
    """
    from letify.config.secrets import write_secret
    from letify.providers.kaggle import KaggleSessionEnded

    write_secret("kaggle_a", "jupyter_url", fake_kaggle.url.replace("/k/123/", "/k/999/"))
    with pytest.raises(KaggleSessionEnded) as caught:
        session_channel(fake_kaggle)
    message = str(caught.value)

    assert "has ended" not in message
    assert "did not answer" in message
    assert "no longer route" in message
    # The remedy is the same either way, so it is still spelled out.
    assert "letify login kaggle kaggle_a --connect" in message
    assert "20 minutes" in message and "12 hour" in message


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


def test_kaggle_never_opts_out_of_preparing_the_runtime(fake_kaggle) -> None:
    from conftest import provider_of

    from letify.providers import Kaggle

    provider = provider_of(Kaggle, "kaggle_a")
    assert provider.prepares_workspace is True
    assert provider.remote_env is True
