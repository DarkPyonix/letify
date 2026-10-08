"""The Kaggle account: the cookie login, the token chain and the session channel.

Spec sections pinned here: "Kaggle account", "Kaggle Jupyter Server session" and "Remaining
usage, Kaggle". A live Kaggle account, its Firebase and Firestore, and a GPU are the only
things a fake stands in for, in ``conftest.FakeKaggleCloud``; the token chain, the channel
and the worker are the real code.
"""

from __future__ import annotations

import base64
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from letify.cli import main
from letify.config import login


def home_config() -> dict:
    return tomllib.loads((Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8"))


def account(alias: str) -> Path:
    return Path.home() / ".letify" / "accounts" / alias


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


#: The API token arguments every successful login test needs now that the token is
#: mandatory, alongside the cookie.
API_TOKEN = "KGAT_" + "a" * 32
TOKEN_ARGS = ["--username", "irack000", "--key", API_TOKEN]


def test_a_cookie_and_a_token_are_kept_owner_only_and_never_in_the_config(
    isolated_home, accept_cookie, capsys
) -> None:
    cookie = make_cookie()
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", cookie, *TOKEN_ARGS, "--no-input"]
    ) == 0

    stored = account("kaggle_a") / "cookie"
    assert stored.read_text(encoding="utf-8").strip() == cookie
    token_path = account("kaggle_a") / "api_token"
    assert token_path.read_text(encoding="utf-8") == API_TOKEN
    owner_path = account("kaggle_a") / "username"
    assert owner_path.read_text(encoding="utf-8") == "irack000"
    assert not (account("kaggle_a") / "kaggle.json").exists()
    if sys.platform != "win32":
        assert stored.stat().st_mode & 0o777 == 0o600
        assert token_path.stat().st_mode & 0o777 == 0o600
        assert owner_path.stat().st_mode & 0o777 == 0o600
        assert account("kaggle_a").stat().st_mode & 0o777 == 0o700
    assert home_config()["kaggle_a"] == {"kind": "kaggle"}
    text = (Path.home() / ".letify" / "config.toml").read_text(encoding="utf-8")
    assert "CLIENT-TOKEN" not in text
    assert API_TOKEN not in text
    out = capsys.readouterr()
    assert "ka_sessionid" not in out.out + out.err
    assert API_TOKEN not in out.out + out.err
    assert accept_cookie == [cookie]


def test_a_login_missing_the_api_token_is_refused_with_nothing_written(
    isolated_home, accept_cookie, capsys
) -> None:
    """Spec "Kaggle account": the cookie alone cannot declare the account any more, since
    deleting the notebook and reading the quota both need the official CLI's token."""
    cookie = make_cookie()
    assert main(["login", "kaggle", "kaggle_a", "--cookie", cookie, "--no-input"]) == 1
    assert "API token" in capsys.readouterr().err
    assert not (account("kaggle_a") / "cookie").exists()
    assert not (Path.home() / ".letify" / "config.toml").exists()
    assert accept_cookie == []  # the cookie is never checked once the token is missing


def test_a_login_missing_only_the_key_is_refused(isolated_home, accept_cookie, capsys) -> None:
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", make_cookie(), "--username", "irack000",
         "--no-input"]
    ) == 1
    assert "API token" in capsys.readouterr().err
    assert not (account("kaggle_a") / "cookie").exists()


def test_an_expired_cookie_is_refused_with_nothing_written(
    isolated_home, accept_cookie, capsys
) -> None:
    expired = make_cookie("2000-01-01T00:00:00Z")
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", expired, *TOKEN_ARGS, "--no-input"]
    ) == 1
    assert "expired" in capsys.readouterr().err
    assert not (Path.home() / ".letify" / "config.toml").exists()
    assert not (account("kaggle_a") / "cookie").exists()
    assert accept_cookie == []  # an expired cookie is never sent to be checked


def test_a_cookie_missing_a_required_name_is_refused(isolated_home, accept_cookie, capsys) -> None:
    partial = make_cookie(drop=["ka_sessionid"])
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", partial, *TOKEN_ARGS, "--no-input"]
    ) == 1
    assert "missing" in capsys.readouterr().err
    assert not (account("kaggle_a") / "cookie").exists()


def test_a_cookie_the_account_check_rejects_writes_nothing(
    isolated_home, monkeypatch, capsys
) -> None:
    def reject(cookie: str) -> str:
        raise ValueError("the Kaggle cookie was refused; log in to kaggle.com for a fresh one")

    monkeypatch.setattr("letify.providers.kaggle.verify_cookie", reject)
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", make_cookie(), *TOKEN_ARGS, "--no-input"]
    ) == 1
    assert "refused" in capsys.readouterr().err
    assert not (account("kaggle_a") / "cookie").exists()
    assert not (Path.home() / ".letify" / "config.toml").exists()


def test_no_input_without_a_cookie_refuses(isolated_home, accept_cookie, capsys) -> None:
    assert main(["login", "kaggle", "kaggle_a", *TOKEN_ARGS, "--no-input"]) == 1
    assert "--cookie" in capsys.readouterr().err


def test_with_a_terminal_the_cookie_and_token_are_asked_for(
    isolated_home, accept_cookie, monkeypatch
) -> None:
    cookie = make_cookie()
    hidden_prompts: list[str] = []
    line_prompts: list[str] = []

    def hidden(prompt: str) -> str:
        hidden_prompts.append(prompt)
        return cookie if prompt == login.KAGGLE_COOKIE_PROMPT else API_TOKEN

    def line(prompt: str) -> str:
        line_prompts.append(prompt)
        return "irack000"

    monkeypatch.setattr(login, "read_password", hidden)
    monkeypatch.setattr(login, "read_line", line)
    assert main(["login", "kaggle", "kaggle_a"]) == 0
    # The token is asked for first, because it comes from a page rather than a tab.
    assert hidden_prompts == [login.KAGGLE_KEY_PROMPT, login.KAGGLE_COOKIE_PROMPT]
    assert line_prompts == [login.KAGGLE_USERNAME_PROMPT]
    assert (account("kaggle_a") / "cookie").is_file()
    assert (account("kaggle_a") / "api_token").is_file()


def test_a_token_pasted_with_the_kgat_prefix_is_stored_verbatim(
    isolated_home, accept_cookie
) -> None:
    """Spec "Kaggle account": kaggle.com shows the token prefixed, the CLI needs it intact."""
    thirty_two = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", make_cookie(),
         "--username", "irack000", "--key", "KGAT_" + thirty_two,
         "--no-input"]
    ) == 0
    stored = (account("kaggle_a") / "api_token").read_text(encoding="utf-8")
    assert stored == "KGAT_" + thirty_two


def test_a_cookie_can_be_read_from_a_file(isolated_home, accept_cookie, tmp_path) -> None:
    cookie = make_cookie()
    path = tmp_path / "kaggle_cookie.txt"
    path.write_text(cookie, encoding="utf-8")
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", str(path), *TOKEN_ARGS, "--no-input"]
    ) == 0
    assert (account("kaggle_a") / "cookie").read_text(encoding="utf-8").strip() == cookie


def test_a_kaggle_login_records_the_workspace(isolated_home, accept_cookie) -> None:
    assert main(
        ["login", "kaggle", "kaggle_a", "--cookie", make_cookie(), *TOKEN_ARGS,
         "--workspace", "/kaggle/working/letify", "--no-input"]
    ) == 0
    assert home_config()["kaggle_a"]["workspace"] == "/kaggle/working/letify"


# -- Spec: Remaining usage, Kaggle -------------------------------------------------


def kaggle_provider():
    from conftest import provider_of

    from letify.providers import Kaggle

    return provider_of(Kaggle, "kaggle_a")


def test_a_kaggle_account_is_a_known_provider_kind() -> None:
    from letify.providers import KINDS, Kaggle

    assert KINDS["kaggle"] is Kaggle


def test_the_account_note_reports_the_exp_claim_without_checking_it_live(isolated_home) -> None:
    """Spec "Kaggle account": the listing reads only the exp claim, with no network call,
    and says so, since that claim is not proof the cookie still works."""
    from letify.config.secrets import write_secret

    write_secret("kaggle_a", "cookie", make_cookie())  # far-future expiry
    note = kaggle_provider().account_note() or ""
    assert "exp claim says" in note
    assert "not checked live" in note


def test_the_account_note_flags_a_missing_cookie(isolated_home) -> None:
    assert "no cookie" in (kaggle_provider().account_note() or "")


def test_a_run_refuses_to_start_with_under_an_hour_left_on_the_cookie(isolated_home) -> None:
    """Spec "Kaggle account": starting a session the cookie cannot outlive is refused.

    This is the offline exp check, so it raises before any network call is attempted.
    """
    from datetime import UTC, datetime, timedelta

    import letify
    from letify.config.secrets import write_secret

    soon = datetime.now(UTC) + timedelta(minutes=30)
    write_secret("kaggle_a", "cookie", make_cookie(soon.strftime("%Y-%m-%dT%H:%M:%SZ")))
    runtime = type("R", (), {"name": "letify-t4-1"})()
    with pytest.raises(letify.ConfigError, match="hour"):
        kaggle_provider().open_channel(runtime)


def test_the_account_note_flags_an_expired_cookie(isolated_home) -> None:
    from letify.config.secrets import write_secret

    write_secret("kaggle_a", "cookie", make_cookie("2000-01-01T00:00:00Z"))
    assert "EXPIRED" in (kaggle_provider().account_note() or "")


def test_kaggle_usage_reads_the_weekly_quota_from_the_cli(
    fake_kaggle_cli, kaggle_api_token
) -> None:
    """Spec "Remaining usage, Kaggle": the quota comes from the official CLI's ``quota``
    command, never the cookie, since reading it is not part of the interactive session.
    """
    usage = kaggle_provider().usage()

    assert usage.unit == "GPU hours"
    assert usage.used == 0.0
    assert usage.limit == 60.0
    assert usage.remaining == 60.0
    assert usage.resets_at is None
    assert usage.resources == ({"name": "TPU", "unit": "TPU hours", "used": 0.0,
                                "remaining": 20.0, "limit": 20.0, "resets_at": None},)
    assert "TPU 0 h used, 20 h left of 20" in (usage.note or "")
    quota_calls = [call for call in fake_kaggle_cli.calls if "quota" in call]
    assert len(quota_calls) == 1


def test_a_failed_quota_cli_call_reports_unknown_rather_than_an_error(
    fake_kaggle_cli, kaggle_api_token
) -> None:
    """A CLI call that fails is not an infrastructure error and is not answered from the
    cookie instead: it is simply nothing to report."""
    fake_kaggle_cli.fail.add("quota")
    usage = kaggle_provider().usage()
    assert usage.remaining is None
    assert "could not be read" in (usage.note or "")


def test_an_unrecognized_quota_shape_reports_unknown(fake_kaggle_cli, kaggle_api_token) -> None:
    """Spec "Remaining usage, Kaggle": unexpected shapes are unread, never guessed at."""
    fake_kaggle_cli.quota_json = json.dumps({"unexpected": "shape"})
    usage = kaggle_provider().usage()
    assert usage.remaining is None
    assert "could not be read" in (usage.note or "")


def test_usage_without_a_token_says_to_log_in_again(isolated_home) -> None:
    import letify
    from letify.config.secrets import write_secret

    write_secret("kaggle_a", "cookie", make_cookie())
    with pytest.raises(letify.ConfigError, match="login kaggle"):
        kaggle_provider().usage()


def test_quota_and_notebook_deletion_never_touch_the_cookie_api(
    fake_kaggle, fake_kaggle_cli
) -> None:
    """Spec "Kaggle account": the credential boundary is locked here, not just described.

    Neither reading the quota nor deleting the notebook is allowed to fall back to a
    cookie-authenticated internal call, so both are asserted against the cloud's own call
    log, not just against what the CLI fake was asked to do.
    """
    provider, runtime, _channel = session_channel(fake_kaggle)
    fake_kaggle.cloud_calls.clear()
    provider.usage()
    provider.stop(runtime)
    forbidden = {"GetAcceleratorQuotaStatistics", "DeleteKernel"}
    touched = {path.rsplit("/", 1)[-1] for path, _ in fake_kaggle.cloud_calls}
    assert not (touched & forbidden)


def test_a_session_is_refused_the_same_way_an_already_dead_cookie_is(
    fake_kaggle, monkeypatch
) -> None:
    """The same online check guards starting a session, not only reading usage."""
    from conftest import _Reply

    import letify
    from letify.providers import kaggle as kaggle_module

    def answer(request, timeout=None):
        if request.full_url.endswith("GetCurrentUser"):
            return _Reply({})
        return fake_kaggle.urlopen(request, timeout)

    monkeypatch.setattr(kaggle_module, "urlopen", answer)
    with pytest.raises(letify.ConfigError, match="Kaggle no longer accepts"):
        session_channel(fake_kaggle)


def test_the_online_check_is_not_repeated_within_the_liveness_ttl(fake_kaggle) -> None:
    """Spec "Kaggle account": one GetCurrentUser call covers several session starts made
    close together, not one per start."""
    provider = kaggle_provider()
    runtime_a = type("R", (), {"name": "letify-t4-a"})()
    runtime_b = type("R", (), {"name": "letify-t4-b"})()
    provider.open_channel(runtime_a)
    provider.open_channel(runtime_b)

    checks = [path for path, _body in fake_kaggle.cloud_calls if path.endswith("GetCurrentUser")]
    assert len(checks) == 1


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


def test_a_fresh_notebook_is_created_for_every_run_not_reused(fake_kaggle) -> None:
    """Spec "Kaggle session token chain": no notebook id is kept across runs.

    Reusing a notebook on disk was what let a notebook deleted on kaggle.com wedge the
    account. A fresh notebook is created for each run instead, so there is nothing to go
    stale and nothing kept in the account directory.
    """
    from letify.config.secrets import account_directory

    session_channel(fake_kaggle, name="letify-t4-1")
    session_channel(fake_kaggle, name="letify-t4-2")

    created = [path for path, _body in fake_kaggle.cloud_calls if path.endswith("WithSettings")]
    assert len(created) == 2, "each run creates its own notebook"
    assert not (account_directory("kaggle_a") / "notebook_id").exists()


def test_a_refused_start_is_raised_without_retrying(fake_kaggle) -> None:
    """Spec "Kaggle session token chain": a refusal on a notebook created moments ago for
    this run alone is the account's own answer, so it is raised rather than retried."""
    from letify.errors import RuntimeFailure

    fake_kaggle.refuse["GetOrCreateKernelSession"] = (403, "Permission denied")
    with pytest.raises(RuntimeFailure) as raised:
        session_channel(fake_kaggle)
    assert "403" in str(raised.value)
    created = [path for path, _body in fake_kaggle.cloud_calls if path.endswith("WithSettings")]
    assert len(created) == 1


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


def test_the_worker_source_reaches_the_bridge_when_a_pipe_write_returns_short(
    fake_kaggle, monkeypatch
) -> None:
    """Spec "Kaggle Jupyter Server session": a line the pipe took only part of is continued.

    The bridge's standard input is an unbuffered pipe. A blocking write into a full pipe
    returns the count it managed when a signal arrives while it waits, and SIGCHLD arrives
    exactly then in a full suite run: PyTorch leaves a SIGCHLD handler in this process once
    a DataLoader has forked workers, and an earlier test's worker exits while the 160 KB
    worker source is going in. A channel that takes the count for the whole line hands the
    bridge a source with no end, and the worker never says hello.
    """
    import subprocess

    from letify.runtime import channel as channel_module

    class ShortWriting:
        """A raw pipe that takes at most 4 KiB per write, as one interrupted by a signal does."""

        def __init__(self, raw: Any):
            self._raw = raw

        def write(self, data: Any) -> int:
            return self._raw.write(memoryview(data)[:4096])

        def __getattr__(self, name: str) -> Any:
            return getattr(self._raw, name)

    real_popen = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        if kwargs.get("bufsize") == 0 and process.stdin is not None:
            process.stdin = ShortWriting(process.stdin)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    # Only so the run without the fix fails in seconds rather than in the startup timeout.
    monkeypatch.setattr(channel_module, "STARTUP_TIMEOUT", 5.0)
    _provider, _runtime, channel = session_channel(fake_kaggle)
    value, _logs = channel.request({"op": "eval", "source": "__letify_value__ = 6 * 7"})
    assert value == 42


def test_a_session_is_started_with_internet_access(fake_kaggle) -> None:
    """Spec "Kaggle Jupyter Server session": the run asks for internet access.

    A Kaggle session has no network unless the run asks for it, and the environment step
    then fails downloading from PyPI, which is how a live run on develop died.
    """
    session_channel(fake_kaggle)
    runs = [body for path, body in fake_kaggle.cloud_calls if path.endswith("CommitAndRun")]
    assert len(runs) == 1
    assert runs[0]["compute"]["internet"] == {"isEnabled": True}


def test_a_refused_kaggle_call_names_the_status_and_the_message(fake_kaggle) -> None:
    """Spec "Kaggle session token chain": a refused call says what Kaggle answered.

    An error that says only "HTTPError" hides whether the notebook's session is taken, the
    cookie is stale or the account lacks a right, which are three different next steps.
    """
    from letify.errors import RuntimeFailure

    fake_kaggle.refuse["GetOrCreateKernelSession"] = (409, "Kernel session already running")
    with pytest.raises(RuntimeFailure) as raised:
        session_channel(fake_kaggle)
    text = str(raised.value)
    assert "GetOrCreateKernelSession" in text
    assert "409" in text
    assert "Kernel session already running" in text
    assert "ka_sessionid" not in text


def test_every_rest_request_carries_the_session_token_in_the_path_not_a_header(
    fake_kaggle,
) -> None:
    """Spec "Kaggle Jupyter Server session": the routed URL carries the token in its path.

    The proxy rejects a token sent only as an Authorization header, so the token rides in the
    URL path. Every REST request the provider makes therefore carries the path token and no
    ``token`` Authorization header.
    """
    session_channel(fake_kaggle)
    assert fake_kaggle.requests
    assert all(r["path_token"] == fake_kaggle.token for r in fake_kaggle.requests)
    assert all(r["authorization"] is None for r in fake_kaggle.requests)


def test_an_account_with_no_cookie_refuses_the_call(isolated_home) -> None:
    from letify.errors import ConfigError

    provider = kaggle_provider()
    with pytest.raises(ConfigError) as caught:
        provider.open_channel(type("R", (), {"name": "n", "instance": provider.CPU})())
    message = str(caught.value)
    assert "kaggle_a" in message
    assert "login kaggle" in message


def test_an_ended_session_at_start_says_to_run_again(fake_kaggle) -> None:
    import letify
    from letify.providers.kaggle import KaggleSessionEnded

    fake_kaggle.end_session()
    with pytest.raises(KaggleSessionEnded) as caught:
        session_channel(fake_kaggle)
    message = str(caught.value)
    assert isinstance(caught.value, letify.RuntimeLost)
    assert "did not answer" in message
    assert "20 minutes" in message and "12 hour" in message
    assert "Run the function again" in message
    assert fake_kaggle.token not in message


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


def test_a_bridge_that_dies_before_hello_reports_its_standard_error(
    fake_kaggle, monkeypatch
) -> None:
    """Spec "Kaggle Jupyter Server session": a worker death is reported with the bridge's
    standard error.

    The bridge is a subprocess whose standard error is a pipe the channel owns. What it
    prints before exiting, such as uv failing to resolve the kernel client or the kernel
    refusing the cell, is the only account of why the worker never said hello, so the
    failure has to carry it. A report that says only that the worker stopped leaves the
    user, and the next reader of a flaky suite, with nothing to act on.
    """
    from letify.errors import RuntimeFailure

    monkeypatch.setenv("FAKE_KAGGLE_DIE", "the bridge refused to start: no kernel client")
    _provider, _runtime, channel = session_channel(fake_kaggle)
    with pytest.raises(RuntimeFailure) as raised:
        channel.request({"op": "eval", "source": "__letify_value__ = 1"})
    assert "the bridge refused to start: no kernel client" in str(raised.value)


def test_a_worker_killed_by_the_startup_watchdog_says_so(fake_kaggle, monkeypatch) -> None:
    """Spec "Kaggle Jupyter Server session", Failure: a startup timeout is reported as one.

    When the bridge never says hello, the watchdog kills it, the bridge's shim then dies on
    an empty read, and a report that shows only that traceback reads as a crash inside an
    answering session. The report has to name the timeout, because the next step is a
    slower machine or a longer wait, not a look at the session.
    """
    from letify.errors import RuntimeFailure
    from letify.runtime import channel as channel_module

    monkeypatch.setenv("FAKE_KAGGLE_HANG", "1")
    monkeypatch.setattr(channel_module, "STARTUP_TIMEOUT", 1.0)
    _provider, _runtime, channel = session_channel(fake_kaggle)
    with pytest.raises(RuntimeFailure) as raised:
        channel.request({"op": "eval", "source": "__letify_value__ = 1"})
    message = str(raised.value)
    assert "did not say hello within 1 s" in message
    assert "the bridge is waiting on the proxy" in message


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


def test_stopping_cancels_the_run_through_the_cookie_and_deletes_the_notebook_via_cli(
    fake_kaggle, fake_kaggle_cli
) -> None:
    """Spec "Kaggle Jupyter Server session": the run is letify's own, so stop cancels it
    through the cookie, the cookie's last job. The notebook created for the run is then
    deleted through the official CLI alone; the cookie is never asked to delete anything.
    """
    provider, runtime, _channel = session_channel(fake_kaggle)
    assert len(fake_kaggle.kernels) == 1
    provider.stop(runtime)
    assert fake_kaggle.kernels == set()
    assert len(fake_kaggle.made("DELETE", "/api/kernels/")) == 1
    assert fake_kaggle.cancelled == [fake_kaggle.RUN_ID]

    deletes = [call for call in fake_kaggle_cli.calls if "delete" in call]
    assert len(deletes) == 1
    assert deletes[0][-2:] == ["-y", "irack000/letify-runtime"]
    assert "kernels" in deletes[0]


def test_a_notebook_is_warned_about_by_name_when_the_cli_delete_fails(
    fake_kaggle, fake_kaggle_cli, capsys
) -> None:
    """Spec "Kaggle session token chain": a failed CLI delete is reported with a warning
    naming the notebook, never its cookie, and never retried against the cookie."""
    fake_kaggle_cli.fail.add("delete")
    provider, runtime, _channel = session_channel(fake_kaggle)
    provider.stop(runtime)

    err = capsys.readouterr().err
    assert "letify-runtime" in err
    assert "delete it by hand" in err
    assert fake_kaggle.token not in err


def test_a_notebook_without_a_token_is_warned_about_not_deleted_via_the_cookie(
    fake_kaggle, fake_kaggle_cli
) -> None:
    """An account declared before the token became required cannot delete its notebook
    through the CLI, and the cookie never substitutes for it."""
    (account("kaggle_a") / "api_token").unlink()
    provider, runtime, _channel = session_channel(fake_kaggle)
    provider.stop(runtime)
    assert fake_kaggle_cli.calls == []


def test_kaggle_never_opts_out_of_preparing_the_runtime(fake_kaggle) -> None:
    from conftest import provider_of

    from letify.providers import Kaggle

    provider = provider_of(Kaggle, "kaggle_a")
    assert provider.prepares_workspace is True
    assert provider.remote_env is True


@pytest.mark.parametrize("token", ["a" * 32, "KGAT_", "wrong_token", "KGAT_has space",
                                  " KGAT_value", "KGAT_value\n"])
def test_a_malformed_api_token_is_refused_before_cookie_verification(
    isolated_home, accept_cookie, capsys, token
) -> None:
    """Spec "Kaggle": login refuses malformed tokens without reformatting them."""
    assert main(["login", "kaggle", "kaggle_a", "--cookie", make_cookie(),
                 "--username", "irack000", "--key", token, "--no-input"]) == 1
    assert "KGAT_" in capsys.readouterr().err
    assert not account("kaggle_a").exists()
    assert not (Path.home() / ".letify" / "config.toml").exists()
    assert accept_cookie == []


def test_every_kaggle_cli_call_receives_only_the_account_token_in_the_environment(
    fake_kaggle_cli, kaggle_api_token, monkeypatch
) -> None:
    """Spec "Kaggle": credentials stay in the environment for quota and deletion."""
    from letify.config.secrets import write_secret
    from letify.providers.kaggle import delete_notebook_via_cli, run_cli_quota

    monkeypatch.setenv("KAGGLE_API_TOKEN", "unrelated-parent-token")
    write_secret("other", "api_token", "KGAT_other")
    write_secret("other", "username", "other_owner")
    assert run_cli_quota("kaggle_a") == fake_kaggle_cli.quota_json
    assert delete_notebook_via_cli("kaggle_a", "notebook63516d2758") is True
    assert run_cli_quota("other") == fake_kaggle_cli.quota_json
    assert len(fake_kaggle_cli.calls) == 3
    assert fake_kaggle_cli.calls[1][-1] == "irack000/notebook63516d2758"
    for command, environment, token in zip(fake_kaggle_cli.calls,
                                         fake_kaggle_cli.environments,
                                         [API_TOKEN, API_TOKEN, "KGAT_other"], strict=True):
        assert environment["KAGGLE_API_TOKEN"] == token
        assert "KAGGLE_CONFIG_DIR" not in environment
        assert all(token not in argument for argument in command)


def test_the_fake_cli_refuses_missing_token_environment(fake_kaggle_cli) -> None:
    """Spec "Kaggle": the fake must enforce the CLI authentication boundary."""
    assert fake_kaggle_cli.run(["kaggle", "quota"], env={}).returncode != 0


def test_quota_uses_reported_hour_values_without_guessing_the_allowance(
    fake_kaggle_cli, kaggle_api_token
) -> None:
    """Spec "Remaining usage": remaining is read directly from the real schema."""
    fake_kaggle_cli.quota_json = json.dumps([
        {"resource": "GPU", "used": "3.25h", "remaining": "56.74h", "total": "60.00h",
         "refreshAt": "2026-10-10T00:00:00"},
    ])
    usage = kaggle_provider().usage()
    assert usage.used == 3.25
    assert usage.remaining == 56.74
    assert usage.limit == 60.0


@pytest.mark.parametrize("output", ["not JSON", "null", "[]",
    '[{"name":"GPU","totalTimeAllowed":"108000s","timeUsed":"0s"}]',
    *[json.dumps([{"resource": "GPU", "used": used, "remaining": "60.00h",
                   "total": "60.00h", "refreshAt": "2026-10-10T00:00:00"}])
      for used in ["0s", 0, "NaNh", "infh", "-1h"]],
    '[{"resource":"GPU","used":"0h","remaining":"60h","total":"60h"}]',
    '[{"resource":"GPU","used":"0h","remaining":"60h","total":"60h",'
    '"refreshAt":"bad date"}]',
])
def test_unreadable_quota_answers_report_unknown(
    fake_kaggle_cli, kaggle_api_token, output
) -> None:
    """Spec "Remaining usage": unreadable answers do not raise or guess a value."""
    fake_kaggle_cli.quota_json = output
    usage = kaggle_provider().usage()
    assert usage.remaining is None
    assert "could not be read" in (usage.note or "")


def test_the_quota_parser_reads_the_measured_cli_list(fake_kaggle_cli) -> None:
    """Spec "Remaining usage": parse the measured schema independently of credentials."""
    from letify.providers.kaggle import _parse_cli_quota

    parsed = _parse_cli_quota(fake_kaggle_cli.quota_json)
    assert parsed is not None
    assert parsed["gpu"] == {"used": 0.0, "remaining": 60.0, "total": 60.0}
    assert parsed["tpu"] == {"used": 0.0, "remaining": 20.0, "total": 20.0}


def test_a_token_body_is_not_restricted_to_the_measured_length(
    isolated_home, accept_cookie
) -> None:
    """Spec "Kaggle": prefix and shape validation does not impose an unproven length."""
    assert main(["login", "kaggle", "kaggle_a", "--cookie", make_cookie(),
                 "--username", "irack000", "--key", "KGAT_synthetic", "--no-input"]) == 0
    assert (account("kaggle_a") / "api_token").read_text() == "KGAT_synthetic"
