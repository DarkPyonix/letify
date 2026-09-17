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


def test_kaggle_usage_reads_the_weekly_quota_from_the_cookie(fake_kaggle) -> None:
    """Spec "Remaining usage, Kaggle": the quota comes from the cookie, not an API key.

    ``GetAcceleratorQuotaStatistics`` answers the weekly GPU and TPU quota in seconds, which
    the provider turns into hours. No Kaggle CLI and no API token are used anywhere.
    """
    usage = kaggle_provider().usage()

    assert usage.unit == "GPU hours"
    assert usage.used == 11700 / 3600
    assert usage.limit == 30.0
    assert usage.remaining == 30.0 - 11700 / 3600
    assert "2026-09-19T00:00:00Z" in (usage.note or "")
    assert "TPU 0 h used, 20 h left of 20" in (usage.note or "")
    called = [path for path, _ in fake_kaggle.cloud_calls]
    assert any(path.endswith("GetAcceleratorQuotaStatistics") for path in called)


def test_a_failed_quota_call_is_an_infrastructure_error(fake_kaggle, monkeypatch) -> None:
    import letify
    from letify.providers import kaggle as kaggle_module

    def refuse(request, timeout=None):
        raise OSError("connection reset")

    monkeypatch.setattr(kaggle_module, "urlopen", refuse)
    with pytest.raises(letify.RuntimeFailure):
        kaggle_provider().usage()


def test_quota_with_no_gpu_figure_is_refused(fake_kaggle, monkeypatch) -> None:
    from conftest import _Reply

    import letify
    from letify.providers import kaggle as kaggle_module

    def answer(request, timeout=None):
        if request.full_url.endswith("GetAcceleratorQuotaStatistics"):
            return _Reply({"quotaRefreshTime": "2026-09-19T00:00:00Z"})
        return fake_kaggle.urlopen(request, timeout)

    monkeypatch.setattr(kaggle_module, "urlopen", answer)
    with pytest.raises(letify.RuntimeFailure, match="GPU"):
        kaggle_provider().usage()


def test_usage_without_a_cookie_says_to_log_in(isolated_home) -> None:
    import letify

    with pytest.raises(letify.ConfigError, match="login kaggle"):
        kaggle_provider().usage()


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


def test_stopping_deletes_the_kernel_and_cancels_the_run(fake_kaggle) -> None:
    """Spec "Kaggle Jupyter Server session": the run is letify's own, so stop cancels it.

    letify starts the session for the runtime, so ending the runtime cancels the run to
    release the accelerator quota it holds, after deleting the kernel it created.
    """
    provider, runtime, _channel = session_channel(fake_kaggle)
    assert len(fake_kaggle.kernels) == 1
    provider.stop(runtime)
    assert fake_kaggle.kernels == set()
    assert len(fake_kaggle.made("DELETE", "/api/kernels/")) == 1
    assert fake_kaggle.cancelled == [fake_kaggle.RUN_ID]


def test_kaggle_never_opts_out_of_preparing_the_runtime(fake_kaggle) -> None:
    from conftest import provider_of

    from letify.providers import Kaggle

    provider = provider_of(Kaggle, "kaggle_a")
    assert provider.prepares_workspace is True
    assert provider.remote_env is True
