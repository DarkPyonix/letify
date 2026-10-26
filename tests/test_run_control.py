"""Controlling a running call: run ids, interrupt, exec, Ctrl-C and exit hooks.

Spec section pinned here: "Controlling a running call" (id run-control). Every test that
can goes through the Local provider, which starts the same worker behind the same framed
protocol a remote runtime uses, so the worker ops and the owner's control socket are the
real ones. The Local provider shares the file system with the test, so a body and the test
talk through marker files.
"""

from __future__ import annotations

import _thread
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import letify
from letify import paths
from letify.cli import main
from letify.protocol import wire
from letify.runtime import control

LOCAL = '[a]\nkind = "local"\n'


@pytest.fixture
def worker_sees_this_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the Local worker import the letify under test, not another installed checkout."""
    root = str(Path(letify.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", root)


@pytest.fixture
def short_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A directory with a short path, because a Unix socket path holds about 100 bytes.

    A test's own temporary directory is longer than that when the suite runs in parallel.
    It is the runtime directory the control socket falls back to.
    """
    directory = Path(tempfile.mkdtemp(prefix="lt-"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(directory))
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def let(isolated_home, launcher_from, worker_sees_this_checkout, short_dir) -> letify.Launcher:
    return launcher_from(LOCAL)


def until(predicate: Callable[[], Any], what: str, timeout: float = 30.0) -> Any:
    """Poll until ``predicate`` is truthy and return its value, or fail naming ``what``."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


class Running:
    """A declared function called on a thread of its own, so the test stays free."""

    def __init__(self, function: Callable[..., Any], *args: Any) -> None:
        self.value: Any = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(function, args), daemon=True)
        self._thread.start()

    def _run(self, function: Callable[..., Any], args: tuple) -> None:
        try:
            self.value = function(*args)
        except BaseException as exc:
            self.error = exc

    def join(self, timeout: float = 30.0) -> None:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the call did not end"


def the_run() -> str:
    """The one live run id, once it is listed."""
    records = until(control.list_runs, "a run record")
    assert len(records) == 1
    return records[0]["run"]


# -- run ids and the registry ---------------------------------------------------


def test_a_running_call_is_listed_by_runs_and_gone_after_it_ends(let, tmp_path, capsys) -> None:
    stop = str(tmp_path / "stop")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def spin(stop_file: str) -> str:
        import os
        import time

        while not os.path.exists(stop_file):
            time.sleep(0.02)
        return "done"

    call = Running(spin, stop)
    run = the_run()
    assert len(run) == 12 and set(run) <= set("0123456789abcdef")

    assert main(["runs", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["run"] for row in rows] == [run]
    row = rows[0]
    assert row["function"].endswith("spin")
    assert row["provider"] == "a"
    assert row["pid"] == os.getpid()
    assert time.time() - 60 < row["started_at"] <= time.time()
    assert row["runtime"]

    runs_dir = paths.letify_root() / "runs"
    record = runs_dir / f"{run}.json"
    assert stat.S_IMODE(runs_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert Path(row["socket"]).exists()
    assert row["socket"] == str(control.socket_path(os.getpid()))

    assert main(["runs"]) == 0
    assert run in capsys.readouterr().out

    Path(stop).write_text("x", encoding="utf-8")
    call.join()
    assert call.value == "done"
    assert not record.exists()
    assert control.list_runs() == []
    assert main(["runs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_a_record_whose_owner_is_gone_is_removed_when_read(isolated_home, short_dir) -> None:
    runs_dir = control.runs_directory()
    runs_dir.mkdir(parents=True, mode=0o700)
    record = runs_dir / "abcdef012345.json"
    record.write_text(
        json.dumps(
            {
                "run": "abcdef012345",
                "function": "f",
                "runtime": "r",
                "provider": "a",
                "started_at": time.time(),
                "pid": 2**22 + 12345,
                "socket": str(runs_dir / "x.sock"),
            }
        ),
        encoding="utf-8",
    )
    assert control.list_runs() == []
    assert not record.exists()
    assert main(["interrupt", "abcdef012345"]) == 3


def test_a_long_home_puts_the_control_socket_in_the_ssh_control_directory(
    monkeypatch, tmp_path, short_dir
) -> None:
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    monkeypatch.setenv("LETIFY_HOME", str(deep))
    chosen = control.socket_path(4321)
    assert chosen.parent == paths.ssh_control_directory()
    assert chosen.name == "run-4321.sock"
    assert len(os.fsencode(chosen)) <= 100

    monkeypatch.delenv("LETIFY_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: short_dir))
    assert control.socket_path(4321) == short_dir / ".letify" / "runs" / "4321.sock"


def test_unknown_run_ids_exit_with_status_three(isolated_home, capsys) -> None:
    assert main(["interrupt", "000000000000"]) == 3
    assert main(["exec", "000000000000", "1 + 1"]) == 3
    assert main(["interrupt", "000000000000", "--json"]) == 3
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["error"]


def test_exec_needs_source_from_the_argument_or_a_file(isolated_home) -> None:
    assert main(["exec", "000000000000"]) == 2
    with pytest.raises(SystemExit) as stopped:
        main(["exec"])
    assert stopped.value.code == 2


def test_a_run_on_a_one_shot_channel_is_refused_naming_the_provider(
    isolated_home, short_dir, capsys
) -> None:
    class Provider:
        alias = "kaggle_a"

    class OneShot:
        name = "kaggle-1"
        provider = Provider()
        persistent_channel = False

    run = control.begin(OneShot(), "train")
    try:
        assert main(["interrupt", run.id]) == 1
        assert "UnsupportedMode" in capsys.readouterr().err
        assert main(["exec", run.id, "1", "--json"]) == 1
        answer = json.loads(capsys.readouterr().out)
        assert answer["type"] == "UnsupportedMode"
        assert "kaggle_a" in answer["error"]
    finally:
        run.end()
    assert control.list_runs() == []


# -- interrupt ------------------------------------------------------------------


def test_interrupt_raises_keyboard_interrupt_in_the_body_and_its_finally_block_runs(
    let, tmp_path, capsys
) -> None:
    ready, cleaned = str(tmp_path / "ready"), str(tmp_path / "cleaned")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def spin(ready_file: str, cleaned_file: str) -> None:
        import time

        try:
            Path(ready_file).touch()
            while True:
                time.sleep(0.02)
        finally:
            Path(cleaned_file).write_text("finally ran")

    call = Running(spin, ready, cleaned)
    run = the_run()
    until(lambda: os.path.exists(ready), "the body to start")
    assert main(["interrupt", run]) == 0
    assert run in capsys.readouterr().out
    call.join()
    assert isinstance(call.error, letify.RemoteError)
    assert "KeyboardInterrupt" in str(call.error)
    assert Path(cleaned).read_text(encoding="utf-8") == "finally ran"


def test_interrupt_wakes_a_body_blocked_in_a_long_sleep(let, tmp_path) -> None:
    ready = str(tmp_path / "ready")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def sleeper(ready_file: str) -> None:
        import time

        Path(ready_file).touch()
        time.sleep(300)

    call = Running(sleeper, ready)
    run = the_run()
    until(lambda: os.path.exists(ready), "the body to start")
    started = time.monotonic()
    assert main(["interrupt", run, "--json"]) == 0
    call.join(20)
    assert "KeyboardInterrupt" in str(call.error)
    assert time.monotonic() - started < 20


def test_a_body_that_catches_the_interrupt_keeps_running(let, tmp_path) -> None:
    ready, caught, stop = (str(tmp_path / n) for n in ("ready", "caught", "stop"))

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def stubborn(ready_file: str, caught_file: str, stop_file: str) -> str:
        import os
        import time

        Path(ready_file).touch()
        while not os.path.exists(stop_file):
            try:
                time.sleep(0.02)
            except KeyboardInterrupt:
                Path(caught_file).write_text("caught")
        return "still here"

    call = Running(stubborn, ready, caught, stop)
    run = the_run()
    until(lambda: os.path.exists(ready), "the body to start")
    assert main(["interrupt", run]) == 0
    until(lambda: os.path.exists(caught), "the body to catch it")
    assert call.error is None and control.list_runs()
    Path(stop).write_text("x", encoding="utf-8")
    call.join()
    assert call.value == "still here"


def test_interrupt_with_no_call_running_says_nothing_is_running(let, tmp_path) -> None:
    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def quick() -> int:
        return 1

    with let.keep_alive():
        assert quick() == 1
        runtime = let.pool.live[0]
        answer = runtime.request({"op": "interrupt"})
    assert answer["interrupted"] is False
    assert "nothing is running" in answer["reason"]


# -- exec -----------------------------------------------------------------------


def spin_with_counter(let):
    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def counting(ready_file: str, stop_file: str) -> int:
        import os
        import time

        step = 0
        while not os.path.exists(stop_file):
            step += 1
            Path(ready_file).touch()
            time.sleep(0.01)
        return step

    return counting


def test_exec_reads_a_local_variable_of_the_running_loop(let, tmp_path, capsys) -> None:
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")
    call = Running(spin_with_counter(let), ready, stop)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")

    answer = control.ask(run, {"cmd": "exec", "source": "L['step']"})
    assert answer["ok"] and answer["has_value"]
    assert int(answer["value"]) > 0
    named = control.ask(run, {"cmd": "exec", "source": "frame.f_code.co_name"})
    assert named["value"] == "'counting'"
    chain = control.ask(run, {"cmd": "exec", "source": "[f.f_code.co_name for f in frames]"})
    assert chain["value"] == "['counting']"
    assert control.ask(run, {"cmd": "exec", "source": "'ready_file' in G or 'ready_file' in L"})

    assert main(["exec", run, "L['step']"]) == 0
    assert capsys.readouterr().out.strip().isdigit()
    Path(stop).write_text("x", encoding="utf-8")
    call.join()
    assert call.value > 0


def test_exec_mutating_an_object_changes_the_running_loop(let, tmp_path) -> None:
    ready = str(tmp_path / "ready")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def watch(ready_file: str) -> str:
        import time
        import types

        config = types.SimpleNamespace(stop=False)
        Path(ready_file).touch()
        while not config.stop:
            time.sleep(0.02)
        return "stopped by exec"

    call = Running(watch, ready)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")
    answer = control.ask(run, {"cmd": "exec", "source": "L['config'].stop = True"})
    assert answer["ok"] and not answer["has_value"] and answer["error"] is None
    call.join()
    assert call.value == "stopped by exec"


@pytest.mark.skipif(sys.version_info < (3, 13), reason="f_locals writes through from 3.13")
def test_exec_assigning_a_name_reaches_the_frame_on_python_3_13(let, tmp_path) -> None:
    ready = str(tmp_path / "ready")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def flagged(ready_file: str) -> str:
        import time

        done = False
        Path(ready_file).touch()
        while not done:
            time.sleep(0.02)
        return "assigned"

    call = Running(flagged, ready)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")
    answer = control.ask(run, {"cmd": "exec", "source": "L['done'] = True"})
    assert answer["assigned"] == ["done"] and answer["applied"] is True
    call.join()
    assert call.value == "assigned"


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="f_locals writes through from 3.13")
def test_exec_assigning_a_name_is_reported_as_not_applied_before_python_3_13(let, tmp_path) -> None:
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")
    call = Running(spin_with_counter(let), ready, stop)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")
    answer = control.ask(run, {"cmd": "exec", "source": "L['step'] = 10**9"})
    assert answer["assigned"] == ["step"] and answer["applied"] is False
    Path(stop).write_text("x", encoding="utf-8")
    call.join()
    assert call.value < 10**9


def test_exec_output_is_captured_without_swallowing_the_bodys_own_prints(
    let, tmp_path, capfd
) -> None:
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def chatty(ready_file: str, stop_file: str) -> None:
        import os
        import sys
        import time

        while not os.path.exists(stop_file):
            print("body-tick", flush=True)
            print("body-warn", file=sys.stderr, flush=True)
            Path(ready_file).touch()
            time.sleep(0.02)

    call = Running(chatty, ready, stop)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")
    answer = control.ask(
        run,
        {
            "cmd": "exec",
            "source": "import sys, time\n"
            "print('exec-out')\nprint('exec-err', file=sys.stderr)\ntime.sleep(0.3)\n",
        },
    )
    assert answer["stdout"] == "exec-out\n"
    assert answer["stderr"] == "exec-err\n"
    Path(stop).write_text("x", encoding="utf-8")
    call.join()
    seen = capfd.readouterr()
    assert "body-tick" in seen.out and "body-warn" in seen.err
    assert "exec-out" not in seen.out and "exec-err" not in seen.err


def test_exec_reports_the_traceback_of_source_that_raises(let, tmp_path, capsys) -> None:
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")
    call = Running(spin_with_counter(let), ready, stop)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")
    answer = control.ask(run, {"cmd": "exec", "source": "1 / 0"})
    assert answer["ok"] is True and "ZeroDivisionError" in answer["error"]
    assert "letify-worker" not in answer["error"]
    assert main(["exec", run, "1 / 0"]) == 1
    assert "ZeroDivisionError" in capsys.readouterr().err
    assert main(["exec", run, "-f", str(tmp_path / "missing.py")]) == 1
    Path(stop).write_text("x", encoding="utf-8")
    call.join()


def test_exec_runs_a_file_and_json_carries_the_whole_answer(let, tmp_path, capsys) -> None:
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")
    call = Running(spin_with_counter(let), ready, stop)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")
    script = tmp_path / "probe.py"
    script.write_text("x = 20\nprint(x + 1)\n", encoding="utf-8")
    assert main(["exec", run, "-f", str(script), "--json", "--timeout", "30"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["stdout"] == "21\n" and answer["has_value"] is False
    Path(stop).write_text("x", encoding="utf-8")
    call.join()


def test_exec_can_move_a_file_to_the_drive_store_with_stash(
    fake_drive, launcher_from, worker_sees_this_checkout, tmp_path
) -> None:
    from test_drive_store import declare_drive

    declare_drive(fake_drive)
    let = launcher_from(LOCAL)
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights" * 1000)

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def training(ready_file: str, stop_file: str) -> None:
        import os
        import time

        Path(ready_file).touch()
        while not os.path.exists(stop_file):
            time.sleep(0.02)

    call = Running(training, ready, stop)
    run = the_run()
    until(lambda: os.path.exists(ready), "the loop to start")
    answer = control.ask(
        run,
        {"cmd": "exec", "source": f"import letify\nletify.stash('rescued', {str(checkpoint)!r})"},
    )
    assert answer["ok"] and answer["error"] is None, answer
    assert "stash/rescued" in fake_drive.refs()
    Path(stop).write_text("x", encoding="utf-8")
    call.join()


# -- Ctrl-C in the owning process -------------------------------------------------


def _interrupt_main_when(condition: Callable[[], Any], times: int = 1) -> threading.Thread:
    def fire() -> None:
        for _ in range(times):
            until(condition, "the moment to press Ctrl-C")
            _thread.interrupt_main()
            time.sleep(1.5)

    thread = threading.Thread(target=fire, daemon=True)
    thread.start()
    return thread


def test_the_first_ctrl_c_interrupts_the_body_and_the_outputs_come_back(
    let, tmp_path, capsys
) -> None:
    ready = str(tmp_path / "ready")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def saver(ready_file: str) -> str:
        import time

        try:
            Path(ready_file).touch()
            while True:
                time.sleep(0.02)
        except KeyboardInterrupt:
            return "saved on interrupt"

    pressed = _interrupt_main_when(lambda: os.path.exists(ready))
    assert saver(ready) == "saved on interrupt"
    pressed.join(5)
    assert "interrupting" in capsys.readouterr().err
    assert control.list_runs() == []


def test_the_message_names_the_run_and_the_second_ctrl_c_stops_waiting(
    let, tmp_path, capsys
) -> None:
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def stubborn(ready_file: str, stop_file: str) -> str:
        import os
        import time

        Path(ready_file).touch()
        while not os.path.exists(stop_file):
            try:
                time.sleep(0.02)
            except KeyboardInterrupt:
                pass
        return "finished"

    recorded: list[str] = []

    def note_run() -> bool:
        records = control.list_runs()
        if records and os.path.exists(ready):
            recorded.append(records[0]["run"])
            return True
        return False

    pressed = _interrupt_main_when(note_run, times=2)
    with let.keep_alive():
        with pytest.raises(KeyboardInterrupt):
            stubborn(ready, stop)
        pressed.join(10)
        err = capsys.readouterr().err
        assert f"letify: interrupting {recorded[0]}; press Ctrl-C again to stop waiting" in err
        # The body is still running on the runtime. Ending it lets the session close.
        Path(stop).write_text("x", encoding="utf-8")
        until(lambda: not control.list_runs(), "the abandoned call's record to go")


# -- exit hooks -------------------------------------------------------------------


def hooked_body(let):
    @let.function(device=let.providers.a.CPU, host=letify.remote, retries=0)
    def hooked(ready_file: str, marker: str) -> None:
        import time

        import letify as inner

        def first() -> None:
            Path(marker).open("a").write("first\n")  # noqa: SIM115

        def second() -> None:
            Path(marker).open("a").write("second\n")  # noqa: SIM115

        inner.on_exit(first)
        inner.on_exit(second)
        Path(ready_file).touch()
        while True:
            time.sleep(0.02)

    return hooked


def test_hooks_run_in_reverse_order_when_the_worker_gets_sigterm(let, tmp_path) -> None:
    ready, marker = str(tmp_path / "ready"), tmp_path / "marker"
    call = Running(hooked_body(let), ready, str(marker))
    the_run()
    until(lambda: os.path.exists(ready), "the body to register its hooks")
    pid = let.pool.live[0].stat()["pid"]
    os.kill(pid, signal.SIGTERM)
    until(lambda: marker.exists() and marker.read_text().count("\n") == 2, "the hooks to run")
    assert marker.read_text(encoding="utf-8") == "second\nfirst\n"
    call.join()
    assert isinstance(call.error, letify.LetifyError)


def test_hooks_run_when_the_lease_deadline_passes(let, tmp_path) -> None:
    ready, marker = str(tmp_path / "ready"), tmp_path / "marker"
    call = Running(hooked_body(let), ready, str(marker))
    the_run()
    until(lambda: os.path.exists(ready), "the body to register its hooks")
    let.pool.live[0].request({"op": "lease", "grace": 0.5})
    until(lambda: marker.exists() and marker.read_text().count("\n") == 2, "the hooks to run")
    assert marker.read_text(encoding="utf-8") == "second\nfirst\n"
    call.join()
    assert isinstance(call.error, letify.LetifyError)


def test_hooks_run_when_the_channel_closes_during_a_call(let, tmp_path) -> None:
    ready, marker = str(tmp_path / "ready"), tmp_path / "marker"
    call = Running(hooked_body(let), ready, str(marker))
    the_run()
    until(lambda: os.path.exists(ready), "the body to register its hooks")
    channel = let.pool.live[0].channel
    channel.connection.sender.frame(wire.SHUTDOWN, 0)
    until(lambda: marker.exists() and marker.read_text().count("\n") == 2, "the hooks to run")
    assert marker.read_text(encoding="utf-8") == "second\nfirst\n"
    channel.declare_lost("test is done with this worker")
    call.join()


def test_a_hook_that_raises_is_reported_and_the_next_one_still_runs(let, tmp_path, capfd) -> None:
    ready, marker = str(tmp_path / "ready"), tmp_path / "marker"

    @let.function(device=let.providers.a.CPU, host=letify.remote, retries=0)
    def hooked(ready_file: str, marker_file: str) -> None:
        import time

        import letify as inner

        inner.on_exit(lambda: Path(marker_file).write_text("ran\n"))

        def broken() -> None:
            raise ValueError("hook failure text")

        inner.on_exit(broken)
        Path(ready_file).touch()
        while True:
            time.sleep(0.02)

    call = Running(hooked, ready, str(marker))
    the_run()
    until(lambda: os.path.exists(ready), "the body to register its hooks")
    let.pool.live[0].request({"op": "lease", "grace": 0.5})
    until(lambda: marker.exists(), "the surviving hook to run")
    call.join()
    assert "hook failure text" in capfd.readouterr().err


def test_hooks_of_a_call_that_ended_normally_do_not_run(let, tmp_path) -> None:
    marker = tmp_path / "marker"

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def registers_and_returns(marker_file: str) -> str:
        import letify as inner

        inner.on_exit(lambda: Path(marker_file).write_text("ran\n"))
        return "returned"

    with let.keep_alive():
        assert registers_and_returns(str(marker)) == "returned"
        runtime = let.pool.live[0]
        pid = runtime.stat()["pid"]
        os.kill(pid, signal.SIGTERM)
        until(lambda: not _alive(pid), "the worker to exit")
    assert not marker.exists()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        done, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return done == 0


def test_on_exit_outside_a_body_raises_unsupported_mode() -> None:
    with pytest.raises(letify.UnsupportedMode, match="atexit"):
        letify.on_exit(lambda: None)
