"""Taking over a running call: a second process becomes the owner of a call and its session.

Spec section pinned here: "Taking over a running call" (id run-takeover) in "Controlling a
running call". Every test goes through the Local provider, which starts the same worker
behind the same framed protocol a remote runtime uses, so the second door, the adoption and
the relay are the real ones. The old owner is a separate process, so ownership really
crosses processes. Only the SSH link of the Shell test is a loopback stand-in, because a
real one needs a host.
"""

from __future__ import annotations

import json
import os
import pickle
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import provider_of

import letify
from letify import errors
from letify.cli import main
from letify.protocol import wire
from letify.providers.local import Local
from letify.runtime import control

LOCAL = '[a]\nkind = "local"\n'

#: The old owner: starts one call and waits for it, writing how it ended to a file.
OWNER = textwrap.dedent(
    '''
    import json, sys
    from pathlib import Path

    import letify

    config, ready, stop, result, mode, out, grace = sys.argv[1:8]
    if float(grace):
        # A lease, so the record carries a deadline and the worker outlives this process.
        from letify.providers.local import Local
        from letify.runtime import lease

        Local.needs_lease = True
        original = lease.Lease.__init__
        lease.Lease.__init__ = lambda self, runtime: original(
            self, runtime, interval=1.0, grace=float(grace)
        )
    let = letify.Launcher(config, home=False, announce=False)


    @let.function(device=let.providers.a.CPU, host=letify.remote, retries=2)
    def train(ready_file, stop_file, mode, out):
        import os
        import time
        from pathlib import Path

        if mode == "hook":
            import letify

            letify.on_exit(lambda: Path(ready_file + ".hook").write_text("the hook ran"))
        if mode == "child":
            import subprocess

            child = subprocess.Popen(["sleep", "600"])
            Path(ready_file + ".child").write_text(str(child.pid))
        print("step 1", flush=True)
        Path(ready_file + ".pid").write_text(str(os.getpid()))
        Path(ready_file).touch()
        while not os.path.exists(stop_file):
            time.sleep(0.02)
        print("step 2", flush=True)
        if mode == "raise":
            raise ValueError("boom from the body")
        if mode == "write":
            (out / "result.txt").write_text("written by the body")
        return 42


    try:
        value = train(ready, stop, mode, Path(out))
        answer = {"value": value}
    except letify.OwnershipMoved as exc:
        answer = {"moved": str(exc), "type": type(exc).__mro__[1].__name__}
    except BaseException as exc:
        answer = {"error": repr(exc)}
    Path(result).write_text(json.dumps(answer))
    '''
)


@pytest.fixture
def worker_sees_this_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every Local worker and old owner import the letify under test."""
    root = str(Path(letify.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", root)


@pytest.fixture
def short_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A short directory, because a Unix socket path holds about 100 bytes."""
    directory = Path(tempfile.mkdtemp(prefix="lt-"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(directory))
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def world(isolated_home, config_file, worker_sees_this_checkout, short_dir, tmp_path) -> Path:
    """The project directory, with the Local account `a` declared for it."""
    config_file(LOCAL, name="config")
    return isolated_home


def until(predicate: Callable[[], Any], what: str, timeout: float = 30.0) -> Any:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


class Running:
    """A function called on a thread of its own, so the test stays free."""

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

    def join(self, timeout: float = 60.0) -> None:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the call did not end"


def gone(pid: int) -> bool:
    """Whether the process has ended, a zombie nobody reaps counting as ended."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return True
    return text.rsplit(")", 1)[1].split()[0] == "Z"


class Owner:
    """The old owner process and the files the test uses to talk to its body."""

    def __init__(self, tmp_path: Path, mode: str = "return", grace: float = 0.0) -> None:
        self.ready = tmp_path / "ready"
        self.stop = tmp_path / "stop"
        self.result = tmp_path / "owner-result.json"
        self.out = tmp_path / "project" / "out"
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = tmp_path / "owner.log"
        self.stdout = open(self.log, "w", encoding="utf-8")  # noqa: SIM115
        script = tmp_path / "owner.py"
        script.write_text(OWNER, encoding="utf-8")
        config = tmp_path / "config"
        self.process = subprocess.Popen(
            [
                sys.executable, str(script), str(config), str(self.ready), str(self.stop),
                str(self.result), mode, str(self.out), str(grace),
            ],
            stdout=self.stdout,
            stderr=subprocess.STDOUT,
        )
        until(self.ready.exists, "the body to start")
        self.worker = int(Path(f"{self.ready}.pid").read_text(encoding="utf-8"))
        self.run = until(self._find, "the run record")

    def _find(self) -> str | None:
        for row in control.list_runs(orphans=True):
            if row["pid"] == self.process.pid:
                return str(row["run"])
        return None

    def finish_body(self) -> None:
        self.stop.write_text("go", encoding="utf-8")

    def answer(self) -> dict[str, Any]:
        until(self.result.exists, "the old owner to finish", 60)
        return json.loads(self.result.read_text(encoding="utf-8"))

    def output(self) -> str:
        return self.log.read_text(encoding="utf-8")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(10)
        self.stdout.close()
        if not gone(self.worker):
            os.kill(self.worker, signal.SIGKILL)


def taking(owner: Owner, function: Callable[..., Any], *args: Any) -> Running:
    """Take the run over on a thread, let the body finish once the old owner let go, and join."""
    call = Running(function, *args)
    owner.answer()
    owner.finish_body()
    call.join()
    return call


@pytest.fixture
def owners(tmp_path: Path) -> Iterator[Callable[..., Owner]]:
    made: list[Owner] = []

    def start(**kwargs: Any) -> Owner:
        owner = Owner(tmp_path, **kwargs)
        made.append(owner)
        return owner

    yield start
    for owner in made:
        owner.close()


# -- the old owner lets go ------------------------------------------------------


def test_a_call_taken_over_from_a_live_owner_finishes_at_the_new_owner(
    world, owners, capsys
) -> None:
    owner = owners()
    taking = Running(main, ["takeover", owner.run, "--json"])
    owner.answer()
    assert not gone(owner.worker), "the body must keep running through the handover"
    owner.finish_body()
    taking.join()
    assert taking.value == 0
    out = capsys.readouterr().out
    assert "step 2" in out
    last = json.loads(out.strip().splitlines()[-1])
    assert last == {"run": owner.run, "ok": True, "value_repr": "42", "result_path": None}
    assert "step 2" not in owner.output()


def test_the_old_owners_call_raises_ownership_moved_naming_the_run_and_the_new_owner(
    world, owners
) -> None:
    owner = owners()
    taking = Running(main, ["takeover", owner.run])
    answer = owner.answer()
    assert owner.run in answer["moved"]
    assert str(os.getpid()) in answer["moved"]
    owner.finish_body()
    taking.join()
    assert issubclass(errors.OwnershipMoved, letify.LetifyError)
    assert not issubclass(errors.OwnershipMoved, letify.RuntimeFailure)
    assert letify.OwnershipMoved is errors.OwnershipMoved


def test_the_old_owner_does_not_end_the_session_and_the_new_owner_does(world, owners) -> None:
    owner = owners()
    taking = Running(main, ["takeover", owner.run])
    owner.answer()
    owner.process.wait(30)
    assert not gone(owner.worker), "the old owner left, and the session must survive it"
    owner.finish_body()
    taking.join()
    assert taking.value == 0
    until(lambda: gone(owner.worker), "the new owner to end the session")


def test_the_new_owner_takes_the_record_and_removes_it_when_the_call_ends(world, owners) -> None:
    owner = owners()
    taking = Running(main, ["takeover", owner.run])
    until(lambda: any(r["pid"] == os.getpid() for r in control.list_runs()), "the new record")
    row = next(r for r in control.list_runs() if r["pid"] == os.getpid())
    assert row["run"] == owner.run
    owner.finish_body()
    taking.join()
    assert control.list_runs(orphans=True) == []


def test_a_wedged_owner_that_does_not_answer_is_treated_as_gone(
    world, owners, monkeypatch
) -> None:
    monkeypatch.setattr(control, "RELEASE_WAIT_S", 0.5)
    owner = owners()
    os.kill(owner.process.pid, signal.SIGSTOP)
    try:
        taking = Running(main, ["takeover", owner.run])
        until(lambda: any(r["pid"] == os.getpid() for r in control.list_runs()), "the new record")
        # The new owner controls the run through its own socket only once the call is adopted.
        until(lambda: main(["exec", owner.run, "1"]) == 0, "the call to be adopted")
        owner.finish_body()
        taking.join()
        assert taking.value == 0
    finally:
        os.kill(owner.process.pid, signal.SIGCONT)


# -- the result ------------------------------------------------------------------


def test_result_writes_the_value_as_a_pickle(world, owners, tmp_path, capsys) -> None:
    owner = owners()
    target = tmp_path / "value.pkl"
    call = taking(owner, main, ["takeover", owner.run, "--result", str(target), "--json"])
    assert call.value == 0
    assert pickle.loads(target.read_bytes()) == 42
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["result_path"] == str(target)
    assert last["ok"] is True


def test_without_result_the_repr_of_the_value_is_printed(world, owners, capsys) -> None:
    owner = owners()
    assert taking(owner, main, ["takeover", owner.run]).value == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "42"


def test_a_body_that_raises_exits_one_and_prints_the_traceback(world, owners, capsys) -> None:
    owner = owners(mode="raise")
    assert taking(owner, main, ["takeover", owner.run, "--json"]).value == 1
    captured = capsys.readouterr()
    assert "ValueError: boom from the body" in captured.err
    assert "Traceback" in captured.err
    last = json.loads(captured.out.strip().splitlines()[-1])
    assert last["ok"] is False
    assert "boom from the body" in last["error"]
    assert "value_repr" not in last


def test_output_files_the_body_wrote_land_under_the_records_project(
    world, owners, tmp_path, monkeypatch
) -> None:
    owner = owners(mode="write")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert taking(owner, main, ["takeover", owner.run]).value == 0
    assert (owner.out / "result.txt").read_text(encoding="utf-8") == "written by the body"
    assert list(elsewhere.iterdir()) == []


def test_takeover_from_python_returns_the_value(world, owners) -> None:
    owner = owners()
    assert taking(owner, letify.takeover, owner.run).value == 42
    assert control.list_runs(orphans=True) == []


def test_takeover_from_python_raises_the_bodys_exception(world, owners) -> None:
    owner = owners(mode="raise")
    call = taking(owner, letify.takeover, owner.run)
    assert isinstance(call.error, letify.RemoteError)
    assert "boom from the body" in str(call.error)


# -- the registry ----------------------------------------------------------------


def write_record(pid: int, deadline: float | None, run: str = "abcdef012345") -> Path:
    directory = control.runs_directory()
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    record = directory / f"{run}.json"
    record.write_text(
        json.dumps(
            {
                "run": run,
                "function": "train",
                "runtime": "r",
                "provider": "a",
                "started_at": time.time() - 5,
                "pid": pid,
                "socket": None,
                "project": str(Path.cwd()),
                "lease_deadline": deadline,
            }
        ),
        encoding="utf-8",
    )
    return record


def test_an_orphan_record_is_listed_with_owner_gone_and_the_seconds_left(
    world, capsys
) -> None:
    write_record(2**22 + 777, time.time() + 120)
    assert main(["runs", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["run"] for row in rows] == ["abcdef012345"]
    assert rows[0]["owner"] == "gone"
    assert 100 < rows[0]["seconds_left"] <= 120
    assert main(["runs"]) == 0
    text = capsys.readouterr().out
    assert "gone" in text
    assert "abcdef012345" in text


def test_an_orphan_record_is_removed_once_its_deadline_has_passed(world, capsys) -> None:
    record = write_record(2**22 + 777, time.time() - 1)
    assert main(["runs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert not record.exists()


def test_a_record_with_no_deadline_whose_owner_is_gone_is_removed(world) -> None:
    record = write_record(2**22 + 777, None)
    assert control.list_runs(orphans=True) == []
    assert not record.exists()


def test_an_orphan_cannot_be_asked_to_interrupt(world, capsys) -> None:
    write_record(2**22 + 777, time.time() + 120)
    assert main(["interrupt", "abcdef012345"]) == 3
    assert "gone" in capsys.readouterr().err


def test_takeover_of_an_unknown_run_exits_three(world, capsys) -> None:
    assert main(["takeover", "000000000000"]) == 3
    assert main(["takeover", "000000000000", "--json"]) == 3
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["error"]
    with pytest.raises(letify.errors.RunNotFound):
        letify.takeover("000000000000")


def test_takeover_of_a_run_this_process_owns_is_refused(world) -> None:
    class Provider:
        alias = "a"

    class Fake:
        name = "r"
        provider = Provider()
        persistent_channel = True

    run = control.begin(Fake(), "train")
    try:
        with pytest.raises(letify.UnsupportedMode, match="this process"):
            letify.takeover(run.id)
    finally:
        run.end()


def test_the_record_carries_the_project_and_the_lease_deadline_rewritten_on_renewal(
    world, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(Local, "needs_lease", True)
    let = letify.Launcher(tmp_path / "config", home=False, announce=False)
    stop = tmp_path / "stop"

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def spin(stop_file: str) -> None:
        import os
        import time

        while not os.path.exists(stop_file):
            time.sleep(0.02)

    call = Running(spin, str(stop))
    rows = until(control.list_runs, "a run record")
    assert rows[0]["project"] == str(world)
    first = rows[0]["lease_deadline"]
    assert first is not None and time.time() < first <= time.time() + 301
    time.sleep(0.05)
    runtime = let.pool.live[0]
    assert runtime.lease is not None
    runtime.lease._renew()
    again = control.list_runs()[0]["lease_deadline"]
    assert again > first
    stop.write_text("go", encoding="utf-8")
    call.join()


# -- a dying owner ----------------------------------------------------------------


def test_a_channel_closing_during_a_call_does_not_end_the_worker_and_a_takeover_finishes_it(
    world, owners, capsys, monkeypatch
) -> None:
    owner = owners(grace=120.0)
    owner.process.kill()
    owner.process.wait(10)
    time.sleep(1.0)
    assert not gone(owner.worker), "the body must keep running when its channel closes"
    rows = control.list_runs(orphans=True)
    assert [(row["run"], row["owner"]) for row in rows] == [(owner.run, "gone")]
    owner.finish_body()
    monkeypatch.setattr(Local, "needs_lease", True)
    assert main(["takeover", owner.run, "--json"]) == 0
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["value_repr"] == "42"
    until(lambda: gone(owner.worker), "the new owner to end the session")


def test_a_call_that_finished_while_nobody_listened_is_delivered_with_its_output(
    world, owners, capsys, monkeypatch
) -> None:
    owner = owners(grace=120.0)
    owner.process.kill()
    owner.process.wait(10)
    owner.finish_body()
    time.sleep(1.5)
    assert not gone(owner.worker), "the finished call waits for an owner until the deadline"
    monkeypatch.setattr(Local, "needs_lease", True)
    assert main(["takeover", owner.run]) == 0
    out = capsys.readouterr().out
    assert "step 2" in out
    assert out.strip().splitlines()[-1] == "42"


def test_the_worker_ends_at_the_lease_deadline_when_nobody_takes_over(world, owners) -> None:
    owner = owners(grace=2.0)
    owner.process.kill()
    owner.process.wait(10)
    until(lambda: gone(owner.worker), "the worker to end at its deadline", 30)
    assert control.list_runs(orphans=True) == []


# -- the second door --------------------------------------------------------------


def test_the_record_names_the_door_and_the_door_is_private(world, owners) -> None:
    owner = owners()
    row = next(r for r in control.list_runs() if r["run"] == owner.run)
    door = Path(row["door"])
    assert door.name == f"run-{owner.run}.sock"
    assert door.parent.name == ".letify"
    assert oct(door.stat().st_mode & 0o777) == "0o600"
    owner.finish_body()
    owner.answer()
    until(lambda: not door.exists(), "the door to close with the call")


def test_a_unix_address_beyond_the_kernel_limit_goes_through_a_directory_descriptor(
    tmp_path,
) -> None:
    deep = tmp_path
    for _ in range(4):
        deep = deep / ("d" * 40)
    deep.mkdir(parents=True)
    path = str(deep / "run-0123456789ab.sock")
    assert len(os.fsencode(path)) > 107
    import socket

    address, descriptor = wire.unix_address(path)
    assert len(os.fsencode(address)) <= 107
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(address)
    server.listen(1)
    if descriptor is not None:
        os.close(descriptor)
    assert Path(path).exists()
    address, descriptor = wire.unix_address(path)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(address)
    if descriptor is not None:
        os.close(descriptor)
    peer, _ = server.accept()
    client.sendall(b"x")
    assert peer.recv(1) == b"x"
    for sock in (client, peer, server):
        sock.close()


def test_a_short_unix_address_is_used_as_it_is(tmp_path) -> None:
    assert wire.unix_address("/run/x.sock") == ("/run/x.sock", None)


# -- the other providers ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["kaggle", "modal", "elice"])
def test_a_provider_whose_runtime_a_second_process_cannot_reach_refuses_naming_it(kind) -> None:
    from letify.providers import KINDS

    provider = provider_of(KINDS[kind], f"{kind}_x")
    with pytest.raises(letify.UnsupportedMode, match=f"{kind}_x"):
        provider.attach_channel("runtime-1", {"door": "/x"})


def test_a_colab_runtime_on_the_exec_channel_refuses_naming_the_provider() -> None:
    from letify.providers.colab import Colab

    provider = provider_of(Colab, "colab_x", channel="exec")
    with pytest.raises(letify.UnsupportedMode, match="colab_x"):
        provider.attach_channel("runtime-1", {"door": "/x"})


class LoopbackLink:
    """A link whose SSH command runs the remote command in a local shell."""

    persistent = True
    strategy = "loopback"

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        return ["/bin/sh", "-c", remote_command or "true"]


def test_a_shell_runtime_is_taken_over_through_the_relay_program(
    world, owners, tmp_path, monkeypatch, config_file
) -> None:
    from letify.providers.shell import Shell

    monkeypatch.setattr(Shell, "link", lambda self, runtime=None: LoopbackLink())
    monkeypatch.setattr(Shell, "remote_python", property(lambda self: shlex.quote(sys.executable)))
    shell_dir = config_file('[a]\nkind = "shell"\naddress = "127.0.0.1"\n', name="shell-config")
    taker = letify.Launcher(shell_dir, home=False, announce=False)
    owner = owners()
    taking = Running(lambda: letify.takeover(owner.run, launcher=taker))
    owner.answer()
    assert not gone(owner.worker)
    owner.finish_body()
    taking.join()
    assert taking.error is None
    assert taking.value == 42
    until(lambda: gone(owner.worker), "the new owner to end the session")


# -- a deliberate shutdown ----------------------------------------------------------
# Spec "A deliberate shutdown".


def test_a_shutdown_during_a_call_ends_the_body_and_runs_the_exit_hooks(world, owners) -> None:
    owner = owners(mode="hook", grace=120.0)
    owner.process.terminate()
    until(lambda: gone(owner.worker), "the worker to end after the owner shut down", 20)
    assert Path(f"{owner.ready}.hook").read_text(encoding="utf-8") == "the hook ran"


def test_a_shutdown_ends_the_processes_the_body_started(world, owners) -> None:
    owner = owners(mode="child", grace=120.0)
    child = int(Path(f"{owner.ready}.child").read_text(encoding="utf-8"))
    assert not gone(child)
    owner.process.terminate()
    until(lambda: gone(owner.worker), "the worker to end after the owner shut down", 20)
    until(lambda: gone(child), "the body's subprocess to end", 20)


def test_a_lost_owner_without_shutdown_leaves_the_body_for_a_takeover(
    world, owners, capsys, monkeypatch
) -> None:
    owner = owners(mode="hook", grace=120.0)
    owner.process.kill()
    owner.process.wait(10)
    time.sleep(1.0)
    assert not gone(owner.worker), "an owner that vanished is not a shutdown"
    owner.finish_body()
    monkeypatch.setattr(Local, "needs_lease", True)
    assert main(["takeover", owner.run, "--json"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["value_repr"] == "42"


def test_pool_shutdown_leaves_no_worker_running(world, owners) -> None:
    owner = owners(grace=120.0)
    owner.process.terminate()
    owner.process.wait(20)
    until(lambda: gone(owner.worker), "no worker to be left running", 10)


#: Runs the worker program without serving, then the probe, in a process of its own: the
#: worker takes over the process's output descriptors, so it cannot run in the test process.
PROBE = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from letify.protocol import worker

    out, moved = sys.argv[1], sys.argv[2] == "1"
    text = worker.SOURCE.rstrip()
    assert text.endswith("_serve()")
    space = {"__name__": "probe"}
    exec(compile(text[: -len("_serve()")], "worker", "exec"), space)
    ended = []
    space["_end_by_owner"] = lambda: ended.append("ended")
    space["_run_exit_hooks"] = lambda reason: None
    space["_TAKE"].update(moved=moved, run="abcdef012345", live=True)
    space["_reader_ended"](None, True)
    space["_TAKE"].update(live=True)
    space["_reader_ended"](None, False)
    Path(out).write_text(",".join(ended))
    """
)


def probe_reader_ended(tmp_path: Path, *, moved: bool) -> str:
    script = tmp_path / "probe.py"
    script.write_text(PROBE, encoding="utf-8")
    out = tmp_path / "probe.out"
    root = str(Path(letify.__file__).resolve().parent.parent)
    subprocess.run(
        [sys.executable, str(script), str(out), "1" if moved else "0"],
        check=True, timeout=60, env={**os.environ, "PYTHONPATH": root},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out.read_text(encoding="utf-8")


def test_a_shutdown_on_a_channel_the_call_moved_from_does_not_end_it(tmp_path) -> None:
    assert probe_reader_ended(tmp_path, moved=True) == ""


def test_a_shutdown_while_a_call_runs_on_its_channel_ends_it(tmp_path) -> None:
    assert probe_reader_ended(tmp_path, moved=False) == "ended"


def test_detaching_a_persistent_channel_sends_no_shutdown(monkeypatch) -> None:
    from letify.runtime.channel import PersistentChannel

    sent: list[int] = []

    class Sender:
        def frame(self, kind: int, stream: int) -> None:
            sent.append(kind)

    class Pipe:
        def close(self) -> None:
            pass

    class Process:
        stdin = Pipe()
        stdout = Pipe()

    channel = PersistentChannel.__new__(PersistentChannel)
    channel._process = Process()
    channel._connection = type("C", (), {"sender": Sender()})()
    channel._drop_ssh_log = lambda: None
    channel.detach()
    assert wire.SHUTDOWN not in sent
    assert sent == []


#: The old owner's channel closes while the worker is still adopting the call to the new owner.
#: The adoption holds the call's lock, so the close has to wait for it and then see the move.
RACE_PROBE = textwrap.dedent(
    """
    import json, sys, threading, time
    from pathlib import Path

    from letify.protocol import worker

    out = sys.argv[1]
    text = worker.SOURCE.rstrip()
    assert text.endswith("_serve()")
    space = {"__name__": "probe"}
    exec(compile(text[: -len("_serve()")], "worker", "exec"), space)
    hooks = []
    space["_run_exit_hooks"] = lambda reason: hooks.append(reason)
    take, cond = space["_TAKE"], space["_TAKE_COND"]
    take.update(run="abcdef012345", live=True, moved=False)
    new_owner = object()
    with cond:
        closing = threading.Thread(target=space["_reader_ended"], args=(None, False))
        closing.start()
        time.sleep(0.5)
        # The adoption finishes under the lock, after the close began.
        take.update(owner=new_owner, moved=True, live=True, ended_input=False)
    closing.join(30)
    Path(out).write_text(json.dumps({
        "live": take["live"], "ended_input": take["ended_input"], "hooks": hooks,
    }))
    """
)


def test_the_old_owners_channel_closing_during_the_adoption_leaves_the_new_owner_live(
    tmp_path,
) -> None:
    script = tmp_path / "race.py"
    script.write_text(RACE_PROBE, encoding="utf-8")
    out = tmp_path / "race.out"
    root = str(Path(letify.__file__).resolve().parent.parent)
    subprocess.run(
        [sys.executable, str(script), str(out)],
        check=True, timeout=60, env={**os.environ, "PYTHONPATH": root},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "live": True, "ended_input": False, "hooks": [],
    }
