"""Reattaching after a lost channel: the owner adopts its own run instead of discarding it.

Spec section pinned here: "Reattaching after a lost channel" (id run-reattach). Every test
goes through the Local provider, which starts the same worker behind the same framed
protocol a remote runtime uses. The link that ssh would be is a small proxy process: the
worker runs in a session of its own behind it, so ending the proxy cuts the owner's channel
the way a dropped ssh connection does, while the worker and its body keep running.
"""

from __future__ import annotations

import os
import shutil
import signal
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import letify
from letify import errors
from letify.providers.local import Local
from letify.runtime import control, lease, reattach

LOCAL = '[a]\nkind = "local"\n'

#: Stands in for ssh: starts the worker in a session of its own and carries the frames. On
#: SIGTERM it exits 255, as ssh does when the link breaks. Otherwise it exits with the
#: worker's own status, as ssh passes on the status of the remote command.
PROXY = textwrap.dedent(
    """
    import os, signal, subprocess, sys, threading

    signal.signal(signal.SIGTERM, lambda *_: os._exit(255))
    worker = subprocess.Popen(
        sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True
    )


    def up():
        try:
            while True:
                data = os.read(0, 65536)
                if not data:
                    break
                worker.stdin.write(data)
                worker.stdin.flush()
        except OSError:
            pass
        try:
            worker.stdin.close()
        except OSError:
            pass


    threading.Thread(target=up, daemon=True).start()
    while True:
        data = os.read(worker.stdout.fileno(), 65536)
        if not data:
            break
        os.write(1, data)
    sys.exit(worker.wait())
    """
)


@pytest.fixture
def worker_sees_this_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
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
def let(
    isolated_home, launcher_from, worker_sees_this_checkout, short_dir, tmp_path, monkeypatch
) -> letify.Launcher:
    """A Local account with a lease and a proxy in front of every worker."""
    script = tmp_path / "proxy.py"
    script.write_text(PROXY, encoding="utf-8")

    def open_channel(self: Local, runtime: Any) -> Any:
        from letify.protocol.worker import BOOTSTRAP
        from letify.runtime.channel import PersistentChannel

        return PersistentChannel(
            [sys.executable, str(script), sys.executable, "-u", "-c", BOOTSTRAP],
            name=runtime.name,
        )

    monkeypatch.setattr(Local, "open_channel", open_channel)
    # A lease, so the worker outlives its channel until the deadline and the owner knows it.
    monkeypatch.setattr(Local, "needs_lease", True)
    original = lease.Lease.__init__
    monkeypatch.setattr(
        lease.Lease,
        "__init__",
        lambda self, runtime: original(self, runtime, interval=1.0, grace=60.0),
    )
    monkeypatch.setattr(reattach, "DELAYS_S", (0.2,))
    return launcher_from(LOCAL)


def until(predicate: Callable[[], Any], what: str, timeout: float = 30.0) -> Any:
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

    def join(self, timeout: float = 60.0) -> None:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the call did not end"


def cut_the_link(let: letify.Launcher) -> None:
    """End the proxy of the one live runtime, the way a broken ssh connection ends."""
    process = let.pool.live[0].channel._process
    os.kill(process.pid, signal.SIGTERM)
    process.wait(10)


def declare(let: letify.Launcher, retries: int = 1) -> Callable[..., Any]:
    @let.function(device=let.providers.a.CPU, host=letify.remote, retries=retries)
    def train(ready: str, stop: str, ledger: str) -> int:
        import os
        import time
        from pathlib import Path

        with open(ledger, "a", encoding="utf-8") as handle:
            handle.write("ran\n")
        print("before the gap", flush=True)
        Path(ready).touch()
        while not os.path.exists(stop):
            time.sleep(0.02)
        print("during the gap", flush=True)
        time.sleep(0.5)
        return 7

    return train


def test_a_call_whose_channel_broke_is_adopted_again_and_returns_its_value(
    let, tmp_path, capfd
) -> None:
    ready, stop, ledger = tmp_path / "ready", tmp_path / "stop", tmp_path / "ledger"
    call = Running(declare(let), str(ready), str(stop), str(ledger))
    until(ready.exists, "the body to start")
    run = next(iter(control.list_runs()))["run"]
    cut_the_link(let)
    stop.touch()
    call.join()
    assert call.error is None
    assert call.value == 7
    assert ledger.read_text(encoding="utf-8") == "ran\n", "the body must run once"
    captured = capfd.readouterr()
    assert "before the gap" in captured.out
    assert "during the gap" in captured.out
    assert f"during {run}; reattaching (attempt 1)" in captured.err
    assert f"letify: reattached to {run} on " in captured.err


def test_a_reattached_call_goes_on_renewing_the_lease_over_the_new_channel(
    let, tmp_path
) -> None:
    ready, stop, ledger = tmp_path / "ready", tmp_path / "stop", tmp_path / "ledger"
    call = Running(declare(let), str(ready), str(stop), str(ledger))
    until(ready.exists, "the body to start")
    runtime = let.pool.live[0]
    cut_the_link(let)
    until(lambda: type(runtime.channel).__name__ == "SocketChannel", "the new channel")
    deadline = runtime.lease.deadline
    until(lambda: runtime.lease.deadline > deadline, "a renewal on the new channel", 10)
    stop.touch()
    call.join()
    assert call.value == 7


def test_a_body_that_raises_after_the_reattach_is_not_retried(let, tmp_path) -> None:
    ledger = tmp_path / "ledger"
    ready, stop = tmp_path / "ready", tmp_path / "stop"

    @let.function(device=let.providers.a.CPU, host=letify.remote, retries=2)
    def boom(ready_file: str, stop_file: str, ledger_file: str) -> None:
        import os
        import time
        from pathlib import Path

        with open(ledger_file, "a", encoding="utf-8") as handle:
            handle.write("ran\n")
        Path(ready_file).touch()
        while not os.path.exists(stop_file):
            time.sleep(0.02)
        raise ValueError("the body failed")

    call = Running(boom, str(ready), str(stop), str(ledger))
    until(ready.exists, "the body to start")
    cut_the_link(let)
    stop.touch()
    call.join()
    assert isinstance(call.error, letify.RemoteError)
    assert "the body failed" in str(call.error)
    assert ledger.read_text(encoding="utf-8") == "ran\n"


def test_a_provider_that_refuses_takeover_gets_the_discard_and_retry(
    let, tmp_path, monkeypatch, capfd
) -> None:
    def refuse(self: Local, name: str, record: Any) -> Any:
        raise errors.UnsupportedMode("this provider cannot be reattached")

    monkeypatch.setattr(Local, "attach_channel", refuse)
    ready, stop, ledger = tmp_path / "ready", tmp_path / "stop", tmp_path / "ledger"
    stop.touch()
    call = Running(declare(let), str(ready), str(stop), str(ledger))
    until(ready.exists, "the body to start")
    cut_the_link(let)
    call.join()
    assert call.value == 7
    assert ledger.read_text(encoding="utf-8").count("ran") == 2
    assert "reattached" not in capfd.readouterr().err


def test_a_worker_that_exited_with_its_own_status_is_not_reattached(
    let, tmp_path, monkeypatch, capfd
) -> None:
    attempts: list[str] = []
    original = Local.attach_channel

    def counting(self: Local, name: str, record: Any) -> Any:
        attempts.append(name)
        return original(self, name, record)

    monkeypatch.setattr(Local, "attach_channel", counting)
    ledger = tmp_path / "ledger"

    @let.function(device=let.providers.a.CPU, host=letify.remote, retries=1)
    def dies(ledger_file: str) -> int:
        import os

        with open(ledger_file, "a", encoding="utf-8") as handle:
            handle.write("ran\n")
        if open(ledger_file, encoding="utf-8").read().count("ran") == 1:
            os._exit(3)
        return 5

    assert dies(str(ledger)) == 5
    assert attempts == []
    assert "reattaching" not in capfd.readouterr().err


def test_a_reattach_that_cannot_reach_the_worker_falls_back_to_the_retry(
    let, tmp_path, monkeypatch, capfd
) -> None:
    # The door is not where the record says, as when the worker died with its channel. Every
    # attempt fails and the call is retried on a fresh session, as it is without reattaching.
    monkeypatch.setattr(reattach, "DELAYS_S", (0.05,))
    monkeypatch.setattr(reattach, "MAX_ATTEMPTS", 2)
    original = Local.attach_channel

    def to_nowhere(self: Local, name: str, record: Any) -> Any:
        return original(self, name, {**record, "door": str(tmp_path / "no-such-door.sock")})

    monkeypatch.setattr(Local, "attach_channel", to_nowhere)
    ready, stop, ledger = tmp_path / "ready", tmp_path / "stop", tmp_path / "ledger"
    stop.touch()
    call = Running(declare(let), str(ready), str(stop), str(ledger))
    until(ready.exists, "the body to start")
    run = next(iter(control.list_runs()))["run"]
    cut_the_link(let)
    call.join()
    assert call.value == 7
    err = capfd.readouterr().err
    assert f"during {run}; reattaching (attempt 2)" in err
    assert "reattached to" not in err
