"""Spec Colab: daemon readiness, serialized wakes and short failure retries."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from conftest import FakeCompleted


def test_failed_wakes_retry_after_five_seconds_and_log_stderr(tmp_path, patch_run, capsys):
    from letify.providers import colab_keepalive as daemon

    recorder = patch_run(
        daemon, result=FakeCompleted(returncode=2, stderr="kernel busy\ntry again")
    )
    config = {"command": ["colab"], "session": "live", "directory": str(tmp_path)}
    assert daemon.wake(config) == 5.0
    log = capsys.readouterr().err
    assert "colab exec -s live" in log
    assert "exited 2" in log and "kernel busy\\ntry again" in log
    assert recorder.calls[-1]["timeout"] == 120
    recorder.result = FakeCompleted()
    assert daemon.wake(config) == 60.0


def test_account_wakes_hold_the_same_file_lock_until_the_command_finishes(tmp_path):
    from letify.providers import colab_keepalive as daemon

    acquired = threading.Event()
    attempted = threading.Event()
    finished = threading.Event()

    def contender():
        attempted.set()
        with daemon.account_lock(tmp_path):
            acquired.set()
        finished.set()

    with daemon.account_lock(tmp_path):
        thread = threading.Thread(target=contender)
        thread.start()
        assert attempted.wait(2)
        assert not acquired.is_set()
    assert finished.wait(2)
    thread.join(2)
    assert acquired.is_set()


def test_the_detached_daemon_records_startup_before_acknowledging(tmp_path):
    from letify.providers import colab_keepalive as daemon

    config = {
        "command": [sys.executable, "-c", "pass"],
        "session": "live",
        "directory": str(tmp_path),
        "owner": __import__("os").getpid(),
    }
    process = subprocess.Popen(
        [sys.executable, str(Path(daemon.__file__)), json.dumps(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout.readline().strip() == "LETIFY-KEEP-ALIVE ready"
        event = json.loads((tmp_path / "history" / "live.jsonl").read_text().splitlines()[0])
        assert event["event_type"] == "keep_alive_started"
        assert event["pid"] == process.pid
        assert process.poll() is None
        assert __import__("os").getsid(process.pid) == process.pid
    finally:
        process.terminate()
        process.communicate(timeout=5)
    assert process.returncode == 0
