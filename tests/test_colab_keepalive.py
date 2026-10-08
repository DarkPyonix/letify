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


def test_every_wake_records_history_and_logs_duration_with_timestamp(tmp_path, patch_run, capsys):
    from letify.providers import colab_keepalive as daemon

    patch_run(daemon, result=FakeCompleted())
    config = {"command": ["colab"], "session": "live", "directory": str(tmp_path)}
    assert daemon.wake(config) == 60.0
    history_file = tmp_path / "history" / "live.jsonl"
    assert history_file.is_file()
    lines = [json.loads(line) for line in history_file.read_text().splitlines()]
    wake_event = lines[-1]
    assert wake_event["event_type"] == "wake"
    assert "colab exec -s live" in wake_event["command"]
    assert wake_event["returncode"] == 0
    assert "duration_s" in wake_event and isinstance(wake_event["duration_s"], float)
    assert "timestamp" in wake_event
    log = capsys.readouterr().err
    assert "colab exec -s live" in log
    assert "duration=" in log
    assert "exit=0" in log

