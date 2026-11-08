"""The suite must not leave letify-link processes behind.

Pins the autouse ``no_link_process_survives`` fixture in conftest.py. A sidecar, shim or
pump that outlives its test stays for the lease window, and a full run ends with orphans.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conftest import TEST_MARK, leaked_link_processes


def _fake_sidecar(directory: Path, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    """A detached process whose command line names letify-link, as a leaked sidecar's does."""
    script = directory / "letify-link"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    return subprocess.Popen([sys.executable, str(script)], env=env, start_new_session=True)


def test_a_link_process_left_running_by_a_test_is_found(tmp_path) -> None:
    leaked = _fake_sidecar(tmp_path)
    try:
        found = leaked_link_processes(os.environ[TEST_MARK], wait=0.2)
        assert list(found) == [leaked.pid]
    finally:
        leaked.kill()
        leaked.wait()
    assert leaked_link_processes(os.environ[TEST_MARK], wait=1.0) == {}


def test_a_process_started_under_another_mark_is_not_counted(tmp_path) -> None:
    other = _fake_sidecar(tmp_path, dict(os.environ, **{TEST_MARK: "another-test"}))
    try:
        time.sleep(0.2)
        assert leaked_link_processes(os.environ[TEST_MARK], wait=0.2) == {}
    finally:
        with contextlib.suppress(OSError):
            os.kill(other.pid, signal.SIGKILL)
        other.wait()
