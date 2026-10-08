"""Supervise Colab kernel wakes outside the short-lived provider CLI.

Owns readiness records, account wake locking and retry timing. It does not decide
whether Colab retains a VM, or replace a runtime after a failed wake. This file
runs by absolute path with only the standard library.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SUCCESS_INTERVAL = 60.0
FAILURE_INTERVAL = 5.0
WAKE_TIMEOUT = 120.0
READY = "LETIFY-KEEP-ALIVE ready"


@contextlib.contextmanager
def account_lock(directory: Path) -> Iterator[None]:
    """Serialize full kernel commands sharing one account's CLI state."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "letify-wake.lock").open("a+b") as handle:
        if os.name == "nt":  # pragma: no cover - Windows file locking
            import msvcrt

            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":  # pragma: no cover - Windows file locking
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def wake(config: dict[str, Any]) -> float:
    """Wake once and return the delay until the next attempt."""
    command = [*config["command"], "exec", "-s", config["session"]]
    try:
        with account_lock(Path(config["directory"])):
            result = subprocess.run(
                command, input="pass\n", capture_output=True, text=True, timeout=WAKE_TIMEOUT
            )
        if result.returncode == 0:
            return SUCCESS_INTERVAL
        detail = f"exited {result.returncode}; stderr={json.dumps(result.stderr or '')}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        detail = f"{type(exc).__name__}: {exc}; stderr={json.dumps(stderr)}"
    print(
        f"letify: wake {config['session']} command={shlex.join(command)} failed: {detail}; "
        f"retry in {FAILURE_INTERVAL:g} s",
        file=sys.stderr,
        flush=True,
    )
    return FAILURE_INTERVAL


def record(config: dict[str, Any], event: str) -> None:
    """Append a lifecycle record to the CLI account's session history."""
    directory = Path(config["directory"]) / "history"
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "event_type": event,
        "pid": os.getpid(),
        "source": "letify",
    }
    with (directory / f"{config['session']}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(config: dict[str, Any]) -> None:
    """Acknowledge recorded startup, then wake until stopped or the owner disappears."""

    def stop(signum: int, frame: Any) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    record(config, "keep_alive_started")
    print(READY, flush=True)
    try:
        while True:
            if os.name == "nt":  # pragma: no cover - Windows process liveness
                import ctypes
                from ctypes import wintypes

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.restype = wintypes.HANDLE
                handle = kernel.OpenProcess(0x00100000, False, config["owner"])
                if not handle:
                    return
                try:
                    if kernel.WaitForSingleObject(wintypes.HANDLE(handle), 0) == 0:
                        return
                finally:
                    kernel.CloseHandle(wintypes.HANDLE(handle))
            else:
                try:
                    os.kill(config["owner"], 0)
                except ProcessLookupError:
                    return
            delay = wake(config)
            threading.Event().wait(delay)
    finally:
        record(config, "keep_alive_stopped")


if __name__ == "__main__":
    run(json.loads(sys.argv[1]))
