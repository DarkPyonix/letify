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
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SUCCESS_INTERVAL = 60.0
FAILURE_INTERVAL = 5.0
WAKE_TIMEOUT = 120.0
READY = "LETIFY-KEEP-ALIVE ready"


@contextlib.contextmanager
def account_lock(directory: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Serialize wakes sharing account CLI state, optionally without waiting."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "letify-wake.lock").open("a+b") as handle:
        if os.name == "nt":  # pragma: no cover - Windows file locking
            import msvcrt

            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            except OSError:
                if blocking:
                    raise
                yield False
                return
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
        try:
            yield True
        finally:
            if os.name == "nt":  # pragma: no cover - Windows file locking
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def wake(config: dict[str, Any]) -> float:
    """Wake once and return the delay until the next attempt."""
    command = [*config["command"], "exec", "-s", config["session"]]
    start = time.monotonic()
    result = None
    exc = None
    try:
        with account_lock(Path(config["directory"]), blocking=False) as acquired:
            if not acquired:
                entry = record(config, "wake_skipped", reason="account_busy")
                print(
                    f"{entry['timestamp']} letify: wake {config['session']} skipped: account busy",
                    file=sys.stderr,
                    flush=True,
                )
                return SUCCESS_INTERVAL
            result = subprocess.run(
                command, input="pass\n", capture_output=True, text=True, timeout=WAKE_TIMEOUT
            )
    except (OSError, subprocess.TimeoutExpired) as caught:
        exc = caught
    duration = time.monotonic() - start

    if result is not None:
        returncode = result.returncode
        stderr_raw = result.stderr or ""
        stderr_short = stderr_raw.strip().splitlines()[-1] if stderr_raw.strip() else ""
    else:
        returncode = getattr(exc, "returncode", -1)
        stderr_raw = getattr(exc, "stderr", "") or ""
        if isinstance(stderr_raw, bytes):
            stderr_raw = stderr_raw.decode(errors="replace")
        stderr_short = f"{type(exc).__name__}: {exc}"

    entry = record(
        config,
        "wake",
        command=shlex.join(command),
        returncode=returncode,
        stderr=stderr_short,
        duration_s=round(duration, 3),
    )
    timestamp = entry["timestamp"]

    if returncode == 0:
        print(
            f"{timestamp} letify: wake {config['session']} command={shlex.join(command)} "
            f"exit=0 duration={duration:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        return SUCCESS_INTERVAL

    if exc is not None:
        detail = f"{type(exc).__name__}: {exc}; stderr={json.dumps(stderr_raw)}"
    else:
        detail = f"exited {returncode}; stderr={json.dumps(stderr_raw)}"
    print(
        f"{timestamp} letify: wake {config['session']} command={shlex.join(command)} "
        f"failed: {detail}; duration={duration:.2f}s; retry in {FAILURE_INTERVAL:g} s",
        file=sys.stderr,
        flush=True,
    )
    return FAILURE_INTERVAL


def record(config: dict[str, Any], event: str, **extra: Any) -> dict[str, Any]:
    """Append a lifecycle record to the CLI account's session history."""
    directory = Path(config["directory"]) / "history"
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "event_type": event,
        "pid": os.getpid(),
        "source": "letify",
        **extra,
    }
    with (directory / f"{config['session']}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return body



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
