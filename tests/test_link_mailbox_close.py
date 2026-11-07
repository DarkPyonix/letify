"""Closing a mailbox bootstrap ends what it started on the runtime.

Spec "Sidecar interface" item 8, "Close before the hand-over ends the runtime side". The
command channel is this machine, as in tests/test_link_command_channel.py. The pump, the shim
and the sidecar are real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from letify.transport import linkd, linkmailbox


@pytest.fixture
def peer_id(sidecar_binary, tmp_path: Path):
    daemon = linkd.LinkDaemon(tmp_path / "peer.key", binary=sidecar_binary)
    try:
        yield daemon.node_id
    finally:
        daemon.close()


def run_here(source: str, timeout: float) -> str:
    done = subprocess.run(
        [sys.executable, "-"], input=source, capture_output=True, text=True, timeout=timeout
    )
    return done.stdout + done.stderr


def pids_of(directory: str) -> list[int]:
    listing = subprocess.run(
        ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True
    ).stdout
    found = []
    for row in listing.splitlines():
        pid, _, args = row.strip().partition(" ")
        if directory in args and int(pid) != os.getpid():
            found.append(int(pid))
    return found


def sidecars(binary: Path) -> set[int]:
    found = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            raw = Path(f"/proc/{name}/cmdline").read_bytes()
        except OSError:
            continue
        if raw.startswith(str(binary).encode() + b"\0serve"):
            found.add(int(name))
    return found


def wait_until(check, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.1)
    return bool(check())


def started(sidecar_binary: Path, peer: str) -> linkmailbox.MailboxBootstrap:
    bootstrap = linkmailbox.MailboxBootstrap(run_here, python=sys.executable)
    bootstrap.start()
    config = {
        "binary": str(sidecar_binary),
        "allow": peer,
        "bind": "127.0.0.1:0",
        "wait_s": 30,
        "resume_s": 30,
    }
    bootstrap.send_line(json.dumps(config))
    while True:
        line = bootstrap.recv_line()
        assert line is not None, bootstrap.stderr_text()
        if json.loads(line).get("event") == "ready":
            return bootstrap


def test_closing_before_the_hand_over_ends_the_pump_the_shim_and_the_sidecar(
    sidecar_binary, peer_id
) -> None:
    before = sidecars(sidecar_binary)
    bootstrap = started(sidecar_binary, peer_id)
    directory = bootstrap._directory
    assert len(pids_of(directory)) >= 2
    mine = sidecars(sidecar_binary) - before
    assert mine
    bootstrap.close()
    assert wait_until(lambda: not pids_of(directory)), pids_of(directory)
    assert wait_until(lambda: not (sidecars(sidecar_binary) & mine))
    assert wait_until(lambda: not os.path.exists(directory))


def test_a_pump_that_no_call_reaches_ends_its_runtime_side(
    sidecar_binary, peer_id, monkeypatch
) -> None:
    source = linkmailbox.pump_source().replace("IDLE_S = 120.0", "IDLE_S = 1.0")
    monkeypatch.setattr(linkmailbox, "pump_source", lambda: source)
    before = sidecars(sidecar_binary)
    bootstrap = started(sidecar_binary, peer_id)
    directory = bootstrap._directory
    mine = sidecars(sidecar_binary) - before
    # Stop the poll thread without the close call, as a client that died would.
    bootstrap._stop.set()
    assert wait_until(lambda: not pids_of(directory), 15.0), pids_of(directory)
    assert wait_until(lambda: not (sidecars(sidecar_binary) & mine))
    assert wait_until(lambda: not os.path.exists(directory))
