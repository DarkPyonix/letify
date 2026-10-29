"""The letify workers on a host: the programs that list and stop them, and how a row reads.

Spec "Workers on a shell host". This module owns the record's directory name, the two small
programs a provider runs on the host, and the local judgement of who owns a worker. It does
not own writing the record, which is the worker's (``letify.protocol.worker``), and it does
not own the transport that carries a program to a host, which is the provider's.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

from ..errors import RuntimeFailure

#: Under the workspace root, where each worker keeps ``<pid>.json``.
RECORD_DIR = ".letify/workers"

#: Seconds a stopped worker gets to end after ``SIGTERM`` before it is sent ``SIGKILL``.
STOP_GRACE_S = 30.0

#: Seconds to wait for a worker to disappear after ``SIGKILL``.
KILL_WAIT_S = 10.0

#: How an owner reads in a row.
ALIVE, GONE, REMOTE = "alive", "gone", "remote"

#: Run on the host with the workspace root as its one argument. It prints one JSON object,
#: ``{"now": <host Unix time>, "workers": [<record>, ...]}``, after deleting the records whose
#: process is not a live worker. A process is that worker when it runs, is not a zombie and
#: has the start time the record names, so a reused pid does not read as one.
LIST_PROGRAM = r'''
import glob, json, os, sys, time

root = os.path.expanduser(sys.argv[1])
directory = os.path.join(root, ".letify", "workers")


def proc_start(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            rest = handle.read().rsplit(")", 1)[1].split()
        return rest[0], int(rest[19])
    except (OSError, ValueError, IndexError):
        return None, None


def live(record):
    pid = record["pid"]
    state, start = proc_start(pid)
    if state is None:
        if os.path.isdir("/proc/self"):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    if state in ("Z", "X"):
        return False
    wanted = record.get("proc_start")
    return wanted is None or wanted == start


found = []
for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
    try:
        with open(path) as handle:
            record = json.load(handle)
        int(record["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        continue
    if live(record):
        found.append(record)
    else:
        try:
            os.unlink(path)
        except OSError:
            pass
print(json.dumps({"now": time.time(), "workers": found}))
'''

#: Run on the host with the workspace root, a pid and the grace in seconds. It refuses a pid
#: with no record or no live worker, then sends ``SIGTERM``, waits the grace and sends
#: ``SIGKILL``. It prints one JSON object, ``{"ok": true, "gone": <bool>}`` or
#: ``{"ok": false, "reason": <text>}``.
STOP_PROGRAM = r'''
import json, os, signal, sys, time

root = os.path.expanduser(sys.argv[1])
pid = int(sys.argv[2])
grace, kill_wait = float(sys.argv[3]), float(sys.argv[4])
path = os.path.join(root, ".letify", "workers", "%d.json" % pid)


def refuse(reason):
    print(json.dumps({"ok": False, "reason": reason}))
    sys.exit(0)


def state_of(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            rest = handle.read().rsplit(")", 1)[1].split()
        return rest[0], int(rest[19])
    except (OSError, ValueError, IndexError):
        return None, None


def running(record):
    state, start = state_of(pid)
    if state is None:
        if os.path.isdir("/proc/self"):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    if state in ("Z", "X"):
        return False
    wanted = record.get("proc_start")
    return wanted is None or wanted == start


try:
    with open(path) as handle:
        record = json.load(handle)
except (OSError, ValueError):
    refuse("pid %d is not a letify worker: no record names it" % pid)
if not running(record):
    try:
        os.unlink(path)
    except OSError:
        pass
    refuse("pid %d is not a letify worker: it is not running" % pid)


def wait(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not running(record):
            return True
        time.sleep(0.1)
    return not running(record)


try:
    os.kill(pid, signal.SIGTERM)
except ProcessLookupError:
    pass
if not wait(grace):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    wait(kill_wait)
gone = not running(record)
if gone:
    try:
        os.unlink(path)
    except OSError:
        pass
print(json.dumps({"ok": True, "gone": gone}))
'''


def proc_start(pid: int) -> int | None:
    """The start time of ``pid`` in clock ticks since boot, or None where ``/proc`` lacks it."""
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return int(handle.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def parse_listing(output: str) -> tuple[float, list[dict[str, Any]]]:
    """The host's clock and its live worker records, from what ``LIST_PROGRAM`` printed."""
    try:
        data = json.loads(output)
        return float(data["now"]), list(data["workers"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeFailure(
            "the worker listing on the host was not readable", stderr=output[-400:]
        ) from exc


def owner_state(record: dict[str, Any], host: str | None = None) -> str:
    """Whether the process that opened a worker's session is alive, gone or on another host.

    Only an owner on this machine can be judged. Spec "Workers on a shell host", owner state.
    """
    owner = record.get("owner") or {}
    if owner.get("host") != (host or socket.gethostname()):
        return REMOTE
    try:
        os.kill(int(owner["pid"]), 0)
    except ProcessLookupError:
        return GONE
    except (PermissionError, KeyError, TypeError, ValueError):
        return ALIVE
    return ALIVE


def is_orphan(record: dict[str, Any], state: str, now: float) -> bool:
    """A passed lease, or a gone owner with no call running. Spec "Workers on a shell host"."""
    deadline = record.get("lease_deadline")
    if deadline is not None and now > deadline:
        return True
    return state == GONE and not record.get("run")


def describe(record: dict[str, Any], state: str) -> str:
    """One readable line for a worker."""
    owner = record.get("owner") or {}
    devices = record.get("devices")
    return (
        f"pid {record['pid']}  gpu {devices if devices else 'none'}  "
        f"started {record.get('started') or 'unknown'}  runtime {record.get('runtime')}  "
        f"key {record.get('key')}  run {record.get('run') or 'none'}  "
        f"owner {owner.get('host')}:{owner.get('pid')} {state}"
    )


__all__ = [
    "ALIVE",
    "GONE",
    "KILL_WAIT_S",
    "LIST_PROGRAM",
    "RECORD_DIR",
    "REMOTE",
    "STOP_GRACE_S",
    "STOP_PROGRAM",
    "describe",
    "is_orphan",
    "owner_state",
    "parse_listing",
    "proc_start",
]
