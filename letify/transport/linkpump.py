"""The remote half of a mailbox: a file pair that stands in for the shim's standard streams.

Owns running the shim of ``linkshim.py`` as a detached process whose standard input is an inbox
file and whose standard output is an outbox file, and the one call that appends lines to the
inbox and returns what the outbox gained. It does not own the shim, the command channel that
carries the calls (``linkmailbox.py`` is its local half) or any path decision. Spec "Sidecar
interface" item 8. This file is sent to the runtime as a program that a provider's command
channel runs, so it imports nothing from letify and avoids syntax newer than Python 3.6.
"""

# ruff: noqa: SIM115, UP031
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

MARKER = "LETIFY-MAILBOX "
POLL_S = 0.05
STDERR_TAIL = 400
EXIT_WAIT_S = 2.0
#: Seconds without a call after which the pump ends its runtime side, whoever is gone.
IDLE_S = 120.0
#: Seconds the shim gets to stop its sidecar after its input ends, before it is killed.
STOP_WAIT_S = 3.0


def answer(**fields):
    sys.stdout.write(MARKER + json.dumps(fields) + "\n")
    sys.stdout.flush()


def start(request, source):
    """Write the shim and the pump into ``dir`` and start the pump, detached."""
    directory = request.get("dir") or tempfile.mkdtemp(prefix="letify-linkmail-")
    python = sys.executable or request["python"]
    for name, text in (
        ("shim.py", request["shim"]),
        ("pump.py", source + "\npump(%r)\n" % directory),
    ):
        with open(os.path.join(directory, name), "w") as handle:
            handle.write(text)
    for name in ("inbox", "outbox", "seen"):
        open(os.path.join(directory, name), "wb").close()
    errors = open(os.path.join(directory, "pump.err"), "wb")
    subprocess.Popen(
        [python, "-u", os.path.join(directory, "pump.py")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=errors,
        start_new_session=True,
        close_fds=True,
    )
    answer(ok=True, dir=directory, python=python)


def pump(directory):
    """Run the shim; copy the inbox to its standard input and its standard output to the outbox."""
    errors = open(os.path.join(directory, "shim.err"), "wb")
    shim = subprocess.Popen(
        [sys.executable, "-u", os.path.join(directory, "shim.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=errors,
        close_fds=True,
        # Its own process group, so that everything the shim started can be ended together.
        start_new_session=True,
    )

    state = {"handed": False}

    def stop_shim():
        if shim.poll() is None and not state["handed"]:
            try:
                shim.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                shim.wait(timeout=STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                shim.terminate()
                try:
                    shim.wait(timeout=STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    shim.kill()
            # Sweep the group: the sidecar must not outlive a shim that exited without it.
            try:
                os.killpg(shim.pid, signal.SIGKILL)
            except OSError:
                pass

    def watch():
        """End this side when the client closed the mailbox or no call has come for IDLE_S."""
        seen = os.path.join(directory, "seen")
        while True:
            time.sleep(0.2)
            try:
                idle = time.time() - os.stat(seen).st_mtime > IDLE_S
            except OSError:
                idle = True
            if idle or os.path.exists(os.path.join(directory, "closed")):
                stop_shim()
                shutil.rmtree(directory, ignore_errors=True)
                os._exit(0)

    def feed():
        offset = 0
        pending = b""
        path = os.path.join(directory, "inbox")
        try:
            while shim.poll() is None:
                with open(path, "rb") as handle:
                    handle.seek(offset)
                    data = handle.read()
                offset += len(data)
                pending += data
                cut = pending.rfind(b"\n")
                if cut >= 0:
                    shim.stdin.write(pending[: cut + 1])
                    shim.stdin.flush()
                    pending = pending[cut + 1 :]
                time.sleep(POLL_S)
        except (OSError, ValueError):
            pass

    threading.Thread(target=feed, daemon=True).start()
    threading.Thread(target=watch, daemon=True).start()
    with open(os.path.join(directory, "outbox"), "ab") as outbox:
        for line in shim.stdout:
            if b"handed_over" in line:
                # Known before the client can read the line, so that a close that follows at
                # once finds the shim handed over and leaves it, and what it started, alone.
                try:
                    if json.loads(line).get("event") == "handed_over":
                        state["handed"] = True
                except ValueError:
                    pass
            outbox.write(line)
            outbox.flush()
        try:
            shim.wait(timeout=EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            # The shim closed its output and runs on: the hand-over, which ends the mailbox.
            # The directory stays until the client's close call or IDLE_S, so a late read works.
            state["handed"] = True
            while True:
                time.sleep(1.0)
        try:
            with open(os.path.join(directory, "shim.err"), "rb") as handle:
                tail = handle.read()[-STDERR_TAIL:].decode("utf-8", "replace")
        except OSError:
            tail = ""
        outbox.write((json.dumps({"event": "eof", "stderr": tail}) + "\n").encode())
        outbox.flush()
    while True:
        time.sleep(1.0)


def exchange(request, source=None):
    """Append ``send`` to the inbox, wait up to ``wait_s`` for the outbox to pass ``offset``."""
    directory = request["dir"]
    os.utime(os.path.join(directory, "seen"), None)
    if request.get("send"):
        with open(os.path.join(directory, "inbox"), "ab") as handle:
            for line in request["send"]:
                handle.write(line.encode("utf-8") + b"\n")
    offset = int(request["offset"])
    deadline = time.time() + float(request.get("wait_s", 0))
    path = os.path.join(directory, "outbox")
    while True:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
        cut = data.rfind(b"\n")
        if cut >= 0:
            lines = data[: cut + 1].decode("utf-8", "replace").splitlines()
            return answer(ok=True, offset=offset + cut + 1, lines=lines)
        if time.time() >= deadline:
            return answer(ok=True, offset=offset, lines=[])
        time.sleep(POLL_S)


def close(request, source=None):
    """Tell the pump to end its side: stop a shim that has not handed over, remove the directory."""
    open(os.path.join(request["dir"], "closed"), "wb").close()
    answer(ok=True)


def run(request, source):
    try:
        {"start": start, "exchange": exchange, "close": close}[request["op"]](request, source)
    except Exception as error:
        answer(ok=False, error="%s: %s" % (type(error).__name__, error))
