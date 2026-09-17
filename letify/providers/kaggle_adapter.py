"""The Kaggle adapter: one bridge between letify's frames and one kernel cell.

Run by letify through ``uv run --no-project --with "jupyter-kernel-client<1" python -P`` so the
letify process never imports ``jupyter_kernel_client``, as spec "Kaggle Jupyter Server session"
describes. It lives for the runtime, not for one program: it executes a single cell that keeps
the letify worker alive, then moves base64 frame lines in both directions.

A kernel answers ``input()`` and nothing else. The cell's ``sys.stdin`` is a ``TextIOWrapper``
whose ``buffer`` reads empty rather than raising ``input_request``, so the cell carries a shim
that reads base64 lines with ``input()``, decodes them, and writes the bytes into a pipe the
worker reads as its standard input. The worker already writes its frames as base64 lines,
because the channel sets ``text_frames``, so the shim passes those out unchanged.

Input: ``LETIFY_JUPYTER_URL`` and ``LETIFY_KERNEL_ID``. The URL is read from the environment so
it never appears in a process list. The URL is the routed proxy URL, whose token rides in the
path, because the proxy rejects a token sent only as a header.

Exit 4 when the server or kernel could not be reached, and 0 when letify closed the bridge's
standard input.
"""

from __future__ import annotations

import os
import sys
import threading
import urllib.parse

UNREACHABLE = 4

#: The stub the shim starts the worker with, the one Channels describes: a byte count line,
#: then that many bytes of source.
STUB = (
    "import sys;"
    "b=sys.stdin.buffer;"
    "n=int(b.readline());"
    "exec(compile(b.read(n).decode(),'letify-worker','exec'))"
)

#: The cell that keeps the worker alive. It reads base64 lines with input(), because that is
#: the only read a kernel answers, and copies the worker's own lines straight back out.
SHIM = """
import base64, subprocess, sys, threading

_worker = subprocess.Popen(
    [sys.executable, "-u", "-c", %(stub)r],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)


def _out():
    for _line in _worker.stdout:
        sys.stdout.write(_line.decode("utf-8", "replace"))
        sys.stdout.flush()


threading.Thread(target=_out, daemon=True).start()

while True:
    try:
        _text = input()
    except EOFError:
        break
    if not _text:
        continue
    try:
        _worker.stdin.write(base64.b64decode(_text))
        _worker.stdin.flush()
    except (OSError, ValueError):
        break

try:
    _worker.stdin.close()
except OSError:
    pass
_worker.wait()
"""


def main() -> int:
    url = os.environ["LETIFY_JUPYTER_URL"]
    kernel = os.environ["LETIFY_KERNEL_ID"]
    parts = urllib.parse.urlsplit(url)
    base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    token = dict(urllib.parse.parse_qsl(parts.query)).get("token", "")

    try:
        from jupyter_kernel_client import KernelClient

        client = KernelClient(server_url=base, token=token, kernel_id=kernel)
        client.start()
    except Exception as exc:  # any failure to attach means the session cannot be used
        print(f"the kernel could not be reached: {type(exc).__name__}", file=sys.stderr)
        return UNREACHABLE

    return bridge(client)


def bridge(client) -> int:
    """Carry frames between this process and one long-lived cell, until standard input ends.

    The two directions run on two threads, each owning one kernel channel, because the one
    thing the bridge must never do is answer an input request from the same loop that reads
    the cell's output. The client's own ``execute_interactive`` does exactly that: its default
    stdin hook blocks in ``input()`` inside the execute loop, and while it blocks the loop
    cannot drain iopub. The worker announces itself with a stream message on iopub, so the
    first blocked read holds that hello behind an input request the channel will not answer
    until it has seen the hello. That is a deadlock, and it returns on every large response
    too, where the cell reads the next request while the worker is still writing output.

    So iopub is drained on this thread and stdin is answered on another. Each ZMQ socket is
    then touched by one thread only, which the sockets require. ``blocking`` is the low-level
    client under the wrapper, the one that carries ``execute``, ``input`` and the per-channel
    reads.
    """
    from queue import Empty

    blocking = client._manager.client
    msg_id = blocking.execute(SHIM % {"stub": STUB}, allow_stdin=True)
    done = threading.Event()

    def answer_stdin() -> None:
        """One line of this process's standard input answers one input request, in order.

        End of input sends the kernel's EOF character, which the shim's ``input()`` raises on,
        so its loop breaks and the cell ends. The pairing the shim relies on holds because the
        cell issues one request per read and this sends one reply per request.
        """
        while not done.is_set():
            try:
                request = blocking.get_stdin_msg(timeout=1)
            except Empty:
                continue
            if request["parent_header"].get("msg_id") != msg_id:
                continue
            if request.get("msg_type") not in ("input_request", None):
                continue
            line = sys.stdin.readline()
            try:
                blocking.input(line.rstrip("\n") if line else "\x04")
            except Exception:
                return

    reader = threading.Thread(target=answer_stdin, daemon=True)
    reader.start()
    try:
        while True:
            try:
                message = blocking.get_iopub_msg(timeout=1)
            except Empty:
                continue
            if message["parent_header"].get("msg_id") != msg_id:
                continue
            if message.get("msg_type") == "stream":
                text = message["content"].get("text") or ""
                stream = sys.stdout if message["content"].get("name") == "stdout" else sys.stderr
                stream.write(text)
                stream.flush()
            elif (
                message.get("msg_type") == "status"
                and message["content"].get("execution_state") == "idle"
            ):
                break
    except Exception as exc:
        print(f"the cell did not finish: {type(exc).__name__}: {exc}", file=sys.stderr)
        return UNREACHABLE
    finally:
        done.set()
        try:
            client.stop(shutdown_kernel=False)
        except Exception:
            pass

    # The cell ends when letify closes this process's standard input, which the stdin thread
    # turns into the EOF character the shim's loop breaks on.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
