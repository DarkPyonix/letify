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
it never appears in a process list.

Exit 4 when the server or kernel could not be reached, 5 when the cell ended by itself, and 0
when letify closed the bridge's standard input.
"""

from __future__ import annotations

import os
import sys
import threading
import urllib.parse

UNREACHABLE = 4
CELL_ENDED = 5
RAISED = 3

#: Seconds one program may run when the caller names none.
DEFAULT_PROGRAM_TIMEOUT = 3600.0

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


def one_program(client, source: str, timeout: float) -> int:
    """Run one program in the kernel and exit, which is what a login needs.

    Recording the session's devices happens before any runtime exists, so there is no
    channel to carry frames and no worker to keep alive. Selected by
    ``LETIFY_ADAPTER_MODE=program``, never by the presence of another variable: a duration
    that also decides which mode runs is two meanings in one name, which is how a provider
    once came to skip a check it should have run.
    """
    try:
        reply = client.execute(source, timeout=timeout)
    except Exception as exc:
        print(f"the execution did not finish: {type(exc).__name__}: {exc}", file=sys.stderr)
        return UNREACHABLE

    for output in reply.get("outputs") or []:
        kind = output.get("output_type")
        if kind == "stream":
            stream = sys.stdout if output.get("name") == "stdout" else sys.stderr
            stream.write(output.get("text") or "")
        elif kind == "error":
            sys.stderr.write("\n".join(output.get("traceback") or []) + "\n")
    sys.stdout.flush()
    return 0 if reply.get("status") == "ok" else RAISED


def main() -> int:
    url = os.environ["LETIFY_JUPYTER_URL"]
    kernel = os.environ["LETIFY_KERNEL_ID"]
    mode = os.environ.get("LETIFY_ADAPTER_MODE", "bridge")
    timeout = float(os.environ.get("LETIFY_TIMEOUT") or DEFAULT_PROGRAM_TIMEOUT)
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

    if mode == "program":
        try:
            return one_program(client, sys.stdin.read(), timeout)
        finally:
            try:
                client.stop(shutdown_kernel=False)
            except Exception:
                pass

    def output_hook(message: dict) -> None:
        """Every stream message the cell writes goes out as it arrives."""
        if message.get("msg_type") != "stream":
            return
        text = message.get("content", {}).get("text") or ""
        stream = sys.stdout if message["content"].get("name") == "stdout" else sys.stderr
        stream.write(text)
        stream.flush()

    closed = threading.Event()

    def stdin_hook(_message: dict) -> None:
        """Answer the cell's read with the next line letify wrote to this process."""
        line = sys.stdin.readline()
        if not line:
            closed.set()
            # An empty reply ends the cell's loop through EOFError on its next read.
            client.input("")
            return
        client.input(line.rstrip("\n"))

    try:
        client.execute_interactive(
            SHIM % {"stub": STUB},
            allow_stdin=True,
            output_hook=output_hook,
            stdin_hook=stdin_hook,
            timeout=None,
        )
    except Exception as exc:
        print(f"the cell did not finish: {type(exc).__name__}: {exc}", file=sys.stderr)
        return UNREACHABLE
    finally:
        try:
            client.stop(shutdown_kernel=False)
        except Exception:
            pass

    return 0 if closed.is_set() else CELL_ENDED


if __name__ == "__main__":
    raise SystemExit(main())
