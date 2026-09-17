"""A standard library stand-in for letify's Kaggle adapter bridge.

It keeps the bridge's contract from spec "Kaggle Jupyter Server session": the URL and the
kernel id arrive in ``LETIFY_JUPYTER_URL`` and ``LETIFY_KERNEL_ID``, and the process lives
for the runtime rather than for one program. Base64 frame lines arrive on standard input and
leave on standard output, so the worker the bridge carries is the real one.

The one thing it does not do is speak to a kernel. It asks the fake server whether the
kernel exists, then runs the worker in a local interpreter and moves lines between that
interpreter and its own standard input and output. What a real kernel adds is the
``input_request`` and ``stream`` hops, which change who carries a line and not what a line
is.

Exit 4 when the server or kernel cannot be reached, otherwise the worker's own exit status.
Every invocation's kernel id is appended to ``FAKE_KAGGLE_LOG``.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

url = os.environ["LETIFY_JUPYTER_URL"]
kernel = os.environ["LETIFY_KERNEL_ID"]
parts = urllib.parse.urlsplit(url)
base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
token = dict(urllib.parse.parse_qsl(parts.query)).get("token", "")

log = os.environ.get("FAKE_KAGGLE_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kernel": kernel, "argv": sys.argv[1:]}) + "\n")

try:
    request = urllib.request.Request(f"{base}/api/kernels/{kernel}?token={token}")
    urllib.request.urlopen(request, timeout=10).read()
except (urllib.error.URLError, OSError) as exc:
    print(f"cannot reach the kernel: {exc}", file=sys.stderr)
    raise SystemExit(4) from None

# The cell a real bridge executes carries a shim, because a kernel answers input() and not a
# binary read. Here the same shape is a subprocess whose standard input and output are pipes:
# the bridge copies lines in and out, exactly as it would copy them to input_request replies
# and from stream messages.
STUB = (
    "import sys;"
    "b=sys.stdin.buffer;"
    "n=int(b.readline());"
    "exec(compile(b.read(n).decode(),'letify-worker','exec'))"
)

worker = subprocess.Popen(
    [sys.executable, "-u", "-c", STUB],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=None,
)
assert worker.stdin is not None and worker.stdout is not None


def pump_out() -> None:
    """Every line the worker writes goes out, as a stream message would carry it."""
    for line in worker.stdout:
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()


reader = threading.Thread(target=pump_out, daemon=True)
reader.start()

try:
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        # Decoded here, as the cell's shim does in the real bridge: the channel writes
        # base64 lines, and the worker reads raw bytes from its standard input.
        worker.stdin.write(base64.b64decode(text))
        worker.stdin.flush()
except (OSError, ValueError):
    pass

try:
    worker.stdin.close()
except OSError:
    pass
status = worker.wait()
reader.join(5)
raise SystemExit(status)
