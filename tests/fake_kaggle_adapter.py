"""A standard library stand-in for letify's Kaggle adapter.

It keeps the adapter's contract from spec "Kaggle Jupyter Server session": the URL, kernel id
and timeout arrive in ``LETIFY_JUPYTER_URL``, ``LETIFY_KERNEL_ID`` and ``LETIFY_TIMEOUT``, the
source on standard input. It asks the fake server whether that kernel exists, then runs the
source in a fresh local interpreter, so the driver programs letify sends are the real ones.

Exit 0 on success, 3 when the program raised, 4 when the server or kernel cannot be reached.
Every invocation's environment keys and kernel id are appended to ``FAKE_KAGGLE_LOG``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

url = os.environ["LETIFY_JUPYTER_URL"]
kernel = os.environ["LETIFY_KERNEL_ID"]
timeout = float(os.environ.get("LETIFY_TIMEOUT") or 120)
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

source = sys.stdin.read()
try:
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=timeout
    )
except subprocess.TimeoutExpired:
    print("timed out", file=sys.stderr)
    raise SystemExit(4) from None
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
raise SystemExit(0 if result.returncode == 0 else 3)
