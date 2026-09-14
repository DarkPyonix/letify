"""The Kaggle adapter: runs one program in one kernel of a Kaggle Jupyter Server session.

Run by letify through ``uv run --no-project --with "jupyter-kernel-client<1" python -P`` so the
letify process never imports ``jupyter_kernel_client``, as spec "Kaggle Jupyter Server session"
describes. It owns only this one execution. Creating and deleting the kernel, and deciding
whether a failure means the session ended, belong to ``letify.providers.kaggle``.

Input: ``LETIFY_JUPYTER_URL`` (the Colab Compatible URL), ``LETIFY_KERNEL_ID``,
``LETIFY_TIMEOUT`` in seconds, and the program source on standard input. The URL is read from
the environment so it never appears in a process list.

Exit 0 when the execution reply status is ``ok``, 3 when the program raised, 4 when the
server or kernel could not be reached or the execution timed out.
"""

from __future__ import annotations

import os
import sys
import urllib.parse

UNREACHABLE = 4
RAISED = 3


def main() -> int:
    url = os.environ["LETIFY_JUPYTER_URL"]
    kernel = os.environ["LETIFY_KERNEL_ID"]
    timeout = float(os.environ.get("LETIFY_TIMEOUT") or 3600)
    parts = urllib.parse.urlsplit(url)
    base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    token = dict(urllib.parse.parse_qsl(parts.query)).get("token", "")
    source = sys.stdin.read()

    try:
        from jupyter_kernel_client import KernelClient

        client = KernelClient(server_url=base, token=token, kernel_id=kernel)
        client.start()
    except Exception as exc:  # any failure to attach means the session cannot be used
        print(f"the kernel could not be reached: {type(exc).__name__}", file=sys.stderr)
        return UNREACHABLE

    try:
        reply = client.execute(source, timeout=timeout)
    except Exception as exc:
        print(f"the execution did not finish: {type(exc).__name__}: {exc}", file=sys.stderr)
        return UNREACHABLE
    finally:
        try:
            client.stop(shutdown_kernel=False)
        except Exception:
            pass

    for output in reply.get("outputs") or []:
        kind = output.get("output_type")
        if kind == "stream":
            stream = sys.stdout if output.get("name") == "stdout" else sys.stderr
            stream.write(output.get("text") or "")
        elif kind == "error":
            sys.stderr.write("\n".join(output.get("traceback") or []) + "\n")
    sys.stdout.flush()
    return 0 if reply.get("status") == "ok" else RAISED


if __name__ == "__main__":
    raise SystemExit(main())
