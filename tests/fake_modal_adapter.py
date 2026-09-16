"""A standard library stand-in for letify's Modal adapter.

It speaks the JSON-lines protocol of spec "Modal adapter" and needs no Modal account.
A sandbox is a local subprocess started from the requested arguments, with ``python3``
replaced by this interpreter, so the worker bootstrap and framing that run inside it are
the real ones. A volume is a directory under ``FAKE_MODAL_STATE``.

Every request is appended to ``requests.jsonl`` in that directory, and the ``MODAL_*``
environment the process started with is written to ``env.json``, so a test can assert on
what letify asked for and as which account.

``FAKE_MODAL_FAIL`` names ops answered with a failure reply. ``FAKE_MODAL_EXIT`` names ops
on which the process prints to standard error and exits, as a crashed adapter would.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

STATE = Path(os.environ["FAKE_MODAL_STATE"])
FAIL = set(filter(None, os.environ.get("FAKE_MODAL_FAIL", "").split(",")))
EXIT = set(filter(None, os.environ.get("FAKE_MODAL_EXIT", "").split(",")))


class NotFound(Exception):
    pass


class Sandbox:
    def __init__(self, args: list[str]):
        command = [sys.executable if args[0] == "python3" else args[0], *args[1:]]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # A Modal sandbox does not end when the adapter process dies, so the stand-in's
            # sandbox is kept out of the adapter's process group.
            start_new_session=True,
        )

    def write(self, data: str) -> None:
        assert self.process.stdin is not None
        decoded = base64.b64decode(data)
        # Modal's sandbox stdin writer refuses to buffer more than 2 MiB before a drain.
        if len(decoded) > 2 * 1024 * 1024:
            raise BufferError("Buffer size exceed limit. Call drain to flush the buffer.")
        self.process.stdin.write(decoded)
        self.process.stdin.flush()

    #: Modal breaks a sandbox stdout line longer than this into pieces that are not base64
    #: on their own, and ends the stream on a longer one. The stand-in ends the stream.
    OUTPUT_LINE = 64 * 1024

    def read_until(self, prefixes: list[str]) -> dict[str, object]:
        assert self.process.stdout is not None
        lines: list[str] = []
        while True:
            raw = self.process.stdout.readline()
            if not raw or len(raw.rstrip(b"\n")) > self.OUTPUT_LINE:
                return {"lines": lines, "eof": True}
            line = raw.decode()
            lines.append(line)
            if any(line.startswith(prefix) for prefix in prefixes):
                return {"lines": lines, "eof": False}

    def terminate(self) -> None:
        self.process.kill()
        self.process.wait()


#: Apps started, in order. Each is stopped when the process's input closes, as the real
#: adapter stops its ephemeral apps.
APPS: list[str] = []


def log_app(event: str, name: str) -> None:
    with (STATE / "apps.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps([event, name]) + "\n")


def known_sandboxes() -> dict[str, int]:
    """Every sandbox any stand-in process started under this state directory, with its pid."""
    log = STATE / "sandboxes.jsonl"
    if not log.is_file():
        return {}
    return dict(json.loads(line) for line in log.read_text(encoding="utf-8").splitlines())


def volume_file(volume: str, path: str) -> Path:
    return STATE / "volumes" / volume / path.lstrip("/")


def handle(request: dict, sandboxes: dict[str, Sandbox]) -> object:
    op = request["op"]
    if op == "hello":
        return {"modal": "fake"}
    if op == "billing_summary":
        summary = os.environ.get("FAKE_MODAL_BILLING")
        if summary is None:
            raise ValueError("no billing summary configured")
        return json.loads(summary)
    if op == "create":
        if request["app"] not in APPS:
            APPS.append(request["app"])
            log_app("run_start", request["app"])
        sandbox_id = f"sb-{len(known_sandboxes()) + 1}"
        sandboxes[sandbox_id] = Sandbox(list(request["args"]))
        with (STATE / "sandboxes.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps([sandbox_id, sandboxes[sandbox_id].process.pid]) + "\n")
        return {"sandbox": sandbox_id}
    if op == "tunnel":
        # FAKE_MODAL_TUNNEL names another address, such as a local TLS proxy in front of the
        # port. Otherwise the sandbox is a local process and its port is reached directly.
        tunnel = os.environ.get("FAKE_MODAL_TUNNEL")
        if tunnel:
            return json.loads(tunnel)
        return {"host": "127.0.0.1", "port": int(request["port"]), "tls": False}
    if op == "write":
        sandboxes[request["sandbox"]].write(request["data"])
        return None
    if op == "read_until":
        return sandboxes[request["sandbox"]].read_until(list(request["prefixes"]))
    if op == "terminate":
        sandbox = sandboxes.pop(request["sandbox"], None)
        if sandbox is not None:
            sandbox.terminate()
            return None
        # Another adapter process created it, as Modal's Sandbox.from_id would find it.
        pid = known_sandboxes().get(request["sandbox"])
        if pid is None:
            raise KeyError(f"no sandbox {request['sandbox']!r}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return None
    if op == "volume_put":
        target = volume_file(request["volume"], request["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(request["data"]))
        return None
    if op == "volume_get":
        target = volume_file(request["volume"], request["path"])
        if not target.is_file():
            raise NotFound(request["path"])
        return base64.b64encode(target.read_bytes()).decode()
    if op == "volume_delete":
        target = volume_file(request["volume"], request["path"])
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        return None
    if op == "volume_list":
        root = STATE / "volumes" / request["volume"]
        target = volume_file(request["volume"], request["path"])
        if target.is_file():
            return [target.relative_to(root).as_posix()]
        if not target.is_dir():
            raise NotFound(request["path"])
        return sorted(p.relative_to(root).as_posix() for p in target.rglob("*") if p.is_file())
    raise ValueError(f"unknown op {op!r}")


def main() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items() if key.startswith("MODAL_")}
    (STATE / "env.json").write_text(json.dumps(env), encoding="utf-8")
    sandboxes: dict[str, Sandbox] = {}
    for line in sys.stdin:
        request = json.loads(line)
        with (STATE / "requests.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(request) + "\n")
        op = request.get("op")
        if op in EXIT:
            sys.stderr.write(f"the fake adapter crashed on {op}\n")
            sys.stderr.flush()
            os._exit(3)
        reply: dict[str, object] = {"id": request.get("id")}
        try:
            if op in FAIL:
                raise RuntimeError(f"{op} was refused by the fake")
            reply.update(ok=True, value=handle(request, sandboxes))
        except NotFound as exc:
            reply.update(ok=False, kind="not_found", error=str(exc))
        except Exception as exc:
            reply.update(ok=False, kind="failure", error=f"{type(exc).__name__}: {exc}")
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    for sandbox in sandboxes.values():
        sandbox.terminate()
    for name in APPS:
        log_app("run_stop", name)


if __name__ == "__main__":
    main()
