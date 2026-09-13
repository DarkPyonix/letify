"""The Modal adapter: Modal's client, run outside the letify process.

letify starts this file with ``uv run --no-project --with modal python modal_adapter.py``
and talks to it in JSON lines, one request and one reply per line, as spec "Modal adapter"
describes. It owns every call into ``modal``. It does not own the worker protocol that runs
inside a sandbox, which it carries as opaque text.

It runs by file path in an environment that holds only Modal, so at load time it imports
the standard library alone, and ``modal`` only when an op needs it. That is also why a
missing Modal answers with a reply of kind ``unavailable`` rather than a crash.

Standard output carries replies only. Anything else that prints, Modal's own messages
included, is sent to standard error.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import traceback
from typing import Any


class Unavailable(Exception):
    """Modal cannot be imported in this environment."""


class NotFound(Exception):
    """A volume path does not exist."""


def load_modal() -> Any:
    try:
        import modal
    except ImportError as exc:
        raise Unavailable(f"the modal package cannot be imported: {exc}") from exc
    return modal


def not_found_errors(modal: Any) -> tuple[type[BaseException], ...]:
    found: list[type[BaseException]] = [FileNotFoundError]
    error = getattr(getattr(modal, "exception", None), "NotFoundError", None)
    if isinstance(error, type):
        found.append(error)
    return tuple(found)


class SandboxStream:
    """One sandbox and the part of its stdout that has been read but not yet returned."""

    def __init__(self, sandbox: Any):
        self.sandbox = sandbox
        self._chunks = iter(sandbox.stdout)
        self._buffer = ""
        self._ended = False

    def next_line(self) -> str | None:
        """The next line with its newline, the unterminated rest at the end, or None."""
        while "\n" not in self._buffer and not self._ended:
            chunk = next(self._chunks, None)
            if chunk is None:
                self._ended = True
                break
            self._buffer += chunk if isinstance(chunk, str) else chunk.decode()
        if "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            return line + "\n"
        rest, self._buffer = self._buffer, ""
        return rest or None


class Adapter:
    """The state one adapter process keeps: its sandboxes and the volumes it opened."""

    def __init__(self) -> None:
        self.sandboxes: dict[str, SandboxStream] = {}
        self.volumes: dict[str, Any] = {}

    def handle(self, request: dict[str, Any]) -> Any:
        op = request.get("op")
        method = getattr(self, f"op_{op}", None)
        if not isinstance(op, str) or method is None:
            raise ValueError(f"unknown op {op!r}")
        return method(request)

    # -- sandboxes ---------------------------------------------------------------

    def op_hello(self, request: dict[str, Any]) -> Any:
        modal = load_modal()
        return {"modal": getattr(modal, "__version__", "unknown")}

    def op_create(self, request: dict[str, Any]) -> Any:
        modal = load_modal()
        app = modal.App.lookup(str(request["app"]), create_if_missing=True)
        image = modal.Image.debian_slim()
        packages = [str(name) for name in request.get("packages") or []]
        if packages:
            image = image.pip_install(*packages)
        sandbox = modal.Sandbox.create(
            *[str(part) for part in request["args"]],
            app=app,
            image=image,
            gpu=request.get("gpu") or None,
            timeout=int(request["timeout"]),
            volumes={
                str(path): modal.Volume.from_name(str(name), create_if_missing=True)
                for path, name in (request.get("volumes") or {}).items()
            },
        )
        sandbox_id = str(getattr(sandbox, "object_id", "") or f"sandbox-{len(self.sandboxes) + 1}")
        self.sandboxes[sandbox_id] = SandboxStream(sandbox)
        return {"sandbox": sandbox_id}

    def _stream(self, request: dict[str, Any]) -> SandboxStream:
        sandbox_id = str(request["sandbox"])
        stream = self.sandboxes.get(sandbox_id)
        if stream is None:
            raise KeyError(f"no sandbox {sandbox_id!r} in this adapter")
        return stream

    def op_write(self, request: dict[str, Any]) -> Any:
        stream = self._stream(request)
        stream.sandbox.stdin.write(str(request["data"]).encode())
        stream.sandbox.stdin.drain()
        return None

    def op_read_until(self, request: dict[str, Any]) -> Any:
        stream = self._stream(request)
        prefixes = tuple(str(prefix) for prefix in request["prefixes"])
        lines: list[str] = []
        while True:
            line = stream.next_line()
            if line is None:
                return {"lines": lines, "eof": True}
            lines.append(line)
            if line.startswith(prefixes):
                return {"lines": lines, "eof": False}

    def op_terminate(self, request: dict[str, Any]) -> Any:
        stream = self.sandboxes.pop(str(request["sandbox"]), None)
        if stream is not None:
            stream.sandbox.terminate()
        return None

    # -- volumes -----------------------------------------------------------------

    def _volume(self, request: dict[str, Any]) -> tuple[Any, Any]:
        modal = load_modal()
        name = str(request["volume"])
        if name not in self.volumes:
            self.volumes[name] = modal.Volume.from_name(name, create_if_missing=True)
        return modal, self.volumes[name]

    def op_volume_put(self, request: dict[str, Any]) -> Any:
        _, volume = self._volume(request)
        payload = base64.b64decode(str(request["data"]))
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(payload), str(request["path"]))
        return None

    def op_volume_get(self, request: dict[str, Any]) -> Any:
        modal, volume = self._volume(request)
        path = str(request["path"])
        try:
            payload = b"".join(volume.read_file(path))
        except not_found_errors(modal) as exc:
            raise NotFound(path) from exc
        return base64.b64encode(payload).decode()

    def op_volume_list(self, request: dict[str, Any]) -> Any:
        modal, volume = self._volume(request)
        path = str(request["path"])
        try:
            entries = volume.listdir(path, recursive=True)
        except not_found_errors(modal) as exc:
            raise NotFound(path) from exc
        return [str(entry.path) for entry in entries]


def serve(stdin: Any, stdout: Any) -> None:
    """Answer requests from ``stdin`` until it closes, then terminate what is left."""
    adapter = Adapter()
    for line in stdin:
        if not line.strip():
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            reply = {"id": request_id, "ok": True, "value": adapter.handle(request)}
        except Unavailable as exc:
            reply = {"id": request_id, "ok": False, "kind": "unavailable", "error": str(exc)}
        except NotFound as exc:
            reply = {"id": request_id, "ok": False, "kind": "not_found", "error": str(exc)}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = {
                "id": request_id,
                "ok": False,
                "kind": "failure",
                "error": f"{type(exc).__name__}: {exc}",
            }
        stdout.write(json.dumps(reply) + "\n")
        stdout.flush()
    for stream in adapter.sandboxes.values():
        try:
            stream.sandbox.terminate()
        except Exception:
            traceback.print_exc(file=sys.stderr)


def main() -> None:
    replies = sys.stdout
    sys.stdout = sys.stderr
    serve(sys.stdin, replies)


if __name__ == "__main__":
    main()
