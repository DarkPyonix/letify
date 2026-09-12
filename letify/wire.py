"""The call protocol.

A declared function is serialized with cloudpickle, base64 encoded and embedded
in a small driver script that the runtime executes. The driver prints the outcome
as base64 between two markers, so the result can be found in a stream that also
carries the user's own prints.

Three details matter for performance and correctness.

Arguments are content addressed. Every large argument is hashed first and sent
only when the runtime does not already hold that digest, so the same weights
passed to ten calls travel once.

Results may stay remote. When a value is registered in the runtime's object
table, the caller receives a ``Handle`` instead of the value, and the next call
resolves it in place. That is what keeps a model on the remote machine instead of
copying it back and forth.

Side effects travel one way. A callback from the runtime to the local process is
fire and forget. Waiting for an acknowledgement would put one round trip in the
token loop, which is the single worst thing this design can do.
"""

from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from typing import Any

import cloudpickle

BEGIN = "__LETIFY_RESULT_BEGIN__"
END = "__LETIFY_RESULT_END__"
CALLBACK = "__LETIFY_CALLBACK__"

#: Raised through the wire when the driver shape changes in a breaking way.
PROTOCOL_VERSION = 1

#: Values below this size travel inline. Larger ones are content addressed.
INLINE_LIMIT = 64 * 1024


@dataclass(frozen=True, slots=True)
class Handle:
    """A reference to an object that lives in a runtime.

    The handle carries the key of the runtime that owns it. Passing it to a call
    on a different runtime raises ``HandleScopeError`` rather than silently
    copying gigabytes across the network.
    """

    runtime: str
    object_id: str
    type_name: str
    summary: str = ""

    #: Read by the remote driver to recognize a handle without importing letify.
    __letify_kind__ = "handle"

    def __repr__(self) -> str:
        return f"<Handle {self.type_name} {self.object_id[:8]} on {self.runtime}>"


@dataclass(frozen=True, slots=True)
class Blob:
    """A content addressed argument that the runtime may already hold."""

    digest: str
    size: int

    def __repr__(self) -> str:
        return f"<Blob {self.digest[:8]} {self.size} bytes>"


_DRIVER = '''
import base64, pickle, sys, traceback
_PAYLOAD = "{payload}"
_BEGIN, _END = "{begin}", "{end}"
_VERSION = {version}

def _emit(obj):
    try:
        blob = pickle.dumps(obj, protocol=5)
    except Exception:
        blob = pickle.dumps(
            {{
                "ok": False,
                "error": "the return value could not be serialized",
                "traceback": traceback.format_exc(),
            }},
            protocol=5,
        )
    sys.stdout.flush()
    sys.stdout.write("\\n" + _BEGIN + base64.b64encode(blob).decode() + _END + "\\n")
    sys.stdout.flush()

try:
    import cloudpickle
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cloudpickle"], check=True)
    import cloudpickle

_TABLE = globals().setdefault("__letify_objects__", {{}})

def _resolve(value):
    """Replace handles with the objects they point at, recursively."""
    kind = getattr(value, "__letify_kind__", None)
    if kind == "handle":
        return _TABLE[value.object_id]
    if isinstance(value, (list, tuple)):
        return type(value)(_resolve(v) for v in value)
    if isinstance(value, dict):
        return {{k: _resolve(v) for k, v in value.items()}}
    return value

try:
    fn, args, kwargs, keep = cloudpickle.loads(base64.b64decode(_PAYLOAD))
    args = _resolve(args)
    kwargs = _resolve(kwargs)
except Exception:
    _emit({{
        "ok": False,
        "error": "the call could not be deserialized",
        "traceback": traceback.format_exc(),
    }})
else:
    try:
        value = fn(*args, **kwargs)
        if hasattr(value, "__await__"):
            # An async declaration makes the local call awaitable. The body still
            # has to run to completion here, and awaiting it remotely is what lets
            # that body use await internally.
            import asyncio
            value = asyncio.run(value)
    except BaseException as exc:
        _emit({{
            "ok": False,
            "error": "{{}}: {{}}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }})
    else:
        if keep:
            import uuid
            oid = uuid.uuid4().hex
            _TABLE[oid] = value
            _emit({{
                "ok": True,
                "handle": {{
                    "object_id": oid,
                    "type_name": type(value).__name__,
                    "summary": repr(value)[:200],
                }},
            }})
        else:
            _emit({{"ok": True, "value": value}})
'''


def encode_call(fn: Any, args: tuple, kwargs: dict, *, keep_remote: bool = False) -> str:
    """Serialize a call into the driver script that runs it remotely.

    When ``keep_remote`` is true the return value is registered in the runtime's
    object table and a handle comes back instead of the value.
    """
    payload = base64.b64encode(
        cloudpickle.dumps((fn, args, kwargs, keep_remote), protocol=5)
    ).decode()
    return _DRIVER.format(
        payload=payload,
        begin=BEGIN,
        end=END,
        version=PROTOCOL_VERSION,
    )


def digest_of(payload: bytes) -> str:
    """Content address of a payload.

    blake3 is used when available because it hashes at several gigabytes per
    second, which keeps hashing far away from being the bottleneck on any real
    network link. blake2b is the fallback.
    """
    try:
        import blake3

        return blake3.blake3(payload).hexdigest(length=16)
    except ImportError:
        import hashlib

        return hashlib.blake2b(payload, digest_size=16).hexdigest()


def split_output(stdout: str) -> tuple[str, str | None]:
    """Return ``(user_output, encoded_result)`` from a runtime's stdout."""
    start = stdout.rfind(BEGIN)
    if start == -1:
        return stdout, None
    stop = stdout.find(END, start)
    if stop == -1:
        return stdout, None
    encoded = stdout[start + len(BEGIN) : stop]
    logs = stdout[:start] + stdout[stop + len(END) :]
    return logs, encoded.strip()


def decode_result(encoded: str, *, runtime_key: str) -> Any:
    """Unpack a remote outcome, re-raising remote failures locally."""
    from .errors import ProtocolError, RemoteError

    try:
        outcome = pickle.loads(base64.b64decode(encoded))
    except Exception as exc:
        raise ProtocolError(f"the remote result could not be decoded: {exc}") from exc

    if not isinstance(outcome, dict) or "ok" not in outcome:
        raise ProtocolError(f"unexpected remote payload: {outcome!r}")
    if not outcome["ok"]:
        raise RemoteError(
            outcome.get("error", "the remote call failed"),
            outcome.get("traceback", ""),
        )
    if "handle" in outcome:
        spec = outcome["handle"]
        return Handle(
            runtime=runtime_key,
            object_id=spec["object_id"],
            type_name=spec["type_name"],
            summary=spec.get("summary", ""),
        )
    return outcome["value"]


def parse(stdout: str, *, runtime_key: str) -> tuple[str, Any]:
    """Parse a runtime's stdout into ``(logs, value)``."""
    from .errors import ProtocolError

    logs, encoded = split_output(stdout)
    if encoded is None:
        raise ProtocolError(
            "the runtime produced no result marker, so the process died before it "
            "finished. The usual causes are an out of memory kill, a preempted "
            "session, or a crash below Python.\n"
            f"--- last remote output ---\n{stdout[-2000:]}"
        )
    return logs, decode_result(encoded, runtime_key=runtime_key)


def check_handles(runtime_key: str, args: tuple, kwargs: dict) -> None:
    """Reject handles that belong to a different runtime.

    A handle is a pointer into one process and one CUDA context. Copying the
    object across would be slow and silent, so this raises instead.
    """
    from .errors import HandleScopeError

    for value in _walk((*args, *kwargs.values())):
        if isinstance(value, Handle) and value.runtime != runtime_key:
            raise HandleScopeError(
                f"{value!r} belongs to runtime {value.runtime!r} but the call targets "
                f"{runtime_key!r}. Route the call to the owning runtime, or return the "
                f"value to this process before passing it on."
            )


def _walk(values: Any) -> Any:
    for value in values:
        if isinstance(value, (list, tuple, set)):
            yield from _walk(value)
        elif isinstance(value, dict):
            yield from _walk(value.values())
        else:
            yield value
