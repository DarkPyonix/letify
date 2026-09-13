"""Serializing a call and reading back its outcome.

Two shapes are supported because two kinds of channel exist.

A persistent channel keeps one worker process alive, so a call is a framed
request and the outcome comes back on the same pipe. That is the normal path, and
it is what makes handles and blob reuse possible.

A one-shot channel has no living process, so the call travels inside a driver
script that prints its outcome between two markers. Anything that can only run a
command and collect its output uses this.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from .._vendor import cloudpickle
from ..errors import ConfigError, ProtocolError, RemoteError
from .handle import Handle

#: Markers the one-shot driver writes around its outcome, so a result can be found
#: in a stream that also carries the user's prints.
BEGIN = "__LETIFY_RESULT_BEGIN__"
END = "__LETIFY_RESULT_END__"

#: Bumped when the request or reply shape changes in a breaking way.
PROTOCOL_VERSION = 2

#: Arguments larger than this are content addressed instead of inlined.
INLINE_LIMIT = 64 * 1024


def ship_by_value(modules: Sequence[str]) -> None:
    """Make these modules travel inside the payload instead of by name.

    cloudpickle sends a function defined in the running script by value and one imported
    from a module by reference, which is right for anything the lock file installs: sending
    numpy by value would mean sending numpy over the network on every call. It is wrong for
    the project's own code, which the machine on the other end does not have, so that has
    to be named here.

    Registration is global to cloudpickle rather than held by the declaration, so this is
    called again on every call and is cheap to repeat.
    """
    for name in modules:
        try:
            module = import_module(name)
        except ImportError as exc:
            raise ConfigError(
                f"cannot ship {name!r}, because it does not import here: {exc}. "
                f"ship() names a module this process can import, not a file path."
            ) from exc
        cloudpickle.register_pickle_by_value(module)


def dumps_call(fn: Any, args: tuple, kwargs: dict) -> bytes:
    """Serialize a call with cloudpickle, which handles locally defined functions."""
    return cloudpickle.dumps((fn, args, kwargs), protocol=5)


def digest_of(payload: bytes) -> str:
    """Content address of a payload: BLAKE2b from the standard library, 16 bytes as hex.

    The standard library rather than a faster wheel, because letify installs nothing else and
    the worker on the far side computes the same address with nothing installed either.
    Hashing still runs at hundreds of megabytes per second, far above any uplink.
    """
    import hashlib

    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def unwrap(outcome: dict[str, Any], *, runtime_key: str) -> Any:
    """Turn a worker outcome into a value, raising what the remote side raised."""
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
    return outcome.get("value")


# -- the one-shot path ---------------------------------------------------------


def split_output(stdout: str) -> tuple[str, str | None]:
    """Return ``(user_output, encoded_outcome)`` from a one-shot driver's stdout."""
    start = stdout.rfind(BEGIN)
    if start == -1:
        return stdout, None
    stop = stdout.find(END, start)
    if stop == -1:
        return stdout, None
    encoded = stdout[start + len(BEGIN) : stop]
    logs = stdout[:start] + stdout[stop + len(END) :]
    return logs, encoded.strip()


def parse(stdout: str, *, runtime_key: str) -> tuple[str, Any]:
    """Parse a one-shot driver's stdout into ``(logs, value)``.

    A missing marker is not a protocol quirk, it means the remote process died.
    The message names the causes worth checking first.
    """
    import base64

    logs, encoded = split_output(stdout)
    if encoded is None:
        raise ProtocolError(
            "the runtime produced no result marker, so the process died before it "
            "finished. The usual causes are an out of memory kill, a preempted "
            "session, or a crash below Python.\n"
            f"--- last remote output ---\n{stdout[-2000:]}"
        )
    try:
        outcome = pickle.loads(base64.b64decode(encoded))
    except Exception as exc:
        raise ProtocolError(f"the remote result could not be decoded: {exc}") from exc
    return logs, unwrap(outcome, runtime_key=runtime_key)


__all__ = [
    "BEGIN",
    "END",
    "INLINE_LIMIT",
    "PROTOCOL_VERSION",
    "digest_of",
    "dumps_call",
    "parse",
    "split_output",
    "unwrap",
]
