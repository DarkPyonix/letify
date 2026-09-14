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
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Any

import cloudpickle

from ..errors import ConfigError, ProtocolError, RemoteError

#: Markers the one-shot driver writes around its outcome, so a result can be found
#: in a stream that also carries the user's prints.
BEGIN = "__LETIFY_RESULT_BEGIN__"
END = "__LETIFY_RESULT_END__"

#: Bumped when the request or reply shape changes in a breaking way.
PROTOCOL_VERSION = 3

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


def dumps_call_parts(fn: Any, args: tuple, kwargs: dict) -> tuple[bytes, list[pickle.PickleBuffer]]:
    """Serialize a call with its out-of-band buffers kept apart, for a binary channel."""
    buffers: list[pickle.PickleBuffer] = []
    head = cloudpickle.dumps((fn, args, kwargs), protocol=5, buffer_callback=buffers.append)
    return head, buffers


def _hasher() -> Any:
    """A fresh incremental hasher: blake3 on every core, or blake2b where blake3 is missing."""
    try:
        import blake3
    except ImportError:
        import hashlib

        return hashlib.blake2b(digest_size=16)
    return blake3.blake3(max_threads=blake3.blake3.AUTO)


def digest_parts(parts: Iterable[Any]) -> str:
    """Content address of the concatenation of ``parts``, hashed without joining them.

    blake3 is preferred because it hashes at several gigabytes per second, which
    keeps hashing orders of magnitude away from being the bottleneck on any real
    network link. blake2b from the standard library is the fallback.
    """
    hasher = _hasher()
    for part in parts:
        hasher.update(part)
    if hasher.name == "blake2b":
        return hasher.hexdigest()
    return hasher.hexdigest(length=16)


def digest_of(payload: bytes) -> str:
    """Content address of one payload."""
    return digest_parts((payload,))


class DigestCache:
    """Digests of immutable arguments, so a repeated argument is not hashed again.

    Spec "Argument addressing": a ``bytes`` entry holds its object, so its id cannot be
    reused while the entry lives, and at most ``limit`` such entries are kept. Any other
    immutable value is held by weak reference and forgotten when it is collected.
    """

    def __init__(self, limit: int = 16):
        self._limit = limit
        self._strong: OrderedDict[int, tuple[bytes, str, int]] = OrderedDict()
        self._weak: dict[int, tuple[weakref.ref, str, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def immutable(value: Any) -> bool:
        """Whether ``value`` cannot change: ``bytes``, or a weakly referenced read-only buffer."""
        if type(value) is bytes:
            return True
        try:
            weakref.ref(value)
            with memoryview(value) as view:
                return view.readonly
        except TypeError:
            return False

    def get(self, value: Any) -> tuple[str, int] | None:
        key = id(value)
        with self._lock:
            if type(value) is bytes:
                strong = self._strong.get(key)
                if strong is None or strong[0] is not value:
                    return None
                self._strong.move_to_end(key)
                return strong[1], strong[2]
            weak = self._weak.get(key)
            if weak is None or weak[0]() is not value:
                return None
            return weak[1], weak[2]

    def put(self, value: Any, digest: str, size: int) -> None:
        key = id(value)
        with self._lock:
            if type(value) is bytes:
                self._strong[key] = (value, digest, size)
                self._strong.move_to_end(key)
                while len(self._strong) > self._limit:
                    self._strong.popitem(last=False)
                return

            def forget(ref: weakref.ref, key: int = key) -> None:
                with self._lock:
                    if self._weak.get(key, (None,))[0] is ref:
                        del self._weak[key]

            self._weak[key] = (weakref.ref(value, forget), digest, size)


def unwrap(outcome: dict[str, Any], *, runtime_key: str) -> Any:
    """Turn a worker outcome into a value, raising what the remote side raised."""
    if not isinstance(outcome, dict) or "ok" not in outcome:
        raise ProtocolError(f"unexpected remote payload: {outcome!r}")
    if not outcome["ok"]:
        raise RemoteError(
            outcome.get("error", "the remote call failed"),
            outcome.get("traceback", ""),
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
    "DigestCache",
    "digest_of",
    "digest_parts",
    "dumps_call",
    "dumps_call_parts",
    "parse",
    "split_output",
    "unwrap",
]
