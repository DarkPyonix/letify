"""The session cache: a value kept for the length of one session.

Owns ``session_cache``, the store behind it and the lock that makes first use build
once. It does not own session lifetime: the store lives in the process that imports
this module, so it ends when that process ends, which in a runtime is when the session
ends.

The store has to live here, in a module the worker imports by reference, because
cloudpickle ships ``__main__`` globals by value with every call. A dict at module level
in the user's script is therefore a fresh copy on every call and keeps nothing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

T = TypeVar("T")

_VALUES: dict[Hashable, Any] = {}
_BUILDING: dict[Hashable, threading.Lock] = {}
_GUARD = threading.Lock()


def session_cache(key: Hashable, factory: Callable[[], T]) -> T:
    """Return the value stored under ``key`` in this session, building it on first use.

    ``factory`` takes no arguments and is called once per key per process while it
    succeeds. Concurrent first use of one key builds once and the other callers
    receive that value. A factory that raises stores nothing, so the next use tries
    again.
    """
    with _GUARD:
        if key in _VALUES:
            return _VALUES[key]
        building = _BUILDING.setdefault(key, threading.Lock())
    with building:
        with _GUARD:
            if key in _VALUES:
                return _VALUES[key]
        value = factory()
        with _GUARD:
            _VALUES[key] = value
            _BUILDING.pop(key, None)
        return value


__all__ = ["session_cache"]
