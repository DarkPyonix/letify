"""Deferred values for single value reads under ``host="local"``.

This module owns the object a forwarded ``Tensor.item()`` returns, as spec "Deferred value
reads" describes: it holds the pending read and the conversion that turns the filled CPU
tensor into a Python value, and it resolves the moment anything asks for that value. It
does not own queueing the read, which is ``client.read_pending``, and it does not decide
which methods are replaced, which is ``tensor.SPECIAL``.
"""

from __future__ import annotations

import dataclasses
import math
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Attributes answered without resolving. Everything else waits for the value first.
_OWN = frozenset({"resolved", "value"})


def auto_fetch_enabled(configured: bool | None) -> bool:
    """``LETIFY_AUTO_FETCH`` when set, else the account's ``auto_fetch``, else on."""
    value = os.environ.get("LETIFY_AUTO_FETCH")
    if value is not None:
        return value.strip().lower() not in ("0", "false", "no")
    if configured is None:
        return _home_setting()
    return configured is not False


def _home_setting() -> bool:
    """``auto_fetch`` at the top level of ``~/.letify/config.toml``, on when absent."""
    home_file = Path.home() / ".letify" / "config.toml"
    try:
        settings = tomllib.loads(home_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return True
    return settings.get("auto_fetch") is not False


class Deferred:
    """One value read that has been queued but not waited for.

    It stands in for the ``float``, ``int`` or ``list`` the read will produce. Every use of
    the value resolves it, which waits for that one read and nothing else.
    """

    __slots__ = ("_client", "_convert", "_read", "_resolved", "_value")

    def __init__(self, client: Any, read: Any, convert: Callable[[Any], Any]):
        self._client = client
        self._read = read
        self._convert = convert
        self._resolved = False
        self._value: Any = None

    @property
    def resolved(self) -> bool:
        """Whether the value is in hand, answered without waiting."""
        return self._resolved

    @property
    def value(self) -> Any:
        """The value, waiting for its own read when it has not arrived."""
        if not self._resolved:
            if not self._read.done:
                self._client.stats.resolved_early += 1
            target = self._client.wait(self._read)
            self._value = self._convert(target)
            self._resolved = True
        return self._value

    # -- conversion ---------------------------------------------------------------------

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return self.value.__index__()

    def __bool__(self) -> bool:
        return bool(self.value)

    def __complex__(self) -> complex:
        return complex(self.value)

    def __round__(self, digits: int | None = None) -> Any:
        return round(self.value) if digits is None else round(self.value, digits)

    def __trunc__(self) -> int:
        return math.trunc(self.value)

    def __floor__(self) -> int:
        return math.floor(self.value)

    def __ceil__(self) -> int:
        return math.ceil(self.value)

    # -- text ---------------------------------------------------------------------------

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)

    # -- leaving the process ------------------------------------------------------------

    def __reduce__(self) -> tuple:
        """Pickling gives the value, so a deferred value never reaches a saved file."""
        return (_identity, (self.value,))

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        import numpy

        if copy is None:
            return numpy.asarray(self.value, dtype=dtype)
        return numpy.array(self.value, dtype=dtype, copy=copy)

    def __copy__(self) -> Any:
        return self.value

    def __deepcopy__(self, memo: dict) -> Any:
        return self.value

    # -- containers ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self) -> Any:
        return iter(self.value)

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __contains__(self, item: Any) -> bool:
        return item in self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in _OWN:
            raise AttributeError(name)
        return getattr(self.value, name)


def _identity(value: Any) -> Any:
    return value


def _binary(name: str) -> Callable[..., Any]:
    def operate(self: Deferred, other: Any) -> Any:
        return getattr(self.value, name)(_plain(other))

    operate.__name__ = name
    return operate


def _unary(name: str) -> Callable[..., Any]:
    def operate(self: Deferred) -> Any:
        return getattr(self.value, name)()

    operate.__name__ = name
    return operate


def _plain(value: Any) -> Any:
    return value.value if type(value) is Deferred else value


_BINARY = (
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__mod__",
    "__rmod__",
    "__pow__",
    "__rpow__",
    "__divmod__",
    "__rdivmod__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
)
_UNARY = ("__neg__", "__pos__", "__abs__")

for _name in _BINARY:
    setattr(Deferred, _name, _binary(_name))
for _name in _UNARY:
    setattr(Deferred, _name, _unary(_name))
del _name


def resolve(value: Any, _seen: set[int] | None = None) -> Any:
    """Replace every deferred value in ``value`` by the value it stands for.

    Walks lists, tuples, sets, dictionaries and dataclasses, so nothing a call returns
    leaves with a read still pending.
    """
    if type(value) is Deferred:
        return value.value
    kind = type(value)
    if kind in (str, bytes, int, float, bool, type(None)):
        return value
    seen = set() if _seen is None else _seen
    if id(value) in seen:
        return value
    if kind in (list, tuple, set, frozenset, dict):
        seen.add(id(value))
        try:
            if kind is dict:
                return {resolve(key, seen): resolve(item, seen) for key, item in value.items()}
            built = [resolve(item, seen) for item in value]
            return kind(built) if kind is not tuple else tuple(built)
        finally:
            seen.discard(id(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(id(value))
        try:
            for field in dataclasses.fields(value):
                current = getattr(value, field.name)
                replaced = resolve(current, seen)
                if replaced is not current:
                    object.__setattr__(value, field.name, replaced)
        finally:
            seen.discard(id(value))
    return value


__all__ = ["Deferred", "auto_fetch_enabled", "resolve"]
