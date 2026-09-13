"""Checks that run before anything is sent.

Catching these locally is the point. A handle that belongs to another runtime
would otherwise be discovered on the remote side, after the payload has already
crossed the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..errors import HandleScopeError
from .handle import Handle


def check_handles(runtime_key: str, args: tuple, kwargs: dict) -> None:
    """Reject handles that belong to a different runtime.

    A handle is a pointer into one process and one CUDA context, so resolving one
    elsewhere would mean copying the whole object across the network without the
    caller asking for it. Raising is the honest answer.
    """
    for value in walk((*args, *kwargs.values())):
        if isinstance(value, Handle) and value.runtime != runtime_key:
            raise HandleScopeError(
                f"{value!r} belongs to runtime {value.runtime!r} but the call targets "
                f"{runtime_key!r}. Route the call to the owning runtime, or return the "
                f"value to this process before passing it on."
            )


def walk(values: Any) -> Iterator[Any]:
    """Yield every leaf value in a nested structure of lists, tuples and dicts."""
    for value in values:
        if isinstance(value, (list, tuple, set)):
            yield from walk(value)
        elif isinstance(value, dict):
            yield from walk(value.values())
        else:
            yield value


__all__ = ["check_handles", "walk"]
