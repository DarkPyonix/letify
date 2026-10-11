"""Where a replica learns which one it is, and what it reduces with.

Owns the per-replica facts a distributed body reads, ``rank`` and ``world_size``, and the
reduction a body asks for. It does not own the loop; spec "Replicas" says so and
``docs/INTENT.md`` keeps that as a non-goal for every loop but one.
"""

from __future__ import annotations

import threading
from typing import Any

from ..errors import UnsupportedMode

#: Set on the thread running a distributed body, so ``rank`` and ``world_size`` answer.
_WHO = threading.local()

_OUTSIDE = (
    "{name}() answers only inside a call declared with host=letify.distributed. A body "
    "under host=letify.remote has one replica and no rank, and answering 0 and 1 here "
    "would hide that it was asked"
)


def rank() -> int:
    """Which replica this is, 0 upward in the order the ``device`` list names.

    Spec "Replicas": raises outside a distributed call rather than answering 0.
    """
    found = getattr(_WHO, "rank", None)
    if found is None:
        raise UnsupportedMode(_OUTSIDE.format(name="rank"))
    return int(found)


def world_size() -> int:
    """How many replicas started. Spec "Replicas"."""
    found = getattr(_WHO, "world_size", None)
    if found is None:
        raise UnsupportedMode(_OUTSIDE.format(name="world_size"))
    return int(found)


class _Replica:
    """Holds the rank for the body's thread, so nothing has to be passed through it."""

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size

    def __enter__(self) -> _Replica:
        _WHO.rank, _WHO.world_size = self.rank, self.world_size
        return self

    def __exit__(self, *_exc: Any) -> None:
        _WHO.rank = _WHO.world_size = None


__all__ = ["rank", "world_size"]
