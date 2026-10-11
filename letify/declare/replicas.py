"""Where a replica learns which one it is, and what it reduces with.

Owns the per-replica facts a distributed body reads, ``rank`` and ``world_size``, and the
reduction a body asks for. It does not own the loop; spec "Replicas" says so and
``docs/INTENT.md`` keeps that as a non-goal for every loop but one.
"""

from __future__ import annotations

import math
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


#: The fraction of a step a reduction may take before it is worth doing less often.
DEFAULT_SHARE = 0.25


def steps_for(
    *,
    param_bytes: int,
    step_seconds: float,
    rate_bps: float | None,
    share: float = DEFAULT_SHARE,
) -> int:
    """The smallest number of local steps for which a reduction is cheap enough.

    Spec "How often to synchronize": a reduction moves the parameters once, so it is
    worth it only when rare enough against the step it interrupts. Raises rather than
    guessing when the link reports no rate, because a guess here is a guess about the
    whole run's shape.
    """
    if not rate_bps or rate_bps <= 0:
        raise UnsupportedMode(
            "local_steps needs the link's measured upload rate and this link reports "
            "none, so there is nothing to divide the parameters by. The rate comes from "
            "the probe, so a link the pipeline never probed, such as a provider "
            "fallback, cannot answer. Pass the count instead of letify.auto"
        )
    if step_seconds <= 0:
        raise ValueError("step_seconds has to be above zero to divide a transfer against")
    allowed = max(share, 0.0) * step_seconds
    if allowed <= 0:
        raise ValueError("share has to be above zero, or no number of steps is ever enough")
    transfer = param_bytes / rate_bps
    return max(1, math.ceil(transfer / allowed))


def local_steps(param_bytes: int, step_seconds: float, share: float = DEFAULT_SHARE) -> int:
    """How many steps to take between reductions, from the link letify already measured.

    Spec "How often to synchronize". Answers for the replica this body runs on, so two
    replicas on links of different speed get different answers.
    """
    rate = getattr(_WHO, "rate_bps", None)
    return steps_for(
        param_bytes=param_bytes,
        step_seconds=step_seconds,
        rate_bps=rate,
        share=share,
    )


class _Replica:
    """Holds the rank for the body's thread, so nothing has to be passed through it."""

    def __init__(self, rank: int, world_size: int, rate_bps: float | None = None) -> None:
        self.rank = rank
        self.world_size = world_size
        self.rate_bps = rate_bps

    def __enter__(self) -> _Replica:
        _WHO.rank, _WHO.world_size = self.rank, self.world_size
        _WHO.rate_bps = self.rate_bps
        return self

    def __exit__(self, *_exc: Any) -> None:
        _WHO.rank = _WHO.world_size = _WHO.rate_bps = None


__all__ = ["local_steps", "rank", "steps_for", "world_size"]
