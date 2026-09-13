"""What a machine can support for CUDA call forwarding."""

from __future__ import annotations

from dataclasses import dataclass

#: Round trip beyond which forwarding is reported as costly. Not a refusal: the
#: declaration decides, and this only sets when letify says something about it.
LATENCY_BUDGET_MS = 40.0


@dataclass(frozen=True, slots=True)
class Capability:
    """Whether forwarding can run here, and what it would cost."""

    core: bool
    agent: bool
    round_trip_ms: float | None
    platform: str

    @property
    def usable(self) -> bool:
        """Whether forwarding can run at all, regardless of speed."""
        return self.core and self.agent

    @property
    def costly(self) -> bool:
        """Whether the round trip is long enough to change the arithmetic."""
        return self.round_trip_ms is not None and self.round_trip_ms > LATENCY_BUDGET_MS

    def explain(self) -> str:
        reasons = []
        if not self.core:
            reasons.append(
                "letify-core is not installed on this machine, so there is "
                "nothing to intercept the CUDA driver"
            )
        if not self.agent:
            reasons.append("the letify agent is not installed on the remote machine")
        if self.round_trip_ms is None:
            reasons.append("the round trip has not been measured")
        elif self.costly:
            reasons.append(
                f"the round trip is {self.round_trip_ms:.0f} ms, so each host "
                f"synchronization pays it"
            )
        return "; ".join(reasons) or "ready"


__all__ = ["LATENCY_BUDGET_MS", "Capability"]
