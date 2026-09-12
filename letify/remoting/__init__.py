"""CUDA call forwarding, the second execution mode.

Function shipping sends the whole loop to the remote machine. Forwarding does the
opposite: Python and the libraries stay in this process, and only the CUDA driver
calls cross the network. That keeps the local machine's data and environment in
place, which is the appeal, but it puts one network round trip in front of every
point where the host has to read a value back from the device.

The arithmetic decides where it is worth offering. Efficiency against a local run
is ``T / (T + k * RTT)``, where ``T`` is the GPU time per step and ``k`` is the
number of host synchronizations in that step. A NVFP4 micro step on an RTX PRO
6000 is near 0.5 s, and a default Hugging Face training step synchronizes about
three times: the trainer's NaN check every step, the attention mask check every
forward, and logging or the gradient scaler. At a 150 ms round trip that is 53
percent of a local run. On a slower card the same overhead matters less, so an L4
in bf16 with a 1.8 s step reaches about 80 percent. A faster GPU makes forwarding
worse, not better.

Token by token decoding is where it fails outright. Generation synchronizes once
or twice per token, and a decode step on an RTX PRO 6000 with 4-bit weights is
only a few milliseconds, so the round trip sets the ceiling at a handful of tokens
per second no matter how fast the card is.

Reducing ``k`` changes the picture: turning off the trainer's NaN filter, removing
the attention mask check with fixed length packing, and moving logging to the
gradient accumulation boundary leaves about one synchronization per optimizer
step, which reaches roughly 96 percent. That is a tuning exercise on the user's
training code, not something letify can do for them, so the default stays
function shipping.

This package is a placeholder for the forwarding client. It currently reports what
it needs and refuses to pretend, because a silent fallback to a slower mode is the
one failure the rest of the design is built to avoid.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from ..errors import UnsupportedMode

#: Round trip beyond which forwarding is not worth offering by default.
LATENCY_BUDGET_MS = 40.0


@dataclass(frozen=True, slots=True)
class Capability:
    """What a machine can support for call forwarding."""

    tun_device: bool
    driver_shim: bool
    round_trip_ms: float | None

    @property
    def usable(self) -> bool:
        if not self.driver_shim:
            return False
        if self.round_trip_ms is None:
            return False
        return self.round_trip_ms <= LATENCY_BUDGET_MS

    def explain(self) -> str:
        reasons = []
        if not self.driver_shim:
            reasons.append("the driver shim is not installed on this machine")
        if not self.tun_device:
            reasons.append("/dev/net/tun is missing, so a layer 3 tunnel cannot be built")
        if self.round_trip_ms is None:
            reasons.append("the round trip has not been measured")
        elif self.round_trip_ms > LATENCY_BUDGET_MS:
            reasons.append(
                f"the round trip is {self.round_trip_ms:.0f} ms, over the "
                f"{LATENCY_BUDGET_MS:.0f} ms budget"
            )
        return "; ".join(reasons) or "ready"


def probe(host: str | None = None) -> Capability:
    """Check whether call forwarding could work against a host.

    Two of the three answers can be had locally. Whether ``/dev/net/tun`` exists
    decides if a layer 3 tunnel is possible at all, and the round trip decides
    whether the per-synchronization cost is tolerable.
    """
    tun = _has_tun()
    shim = shutil.which("letify-cuda-shim") is not None
    rtt = _ping(host) if host else None
    return Capability(tun_device=tun, driver_shim=shim, round_trip_ms=rtt)


def require(host: str | None = None) -> Capability:
    """Raise unless forwarding is actually usable.

    letify never downgrades to a slower mode without saying so. A caller that
    asked for ``cpu="local"`` gets an error explaining what is missing, not a run
    that quietly takes four times as long.
    """
    capability = probe(host)
    if not capability.usable:
        raise UnsupportedMode(
            f"CUDA call forwarding is not available: {capability.explain()}. "
            f"Use cpu='remote' to ship the function instead."
        )
    return capability


def _has_tun() -> bool:
    from pathlib import Path

    return Path("/dev/net/tun").exists()


def _ping(host: str) -> float | None:
    """Measure the round trip in milliseconds, or return None if it cannot be."""
    try:
        result = subprocess.run(
            ["ping", "-n" if _is_windows() else "-c", "3", host],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    values = []
    for token in result.stdout.replace("=", " ").replace("ms", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    plausible = [v for v in values if 0.01 < v < 10000]
    return min(plausible) if plausible else None


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


__all__ = ["LATENCY_BUDGET_MS", "Capability", "probe", "require"]
