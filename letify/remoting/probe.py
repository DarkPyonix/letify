"""Measuring whether CUDA call forwarding is worth using.

Forwarding keeps Python and the libraries in this process and sends only driver calls,
so its cost is one network round trip at every point where the host reads a value back
from the device. Efficiency against a direct run is ``T / (T + k * RTT)``, where ``T``
is GPU time per step and ``k`` is host synchronizations per step.

Both terms are measurable before anything is built. This module measures the round trip
and reports whether letify-core and the agent are present; ``k`` is measured in the user's
own training step with ``torch.cuda.set_sync_debug_mode("warn")``, which does not depend
on where the GPU is.

The numbers that came out of that arithmetic, for an RTX PRO 6000 with NVFP4 and a
0.5 s micro step at a 150 ms round trip: fine-tuning is about 53 percent of a direct run
with default Hugging Face settings and about 96 percent once the per-step
synchronizations are reduced to one per optimizer step. Token by token decoding stays
bad at any useful latency, because a decode step is a few milliseconds and the round
trip sets the ceiling.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ..errors import UnsupportedMode
from .capability import LATENCY_BUDGET_MS, Capability

#: Names letify-core library goes by on each platform.
CORE_NAMES = {
    "win32": "nvcuda.dll",
    "linux": "libcuda.so.1",
    "darwin": "libletify_shim.dylib",
}


def core_path() -> Path | None:
    """Where the built shim library is, if it was installed."""
    import os

    override = os.environ.get("LETIFY_CORE_PATH")
    if override and Path(override).exists():
        return Path(override)
    name = CORE_NAMES.get(sys.platform, "libletify_shim.so")
    candidate = Path(__file__).resolve().parent / "lib" / name
    return candidate if candidate.exists() else None


def probe(host: str | None = None) -> Capability:
    """Report whether forwarding could run against a host, and what it would cost."""
    return Capability(
        core=core_path() is not None,
        agent=shutil.which("letify-agent") is not None or host is None,
        round_trip_ms=ping(host) if host else None,
        platform=sys.platform,
    )


def require(host: str | None = None) -> Capability:
    """Raise unless forwarding can actually run.

    Speed is not a reason to refuse. A declaration that asked for forwarding over a long
    link runs, with a warning carrying the arithmetic. Only a missing shim or agent stops
    it, because then there is nothing to run.
    """
    capability = probe(host)
    if not capability.usable:
        raise UnsupportedMode(
            f"host='local' cannot run here: {capability.explain()}. Build letify-core from "
            f"the letify-core/ directory, or use host='remote' to ship the function instead."
        )
    return capability


def ping(host: str) -> float | None:
    """Measure the round trip in milliseconds, or return None if it cannot be."""
    flag = "-n" if sys.platform.startswith("win") else "-c"
    try:
        result = subprocess.run(
            ["ping", flag, "3", host], capture_output=True, text=True, timeout=30
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
    plausible = [value for value in values if 0.01 < value < 10000]
    return min(plausible) if plausible else None


def efficiency(step_seconds: float, syncs: int, round_trip_ms: float) -> float:
    """Expected fraction of a direct run, from the two measured terms."""
    overhead = syncs * round_trip_ms / 1000.0
    return step_seconds / (step_seconds + overhead)


__all__ = [
    "CORE_NAMES",
    "LATENCY_BUDGET_MS",
    "Capability",
    "core_path",
    "efficiency",
    "ping",
    "probe",
    "require",
]
