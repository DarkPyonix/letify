"""PyTorch forwarding for ``host="local"`` and the arithmetic that prices it.

``remoting/device/`` is the forwarding itself. What is defined here is the efficiency
formula, which needs neither PyTorch nor a connection. It does not own the letify-quic
binary in ``lib/``, which ``letify.transport.quic`` finds.
"""

from __future__ import annotations


def efficiency(step_seconds: float, syncs: int, round_trip_ms: float) -> float:
    """Expected fraction of a direct run, ``T / (T + k * RTT)``.

    ``step_seconds`` is the GPU time per step, ``syncs`` the host synchronizations per
    step and ``round_trip_ms`` the round trip to the runtime. ``syncs`` is measured with
    ``torch.cuda.set_sync_debug_mode("warn")``, which does not depend on where the GPU is.
    """
    overhead = syncs * round_trip_ms / 1000.0
    return step_seconds / (step_seconds + overhead)


__all__ = ["efficiency"]
