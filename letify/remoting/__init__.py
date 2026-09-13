"""CUDA call forwarding: the capability probe and the arithmetic behind it.

letify-core that does the interception is a Rust component under ``letify-core/``, because
intercepting the CUDA driver cannot be done from Python. This package is the part that
stays pure Python: it finds letify-core, measures the round trip, and refuses clearly when
forwarding cannot run.
"""

from __future__ import annotations

from .capability import LATENCY_BUDGET_MS, Capability
from .loader import Injection, core_directory, inject, preload_command
from .probe import CORE_NAMES, core_path, efficiency, ping, probe, require

__all__ = [
    "CORE_NAMES",
    "LATENCY_BUDGET_MS",
    "Capability",
    "Injection",
    "core_directory",
    "core_path",
    "efficiency",
    "inject",
    "ping",
    "preload_command",
    "probe",
    "require",
]
