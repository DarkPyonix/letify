"""CUDA call forwarding: the capability probe and the arithmetic behind it.

The shim that does the interception is a Rust component under ``shim/``, because
intercepting the CUDA driver cannot be done from Python. This package is the part that
stays pure Python: it finds the shim, measures the round trip, and refuses clearly when
forwarding cannot run.
"""

from __future__ import annotations

from .capability import LATENCY_BUDGET_MS, Capability
from .loader import Injection, inject, preload_command, shim_directory
from .probe import SHIM_NAMES, efficiency, ping, probe, require, shim_path

__all__ = [
    "LATENCY_BUDGET_MS",
    "SHIM_NAMES",
    "Capability",
    "Injection",
    "efficiency",
    "inject",
    "ping",
    "preload_command",
    "probe",
    "require",
    "shim_directory",
    "shim_path",
]
