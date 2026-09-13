"""How hard an accelerator is working, read from the machine that owns it.

``nvidia-smi --query-gpu`` is the only source used, because it is present on every machine
letify reaches and it needs no framework loaded: asking PyTorch would mean importing it
into a session that may not have it, which would change the very load being measured.

The reader is a plain module level function so it can be shipped into a session through
the ordinary call protocol. Measuring a remote device therefore needs no extra channel and
no extra dependency on the machine.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

#: What is asked of nvidia-smi, in this order. Kept to readings every driver reports.
SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)

SMI_COMMAND = (
    "nvidia-smi",
    f"--query-gpu={','.join(SMI_FIELDS)}",
    "--format=csv,noheader,nounits",
)

SMI_TIMEOUT = 30


@dataclass(frozen=True, slots=True)
class DeviceLoad:
    """One physical accelerator, as the driver reported it."""

    index: int
    name: str
    utilization_percent: float | None = None
    memory_used_gb: float | None = None
    memory_total_gb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None

    @property
    def memory_percent(self) -> float | None:
        if not self.memory_total_gb:
            return None
        return 100.0 * (self.memory_used_gb or 0.0) / self.memory_total_gb

    def describe(self) -> str:
        load = (
            f"{self.utilization_percent:.0f}%"
            if self.utilization_percent is not None
            else "load unknown"
        )
        memory = (
            f"{self.memory_used_gb:.1f}/{self.memory_total_gb:.1f} GiB"
            if self.memory_total_gb
            else "memory unknown"
        )
        return f"{self.index} {self.name} {load} {memory}"

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "utilization_percent": self.utilization_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "memory_percent": self.memory_percent,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
        }


def _number(field: str) -> float | None:
    """Read one csv field, or None where the driver wrote a placeholder.

    nvidia-smi writes ``[N/A]`` and ``[Not Supported]`` for readings a card does not
    report, which are gaps rather than zeros.
    """
    text = field.strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_smi(output: str) -> list[DeviceLoad]:
    """Turn the csv nvidia-smi writes into one record per device.

    Memory arrives in mebibytes with ``nounits``, and is reported in gibibytes so it can
    be compared against the VRAM figure an instance carries.
    """
    devices: list[DeviceLoad] = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < len(SMI_FIELDS):
            continue
        index = _number(fields[0])
        if index is None:
            continue
        used = _number(fields[3])
        total = _number(fields[4])
        devices.append(
            DeviceLoad(
                index=int(index),
                name=fields[1],
                utilization_percent=_number(fields[2]),
                memory_used_gb=used / 1024 if used is not None else None,
                memory_total_gb=total / 1024 if total is not None else None,
                temperature_c=_number(fields[5]),
                power_w=_number(fields[6]),
            )
        )
    return devices


def read_smi() -> str:
    """Run nvidia-smi here and return its output, or an empty string.

    Shipped into a session to measure a remote device, so it must depend on nothing but
    the standard library. An empty string means there was nothing to ask.
    """
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        result = subprocess.run(
            list(SMI_COMMAND), capture_output=True, text=True, timeout=SMI_TIMEOUT
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def local_load() -> list[DeviceLoad]:
    """What the accelerators in this machine are doing right now."""
    return parse_smi(read_smi())


__all__ = [
    "SMI_COMMAND",
    "SMI_FIELDS",
    "SMI_TIMEOUT",
    "DeviceLoad",
    "local_load",
    "parse_smi",
    "read_smi",
]
