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


#: What nvidia-smi is asked for when the question is who else is on the card. A compute
#: process is the only honest answer: memory can be held by a display server, and a card at
#: zero percent may still be mid step.
APPS_COMMAND = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid",
    "--format=csv,noheader,nounits",
)

UUID_COMMAND = ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits")


def busy_indices(exclude_pids: set[int] | None = None) -> tuple[int, ...]:
    """Device indices another process is currently computing on.

    Registered is permission, not availability: a card a colleague is training on is not
    something to fight over. Compute processes are read rather than utilization, because a
    card between steps reads as idle and is not.

    ``exclude_pids`` leaves out processes letify itself started, so a session asking for a
    second card does not see its own as taken.
    """
    mine = exclude_pids or set()
    uuids = _uuid_to_index()
    if not uuids:
        return ()
    taken: set[int] = set()
    for line in _run(APPS_COMMAND).splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2 or fields[0] not in uuids:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        if pid not in mine:
            taken.add(uuids[fields[0]])
    return tuple(sorted(taken))


def _uuid_to_index() -> dict[str, int]:
    """Map each card's uuid to its index, because compute apps are reported by uuid."""
    table: dict[str, int] = {}
    for line in _run(UUID_COMMAND).splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        table[fields[1]] = int(fields[0])
    return table


def _run(command: tuple[str, ...]) -> str:
    """Run one nvidia-smi query, answering with nothing when there is nothing to ask."""
    if not shutil.which(command[0]):
        return ""
    try:
        result = subprocess.run(list(command), capture_output=True, text=True, timeout=SMI_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def local_load() -> list[DeviceLoad]:
    """What the accelerators in this machine are doing right now."""
    return parse_smi(read_smi())


__all__ = [
    "APPS_COMMAND",
    "SMI_COMMAND",
    "SMI_FIELDS",
    "SMI_TIMEOUT",
    "UUID_COMMAND",
    "DeviceLoad",
    "busy_indices",
    "local_load",
    "parse_smi",
    "read_smi",
]
