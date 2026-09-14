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
from collections.abc import Callable
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

#: One shell script that lists the compute processes, the owner of each one visible here and
#: the login user. The owners are read in the same command as the listing, so a process id
#: cannot be reused in between. A process id with no /proc entry prints no owner line.
OWNERS_SCRIPT = (
    "command -v stat >/dev/null || exit 127; "
    f"apps=$({' '.join(APPS_COMMAND)}) || exit $?; "
    "printf '%s\\n' \"$apps\"; "
    "echo '#owners'; "
    "for pid in $(printf '%s\\n' \"$apps\" | cut -d, -f2); do "
    'owner=$(stat -c %U "/proc/$pid" 2>/dev/null) && echo "$pid $owner"; '
    "done; "
    "echo '#login'; "
    "id -un || exit 1"
)

OWNERS_COMMAND = ("sh", "-c", OWNERS_SCRIPT)

#: The owner named for a process whose owner could not be read.
UNKNOWN_OWNER = "unknown"


def busy_indices(
    exclude_pids: set[int] | None = None,
    run: Callable[[tuple[str, ...]], str] | None = None,
    owners_out: dict[int, tuple[str, ...]] | None = None,
) -> tuple[int, ...]:
    """Device indices another user is currently computing on.

    Registered is permission, not availability: a card a colleague is training on is not
    something to fight over. Compute processes are read rather than utilization, because a
    card between steps reads as idle and is not. A process owned by the login user does not
    count, except when the login user is root, which many people share.

    ``exclude_pids`` leaves out processes letify itself started, so a session asking for a
    second card does not see its own as taken. ``run`` answers one command with its output,
    and defaults to running it on this machine. A remote provider passes a runner that asks
    its own machine and raises when the command cannot run. ``owners_out``, when given, is
    filled with the owners of the processes on each busy index.
    """
    runner = run or _run
    uuids = parse_uuids(runner(UUID_COMMAND))
    if not uuids:
        return ()
    return parse_busy(runner(OWNERS_COMMAND), uuids, exclude_pids, owners_out)


def parse_uuids(output: str) -> dict[str, int]:
    """Map each card's uuid to its index, because compute apps are reported by uuid."""
    table: dict[str, int] = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        table[fields[1]] = int(fields[0])
    return table


def parse_holders(
    output: str,
    uuids: dict[str, int],
    exclude_pids: set[int] | None = None,
) -> dict[int, tuple[str, tuple[str, ...]]]:
    """Who holds each card, read from the output of ``OWNERS_COMMAND``.

    Every index in ``uuids`` gets ``("others", users)`` when another user computes on it,
    ``("mine", ())`` when only the login user's own processes or this client's workers do,
    and ``("free", ())`` otherwise. The login user rule is the busy check's: when the login
    user is root, a root process that is not one of this client's workers is another user's.

    Raises ``RuntimeFailure`` when the output carries no login user, because without it no
    process can be told apart from the login user's own.
    """
    from ..errors import RuntimeFailure

    apps, _, rest = output.partition("#owners")
    owner_lines, _, login_lines = rest.partition("#login")
    login = login_lines.strip()
    if not login:
        raise RuntimeFailure(
            "the busy check could not read the login user, so which cards are free is unknown"
        )
    owner_of: dict[int, str] = {}
    for line in owner_lines.splitlines():
        pid_text, _, user = line.strip().partition(" ")
        if pid_text.isdigit() and user:
            owner_of[int(pid_text)] = user.strip()
    workers = exclude_pids or set()
    taken: dict[int, set[str]] = {}
    mine: set[int] = set()
    for line in apps.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 2 or fields[0] not in uuids:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        index = uuids[fields[0]]
        owner = owner_of.get(pid)
        if pid in workers or (owner is not None and owner == login and login != "root"):
            mine.add(index)
            continue
        taken.setdefault(index, set()).add(owner or UNKNOWN_OWNER)
    holders: dict[int, tuple[str, tuple[str, ...]]] = {}
    for index in sorted(uuids.values()):
        if index in taken:
            holders[index] = ("others", tuple(sorted(taken[index])))
        elif index in mine:
            holders[index] = ("mine", ())
        else:
            holders[index] = ("free", ())
    return holders


def parse_busy(
    output: str,
    uuids: dict[str, int],
    exclude_pids: set[int] | None = None,
    owners_out: dict[int, tuple[str, ...]] | None = None,
) -> tuple[int, ...]:
    """The indices another user computes on, read from the output of ``OWNERS_COMMAND``.

    Raises ``RuntimeFailure`` when the output carries no login user, because without it no
    process can be told apart from the login user's own.
    """
    holders = parse_holders(output, uuids, exclude_pids)
    taken = {index: users for index, (holder, users) in holders.items() if holder == "others"}
    if owners_out is not None:
        owners_out.update(taken)
    return tuple(sorted(taken))


def read_machine(
    run: Callable[[tuple[str, ...]], str],
    exclude_pids: set[int] | None = None,
) -> tuple[list[DeviceLoad], dict[int, tuple[str, tuple[str, ...]]]]:
    """Every card's load and who holds it, from the three read-only nvidia-smi queries.

    ``run`` answers one command with its output. The load query failing raises, because
    then there is nothing to report. The owner queries failing leaves the holders empty,
    which a caller prints as unknown, because the load is still worth showing.
    """
    from ..errors import LetifyError

    devices = parse_smi(run(SMI_COMMAND))
    if not devices:
        return devices, {}
    try:
        uuids = parse_uuids(run(UUID_COMMAND))
        holders = parse_holders(run(OWNERS_COMMAND), uuids, exclude_pids) if uuids else {}
    except LetifyError:
        holders = {}
    return devices, holders


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
    "OWNERS_COMMAND",
    "OWNERS_SCRIPT",
    "SMI_COMMAND",
    "SMI_FIELDS",
    "SMI_TIMEOUT",
    "UNKNOWN_OWNER",
    "UUID_COMMAND",
    "DeviceLoad",
    "busy_indices",
    "local_load",
    "parse_busy",
    "parse_holders",
    "parse_smi",
    "parse_uuids",
    "read_machine",
    "read_smi",
]
