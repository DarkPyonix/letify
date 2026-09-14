"""What is left on an account, asked the same way of every provider.

A researcher renting cheap GPU time runs out of it, so the question that decides what to
run next is how much is left. This module owns the record every provider answers with,
the configured command that can replace a provider's own reading, and the way a record is
printed. It does not own how each service is asked, which lives in each provider module.

Every field except the alias, the kind, the unit and the source may be None, because a
gap is information and a fabricated balance is not: a researcher spends against a number
letify prints.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

#: The last number in a command's output is the remaining amount, so a command that
#: prints a sentence around it still works.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: A usage command answers or it does not. It runs while a person waits for a table.
COMMAND_TIMEOUT = 20


@dataclass(frozen=True, slots=True)
class Usage:
    """What one provider account has left, as far as it can be known."""

    alias: str
    kind: str
    unit: str
    source: str
    remaining: float | None = None
    limit: float | None = None
    used: float | None = None
    rate_per_hour: float | None = None
    resets_at: float | None = None
    unmetered: bool = False
    as_of: float | None = None
    note: str | None = None

    @property
    def known(self) -> bool:
        """Whether any figure came back at all."""
        return self.remaining is not None or self.rate_per_hour is not None

    def describe(self) -> str:
        """One line for the table, saying plainly when there is no number."""
        return describe_row(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "kind": self.kind,
            "unit": self.unit,
            "source": self.source,
            "remaining": self.remaining,
            "limit": self.limit,
            "used": self.used,
            "rate_per_hour": self.rate_per_hour,
            "resets_at": self.resets_at,
            "unmetered": self.unmetered,
            "as_of": self.as_of,
            "note": self.note,
        }


def _number(value: float, unit: str) -> str:
    """The amount alone, at the precision its unit is spent in."""
    if unit == "KRW":
        return f"{value:,.0f}"
    if unit == "USD":
        return f"${value:,.2f}"
    if unit == "compute units":
        return f"{value:,.2f}"
    if unit == "GPU hours":
        return f"{value:,.1f}"
    return f"{value:g}"


def format_amount(value: float, unit: str) -> str:
    """An amount with its unit: ``12,345 KRW``, ``$29.50``, ``12.5 GPU hours``."""
    number = _number(value, unit)
    return number if unit == "USD" else f"{number} {unit}".rstrip()


def describe_row(row: dict[str, Any]) -> str:
    """One line for a usage record given as a dictionary, as ``--json`` prints it."""
    if row.get("unmetered"):
        return "no quota, unmetered"
    unit = str(row.get("unit") or "")
    parts = []
    remaining = row.get("remaining")
    if remaining is not None:
        left = f"{format_amount(float(remaining), unit)} left"
        if row.get("limit") is not None:
            left += f" of {_number(float(row['limit']), unit)}"
        if row.get("resets_at") is not None:
            stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(row["resets_at"])))
            left += f", resets {stamp}"
        parts.append(left)
    if row.get("rate_per_hour") is not None:
        parts.append(f"{format_amount(float(row['rate_per_hour']), unit)}/hour running now")
    return ", ".join(parts) or f"not reported ({row.get('note') or row.get('source')})"


def read_number(text: str) -> float | None:
    """The last number in the text, or None when there is none.

    None rather than zero, because a command that printed an error message has not told
    us the balance is empty.
    """
    found = _NUMBER.findall(text)
    return float(found[-1]) if found else None


def from_command(alias: str, kind: str, command: str, unit: str, limit: float | None) -> Usage:
    """Run a configured command and read the remaining amount out of its output.

    A command that fails or prints no number reports nothing rather than taking the whole
    table down, because one misconfigured account should not hide the others.
    """
    note: str | None = None
    remaining: float | None = None
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        note = f"usage_command failed: {exc}"
    else:
        if result.returncode != 0:
            note = f"usage_command exit {result.returncode}: {result.stderr.strip()}"
        else:
            remaining = read_number(result.stdout)
            if remaining is None:
                note = "usage_command printed no number"
    used = limit - remaining if limit is not None and remaining is not None else None
    return Usage(
        alias=alias,
        kind=kind,
        unit=unit,
        source="usage_command",
        remaining=remaining,
        limit=limit,
        used=used,
        as_of=time.time(),
        note=note,
    )


__all__ = [
    "COMMAND_TIMEOUT",
    "Usage",
    "describe_row",
    "format_amount",
    "from_command",
    "read_number",
]
