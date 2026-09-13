"""What is left on an account, asked the same way of every provider.

A researcher renting cheap GPU time runs out of it, so the question that decides what to
run next is how much is left. Each service answers differently, and most of them do not
answer at all: Colab keeps the compute unit balance in its web console, Modal exposes no
workspace balance through its SDK, and a machine reached over SSH has no account behind
it.

So this module is built around the missing answer. Every field except the alias, the unit
and the source may be None, because a gap is information and a fabricated balance is not:
a researcher spends against a number letify prints. Where a service publishes nothing, a
configuration entry names a command that prints the figure, which keeps the table useful
without letify inventing an endpoint.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass

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
    unmetered: bool = False
    as_of: float | None = None
    note: str | None = None

    @property
    def known(self) -> bool:
        """Whether any figure came back at all."""
        return self.remaining is not None or self.rate_per_hour is not None

    def describe(self) -> str:
        """One line for the table, saying plainly when there is no number."""
        if self.unmetered:
            return "unmetered"
        parts = []
        if self.remaining is not None:
            left = f"{self.remaining:g} {self.unit} left"
            if self.limit:
                left += f" of {self.limit:g}"
            parts.append(left)
        if self.rate_per_hour is not None:
            parts.append(f"{self.rate_per_hour:g} {self.unit}/hour running now")
        return ", ".join(parts) or "not reported"

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
            "unmetered": self.unmetered,
            "as_of": self.as_of,
            "note": self.note,
        }


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


__all__ = ["COMMAND_TIMEOUT", "Usage", "from_command", "read_number"]
