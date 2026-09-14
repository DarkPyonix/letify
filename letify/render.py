"""Terminal rendering shared by the command line.

This module owns how records look on a terminal: the style decision (colour, block
characters, width), the gauge, relative times, and the usage block layout from spec
"Remaining usage". It does not read any provider and does not decide what a record
contains, which belongs to the providers and to ``providers/usage.py``.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from .providers.usage import format_amount

#: The gauge is never narrower or wider than this many cells.
GAUGE_MIN = 16
GAUGE_MAX = 40
#: Columns kept free for the percentage text, sized for " 100% used".
PERCENT_COLUMNS = 10
INDENT = "  "

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"


@dataclass(frozen=True, slots=True)
class Style:
    """How output is drawn: terminal width, whether to colour, which gauge characters."""

    width: int = 80
    color: bool = False
    unicode: bool = True

    @classmethod
    def for_stream(cls, stream: TextIO) -> Style:
        """The style a stream supports: colour on a terminal without NO_COLOR, UTF-8 blocks."""
        try:
            tty = bool(stream.isatty())
        except (AttributeError, ValueError):
            tty = False
        color = tty and not os.environ.get("NO_COLOR")
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        width = shutil.get_terminal_size((80, 24)).columns
        return cls(width=width, color=color, unicode=encoding == "utf8")

    def paint(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def bold(self, text: str) -> str:
        return self.paint(text, _BOLD)

    def dim(self, text: str) -> str:
        return self.paint(text, _DIM)


def share_color(remaining_share: float) -> str:
    """Green above half left, yellow from a fifth to half, red below a fifth."""
    if remaining_share > 0.5:
        return _GREEN
    if remaining_share >= 0.2:
        return _YELLOW
    return _RED


def gauge(fraction: float, cells: int, style: Style) -> str:
    """``[#####-----]`` with ``fraction`` of the cells filled, clamped to 0 through 1."""
    fraction = min(max(fraction, 0.0), 1.0)
    filled = round(fraction * cells)
    full, empty = ("█", "░") if style.unicode else ("#", "-")
    return f"[{full * filled}{empty * (cells - filled)}]"


def gauge_cells(style: Style, taken: int = 0) -> int:
    """How many cells a gauge gets once ``taken`` extra columns are used on its line."""
    room = style.width - len(INDENT) - 2 - PERCENT_COLUMNS - taken
    return max(GAUGE_MIN, min(GAUGE_MAX, room))


def relative(seconds: float) -> str:
    """A duration in its two largest units: ``4 d 6 h``, ``3 h 12 min``, ``45 min``."""
    minutes = max(int(seconds // 60), 0)
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min"


def _stamp(when: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(when))


def _allowance_lines(
    record: Mapping[str, Any], style: Style, now: float, name: str | None
) -> list[str]:
    """Gauge, amount and reset lines for one allowance, the account's or a further one."""
    unit = str(record.get("unit") or "")
    remaining = record.get("remaining")
    limit = record.get("limit")
    used = record.get("used")
    prefix = f"{name} " if name else ""
    pad = " " * len(prefix)
    lines: list[str] = []
    if isinstance(limit, (int, float)) and limit > 0:
        if used is None and remaining is not None:
            used = max(float(limit) - float(remaining), 0.0)
        if used is not None:
            fraction = float(used) / float(limit)
            cells = gauge_cells(style, len(prefix))
            text = f"{gauge(fraction, cells, style)} {round(fraction * 100)}% used"
            lines.append(prefix + style.paint(text, share_color(1.0 - fraction)))
    if remaining is not None:
        amount = f"{format_amount(float(remaining), unit)} left"
        if isinstance(limit, (int, float)):
            amount += f" of {format_amount(float(limit), unit).removesuffix(' ' + unit)}"
        else:
            amount += ", limit unknown"
        lines.append(pad + amount if lines else prefix + amount)
    resets_at = record.get("resets_at")
    if resets_at is not None:
        left = float(resets_at) - now
        when = f"resets in {relative(left)}" if left > 0 else "reset due"
        lines.append(pad + style.dim(f"{when} ({_stamp(float(resets_at))})"))
    return lines


def usage_block(row: Mapping[str, Any], style: Style, now: float | None = None) -> str:
    """One account's block, as spec "Remaining usage" lays it out."""
    now = time.time() if now is None else now
    alias = str(row.get("alias"))
    if "unavailable" in row:
        return f"{style.bold(alias)}\n{INDENT}unavailable: {row['unavailable']}\n"
    lines = [f"{style.bold(alias)}  {row.get('kind')}"]
    if row.get("unmetered"):
        return "\n".join([*lines, f"{INDENT}no quota, unmetered"]) + "\n"
    body = _allowance_lines(row, style, now, None)
    unit = str(row.get("unit") or "")
    rate = row.get("rate_per_hour")
    remaining = row.get("remaining")
    if rate is not None:
        text = f"{format_amount(float(rate), unit)}/hour running now"
        if float(rate) > 0 and remaining is not None:
            text += f", about {relative(float(remaining) / float(rate) * 3600)} at this rate"
        body.append(text)
    for resource in row.get("resources") or ():
        body.extend(_allowance_lines(resource, style, now, str(resource.get("name") or "")))
    if remaining is None and rate is None:
        body.insert(0, "not reported")
    if row.get("note"):
        body.append(style.dim(str(row["note"])))
    return "\n".join([*lines, *(INDENT + line for line in body)]) + "\n"


def usage_blocks(rows: Iterable[Mapping[str, Any]], style: Style, now: float | None = None) -> str:
    """Every account's block, one blank line apart."""
    return "\n".join(usage_block(row, style, now) for row in rows)


__all__ = [
    "GAUGE_MAX",
    "GAUGE_MIN",
    "Style",
    "gauge",
    "gauge_cells",
    "relative",
    "share_color",
    "usage_block",
    "usage_blocks",
]
