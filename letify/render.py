"""Terminal rendering shared by the command line.

This module owns how records look on a terminal: the style decision (colour, block
characters, width), marks, tables, aligned fields, the gauge, relative times, the status
layout from spec "Command line", and the usage block layout from spec
"Remaining usage" and the utilization block layout from spec "GPU utilization". It does
not read any provider and does not decide what a record
contains, which belongs to the providers and to ``providers/usage.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

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
_CYAN = "\x1b[36m"

#: An escape sequence, which takes no column on a terminal.
_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True, slots=True)
class Style:
    """How output is drawn: terminal width, whether to colour, which gauge characters."""

    width: int = 80
    color: bool = False
    unicode: bool = True

    @classmethod
    def for_stream(cls, stream: TextIO) -> Style:
        """The style a stream supports: colour on a terminal without NO_COLOR, UTF-8 blocks."""
        color = color_enabled(stream)
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        width = shutil.get_terminal_size((80, 24)).columns
        return cls(width=width, color=color, unicode=encoding == "utf8")

    def paint(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def bold(self, text: str) -> str:
        return self.paint(text, _BOLD)

    def dim(self, text: str) -> str:
        return self.paint(text, _DIM)


def format_amount(value: float, unit: str) -> str:
    """The unit aware amount from ``providers/usage.py``, imported late to stay light."""
    from .providers.usage import format_amount as formatted

    return formatted(value, unit)


def color_enabled(stream: TextIO) -> bool:
    """Whether a stream gets colour: a terminal, with NO_COLOR unset or empty."""
    try:
        tty = bool(stream.isatty())
    except (AttributeError, ValueError):
        tty = False
    return tty and not os.environ.get("NO_COLOR")


#: Each mark's UTF-8 symbol, ASCII fallback and colour.
_MARKS = {"ok": ("\u2713", "+", _GREEN), "fail": ("\u2717", "x", _RED), "warn": ("!", "!", _YELLOW)}


def mark(kind: str, style: Style) -> str:
    """The success, failure or warning symbol, as spec "Command line" lists them."""
    utf8, ascii_, code = _MARKS[kind]
    return style.paint(utf8 if style.unicode else ascii_, code)


def visible_len(text: str) -> int:
    """Columns a string takes on a terminal, not counting escape sequences."""
    return len(_ESCAPE.sub("", text))


def _pad(text: str, width: int) -> str:
    return text + " " * (width - visible_len(text))


def table(headers: list[str], rows: list[list[str]], style: Style) -> str:
    """Columns left-aligned to their longest cell, two spaces apart, with a bold header."""
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(visible_len(row[column]) for row in [headers, *cells]) for column in range(len(headers))
    ]

    def line(row: list[str], bold: bool) -> str:
        parts = [style.bold(cell) if bold else cell for cell in row]
        padded = [_pad(part, widths[i]) for i, part in enumerate(parts[:-1])]
        return "  ".join([*padded, parts[-1]]).rstrip() + "\n"

    return line(headers, True) + "".join(line(row, False) for row in cells)


def fields(pairs: list[tuple[str, str]], style: Style) -> str:
    """Name and value lines with the values aligned."""
    width = max((len(name) for name, _ in pairs), default=0)
    return "".join(f"{name.ljust(width)}  {value}\n" for name, value in pairs)


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


#: The header text and colour for each card holder, as spec "GPU utilization" lists them.
HOLDERS: dict[str, tuple[str, str | None]] = {
    "letify": ("reserved by letify", _CYAN),
    "others": ("busy", _RED),
    "mine": ("in use by your processes", _YELLOW),
    "free": ("free", _GREEN),
    "unknown": ("holder unknown", None),
}

#: Columns a card's gauge line spends on things other than the gauge.
_CARD_TAKEN = 4 + 7 + 2 + 5 + 16


def _card_lines(device: Mapping[str, Any], style: Style) -> list[str]:
    """A card's header, load and memory gauges, and temperature and power, unindented."""
    head = f"gpu{device.get('index')}  {device.get('name')}"
    holder = device.get("holder")
    if holder in HOLDERS:
        text, code = HOLDERS[holder]
        if holder == "others":
            text = f"busy: {', '.join(device.get('users') or ()) or 'another user'}"
        head += "  " + (style.paint(text, code) if code else text)
    lines = [head]
    cells = max(GAUGE_MIN, min(GAUGE_MAX, style.width - _CARD_TAKEN))
    load = device.get("utilization_percent")
    if load is not None:
        fraction = float(load) / 100
        text = f"{gauge(fraction, cells, style)} {round(float(load)):>3}%"
        lines.append("  load   " + style.paint(text, share_color(1.0 - fraction)))
    total = device.get("memory_total_gb")
    if total:
        used = float(device.get("memory_used_gb") or 0.0)
        fraction = used / float(total)
        text = f"{gauge(fraction, cells, style)} {round(fraction * 100):>3}%"
        painted = style.paint(text, share_color(1.0 - fraction))
        lines.append(f"  memory {painted}  {used:.1f}/{float(total):.1f} GiB")
    extra = []
    if device.get("temperature_c") is not None:
        extra.append(f"{float(device['temperature_c']):.0f}C")
    if device.get("power_w") is not None:
        extra.append(f"{float(device['power_w']):.0f}W")
    if extra:
        lines.append("  " + style.dim("  ".join(extra)))
    return lines


def utilization_block(rows: list[Mapping[str, Any]], style: Style) -> str:
    """One provider's block: its cards, or why it has none to show."""
    first = rows[0]
    alias = str(first.get("alias"))
    if "unavailable" in first:
        return f"{style.bold(alias)}\n{INDENT}unavailable: {first['unavailable']}\n"
    body: list[str] = []
    reasons = {row.get("reason") for row in rows}
    if len(reasons) == 1 and None not in reasons:
        body.append(str(first["reason"]))
    else:
        for row in rows:
            if row.get("reason"):
                named = row.get("scope") == "session" and row.get("accelerator")
                body.append(f"{row['accelerator']}: {row['reason']}" if named else row["reason"])
            for device in row.get("devices") or ():
                body.extend(_card_lines(device, style))
    header = f"{style.bold(alias)}  {first.get('kind')}"
    return "\n".join([header, *(INDENT + line for line in body)]) + "\n"


def utilization_blocks(rows: Iterable[Mapping[str, Any]], style: Style) -> str:
    """Every provider's block, grouping its rows by alias in the order they came."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("alias")), []).append(row)
    return "\n".join(utilization_block(group, style) for group in groups.values())


def _runtime_block(runtime: Mapping[str, Any], style: Style, now: float) -> str:
    """One live runtime: where it runs, how it is reached, how long, and what it costs."""
    state = "busy" if runtime.get("busy") else "idle"
    where_to = f"{runtime.get('provider')}.{runtime.get('accelerator')}"
    head = f"{style.bold(str(runtime.get('name')))}  {where_to}  {state}"
    where = []
    cards = runtime.get("devices")
    if isinstance(cards, (list, tuple)) and cards:
        where.append("cards " + ", ".join(str(card) for card in cards))
    where.append(f"host {runtime.get('placement')}")
    if runtime.get("link"):
        rtt = runtime.get("rtt_ms")
        where.append(
            f"link {runtime['link']}" + (f", {float(rtt):.1f} ms" if rtt is not None else "")
        )
    uptime = float(runtime.get("uptime_seconds") or 0.0)
    idle = float(runtime.get("idle_seconds") or 0.0)
    lines = [head, "  ".join(where), f"up {relative(uptime)}  idle {relative(idle)}"]
    usage = runtime.get("usage")
    if isinstance(usage, Mapping):
        unit = str(usage.get("unit") or "")
        rate = usage.get("rate_per_hour")
        if rate is not None:
            spent = float(rate) * uptime / 3600
            lines.append(
                f"about {format_amount(spent, unit)} so far "
                f"at {format_amount(float(rate), unit)}/hour"
            )
        lines.extend(_allowance_lines(usage, style, now, None))
        if usage.get("note"):
            lines.append(style.dim(str(usage["note"])))
    return "\n".join([lines[0], *(INDENT + line for line in lines[1:])]) + "\n"


def status_text(status: Mapping[str, Any], style: Style, now: float | None = None) -> str:
    """``letify status``: the counts, each live runtime, and the inventory against reservations."""
    now = time.time() if now is None else now
    counts = f"{status.get('live', 0)} live, {status.get('busy', 0)} busy"
    header = f"{style.bold(str(status.get('name')))}  {counts}\n"
    parts = [header]
    runtimes = status.get("runtimes") or []
    if runtimes:
        parts.extend(_runtime_block(runtime, style, now) for runtime in runtimes)
    else:
        parts.append(style.dim("no live session in this process") + "\n")
    rows = [
        [
            alias,
            name,
            f"{entry.get('reserved', 0)}/{entry.get('count', 0)}",
            ", ".join(str(index) for index in entry.get("indices") or ()),
        ]
        for alias, inventory in (status.get("devices") or {}).items()
        for name, entry in inventory.items()
    ]
    if rows:
        parts.append(table(["PROVIDER", "ACCELERATOR", "RESERVED", "INDICES"], rows, style))
    return "\n".join(parts)


__all__ = [
    "GAUGE_MAX",
    "GAUGE_MIN",
    "Style",
    "color_enabled",
    "fields",
    "format_amount",
    "gauge",
    "gauge_cells",
    "mark",
    "relative",
    "share_color",
    "status_text",
    "table",
    "usage_block",
    "usage_blocks",
    "utilization_block",
    "utilization_blocks",
    "visible_len",
]
