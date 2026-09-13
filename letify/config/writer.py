"""Writing one account back into a ``.letify`` file without rewriting the rest of it.

A user edits this file by hand, so the comments and the ordering in it are theirs. Parsing
the whole file into a dictionary and serializing it again would lose both. So an edit here
is textual: the block belonging to one alias is found, replaced or appended, and every
other byte of the file is left exactly as it was.

Only the value types TOML needs for an account are written: string, integer, float,
boolean and a list of strings. A value keeps the type it came in as, because a port
written as a string is a different value to whoever reads it back.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

#: Owner only, because this file holds connection details for the machine.
HOME_FILE_MODE = 0o600


def format_value(value: Any) -> str:
    """Render one value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return _quote(str(value))


def _quote(text: str) -> str:
    """A basic TOML string, with the two characters that cannot appear raw escaped."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_block(alias: str, options: dict[str, Any]) -> str:
    """Render one alias as a TOML table, with ``kind`` first so it reads as a heading."""
    lines = [f"[{alias}]"]
    for key in ["kind", *sorted(k for k in options if k != "kind")]:
        if key not in options or options[key] is None:
            continue
        lines.append(f"{key} = {format_value(options[key])}")
    return "\n".join(lines) + "\n"


def _span(text: str, alias: str) -> tuple[int, int] | None:
    """Where the alias's table starts and ends, or None when it is not there.

    A table runs until the next table header at the start of a line, which is how TOML
    delimits it, so a value containing a bracket cannot end it early.
    """
    opening = re.search(rf"^\[{re.escape(alias)}\]\s*$", text, re.MULTILINE)
    if opening is None:
        return None
    following = re.search(r"^\[", text[opening.end() :], re.MULTILINE)
    if following is None:
        return opening.start(), len(text)
    end = opening.end() + following.start()
    # A comment above a table header belongs to that table, so it is not part of this
    # block and must survive the block being replaced.
    body = text[opening.end() : end].splitlines(keepends=True)
    while body and (not body[-1].strip() or body[-1].lstrip().startswith("#")):
        end -= len(body.pop())
    return opening.start(), end


def write_block(text: str, alias: str, options: dict[str, Any]) -> str:
    """Return the file with this alias declared, replacing any earlier declaration."""
    block = format_block(alias, options)
    span = _span(text, alias)
    if span is None:
        separator = (
            "" if not text or text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n"
        )
        return text + separator + block
    start, end = span
    # One blank line after the block, so the next table is not glued to it.
    trailing = "\n" if end < len(text) else ""
    return text[:start] + block + trailing + text[end:]


def remove_block(text: str, alias: str) -> str:
    """Return the file without this alias, or unchanged when it was not there."""
    span = _span(text, alias)
    if span is None:
        return text
    start, end = span
    return (text[:start] + text[end:]).lstrip("\n")


def has_block(text: str, alias: str) -> bool:
    return _span(text, alias) is not None


def update(path: Path, alias: str, options: dict[str, Any], *, private: bool = False) -> None:
    """Declare one alias in a file, creating the file if it does not exist.

    ``private`` restricts the file to its owner, which is what the home file needs and
    what a file inside a repository must not pretend to have.
    """
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(write_block(text, alias, options), encoding="utf-8")
    if private:
        restrict(path)


def drop(path: Path, alias: str) -> bool:
    """Remove one alias from a file. Returns whether it was there to remove."""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if not has_block(text, alias):
        return False
    path.write_text(remove_block(text, alias), encoding="utf-8")
    return True


def restrict(path: Path) -> None:
    """Make a file readable by its owner alone, where the platform has the concept.

    Windows permissions are access control lists rather than mode bits, and chmod there
    sets only the read-only flag, which is not what this means. Nothing is claimed on that
    platform rather than claiming something false.
    """
    if sys.platform.startswith("win"):
        return
    try:
        os.chmod(path, HOME_FILE_MODE)
    except OSError:
        pass


__all__ = [
    "HOME_FILE_MODE",
    "drop",
    "format_block",
    "format_value",
    "has_block",
    "remove_block",
    "restrict",
    "update",
    "write_block",
]
