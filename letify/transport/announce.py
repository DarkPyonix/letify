"""The one line printers for connection decisions.

Owns where a connection decision is printed: stderr, with the ``letify: `` prefix the
Launcher's session start line uses. It does not own what is printed; the pipeline does.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

#: A printer takes one message without the prefix.
Say = Callable[[str], None]


def say(message: str) -> None:
    """Print one line to stderr, so the user's stdout stays clean."""
    print(f"letify: {message}", file=sys.stderr, flush=True)


def quiet(message: str) -> None:
    """Print nothing, for ``Launcher(announce=False)``."""


def printer(announce: bool) -> Say:
    return say if announce else quiet


__all__ = ["Say", "printer", "quiet", "say"]
