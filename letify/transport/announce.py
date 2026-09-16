"""The one line printers for connection decisions.

Owns where a connection decision is printed: stderr, with the ``letify: `` prefix the
Launcher's session start line uses. It does not own what is printed; the pipeline does.
On a terminal with colour the prefix is dim, as spec "Command line" says, and the
message is unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

#: A printer takes one message without the prefix.
Say = Callable[[str], None]


def say(message: str) -> None:
    """Print one line to stderr, so the user's stdout stays clean."""
    from ..render import color_enabled

    prefix = "\x1b[2mletify:\x1b[0m" if color_enabled(sys.stderr) else "letify:"
    print(f"{prefix} {message}", file=sys.stderr, flush=True)


def quiet(message: str) -> None:
    """Print nothing, for ``Launcher(announce=False)``."""


def printer(announce: bool) -> Say:
    return say if announce else quiet


__all__ = ["Say", "printer", "quiet", "say"]
