"""The User-Agent letify names itself with. Owns the string; it sends no request itself.

Spec "User-Agent": every HTTP request letify sends carries ``User-Agent: letify/<version>``.
"""

from __future__ import annotations

from collections.abc import Mapping


def user_agent() -> str:
    """The User-Agent of every request: ``letify/<version>``."""
    from . import __version__

    return f"letify/{__version__}"


def with_agent(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``headers`` that carries the letify User-Agent, keeping any one already set."""
    named = dict(headers or {})
    if not any(key.lower() == "user-agent" for key in named):
        named["User-Agent"] = user_agent()
    return named
