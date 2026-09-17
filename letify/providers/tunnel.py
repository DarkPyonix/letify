"""Tunnel, a plain machine behind NAT with no provider API.

The machine's user runs ``letify client shell connect`` on it once. That starts the
remote agent behind ``tailcat serve`` and prints a Tailcat address and port, which go into
this account as ``tailcat`` and ``tailcat_port``. From there the connection pipeline in
``Shell`` does the rest: it reaches the agent over Tailcat, exchanges the TCP punch
mapping through it, and races the strategies as the spec describes. This class adds no
transport of its own; it exists so the configuration kind says what the machine is.
"""

from __future__ import annotations

from .shell import Shell


class Tunnel(Shell):
    """A machine behind NAT, reached through the remote agent."""

    kind = "tunnel"
    extra = "shell"
    default_persistence = "ephemeral"
    has_fast_path = True


__all__ = ["Tunnel"]
