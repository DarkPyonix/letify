"""Reattaching to a running call after the owner's channel broke.

Owns the constants of spec "Reattaching after a lost channel" and, once written, the rule
for when a lost channel is worth a reattach and the attempts that follow it. It does not own
how a channel to the worker's second door is built, which ``Provider.attach_channel`` does,
or the adoption itself, which the worker source owns.
"""

from __future__ import annotations

#: Seconds waited before attempt 1, 2, 3 and 4. Every later attempt waits the last value.
DELAYS_S: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)

#: The most attempts made for one lost channel.
MAX_ATTEMPTS = 8

#: Seconds that must remain before the lease deadline for an attempt to be worth making.
MARGIN_S = 10.0
