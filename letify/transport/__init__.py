"""The connection pipeline that reaches a Shell machine.

Owns the objects the spec's Transport section names: ``Rendezvous``, ``Strategy``,
``Link``, ``Probe``, ``Pipeline`` and ``LinkCache``. It does not own providers, which
choose a rendezvous and a strategy list, or channels, which run over a chosen link.
Nothing here is imported at package import time.
"""

from __future__ import annotations
