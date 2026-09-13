"""The lease that stops a session from outliving this process.

A runtime is the only thing in letify that costs money, and the danger is not a
session that runs too long, it is one that nobody is watching. A killed script, a
closed laptop or a lost connection would otherwise leave a GPU billing until the
provider's own timeout, which on Colab can be twelve or twenty four hours.

So the worker holds a deadline and this process keeps pushing it forward. Stop
renewing and the worker exits on its own. The gap between the renewal interval and
the grace period is what a brief network drop fits into, so a flaky link does not
kill a training run.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Runtime

#: How often this process pushes the deadline forward.
INTERVAL = 30.0

#: How long the worker waits for a renewal before exiting. The difference from the
#: interval is the tolerance for a dropped connection.
GRACE = 300.0


class Lease:
    """Renews one runtime's deadline until it is stopped."""

    def __init__(self, runtime: Runtime, *, interval: float = INTERVAL, grace: float = GRACE):
        self.runtime = runtime
        self.interval = interval
        self.grace = grace
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def arm(self) -> None:
        """Set the deadline once and start renewing it."""
        if self._thread is not None:
            return
        self._renew()
        self._thread = threading.Thread(
            target=self._loop, name=f"letify-lease-{self.runtime.name}", daemon=True
        )
        self._thread.start()

    def release(self) -> None:
        self._stop.set()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if not self._renew():
                return

    def _renew(self) -> bool:
        from ..errors import LetifyError

        try:
            self.runtime.request({"op": "lease", "grace": self.grace}, timeout=60)
        except LetifyError:
            # The session is gone. The pool notices on its next call.
            return False
        return True


__all__ = ["GRACE", "INTERVAL", "Lease"]
