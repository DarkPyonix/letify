"""The pool that hands out runtimes and decides when they die.

The rule is that a runtime dies when the work that needed it is done. One
invocation is the unit, and a search space counts as one invocation, so a sweep
boots its runtimes once and releases them when the last point finishes. Nothing is
kept alive on the chance that another call might come.

Keeping one longer is declared, never assumed. A runtime acquired for a declaration
made with ``lifetime="process"`` survives its release, because starting a session is not
free: provider boot plus environment installation is minutes on Colab, so several separate
calls in a row are cheaper with one session than with several.

``hold`` is the other half of the rule, and it is internal. One invocation brackets
itself with it so that a search space, which is many calls, starts its runtimes once and
releases them when the last point finishes.

Two backstops cover a process-lifetime runtime nobody uses any more. A reaper thread tears down
runtimes idle past the timeout, and each runtime's lease makes the remote worker exit if
this process stops renewing.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..declare.instance import Lifetime
from ..errors import LetifyError

if TYPE_CHECKING:
    from ..declare.env import Env
    from ..declare.instance import Instance
    from ..store.volume import Volume
    from .session import Runtime

#: How long a held runtime may sit unused before the reaper tears it down.
DEFAULT_IDLE_TIMEOUT = 600.0

#: How often the reaper looks.
REAP_INTERVAL = 30.0


class RuntimePool:
    """Keeps track of live runtimes and enforces the release rule.

    How many sessions may exist is the provider's inventory and nothing else. Starting one
    reserves the devices its instance asks for, and a call that cannot reserve them waits
    for a session to release some. There is no ceiling here: a number would be a guess about
    hardware the provider entry already describes, and when the two disagreed the smaller
    would win silently.
    """

    def __init__(
        self,
        *,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        on_start: object | None = None,
    ):
        self.idle_timeout = idle_timeout
        self.on_start = on_start
        self._runtimes: dict[str, list[Runtime]] = {}
        self._guard = threading.RLock()
        self._free = threading.Condition(self._guard)
        self._hold_depth = 0
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()

    # -- holding -------------------------------------------------------------

    @property
    def holding(self) -> bool:
        with self._guard:
            return self._hold_depth > 0

    def hold(self) -> None:
        """Keep released runtimes alive until the matching ``unhold``.

        Internal. One invocation uses this so a sweep does not restart a session between
        its points.
        """
        with self._guard:
            self._hold_depth += 1

    def unhold(self) -> None:
        """Drop one hold, shutting runtimes down when the last one goes."""
        with self._guard:
            self._hold_depth = max(0, self._hold_depth - 1)
            closing = self._hold_depth == 0
        if closing:
            self.shutdown_idle()

    # -- acquisition ---------------------------------------------------------

    def acquire(
        self,
        instance: Instance,
        env: Env,
        volumes: Sequence[Volume] = (),
        *,
        lifetime: Lifetime = Lifetime.call,
    ) -> Runtime:
        """Return a runtime for this declaration, starting one if a slot is free.

        ``lifetime`` comes from the declaration and decides whether the runtime survives
        its release.
        """
        key = f"{instance.key}|{env.key}"
        while True:
            with self._guard:
                for runtime in self._runtimes.get(key, ()):
                    if not runtime.busy:
                        runtime.busy = True
                        if lifetime is Lifetime.process:
                            runtime.lifetime = lifetime
                        runtime.last_used = time.monotonic()
                        return runtime

            # No free session, so this needs cards. Taking them outside the pool guard,
            # because the provider may have to ask the machine who else is on them.
            held = instance.provider.reserve(instance)
            if held is not None:
                try:
                    return self._start(instance, env, volumes, key, lifetime=lifetime, held=held)
                except BaseException:
                    instance.provider.unreserve(instance.accelerator, held, instance.devices)
                    with self._free:
                        self._free.notify_all()
                    raise

            # Every card this instance could use is taken. Wait for a session to give some
            # back rather than asking the provider for a machine it would refuse.
            with self._free:
                self._free.wait(timeout=REAP_INTERVAL)

    def release(self, runtime: Runtime) -> None:
        """Give a runtime back. It ends here unless its lifetime says otherwise."""
        with self._guard:
            runtime.busy = False
            runtime.last_used = time.monotonic()
            keep = runtime.lifetime is Lifetime.process or self._hold_depth > 0
        if keep:
            with self._free:
                self._free.notify_all()
            return
        self.discard(runtime)

    def discard(self, runtime: Runtime) -> None:
        """Shut a runtime down and give its devices back, whatever state it was in."""
        with self._guard:
            bucket = self._runtimes.get(runtime.key, [])
            present = runtime in bucket
            if present:
                bucket.remove(runtime)
        if present:
            runtime.provider.unreserve(
                runtime.instance.accelerator, runtime.held_devices, runtime.instance.devices
            )
        try:
            runtime.shutdown()
        except LetifyError:
            pass
        with self._free:
            self._free.notify_all()

    def _start(
        self,
        instance: Instance,
        env: Env,
        volumes: Sequence[Volume],
        key: str,
        held: tuple[int, ...] = (),
        *,
        lifetime: Lifetime = Lifetime.call,
    ) -> Runtime:
        name = f"letify-{instance.accelerator.lower()}-{uuid.uuid4().hex[:6]}"
        if callable(self.on_start):
            self.on_start(instance, name)
        runtime = instance.provider.start(
            instance, env, name=name, volumes=tuple(volumes), held=held
        )
        runtime.busy = True
        runtime.lifetime = lifetime
        with self._guard:
            self._runtimes.setdefault(key, []).append(runtime)
        self._ensure_reaper()
        return runtime

    # -- upkeep --------------------------------------------------------------

    def _ensure_reaper(self) -> None:
        if self._reaper is not None:
            return
        self._stop_reaper.clear()
        self._reaper = threading.Thread(target=self._reap_loop, name="letify-reaper", daemon=True)
        self._reaper.start()

    def _reap_loop(self) -> None:
        while not self._stop_reaper.wait(REAP_INTERVAL):
            self.reap_idle()
            with self._guard:
                if self._count == 0:
                    self._reaper = None
                    return

    def reap_idle(self) -> list[str]:
        """Shut down runtimes that have been idle past the timeout."""
        with self._guard:
            candidates = [
                runtime
                for bucket in self._runtimes.values()
                for runtime in bucket
                if not runtime.busy and runtime.idle_for > self.idle_timeout
            ]
        stopped = []
        for runtime in candidates:
            self.discard(runtime)
            stopped.append(runtime.name)
        return stopped

    def shutdown_idle(self, *, every: bool = False) -> list[str]:
        """Shut down runtimes that are not running a call.

        A runtime whose lifetime is the process is left alone unless ``every`` is set.
        """
        with self._guard:
            candidates = [
                runtime
                for bucket in self._runtimes.values()
                for runtime in bucket
                if not runtime.busy and (every or runtime.lifetime is not Lifetime.process)
            ]
        stopped = []
        for runtime in candidates:
            self.discard(runtime)
            stopped.append(runtime.name)
        return stopped

    def shutdown(self) -> list[str]:
        """Shut everything down, including runtimes still marked busy."""
        with self._guard:
            everything = [r for bucket in self._runtimes.values() for r in bucket]
            self._runtimes.clear()
            self._count = 0
            self._hold_depth = 0
        self._stop_reaper.set()
        self._reaper = None
        for runtime in everything:
            try:
                runtime.shutdown()
            except LetifyError:
                pass
        with self._free:
            self._free.notify_all()
        return [r.name for r in everything]

    @property
    def live(self) -> list[Runtime]:
        with self._guard:
            return [r for bucket in self._runtimes.values() for r in bucket]


__all__ = ["DEFAULT_IDLE_TIMEOUT", "REAP_INTERVAL", "RuntimePool"]
