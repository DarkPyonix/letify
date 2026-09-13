"""The pool that hands out runtimes and decides when they die.

The rule is that a runtime dies when the work that needed it is done. One
invocation is the unit, and a search space counts as one invocation, so a sweep
boots its runtimes once and releases them when the last point finishes. Nothing is
kept alive on the chance that another call might come.

Keeping one longer is declared, never assumed. A runtime acquired for a declaration
made with ``warm=True`` is marked warm and survives its release, and ``hold()`` does
the same for every runtime inside a ``let.warm()`` block. Both exist because starting
a session is not free: provider boot plus environment installation is minutes on
Colab, so several separate calls in a row are cheaper warm than cold.

Two backstops cover a warm runtime nobody released. A reaper thread tears down
runtimes idle past the timeout, and each runtime's lease makes the remote worker exit
if this process stops renewing.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

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

    ``max_runtimes`` is a ceiling for the whole process. It defaults to 3 and is a
    guess: the number of sessions one Colab account may hold at once is not
    documented and moves with tier, credit balance and current demand.
    """

    def __init__(
        self,
        *,
        max_runtimes: int = 3,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        on_start: object | None = None,
    ):
        self.max_runtimes = max_runtimes
        self.idle_timeout = idle_timeout
        self.on_start = on_start
        self._runtimes: dict[str, list[Runtime]] = {}
        self._guard = threading.RLock()
        self._free = threading.Condition(self._guard)
        self._count = 0
        self._hold_depth = 0
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()

    # -- holding -------------------------------------------------------------

    @property
    def holding(self) -> bool:
        with self._guard:
            return self._hold_depth > 0

    def hold(self) -> None:
        """Keep released runtimes alive until the matching ``unhold``."""
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
        warm: bool = False,
    ) -> Runtime:
        """Return a runtime for this declaration, starting one if a slot is free.

        ``warm`` marks the runtime as worth surviving its release, which is what a
        declaration made with ``warm=True`` asks for.
        """
        key = f"{instance.key}|{env.key}"
        while True:
            with self._guard:
                for runtime in self._runtimes.get(key, ()):
                    if not runtime.busy:
                        runtime.busy = True
                        runtime.warm = runtime.warm or warm
                        runtime.last_used = time.monotonic()
                        return runtime
                may_start = self._count < self.max_runtimes
                if may_start:
                    self._count += 1

            if may_start:
                try:
                    return self._start(instance, env, volumes, key, warm=warm)
                except BaseException:
                    with self._guard:
                        self._count -= 1
                        self._free.notify_all()
                    raise

            # Every slot is taken. Wait for one to come free rather than asking the
            # provider for a session it would refuse.
            with self._free:
                self._free.wait(timeout=REAP_INTERVAL)

    def release(self, runtime: Runtime) -> None:
        """Give a runtime back. It dies here unless it was declared warm."""
        with self._guard:
            runtime.busy = False
            runtime.last_used = time.monotonic()
            keep = runtime.warm or self._hold_depth > 0
        if keep:
            with self._free:
                self._free.notify_all()
            return
        self.discard(runtime)

    def discard(self, runtime: Runtime) -> None:
        """Shut a runtime down and free its slot, whatever state it was in."""
        with self._guard:
            bucket = self._runtimes.get(runtime.key, [])
            if runtime in bucket:
                bucket.remove(runtime)
                self._count -= 1
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
        *,
        warm: bool = False,
    ) -> Runtime:
        name = f"letify-{instance.accelerator.lower()}-{uuid.uuid4().hex[:6]}"
        if callable(self.on_start):
            self.on_start(instance, name)
        runtime = instance.provider.start(instance, env, name=name, volumes=tuple(volumes))
        runtime.busy = True
        runtime.warm = warm
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

    def shutdown_idle(self, *, include_warm: bool = False) -> list[str]:
        """Shut down runtimes that are not running a call.

        A runtime declared warm is left alone unless ``include_warm`` is set, which is
        what an explicit release does.
        """
        with self._guard:
            candidates = [
                runtime
                for bucket in self._runtimes.values()
                for runtime in bucket
                if not runtime.busy and (include_warm or not runtime.warm)
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
