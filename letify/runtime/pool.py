"""The pool that hands out runtimes and decides when they die.

The rule is that a runtime dies when the work that needed it is done. One invocation is the
unit, and invocations that overlap in time share one span, so concurrent calls reuse the
runtimes they release until the last of them finishes. Nothing is kept alive on the chance
that another call might come.

``hold`` is how a runtime outlives its release. ``Launcher.keep_alive()`` holds the pool for
the length of a block, and one invocation holds it for its own length. Released runtimes stay
while any hold is open and end when the last one closes.

Nothing ends a runtime on a timer. The lease is the one thing that ends one without being
asked, and it is for a process killed outright: the worker holds a deadline and exits if this
process stops pushing it forward.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..errors import InsufficientDevices, LetifyError

if TYPE_CHECKING:
    from ..declare.env import Env
    from ..declare.instance import Instance
    from ..store.volume import Volume
    from .session import Runtime

#: How long a call waits for a card before looking again. Not a deadline on anything: the
#: wait ends as soon as a session gives devices back, and this only bounds how long a lost
#: notification can go unnoticed.
POLL_INTERVAL = 30.0


class RuntimePool:
    """Keeps track of live runtimes and enforces the release rule.

    How many sessions may exist is the provider's inventory and nothing else. Starting one
    reserves the devices its instance asks for, and a call that cannot reserve them waits
    for a session to release some. There is no ceiling here: a number would be a guess about
    hardware the provider entry already describes, and when the two disagreed the smaller
    would win silently.
    """

    def __init__(self, *, on_start: object | None = None):
        self.on_start = on_start
        self._runtimes: dict[str, list[Runtime]] = {}
        self._guard = threading.RLock()
        self._free = threading.Condition(self._guard)
        self._hold_depth = 0
        # Sessions being started by this process, per provider and accelerator. A start holds
        # its cards before the runtime is registered, so without this count a concurrent call
        # would take those cards for another process's and refuse instead of waiting.
        self._starting: dict[tuple[int, str], int] = {}

    # -- holding -------------------------------------------------------------

    @property
    def holding(self) -> bool:
        with self._guard:
            return self._hold_depth > 0

    def hold(self) -> None:
        """Keep released runtimes alive until the matching ``unhold``.

        Used by ``Launcher.keep_alive()`` for the length of a block, and by one invocation so
        an overlapping call can reuse a session another call released.
        """
        with self._guard:
            self._hold_depth += 1

    def unhold(self) -> None:
        """Drop one hold, shutting runtimes down when the last one goes."""
        with self._guard:
            self._hold_depth = max(0, self._hold_depth - 1)
            closing = self._hold_depth == 0
        if closing:
            self.release_idle()

    # -- acquisition ---------------------------------------------------------

    def acquire(
        self,
        instance: Instance,
        env: Env,
        volumes: Sequence[Volume] = (),
    ) -> Runtime:
        """Return a runtime for this declaration, starting one if a slot is free."""
        key = f"{instance.key}|{env.key}"
        while True:
            with self._guard:
                for runtime in self._runtimes.get(key, ()):
                    if not runtime.busy:
                        runtime.busy = True
                        runtime.last_used = time.monotonic()
                        return runtime

            # No free session, so this needs cards. Taking them outside the pool guard,
            # because the provider may have to ask the machine who else is on them.
            starting = (id(instance.provider), instance.accelerator.casefold())
            with self._guard:
                self._starting[starting] = self._starting.get(starting, 0) + 1
            try:
                held = instance.provider.reserve(instance)
                if held is not None:
                    try:
                        return self._start(instance, env, volumes, key, held=held)
                    except BaseException:
                        instance.provider.unreserve(instance.accelerator, held, instance.devices)
                        raise
            finally:
                with self._free:
                    self._starting[starting] -= 1
                    if not self._starting[starting]:
                        del self._starting[starting]
                    self._free.notify_all()

            # Every card this instance could use is taken. That is worth waiting for only while a
            # session here is serving a call on one, because it gives the card back when the call
            # finishes. Otherwise nothing running would free a card, and waiting would never end.
            with self._free:
                if any(not runtime.busy for runtime in self._runtimes.get(key, ())):
                    continue
                reason = self._unallocatable(instance)
                if reason is None:
                    self._free.wait(timeout=POLL_INTERVAL)
                    continue
            raise InsufficientDevices(reason)

    def _unallocatable(self, instance: Instance) -> str | None:
        """Why the devices cannot be allocated, or ``None`` when a busy session will free one.

        Called with the pool guard held, after a reservation failed.
        """
        provider = instance.provider
        name = instance.accelerator
        try:
            declared = provider.devices_of(name).count
        except LetifyError:
            declared = None
        if declared is not None and instance.devices > declared:
            return (
                f"{instance!r} asks for {instance.devices} {name} but {provider.alias} declares "
                f"{declared}, so it can never be allocated. Ask for fewer, or declare more in "
                f"the account's devices table."
            )

        holders = [
            runtime
            for bucket in self._runtimes.values()
            for runtime in bucket
            if runtime.provider is provider
            and runtime.instance.accelerator.casefold() == name.casefold()
        ]
        if any(runtime.busy for runtime in holders):
            return None
        if self._starting.get((id(provider), name.casefold())):
            # A session this process is still starting holds a card and will serve a call.
            return None
        if holders:
            names = ", ".join(runtime.name for runtime in holders)
            return (
                f"every {name} on {provider.alias} is held by a session that is idle but kept by "
                f"let.keep_alive() ({names}), and this call needs a session with a different "
                f"environment. Nothing running would free a card. Make the call outside the "
                f"block, or give {provider.alias} more {name} in its devices table."
            )
        busy = ", ".join(str(index) for index in provider.last_busy) or "none"
        return (
            f"no {name} on {provider.alias} can be allocated: the cards it may use are taken by "
            f"another process (indices another process is computing on: {busy}), and letify "
            f"cannot know when that process ends."
        )

    def release(self, runtime: Runtime) -> None:
        """Give a runtime back. It ends here unless the pool is held."""
        with self._guard:
            runtime.busy = False
            runtime.last_used = time.monotonic()
            keep = self._hold_depth > 0
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
    ) -> Runtime:
        name = f"letify-{instance.accelerator.lower()}-{uuid.uuid4().hex[:6]}"
        if callable(self.on_start):
            self.on_start(instance, name)
        runtime = instance.provider.start(
            instance, env, name=name, volumes=tuple(volumes), held=held
        )
        runtime.busy = True
        with self._guard:
            self._runtimes.setdefault(key, []).append(runtime)
        return runtime

    def release_idle(self) -> list[str]:
        """End every runtime that is not serving a call, now that the last hold has closed.

        Not a timer: it runs when the block or the invocation that kept the sessions ends,
        which is the moment the caller named. A runtime still serving a call ends when that
        call releases it.
        """
        with self._guard:
            candidates = [
                runtime
                for bucket in self._runtimes.values()
                for runtime in bucket
                if not runtime.busy
            ]
        stopped = []
        for runtime in candidates:
            self.discard(runtime)
            stopped.append(runtime.name)
        return stopped

    def shutdown(self) -> list[str]:
        """Shut everything down, including runtimes still marked busy.

        Registered at process exit, which is the one moment everything goes.
        """
        with self._guard:
            everything = [r for bucket in self._runtimes.values() for r in bucket]
            self._runtimes.clear()
            self._hold_depth = 0
        for runtime in everything:
            runtime.provider.unreserve(
                runtime.instance.accelerator, runtime.held_devices, runtime.instance.devices
            )
            try:
                runtime.shutdown()
            except LetifyError:
                pass
        with self._free:
            self._free.notify_all()
        return [r.name for r in everything]

    # -- upkeep --------------------------------------------------------------

    @property
    def live(self) -> list[Runtime]:
        with self._guard:
            return [r for bucket in self._runtimes.values() for r in bucket]


__all__ = ["POLL_INTERVAL", "RuntimePool"]
