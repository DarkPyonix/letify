"""The declared function.

Calling a declared function runs it. There is no second verb: a decorator that
wraps a ``def`` and then needs ``.remote()`` to actually run it has moved the
declaration out of the declaration.

Whether a call blocks is decided at the ``def`` site. A plain ``def`` blocks and
returns its value; an ``async def`` returns an awaitable, so ``await`` and
``asyncio.gather`` behave as they do for any coroutine and letify contributes no
future type of its own.

Running many configurations is many calls. ``asyncio.gather`` over an async
declaration runs them at once, as wide as the provider's inventory allows.

One invocation is also how long the runtimes it needed live. They are released when the call
finishes, or when the last call overlapping it finishes. Inside a ``let.keep_alive()``
block they are kept until the block ends instead.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ..errors import ProtocolError, RuntimeFailure, RuntimeLost, SpotPreempted, UnsupportedMode
from .instance import AnyInstance, Host, Instance

if TYPE_CHECKING:
    from ..launcher import Launcher
    from ..store.volume import Volume
    from .env import Env

R = TypeVar("R")


class Function(Generic[R]):
    """A function with a declared execution site."""

    def __init__(
        self,
        fn: Callable[..., R],
        launcher: Launcher,
        *,
        device: Instance | AnyInstance,
        env: Env,
        host: Host | str | None = None,
        volumes: Sequence[Volume] = (),
        timeout: float | None = None,
        retries: int = 1,
        data_order: Any = None,
        data_first_wave: int | None = None,
    ):
        self.fn = fn
        self.launcher = launcher
        self.env = env
        self.host = _where(host)
        self.volumes = tuple(volumes)
        self.timeout = timeout
        self.retries = retries
        #: Spec "The send order and the first wave": a declared order outranks the analysis.
        self.data_order = data_order
        self.data_first_wave = data_first_wave
        self.is_async = inspect.iscoroutinefunction(fn)
        #: Spec "Replicas": one Instance per replica, in the order the list named them,
        #: each carrying the declared placement so a provider is not told it is local.
        self.replicas = tuple(
            instance._placed(self.host) for instance in _replicas(device, self.host)
        )
        self.device = self._place(self.replicas[0] if self.replicas else device)
        update_wrapper(self, fn)

    def _place(self, device: Instance | AnyInstance) -> Instance | AnyInstance:
        """Fold the declared host placement into the instance it names.

        A placement the instance's provider cannot serve is refused here, at decoration, as
        spec "Placements a provider cannot serve" describes.
        """
        placed = device._placed(self.host)
        if (
            isinstance(placed, Instance)
            and placed.placement is Host.local
            and not placed.provider.serves_host_local
        ):
            raise UnsupportedMode(
                f"{placed.provider.alias} cannot serve host='local', so {self.fn.__name__} "
                f"cannot be declared on {placed!r}. Declare it with host='remote'."
            )
        return placed

    # -- invocation ----------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the function at its declared site.

        A sync declaration returns its value. An async declaration returns a plain
        coroutine, so ``await`` and ``asyncio.gather`` accept it directly.
        """
        if self.is_async:
            return self._call_one(args, kwargs)
        with self.launcher.invocation():
            return self._run(args, kwargs)

    def session(self) -> Any:
        """The live session this declaration uses, started if it is not up yet.

        Internal to the declaration surface: a user never holds a session, and a volume
        takes the declaration rather than this. It exists because a volume moving a
        checkpoint has to move it through the session the calls actually run in, and only
        the declaration knows which that is. Handing the caller a session instead made it
        possible to name a different one, and a checkpoint written into a session nothing
        runs in is the kind of defect that passes on one machine and fails on a rented one.
        """
        launcher = self.launcher
        if not launcher.pool.holding:
            # The session would end as soon as this returned, so a file moved into it would
            # vanish before the call that needs it.
            raise UnsupportedMode(
                "moving a file through a session needs the session to outlive this call. "
                "Do it inside 'with let.keep_alive():', which keeps the session until the "
                "block ends."
            )
        launcher._register_at_exit()
        runtime = launcher.pool.acquire(launcher.resolve(self.device), self.env, self.volumes)
        # Handed straight back, because the point is for the next call to find it warm.
        launcher.pool.release(runtime)
        return runtime

    def local(self, *args: Any, **kwargs: Any) -> R:
        """Run the body in this process, ignoring the declaration.

        For testing a body with no provider at all. Declaring a ``Local`` provider is
        usually better, because that keeps the production code path.
        """
        return self.fn(*args, **kwargs)

    # -- execution -----------------------------------------------------------

    def _run(self, args: tuple, kwargs: dict) -> Any:
        launcher = self.launcher
        if self.host is Host.distributed:
            return self._run_replicas(args, kwargs)
        instance = launcher.resolve(self.device)
        if instance.placement is Host.local:
            return self._run_here(instance, args, kwargs)
        return self._run_one(instance, args, kwargs)

    def _run_one(
        self,
        instance: Any,
        args: tuple,
        kwargs: dict,
        replica: dict | None = None,
        *,
        on_channel: Callable[[Any], None] | None = None,
    ) -> Any:
        """Run the body on one runtime, retrying on an infrastructure failure.

        One replica of a distributed call runs through here too, so a replica gets the
        same retries and the same reporting as a call that named a single device.
        ``on_channel``, when given, is called with the runtime's channel as soon as one
        is acquired, so a distributed call's broker can watch it for an ``all_reduce``
        join before the call itself is sent.
        """
        launcher = self.launcher
        last: Exception | None = None

        for attempt in range(self.retries + 1):
            runtime = launcher.pool.acquire(instance, self.env, self.volumes)
            if runtime.channel is not None:
                # A persistent channel writes the body's output live, while the call runs.
                runtime.channel.echo = launcher.stream_logs
            if on_channel is not None and runtime.channel is not None:
                on_channel(runtime.channel)
            try:
                value, logs = runtime.call(
                    self.fn,
                    args,
                    kwargs,
                    timeout=self.timeout,
                    data_order=self.data_order,
                    data_first_wave=self.data_first_wave,
                    replica=_with_rate(replica, runtime),
                )
            except (RuntimeFailure, ProtocolError) as failure:
                # The session misbehaved rather than the user's code, so this runtime
                # is no longer trusted and the call may be retried on a fresh one. The
                # provider may name the failure first, such as a spot preemption.
                exc = runtime.provider.diagnose(runtime, failure)
                last = exc
                launcher.pool.discard(
                    runtime, reason=f"infrastructure_failure:{type(exc).__name__}: {exc}"
                )
                if attempt == self.retries:
                    if isinstance(exc, SpotPreempted):
                        raise exc from failure
                    lost = RuntimeLost(
                        f"{self.__name__} failed after {attempt + 1} attempt(s) on "
                        f"{instance!r}: {exc}"
                    )
                    # Kept as attributes only: str(exc) above already carries the tail.
                    lost.command = getattr(exc, "command", "")
                    lost.stderr = getattr(exc, "stderr", "")
                    raise lost from exc
                continue
            except BaseException:
                launcher.pool.release(runtime)
                raise
            else:
                launcher.pool.release(runtime)
                # Output of a one-shot channel arrives only with the outcome.
                if not runtime.persistent_channel and logs.strip() and launcher.stream_logs:
                    print(logs.rstrip(), file=sys.stderr)
                return value

        raise RuntimeLost(str(last))  # pragma: no cover

    def _run_replicas(self, args: tuple, kwargs: dict) -> list:
        """Run the body once per replica, and answer with their values in rank order.

        Spec "Replicas": each replica is told which one it is, the call waits for all of
        them, and a replica that fails fails the call, as a single remote call's failure
        does. The replicas run at the same time, because running them in turn would make
        a reduction between them impossible.
        """
        import threading

        from ..runtime.broker import Broker

        world = len(self.replicas)
        values: list = [None] * world
        failures: list[BaseException] = []
        broker = Broker()

        def run(rank: int, instance: Any) -> None:
            try:
                values[rank] = self._run_one(
                    instance,
                    args,
                    kwargs,
                    {"rank": rank, "world_size": world},
                    on_channel=lambda channel: broker.watch(rank, channel),
                )
            except BaseException as exc:  # reported after every replica has stopped
                failures.append(exc)

        threads = [
            threading.Thread(target=run, args=(rank, instance), daemon=True)
            for rank, instance in enumerate(self.replicas)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        broker.close()
        if failures:
            raise failures[0]
        return values

    def _run_here(self, instance: Any, args: tuple, kwargs: dict) -> Any:
        """Run the body in this process with its PyTorch operators on the runtime's device.

        Not retried: the body has already run its side effects here once.
        """
        launcher = self.launcher
        runtime = launcher.pool.acquire(instance, self.env, self.volumes)
        try:
            client = runtime.device()
            before = client.stats.snapshot()
            with client.activate():
                value = self.fn(*args, **kwargs)
                if inspect.iscoroutine(value):
                    value = asyncio.run(value)
                client.synchronize()
                value = _resolve_deferred(value)
            _report_deferred(client.stats.snapshot() - before)
        except (RuntimeFailure, ProtocolError) as failure:
            named = runtime.provider.diagnose(runtime, failure)
            launcher.pool.discard(
                runtime, reason=f"infrastructure_failure:{type(named).__name__}: {named}"
            )
            if named is failure:
                raise
            raise named from failure
        except BaseException:
            launcher.pool.release(runtime)
            raise
        launcher.pool.release(runtime)
        return value

    async def _run_async(self, args: tuple, kwargs: dict) -> Any:
        return await asyncio.to_thread(self._run, args, kwargs)

    async def _call_one(self, args: tuple, kwargs: dict) -> Any:
        """One awaited call, bracketed so its runtime is released at the end."""
        with self.launcher.invocation():
            return await self._run_async(args, kwargs)

    def __repr__(self) -> str:
        mode = "async" if self.is_async else "sync"
        return f"<Function {self.__name__} {mode} on {self.device!r}>"


def _resolve_deferred(value: Any) -> Any:
    """Resolve every deferred value the call returns, as spec "Deferred value reads" says."""
    from ..remoting.device.value import resolve

    return resolve(value)


def _report_deferred(delta: Any) -> None:
    """One line naming how many value reads were deferred and how many cost a wait."""
    if delta.auto_deferred:
        print(
            f"letify: deferred {delta.auto_deferred} value reads, "
            f"{delta.resolved_early} resolved early",
            file=sys.stderr,
        )


def _with_rate(replica: dict | None, runtime: Any) -> dict | None:
    """The replica's numbers plus the rate its own link measured, for ``local_steps``.

    Spec "How often to synchronize": the rate is the quantity only letify has, and it is
    this replica's link rather than the job's, so it travels with the call.
    """
    if replica is None:
        return None
    rate = None
    try:
        link = runtime.provider.link(runtime)
        rate = getattr(link, "upload_bps", None)
    except Exception:
        rate = None
    return {**replica, "rate_bps": rate}


def _replicas(device: Any, host: Host) -> tuple[Instance, ...]:
    """The replicas a declaration names, or empty when it names a single device.

    Spec "Replicas": ``device`` takes a list only under ``host=letify.distributed``, each
    entry is one replica, and two replicas of one instance are refused because that asks
    one account for twice the devices it has.
    """
    listed = isinstance(device, (list, tuple))
    if listed and host is not Host.distributed:
        raise ValueError(
            f"device is a list of {len(device)} instances, which declares one replica "
            f"each, so it needs host=letify.distributed. Under host={host.value!r} name "
            f"one device."
        )
    if not listed:
        return (device,) if host is Host.distributed else ()
    if not device:
        raise ValueError(
            "device is an empty list, so the declaration names no replica. Name at least "
            "one instance."
        )
    seen: list[str] = []
    for instance in device:
        alias = getattr(getattr(instance, "provider", None), "alias", None)
        name = getattr(instance, "accelerator", None)
        key = f"{alias}.{name}"
        if key in seen:
            raise ValueError(
                f"device names {key} twice, so one account would be asked for twice the "
                f"devices one replica takes. Name each account once, and use a larger "
                f"instance such as {name} * 2 for more devices in one replica."
            )
        seen.append(key)
    return tuple(device)


def _where(host: Host | str | None) -> Host:
    """Validate the declared host placement, naming both options when it is wrong."""
    if host is None:
        return Host.local
    try:
        return Host(host)
    except ValueError:
        raise ValueError(
            f"host={host!r} is not a host placement. Use host='local' to keep Python "
            f"here and forward PyTorch operators, host='remote' to ship the function to "
            f"the machine that holds the device, or host='distributed' to run it once per "
            f"device in a list."
        ) from None


__all__ = ["Function"]
