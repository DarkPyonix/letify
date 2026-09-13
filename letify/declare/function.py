"""The declared function.

Calling a declared function runs it. There is no second verb: a decorator that
wraps a ``def`` and then needs ``.remote()`` to actually run it has moved the
declaration out of the declaration.

Whether a call blocks is decided at the ``def`` site. A plain ``def`` blocks and
returns its value; an ``async def`` returns an awaitable, so ``await`` and
``asyncio.gather`` behave as they do for any coroutine and letify contributes no
future type of its own.

Fan-out is a declared space rather than a map call. Passing a ``Sweep`` where a
scalar is expected says the argument varies, and the two useful orderings are
already in the language: ``await`` gives input order, ``async for`` gives completion
order.

One invocation is also how long the runtimes it needed live. They are released when the call
finishes, including a sweep, which counts as one invocation. Inside a ``let.keep_alive()``
block they are kept until the block ends instead.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ..errors import LetifyError, ProtocolError, RuntimeFailure, RuntimeLost, UnsupportedMode
from .instance import AnyInstance, Host, Instance
from .sweep import Sweep

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
        keep_remote: bool = False,
    ):
        self.fn = fn
        self.launcher = launcher
        self.env = env
        self.host = _where(host)
        self.volumes = tuple(volumes)
        self.timeout = timeout
        self.retries = retries
        self.keep_remote = keep_remote
        self.is_async = inspect.iscoroutinefunction(fn)
        self.device = self._place(device)
        update_wrapper(self, fn)

    def _place(self, device: Instance | AnyInstance) -> Instance | AnyInstance:
        """Fold the declared host placement into the instance it names."""
        return device._placed(self.host)

    # -- invocation ----------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the function at its declared site.

        A sync declaration returns its value, or a list of values when a space was
        passed. An async declaration returns an awaitable that is also
        async-iterable, so ``await`` collects in order and ``async for`` yields by
        completion.
        """
        space = _space_in(args, kwargs)
        if self.is_async:
            if space is None:
                # A plain coroutine, so asyncio.run and asyncio.gather accept it
                # directly. Only the fan-out form needs a type of its own, because
                # that is what async for iterates.
                return self._call_one(args, kwargs)
            return AsyncCall(self, args, kwargs, space)
        with self.launcher.invocation():
            if space is None:
                return self._run(args, kwargs)
            return [self._run(a, k) for a, k in _expand(args, kwargs, space)]

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
        instance = launcher.resolve(self.device)
        last: Exception | None = None

        for attempt in range(self.retries + 1):
            runtime = launcher.pool.acquire(
                instance, self.env, self.volumes
            )
            try:
                value, logs = runtime.call(
                    self.fn,
                    args,
                    kwargs,
                    keep_remote=self.keep_remote,
                    timeout=self.timeout,
                )
            except (RuntimeFailure, ProtocolError) as exc:
                # The session misbehaved rather than the user's code, so this runtime
                # is no longer trusted and the call may be retried on a fresh one.
                last = exc
                launcher.pool.discard(runtime)
                if attempt == self.retries:
                    raise RuntimeLost(
                        f"{self.__name__} failed after {attempt + 1} attempt(s) on "
                        f"{instance!r}: {exc}"
                    ) from exc
                continue
            except BaseException:
                launcher.pool.release(runtime)
                raise
            else:
                launcher.pool.release(runtime)
                if logs.strip() and launcher.stream_logs:
                    print(logs.rstrip(), file=sys.stderr)
                return value

        raise RuntimeLost(str(last))  # pragma: no cover

    async def _run_async(self, args: tuple, kwargs: dict) -> Any:
        return await asyncio.to_thread(self._run, args, kwargs)

    async def _call_one(self, args: tuple, kwargs: dict) -> Any:
        """One awaited call, bracketed so its runtime is released at the end."""
        with self.launcher.invocation():
            return await self._run_async(args, kwargs)

    def __repr__(self) -> str:
        mode = "async" if self.is_async else "sync"
        return f"<Function {self.__name__} {mode} on {self.device!r}>"


class AsyncCall:
    """What an async declaration returns when a search space was passed.

    Awaiting it collects every result in input order. Iterating it with ``async for``
    yields results as they complete, which is what a long sweep wants so early results
    can be logged while the rest is still running.

    A call with no space returns a plain coroutine instead, so the standard library
    accepts it wherever a coroutine is expected.
    """

    def __init__(self, function: Function, args: tuple, kwargs: dict, space: Sweep):
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._space = space

    def __await__(self) -> Any:
        return self._gather().__await__()

    async def _gather(self) -> Any:
        function = self._function
        with function.launcher.invocation():
            return list(await asyncio.gather(*self._tasks()))

    async def __aiter__(self) -> AsyncIterator[Any]:
        function = self._function
        with function.launcher.invocation():
            for finished in asyncio.as_completed(self._tasks()):
                yield await finished

    def _tasks(self) -> list[asyncio.Task]:
        function = self._function
        semaphore = asyncio.Semaphore(_width(function))

        async def one(args: tuple, kwargs: dict) -> Any:
            async with semaphore:
                return await function._run_async(args, kwargs)

        return [
            asyncio.create_task(one(a, k))
            for a, k in _expand(self._args, self._kwargs, self._space)
        ]


def _width(function: Function) -> int:
    """How many points of a space may be in flight at once.

    The provider's inventory and nothing else. A declaration taking two cards halves the
    width on a four card machine, which is arithmetic rather than policy, and is also why a
    number on the declaration could not express it.

    At least one, always. A provider whose inventory cannot be read yet is not a reason to
    run nothing: the pool is what waits when a card is not there.
    """
    instance = function.launcher.resolve(function.device)
    try:
        capacity = instance.provider.capacity(instance.accelerator)
    except LetifyError:
        return 1
    return max(1, capacity // max(1, instance.devices))


def _where(host: Host | str | None) -> Host:
    """Validate the declared host placement, naming both options when it is wrong."""
    if host is None:
        return Host.local
    try:
        return Host(host)
    except ValueError:
        raise ValueError(
            f"host={host!r} is not a host placement. Use host='local' to keep Python "
            f"here and forward CUDA calls, or host='remote' to ship the function to "
            f"the machine that holds the device."
        ) from None


def _space_in(args: tuple, kwargs: dict) -> Sweep | None:
    """Find the declared space in a call, if the caller passed one."""
    spaces = [v for v in (*args, *kwargs.values()) if isinstance(v, Sweep)]
    if not spaces:
        return None
    if len(spaces) > 1:
        raise TypeError(
            "only one search space may be passed per call. Combine them with "
            "let.grid(...) or the | operator instead."
        )
    return spaces[0]


def _expand(args: tuple, kwargs: dict, space: Sweep) -> Iterator[tuple[tuple, dict]]:
    """Turn one call carrying a space into one call per point.

    A space names the arguments it varies, so wherever it appeared it is removed and
    its point is merged into the keyword arguments. That keeps ``train(space)`` and
    ``train(over=space)`` meaning the same thing.
    """
    kept_args = tuple(v for v in args if not isinstance(v, Sweep))
    kept_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Sweep)}
    for point in space:
        yield kept_args, {**kept_kwargs, **point}


__all__ = ["AsyncCall", "Function"]
