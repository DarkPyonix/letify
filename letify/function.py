"""The declared function.

Calling a declared function runs it. There is no second verb: writing
``train(lr=1e-4)`` is the whole invocation, because a decorator that wraps a
``def`` and then needs ``.remote()`` to actually run it has taken the meaning out
of the declaration.

Whether a call blocks is decided at the ``def`` site, not the call site. A plain
``def`` runs and returns its value. An ``async def`` returns a coroutine, so
``await`` and ``asyncio.gather`` work the way they do for any other coroutine, and
letify does not need a future type of its own.

Fan-out is a declared space rather than a map call. Passing a ``Sweep`` where a
scalar is expected says that the argument varies, and the two ways of consuming
the result are already in the language: ``await`` gives a list in input order,
``async for`` yields results as they finish.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .errors import NotRunning, ProtocolError, RuntimeFailure, RuntimeLost
from .instance import AnyInstance, Instance
from .sweep import Sweep

if TYPE_CHECKING:
    from .env import Env
    from .launcher import Launcher
    from .store.volume import Volume

R = TypeVar("R")


class Function(Generic[R]):
    """A function with a declared execution site."""

    def __init__(
        self,
        fn: Callable[..., R],
        launcher: Launcher,
        *,
        gpu: Instance | AnyInstance,
        env: Env,
        volumes: Sequence[Volume] = (),
        concurrency: int = 1,
        timeout: float | None = 3600,
        retries: int = 1,
        keep_remote: bool = False,
    ):
        self.fn = fn
        self.launcher = launcher
        self.gpu = gpu
        self.env = env
        self.volumes = tuple(volumes)
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.retries = retries
        self.keep_remote = keep_remote
        self.is_async = inspect.iscoroutinefunction(fn)
        update_wrapper(self, fn)

    # -- invocation ----------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the function at its declared site.

        The return type follows the declaration. A sync function returns its
        value, or a list of values when a space was passed. An async function
        returns an awaitable that is also async-iterable, so ``await`` collects in
        order and ``async for`` yields by completion.
        """
        space = _space_in(args, kwargs)
        if self.is_async:
            return _AsyncCall(self, args, kwargs, space)
        if space is None:
            return self._run(args, kwargs)
        return [self._run(*point) for point in _expand(args, kwargs, space)]

    def local(self, *args: Any, **kwargs: Any) -> R:
        """Run the body in this process, ignoring the declaration.

        Present for the case where a declared function has to be exercised
        without any provider, such as a unit test of the body itself. Prefer
        declaring a ``Local`` provider, which keeps the production code path.
        """
        return self.fn(*args, **kwargs)

    # -- execution -----------------------------------------------------------

    def _run(self, args: tuple, kwargs: dict) -> Any:
        launcher = self.launcher
        if not launcher.is_running:
            raise NotRunning(
                f"{self.__name__} was called outside a run scope. Wrap the call in "
                f"`with let.run():` so letify knows when to start and stop paying "
                f"for a session."
            )
        instance = launcher.resolve(self.gpu)
        last: Exception | None = None

        for attempt in range(self.retries + 1):
            runtime = launcher.pool.acquire(instance, self.env, self.volumes)
            try:
                value, logs = runtime.call(
                    self.fn,
                    args,
                    kwargs,
                    keep_remote=self.keep_remote,
                    timeout=self.timeout,
                )
            except (RuntimeFailure, ProtocolError) as exc:
                # The session misbehaved rather than the user's code, so the
                # runtime is no longer trusted and the call may be retried.
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

    def __repr__(self) -> str:
        mode = "async" if self.is_async else "sync"
        return f"<Function {self.__name__} {mode} on {self.gpu!r}>"


class _AsyncCall:
    """The object an async declared function returns.

    Awaiting it collects every result in input order. Iterating it with
    ``async for`` yields results as they complete, which is what a long sweep
    wants so that early results can be logged while the rest is still running.
    """

    def __init__(self, function: Function, args: tuple, kwargs: dict, space: Sweep | None):
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._space = space

    def __await__(self) -> Any:
        return self._gather().__await__()

    async def _gather(self) -> Any:
        function = self._function
        if self._space is None:
            return await function._run_async(self._args, self._kwargs)
        semaphore = asyncio.Semaphore(function.concurrency)

        async def one(args: tuple, kwargs: dict) -> Any:
            async with semaphore:
                return await function._run_async(args, kwargs)

        tasks = [
            asyncio.create_task(one(a, k))
            for a, k in _expand(self._args, self._kwargs, self._space)
        ]
        return list(await asyncio.gather(*tasks))

    async def __aiter__(self) -> AsyncIterator[Any]:
        function = self._function
        if self._space is None:
            yield await function._run_async(self._args, self._kwargs)
            return
        semaphore = asyncio.Semaphore(function.concurrency)

        async def one(args: tuple, kwargs: dict) -> Any:
            async with semaphore:
                return await function._run_async(args, kwargs)

        tasks = [
            asyncio.create_task(one(a, k))
            for a, k in _expand(self._args, self._kwargs, self._space)
        ]
        for finished in asyncio.as_completed(tasks):
            yield await finished


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

    A space names the arguments it varies, so wherever it appeared it is removed
    and its point is merged into the keyword arguments. That keeps
    ``train(space)`` and ``train(over=space)`` meaning the same thing.
    """
    kept_args = tuple(v for v in args if not isinstance(v, Sweep))
    kept_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Sweep)}
    for point in space:
        yield kept_args, {**kept_kwargs, **point}
