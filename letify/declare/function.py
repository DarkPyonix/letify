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

from ..errors import ProtocolError, RuntimeFailure, RuntimeLost, UnsupportedMode
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
        instance = launcher.resolve(self.device)
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


__all__ = ["Function"]
