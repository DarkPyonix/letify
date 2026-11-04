"""Reattaching to a running call after the owner's channel broke.

Owns spec "Reattaching after a lost channel": the rule for when a lost channel is worth a
reattach, the attempts that follow it and the messages they print. It does not own how a
channel to the worker's second door is built, which ``Provider.attach_channel`` does, the
adoption itself, which the worker source owns, or the lease, which ``letify.runtime.lease``
owns. The sequence of connecting, adopting and receiving the reply is the one of
``letify.runtime.takeover``, with this process as both the old and the new owner.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from ..errors import LetifyError, ProtocolError, RemoteError, RuntimeFailure, UnsupportedMode

if TYPE_CHECKING:
    from . import control
    from .channel import Channel
    from .session import Runtime

#: Seconds waited before attempt 1, 2, 3 and 4. Every later attempt waits the last value.
DELAYS_S: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)

#: The most attempts made for one lost channel.
MAX_ATTEMPTS = 8

#: Seconds that must remain before the lease deadline for an attempt to be worth making.
MARGIN_S = 10.0


def eligible(runtime: Runtime, run: control.Run, failure: ProtocolError) -> bool:
    """Whether ``failure`` is a lost channel whose worker may still be running its call.

    Spec "Reattaching after a lost channel", the first item. The worker is not shown to have
    died, the lease still has time, the run has a second door and the provider does not say
    the runtime is gone.
    """
    channel, lease = runtime.channel, runtime.lease
    if channel is None or lease is None or lease.deadline is None:
        return False
    if lease.deadline - time.time() <= MARGIN_S:
        return False
    if not run.details.get("door"):
        return False
    if not channel.worker_may_survive():
        return False
    from ..providers.base import RuntimeState

    try:
        state = runtime.provider.runtime_state(runtime)
    except Exception:
        state = RuntimeState()
    if state.status == "ended":
        return False
    try:
        return runtime.provider.diagnose(runtime, failure) is failure
    except Exception:
        return False


def request(
    runtime: Runtime,
    run: control.Run,
    payload: dict[str, Any],
    *,
    timeout: float | None,
    on_channel: Callable[[Channel], None] | None = None,
) -> tuple[Any, str]:
    """Send ``payload`` on the runtime's channel and answer like ``Channel.request``.

    A channel lost while the call runs is replaced by one to the same worker, and the call's
    reply comes over that. ``on_channel`` is told of each new channel once it is the
    runtime's, so a thread that watches the channel can move to it. When reattaching is not
    possible the original failure is raised, for the caller's discard and retry.
    """
    started = time.monotonic()
    assert runtime.channel is not None
    try:
        return runtime.channel.request(payload, timeout=timeout)
    except ProtocolError as exc:
        failure = exc
    while True:
        remaining = None
        if timeout is not None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeFailure(f"{runtime.name}: the call exceeded {timeout}s") from failure
        pieces = _reattach(runtime, run, failure, remaining, on_channel)
        try:
            return next(pieces), ""
        except ProtocolError as exc:
            # Lost again: the same limits apply to the new loss.
            failure = exc
        finally:
            pieces.close()


def _reattach(
    runtime: Runtime,
    run: control.Run,
    failure: ProtocolError,
    timeout: float | None,
    on_channel: Callable[[Channel], None] | None,
) -> Iterator[Any]:
    """Adopt the run again over a new channel and return the stream that carries its reply.

    Raises ``failure`` when the conditions do not hold or no attempt succeeds.
    """
    if not eligible(runtime, run, failure):
        raise failure
    lease = runtime.lease
    assert lease is not None and lease.deadline is not None
    old = runtime.channel
    assert old is not None
    # The lease does not renew on a dead channel. It starts again over the new one.
    lease.release()
    record = {**run.record(run.socket_file), "external_id": runtime.external_id}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        delay = DELAYS_S[min(attempt, len(DELAYS_S)) - 1]
        if lease.deadline - time.time() < delay + MARGIN_S:
            break
        try:
            channel = runtime.provider.attach_channel(runtime.name, record)
        except UnsupportedMode:
            break
        time.sleep(delay)
        print(
            f"letify: lost the channel to {runtime.name} during {run.id}; "
            f"reattaching (attempt {attempt})",
            file=sys.stderr,
            flush=True,
        )
        channel.echo, channel.on_output = old.echo, old.on_output
        pieces = None
        try:
            channel.start()
            pieces = channel.stream(
                {"op": "adopt", "run": run.id, "owner": os.getpid()}, timeout=timeout
            )
            next(pieces)
        except RemoteError:
            # The worker answered that it has no such call: nothing is left to adopt.
            _let_go(channel, pieces)
            break
        except LetifyError:
            _let_go(channel, pieces)
            continue
        runtime.channel = channel
        runtime.reattached = True
        lease.restart()
        if on_channel is not None:
            on_channel(channel)
        _let_go(old)
        print(f"letify: reattached to {run.id} on {runtime.name}", file=sys.stderr, flush=True)
        assert pieces is not None
        return pieces
    # Gave up. The lease stays released, so a worker nobody reaches ends at its deadline.
    raise failure


def _let_go(channel: Channel, pieces: Iterator[Any] | None = None) -> None:
    """Drop a channel without a shutdown frame, which the worker would take as an end."""
    if pieces is not None:
        with contextlib.suppress(Exception):
            pieces.close()
    kill = getattr(channel, "_kill", None)
    if kill is not None:
        with contextlib.suppress(Exception):
            kill()
    with contextlib.suppress(Exception):
        channel.detach()
