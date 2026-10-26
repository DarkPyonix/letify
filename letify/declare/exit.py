"""``letify.on_exit``: a function to run when a body's worker is about to end.

Owns the public registration call and the errors for where it cannot work. It does not own
when hooks run, which is the worker's (``letify.protocol.worker``), nor how they are ordered
and bounded, which spec "Controlling a running call", Exit hooks, states.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from ..errors import UnsupportedMode


def on_exit(fn: Callable[[], object]) -> None:
    """Run ``fn`` if the worker ends while this body is still running.

    Call it inside a body. ``fn`` runs, with no arguments, when the worker receives
    ``SIGTERM``, when its lease deadline passes, or when the channel closes during the call.
    Hooks run newest first, each once, within 120 s in total, and are dropped when the call
    ends. ``SIGKILL`` and the out of memory killer run no hook. Outside a body this raises
    ``UnsupportedMode``, because a local process has ``atexit``.
    """
    if not callable(fn):
        raise TypeError(f"letify.on_exit needs a callable, got {type(fn).__name__}")
    main = sys.modules.get("__main__")
    register = getattr(main, "_exit_register", None)
    if register is None:
        worker_pid = os.environ.get("LETIFY_WORKER_PID")
        if worker_pid is not None and worker_pid != str(os.getpid()):
            raise UnsupportedMode(
                "letify.on_exit must be called in the process letify runs the body in, not in "
                f"a child process it started (a child of worker process {worker_pid})."
            )
        if os.environ.get("LETIFY_ONE_SHOT") is not None:
            raise UnsupportedMode(
                "letify.on_exit needs a persistent channel, and this body runs on a one-shot "
                "channel, which ends its process when the call does."
            )
        raise UnsupportedMode(
            "letify.on_exit registers a hook inside a body running on a runtime. "
            "In a local process use atexit."
        )
    if not register(fn):
        raise UnsupportedMode(
            "letify.on_exit was called while no body is running, so there is no call to "
            "attach the hook to. In a local process use atexit."
        )
