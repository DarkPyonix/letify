"""letify: declarations that become infrastructure.

Declare what a function needs and it runs there:

    import letify

    let = letify.Launcher()
    colab = let.providers.colab_a

    @let.function(device=colab.G4, host=letify.remote)
    def train(lr, bs):
        ...

    train(lr=1e-4, bs=32)

A declaration places two things. ``device`` says where the device is, carrying the provider
and the account with it. ``host`` says where the host code runs: ``local``, the default,
keeps Python here and forwards only PyTorch operators, and ``remote`` ships the function to the
machine that holds the device.

A session ends with the call that needed it. ``with let.keep_alive():`` keeps sessions for the
length of a block, so a run of separate calls does not pay session start each time.

Nothing here imports a provider's optional dependency, so ``import letify`` works with
the base install and a provider whose package is missing reports itself unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, overload

from .declare.cache import session_cache
from .declare.env import Env
from .declare.exit import on_exit
from .declare.function import Function
from .declare.instance import AnyInstance, Host, Instance
from .declare.reduce import all_reduce, mean, sum
from .declare.replicas import local_steps, rank, world_size
from .errors import (
    ConfigError,
    EnvironmentFailure,
    InsufficientDevices,
    InterpreterMismatch,
    LetifyError,
    ProtocolError,
    ProviderUnavailable,
    ReductionFailed,
    RemoteError,
    RuntimeFailure,
    RuntimeLost,
    SpotPreempted,
    UnknownInstance,
    UnknownProvider,
    UnsupportedMode,
)
from .launcher import Launcher, Providers
from .protocol.handle import Blob, RemoteFile
from .store.volume import Volume

#: Where a declaration's host code runs. Two named values rather than the enum class that holds
#: them, because a declaration only ever needs one of the two.
#: ``Final`` so a type checker sees the literal member, which the declaration overloads match on.
local: Final = Host.local
remote: Final = Host.remote
distributed: Final = Host.distributed

# The enum class stays importable from letify.declare.instance for letify's own use.
del Host

__version__ = "2.8.6"


def fetch(tensor):  # type: ignore[no-untyped-def]
    """Queue a read of ``tensor`` and return an awaitable resolving to its CPU copy.

    Under ``host="local"`` the read is queued at the call and awaiting it does not block the
    event loop. A tensor not on the runtime resolves to ``tensor.detach().cpu()``.
    """
    from .remoting.device.client import fetch as queue_fetch

    return queue_fetch(tensor)


def stash(name: str, path: str | os.PathLike[str]) -> str:
    """Put a file or directory on the machine's Drive store under ``name``.

    Returns the digest once the upload has finished and the ref is written. A later call on
    any provider, or the local process, gets it back with ``unstash(name, path)``.
    """
    from .store.handoff import stash as stash_handoff

    return stash_handoff(name, path)


@overload
def unstash(name: str) -> dict[str, bytes] | None: ...
@overload
def unstash(name: str, path: str | os.PathLike[str]) -> Path | None: ...
def unstash(
    name: str, path: str | os.PathLike[str] | None = None
) -> Path | dict[str, bytes] | None:
    """Put the handoff ``name`` at ``path``, replacing what was there.

    Returns ``path``, or None when no handoff has that name, in which case nothing changed.

    With no ``path`` the handoff is returned in memory and nothing is written to disk: a
    ``dict[str, bytes]`` from the path of each regular file in the handoff, as stored
    (``ckpt/model.safetensors``, or ``checkpoint.safetensors`` for a one-file handoff), to
    its bytes. Directories are not entries, and None means no handoff has that name. The
    dict is returned only after the digest matches; a mismatch raises ``RuntimeFailure``.
    Peak memory is about the handoff's size, because members are read straight from the
    download.
    """
    from .store.handoff import unstash as unstash_handoff

    if path is None:
        return unstash_handoff(name)
    return unstash_handoff(name, path)


__all__ = [
    "AnyInstance",
    "Blob",
    "ConfigError",
    "Env",
    "EnvironmentFailure",
    "Function",
    "Instance",
    "InsufficientDevices",
    "InterpreterMismatch",
    "Launcher",
    "LetifyError",
    "ProtocolError",
    "ProviderUnavailable",
    "Providers",
    "ReductionFailed",
    "RemoteError",
    "RemoteFile",
    "RuntimeFailure",
    "RuntimeLost",
    "SpotPreempted",
    "UnknownInstance",
    "UnknownProvider",
    "UnsupportedMode",
    "Volume",
    "__version__",
    "all_reduce",
    "distributed",
    "fetch",
    "local",
    "local_steps",
    "mean",
    "on_exit",
    "rank",
    "remote",
    "session_cache",
    "stash",
    "sum",
    "unstash",
    "world_size",
]
