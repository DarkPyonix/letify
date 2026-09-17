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

from typing import Final

from .declare.cache import session_cache
from .declare.env import Env
from .declare.function import Function
from .declare.instance import AnyInstance, Host, Instance
from .errors import (
    ConfigError,
    EnvironmentFailure,
    InsufficientDevices,
    InterpreterMismatch,
    LetifyError,
    ProtocolError,
    ProviderUnavailable,
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

# The enum class stays importable from letify.declare.instance for letify's own use.
del Host

__version__ = "1.1.0"


def fetch(tensor):  # type: ignore[no-untyped-def]
    """Queue a read of ``tensor`` and return an awaitable resolving to its CPU copy.

    Under ``host="local"`` the read is queued at the call and awaiting it does not block the
    event loop. A tensor not on the runtime resolves to ``tensor.detach().cpu()``.
    """
    from .remoting.device.client import fetch as queue_fetch

    return queue_fetch(tensor)


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
    "local",
    "remote",
    "session_cache",
]
