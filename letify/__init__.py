"""letify: declarations that become infrastructure.

Declare what a function needs and it runs there:

    import letify

    let = letify.Launcher()
    colab = let.providers.colab_a

    @let.function(device=colab.G4, host=letify.Host.remote)
    def train(lr, bs):
        ...

    train(lr=1e-4, bs=32)

A declaration places two things. ``device`` says where the device is, carrying the provider
and the account with it. ``host`` says where the host code runs: ``local``, the default,
keeps Python here and forwards only CUDA calls, and ``remote`` ships the function to the
machine that holds the device.

A session ends with the call that needed it. ``lifetime="process"`` keeps it, so a run of
separate calls does not pay session start each time. Nothing has to be torn down by hand.

Nothing here imports a provider's optional dependency, so ``import letify`` works with
the base install and a provider whose package is missing reports itself unavailable.
"""

from __future__ import annotations

from .declare.env import Env
from .declare.function import Function
from .declare.instance import AnyInstance, Host, Instance, Lifetime
from .declare.sweep import Sweep, grid
from .declare.sweep import zip_ as zip
from .errors import (
    ConfigError,
    HandleScopeError,
    LetifyError,
    ProtocolError,
    ProviderUnavailable,
    RemoteError,
    RuntimeFailure,
    RuntimeLost,
    UnknownInstance,
    UnknownProvider,
    UnsupportedMode,
)
from .launcher import Launcher, Providers
from .protocol.handle import Blob, Handle, RemoteFile
from .store.volume import Volume

__version__ = "1.0.0"

__all__ = [
    "AnyInstance",
    "Blob",
    "ConfigError",
    "Env",
    "Function",
    "Handle",
    "HandleScopeError",
    "Host",
    "Instance",
    "Launcher",
    "LetifyError",
    "Lifetime",
    "ProtocolError",
    "ProviderUnavailable",
    "Providers",
    "RemoteError",
    "RemoteFile",
    "RuntimeFailure",
    "RuntimeLost",
    "Sweep",
    "UnknownInstance",
    "UnknownProvider",
    "UnsupportedMode",
    "Volume",
    "__version__",
    "grid",
    "zip",
]
