"""letify: declarations that become infrastructure.

Declare what a function needs and it runs there:

    import letify

    let = letify.Launcher()
    env = letify.Env()

    colab = let.providers.colab_a

    @let.function(gpu=colab.G4, env=env, concurrency=3)
    def train(lr, bs):
        ...

    with let.run():
        train(lr=1e-4, bs=32)

Nothing in this module imports a provider's optional dependency, so
``import letify`` works with the base install and a provider whose package is
missing simply reports itself unavailable.
"""

from __future__ import annotations

from .env import Env
from .errors import (
    ConfigError,
    HandleScopeError,
    LetifyError,
    NotRunning,
    ProtocolError,
    ProviderUnavailable,
    RemoteError,
    RuntimeFailure,
    RuntimeLost,
    UnknownInstance,
    UnknownProvider,
    UnsupportedMode,
)
from .function import Function
from .instance import AnyInstance, Instance
from .launcher import Launcher, Providers
from .store import Volume
from .sweep import Sweep, grid
from .sweep import zip_ as zip
from .wire import Blob, Handle

__version__ = "1.0.0"

__all__ = [
    "AnyInstance",
    "Blob",
    "ConfigError",
    "Env",
    "Function",
    "Handle",
    "HandleScopeError",
    "Instance",
    "Launcher",
    "LetifyError",
    "NotRunning",
    "ProtocolError",
    "ProviderUnavailable",
    "Providers",
    "RemoteError",
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
