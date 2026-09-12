"""Provider, one account on one kind of infrastructure.

A Provider object is what you get when a configuration entry is loaded. It knows
which accelerators that account can offer, owns the storage that outlives a
session, and creates the session itself.

Two properties on a provider decide how letify runs work there, and neither is a
user-facing switch.

``persistence`` says whether storage outlives a runtime. When it does, the
environment and the data are already on the machine, so shipping the whole loop
function is natural. When it does not, every runtime would have to rebuild them.

``has_fast_path`` says whether a low-latency connection to the machine exists.
CUDA call forwarding puts one network round trip in front of every host
synchronization, so it is only worth offering where that round trip is small.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Literal

from ..config import ProviderConfig
from ..errors import UnknownInstance
from ..instance import CpuPlacement, Instance

if TYPE_CHECKING:
    from ..env import Env
    from ..runtime import Runtime
    from ..store.volume import Volume

Persistence = Literal["persistent", "ephemeral"]


class Provider(abc.ABC):
    """Base class for every provider."""

    #: Value of the ``kind`` field that selects this class.
    kind: str = ""

    #: Extra name to install when the optional dependency is missing.
    extra: str | None = None

    #: Default when the configuration does not declare ``persistent``.
    default_persistence: Persistence = "ephemeral"

    #: Whether CUDA call forwarding can reach this provider at useful latency.
    has_fast_path: bool = False

    #: Whether a runtime has to install the declared environment. False when the
    #: machine already runs in it, which is the case for the local provider.
    prepares_env: bool = True

    #: Whether a runtime needs the heartbeat lease. The lease exists so a session
    #: that bills by the second cannot outlive the process that started it, so a
    #: provider that charges nothing and dies with the call does not need one.
    needs_lease: bool = True

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.alias = config.alias
        self._instances: dict[str, Instance] | None = None
        self._volumes: dict[str, Volume] = {}

    # -- identity ------------------------------------------------------------

    @property
    def persistence(self) -> Persistence:
        declared = self.config.option("persistent")
        if declared is True:
            return "persistent"
        if declared is False:
            return "ephemeral"
        return self.default_persistence

    @property
    def persistent(self) -> bool:
        return self.persistence == "persistent"

    @property
    def default_cpu_placement(self) -> CpuPlacement:
        """Where the Python side runs when the instance does not say.

        Storage decides first. With storage that outlives the runtime, or with a
        volume attached to stand in for it, the loop is shipped. Without either,
        state has to stay on this machine, which means forwarding CUDA calls, and
        that is only offered where a fast path exists.
        """
        if self.persistent or self._volumes:
            return "remote"
        return "local" if self.has_fast_path else "remote"

    # -- instances -----------------------------------------------------------

    @abc.abstractmethod
    def discover(self) -> Mapping[str, Instance]:
        """Return the accelerators this account can offer, keyed by name.

        Called once, lazily, on first access. A provider that has to connect to
        find out should cache the answer rather than connecting at import time.
        """

    @property
    def instances(self) -> Mapping[str, Instance]:
        if self._instances is None:
            self._instances = dict(self.discover())
        return self._instances

    def refresh(self) -> Mapping[str, Instance]:
        """Discard the cached instance list and ask again."""
        self._instances = None
        return self.instances

    def __getattr__(self, name: str) -> Instance:
        # Only reached for names that are not real attributes, so accelerator
        # names such as ``colab.G4`` land here.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            table = self.instances
        except Exception as exc:  # discovery failure should not look like a typo
            raise AttributeError(f"{self.alias}.{name} is unavailable: {exc}") from exc
        if name in table:
            return table[name]
        upper = name.upper()
        if upper in table:
            return table[upper]
        raise UnknownInstance(
            f"{self.alias} does not offer {name!r}. Available: {', '.join(sorted(table))}"
        )

    def gpu(self, name: str, **overrides: object) -> Instance:
        """Look up an accelerator by name, with optional placement overrides."""
        instance = getattr(self, name)
        return instance(**overrides) if overrides else instance

    def __dir__(self) -> Iterator[str]:  # type: ignore[override]
        yield from super().__dir__()
        try:
            yield from self.instances
        except Exception:
            return

    # -- storage -------------------------------------------------------------

    def volume(self, name: str, **options: object) -> Volume:
        """Return the named volume on this provider, creating the binding once.

        A volume is a content addressed blob store on whichever backend this
        provider has. Attaching one to an ephemeral provider is what makes the
        environment and the model cache survive between runtimes.
        """
        from ..store.volume import Volume

        if name not in self._volumes:
            self._volumes[name] = Volume(provider=self, name=name, options=dict(options))
        return self._volumes[name]

    @abc.abstractmethod
    def store_backend(self) -> str:
        """Name of the blob store backend this provider uses."""

    # -- runtimes ------------------------------------------------------------

    @abc.abstractmethod
    def start(self, instance: Instance, env: Env, *, name: str) -> Runtime:
        """Bring up a session and return the live runtime.

        This is where the provider actually costs money. Everything before it is
        declaration.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.alias} ({self.persistence})>"
