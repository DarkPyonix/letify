"""Provider, one account on one kind of infrastructure.

A provider object knows which accelerators an account can offer, owns the storage
that outlives a session, opens the channel to a machine, and starts and stops the
session itself.

Four class attributes set every default, and none of them is a user-facing switch.

``persistence`` says whether storage outlives a runtime, which decides whether an
environment and a dataset have to be rebuilt for every session.

``has_fast_path`` says whether the machine is close enough for CUDA call forwarding to
pay off. It does not gate the mode: a declaration that asks for forwarding over a long
link gets a warning with the arithmetic and then runs, because the choice belongs to
whoever wrote the declaration. A mode is refused only when the provider cannot serve
it at all.

``persistent_channel`` says whether a worker process can be kept alive behind a
pipe. Without one there is no object table, so a handle has nothing to point at,
and no blob table, so a large argument travels again on every call.

``needs_lease`` says whether a session can outlive this process and keep billing.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from ..config import ProviderConfig
from ..declare.instance import Host, Instance
from ..errors import UnknownInstance

if TYPE_CHECKING:
    from ..declare.env import Env
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime
    from ..store.volume import Volume

Persistence = Literal["persistent", "ephemeral"]


class Provider(abc.ABC):
    """Base class for every provider."""

    #: Value of a configuration entry's ``kind`` field that selects this class.
    kind: str = ""

    #: Extra to install when the optional dependency is missing.
    extra: str | None = None

    #: Default when the configuration does not declare ``persistent``.
    default_persistence: Persistence = "ephemeral"

    #: Whether CUDA call forwarding can reach this provider at useful latency.
    has_fast_path: bool = False

    #: Whether a worker process can be kept alive between calls.
    persistent_channel: bool = True

    #: Whether a runtime has to install the declared environment. False when the
    #: machine already runs in it, which is the case for the local provider.
    prepares_env: bool = True

    #: Whether a session can outlive this process and keep billing.
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

    # -- instances -----------------------------------------------------------

    @abc.abstractmethod
    def discover(self) -> Mapping[str, Instance]:
        """Return the accelerators this account can offer, keyed by name.

        Called once, lazily, on first access. A provider that has to connect to
        find out caches the answer rather than connecting at import time.
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
        # Only reached for names that are not real attributes, so accelerator names
        # such as ``colab.G4`` land here.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            table = self.instances
        except Exception as exc:  # a discovery failure should not look like a typo
            raise AttributeError(f"{self.alias}.{name} is unavailable: {exc}") from exc
        if name in table:
            return table[name]
        if name.upper() in table:
            return table[name.upper()]
        raise UnknownInstance(
            f"{self.alias} does not offer {name!r}. Available: {', '.join(sorted(table))}"
        )

    def device(self, name: str) -> Instance:
        """Look up an accelerator by name."""
        return getattr(self, name)

    def __dir__(self) -> Iterator[str]:  # type: ignore[override]
        yield from super().__dir__()
        try:
            yield from self.instances
        except Exception:
            return

    # -- storage -------------------------------------------------------------

    def volume(self, name: str, **options: object) -> Volume:
        """Return the named volume on this provider, binding it once.

        Attaching a volume to an ephemeral provider is what makes the environment
        and the model cache survive between runtimes.
        """
        from ..store.volume import Volume

        if name not in self._volumes:
            self._volumes[name] = Volume(provider=self, name=name, options=dict(options))
        return self._volumes[name]

    @abc.abstractmethod
    def store_backend(self) -> str:
        """Name of the blob store backend this provider uses."""

    # -- sessions ------------------------------------------------------------

    @abc.abstractmethod
    def open_channel(self, runtime: Runtime) -> Channel:
        """Return the channel that talks to this runtime."""

    def create_session(self, instance: Instance, name: str) -> None:
        """Ask the provider for a machine. Nothing to do where one already exists."""
        return None

    def stop(self, runtime: Runtime) -> None:
        """Release whatever the provider allocated for this runtime."""
        return None

    def start(
        self,
        instance: Instance,
        env: Env,
        *,
        name: str,
        volumes: Sequence[Volume] = (),
    ) -> Runtime:
        """Bring a session up and return the live runtime.

        This is where the provider starts costing money. Everything before it is
        declaration.
        """
        from ..runtime.session import Runtime

        self.check_mode(instance)
        self.create_session(instance, name)
        runtime = Runtime(
            name=name,
            provider=self,
            instance=instance,
            env=env,
            volumes=tuple(volumes),
        )
        runtime.boot()
        return runtime

    #: What forwarding would cost here, in milliseconds of round trip. Set where it
    #: has been measured, so the warning can carry a real number.
    expected_round_trip_ms: float | None = None

    def check_mode(self, instance: Instance) -> None:
        """Say what a mode will cost here, and refuse only what cannot work.

        letify never changes the mode on its own, because a four times slowdown with
        no explanation costs more than a warning. It also does not overrule the
        declaration: a slow path that the author asked for still runs.
        """
        if instance.placement is not Host.local:
            return
        if not self.has_fast_path:
            self.warn_slow_forwarding()
        from ..remoting import require

        require(getattr(self, "address", None))

    def warn_slow_forwarding(self) -> None:
        """Report the expected efficiency of forwarding over a long link."""
        import warnings

        round_trip = self.expected_round_trip_ms
        detail = f"about {round_trip:.0f} ms" if round_trip else "long enough to matter"
        warnings.warn(
            f"{self.alias} has a round trip of {detail}, so cpu='local' pays that once "
            f"per host synchronization. Expect roughly half the throughput of a direct "
            f"run for fine-tuning and a handful of tokens per second for decoding. "
            f"cpu='remote' avoids it by running the loop on the machine.",
            stacklevel=3,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.alias} ({self.persistence})>"


__all__ = ["Persistence", "Provider"]
