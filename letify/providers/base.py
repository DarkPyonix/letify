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
import dataclasses
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from ..config import ProviderConfig
from ..config.inventory import Devices, read_table
from ..declare.instance import Host, Instance
from ..errors import ConfigError, LetifyError, UnknownInstance, UnsupportedMode
from .usage import Usage, from_command

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
        self._inventory: dict[str, Devices] | None = None
        self._reserved: dict[str, list[int]] = {}
        self._taken: dict[str, int] = {}
        self._devices_guard = threading.RLock()
        #: Process ids of the workers this client process started here. Their compute
        #: processes are this client's own and do not make a card busy.
        self._worker_pids: set[int] = set()
        #: The busy indices the most recent reservation read, for the refusal message.
        self.last_busy: tuple[int, ...] = ()
        #: The users owning the processes on each index of ``last_busy``, for the same message.
        self.last_busy_owners: dict[int, tuple[str, ...]] = {}

    # -- inventory -----------------------------------------------------------

    @property
    def inventory(self) -> Mapping[str, Devices]:
        """What this account has, read from the entry's device table.

        This is the only thing that bounds how much runs at once. An entry that declares
        nothing falls back to whatever the provider discovers, one of each, because a
        provider that can be asked should not have to be told.
        """
        if self._inventory is None:
            declared = read_table(self.config.options)
            if not declared:
                # Keyed by what an instance calls itself, so a discovered shape and its
                # inventory entry are the same name. A CPU shape reports "cpu".
                declared = {
                    instance.accelerator: Devices(instance.accelerator)
                    for instance in self.discover().values()
                }
            self._inventory = declared
        return self._inventory

    def devices_of(self, accelerator: str) -> Devices:
        """The inventory entry for one accelerator, matched the way attribute access is.

        A device table is keyed as the user wrote it and an instance reports the provider's
        spelling, so the two are compared without case, the same way ``colab.g4`` finds
        ``G4``. An accelerator with no entry at all is one of it: a shape the provider
        registered is a shape it has.
        """
        table = self.inventory
        if accelerator in table:
            return table[accelerator]
        folded = accelerator.casefold()
        for name, entry in table.items():
            if name.casefold() == folded:
                return entry
        if accelerator in {instance.accelerator for instance in self.instances.values()}:
            return Devices(accelerator)
        raise UnknownInstance(
            f"{self.alias} has no {accelerator!r} in its inventory. It has: "
            f"{', '.join(sorted(table)) or 'nothing'}"
        )

    def free(self, accelerator: str) -> tuple[int, ...]:
        """Which registered indices could be reserved right now.

        Registered is permission, not availability: an index another process is computing
        on is skipped rather than fought over, and the answer is read fresh because it
        changes while a run is queued.
        """
        entry = self.devices_of(accelerator)
        if not entry.chooses_indices:
            return ()
        with self._devices_guard:
            held = set(self._reserved.get(entry.accelerator, ()))
        busy = tuple(self.busy())
        self.last_busy = busy
        return tuple(index for index in entry.indices if index not in held | set(busy))

    def busy(self) -> tuple[int, ...]:
        """Device indices another user is computing on. Nothing for a provider letify cannot ask.

        Overridden where the machine can be asked. The base answer is nothing, which is
        right for a provider that assigns the device itself.
        """
        return ()

    def add_worker_pid(self, pid: int) -> None:
        """Record a worker this client process started here, so its card is not read as busy."""
        with self._devices_guard:
            self._worker_pids.add(pid)

    def remove_worker_pid(self, pid: int) -> None:
        """Forget a worker that has shut down."""
        with self._devices_guard:
            self._worker_pids.discard(pid)

    def worker_pids(self) -> set[int]:
        """Process ids of the live workers this client process started here."""
        with self._devices_guard:
            return set(self._worker_pids)

    def reserve(self, instance: Instance) -> tuple[int, ...] | None:
        """Take the devices this instance asks for, or None when they are not there.

        Returns the indices taken, which is empty for a provider that assigns the device
        itself: there the only question is whether the account has a slot left. Half the
        cards a run asked for is not a smaller version of the run, so a request that cannot
        be met in full takes nothing.
        """
        entry = self.devices_of(instance.accelerator)
        wanted = instance.devices
        with self._devices_guard:
            if entry.chooses_indices:
                available = self.free(instance.accelerator)
                if len(available) < wanted:
                    return None
                taken = tuple(available[:wanted])
                self._reserved.setdefault(entry.accelerator, []).extend(taken)
                return taken

            used = self._taken.get(entry.accelerator, 0)
            if used + wanted > entry.count:
                return None
            self._taken[entry.accelerator] = used + wanted
            return ()

    def unreserve(self, accelerator: str, indices: tuple[int, ...], devices: int = 1) -> None:
        """Give devices back, whether they were indices or a count.

        ``devices`` is how many were taken, needed only where the provider assigns the
        device itself and there are no indices to hand back.
        """
        try:
            entry = self.devices_of(accelerator)
        except UnknownInstance:
            entry = Devices(accelerator)
        with self._devices_guard:
            if indices:
                held = self._reserved.get(entry.accelerator, [])
                for index in indices:
                    if index in held:
                        held.remove(index)
                return
            name = entry.accelerator
            self._taken[name] = max(0, self._taken.get(name, 0) - devices)

    def reserved_count(self, accelerator: str) -> int:
        """How much of one accelerator is currently held."""
        with self._devices_guard:
            return len(self._reserved.get(accelerator, ())) or self._taken.get(accelerator, 0)

    def capacity(self, accelerator: str) -> int:
        """How many sessions of one device each this account can hold at once."""
        try:
            return self.devices_of(accelerator).count
        except UnknownInstance:
            return 0

    def visible_devices(self, indices: tuple[int, ...]) -> str | None:
        """What to set CUDA_VISIBLE_DEVICES to, so the session sees its cards as 0 upward."""
        return ",".join(str(index) for index in indices) if indices else None

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
        self._inventory = None
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

    #: The workspace root when the account sets no ``workspace``. None means the default
    #: per-user directory, which needs no elevated rights.
    default_workspace: str | None = None

    #: Whether a session boot expands, creates and enters the workspace root on the runtime.
    prepares_workspace: bool = True

    @property
    def workspace_root(self) -> str:
        """Where letify may write on the runtime, before ``~`` is expanded there.

        Every remote path letify introduces, such as the project directory a sync runs in,
        derives from this one value: the account's ``workspace``, or this kind's default.
        """
        from ..runtime import bootstrap

        value = self.config.option("workspace")
        if value is None:
            return self.default_workspace or bootstrap.DEFAULT_WORKSPACE_ROOT
        if not isinstance(value, str) or not value.startswith(("/", "~")):
            raise ConfigError(
                f"{self.alias}: workspace must be an absolute path or start with ~, not {value!r}"
            )
        return value

    @property
    def managed_python(self) -> str | None:
        """The interpreter the account names with ``python``, which the user manages."""
        value = self.config.option("python")
        return str(value) if value else None

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
        held: tuple[int, ...] = (),
    ) -> Runtime:
        """Bring a session up and return the live runtime.

        This is where the provider starts costing money. Everything before it is
        declaration.
        """
        from ..runtime.session import Runtime

        self.check_mode(instance)
        if self.prepares_env and not self.managed_python:
            from ..runtime.bootstrap import project_files

            # A missing lock file or a Python that cannot match is refused before the
            # provider allocates anything.
            project_files(env)
        self.create_session(instance, name)
        runtime = Runtime(
            name=name,
            provider=self,
            instance=instance,
            env=env,
            volumes=tuple(volumes),
            held_devices=held,
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
        from ..remoting.device.guard import require_torch

        require_torch()

    def device_channel(self, runtime: Runtime) -> Channel:
        """The persistent channel whose call worker hosts the PyTorch device executor.

        A provider whose channel cannot carry the device stream cannot serve
        ``host="local"``, and says so here.
        """
        raise UnsupportedMode(
            f"{self.kind} cannot start a PyTorch device worker, so host='local' cannot run "
            f"on {self.alias}. Use host='remote'."
        )

    #: Whether forwarding needs the agent installed on another machine. False only
    #: where the device is in this machine, so nothing has to be reached.
    needs_remote_agent: bool = True

    def forwarding_host(self) -> str | None:
        """The name to measure the round trip against, if this provider has one.

        A provider that has no address, or whose address depends on configuration it does
        not need, reports None. Reading the attribute directly would let an unrelated
        configuration error surface as a refusal to forward, which says the wrong thing
        about why the mode is unavailable.
        """
        try:
            return getattr(self, "address", None)
        except LetifyError:
            return None

    def warn_slow_forwarding(self) -> None:
        """Report the expected efficiency of forwarding over a long link."""
        import warnings

        round_trip = self.expected_round_trip_ms
        detail = f"about {round_trip:.0f} ms" if round_trip else "long enough to matter"
        warnings.warn(
            f"{self.alias} has a round trip of {detail}, so host='local' pays that once "
            f"per host synchronization. Expect roughly half the throughput of a direct "
            f"run for fine-tuning and a handful of tokens per second for decoding. "
            f"host='remote' avoids it by running the loop on the machine.",
            stacklevel=3,
        )

    # -- remaining usage -----------------------------------------------------

    #: What the account is metered in, where the provider knows. Overridden per provider.
    usage_unit: str = "hours"

    #: Why this provider cannot report a balance of its own. A provider that can report
    #: one overrides ``usage()`` instead.
    usage_source: str = "this provider publishes no balance"

    def usage(self) -> Usage:
        """Report what is left on this account.

        A configured command wins, because a user who wired one up knows something letify
        does not. Otherwise the provider answers for itself, and a provider with no source
        says so rather than returning a number nobody measured.
        """
        command = self.config.option("usage_command")
        if isinstance(command, str) and command:
            limit = self.config.option("usage_limit")
            unit = self.config.option("usage_unit")
            return from_command(
                self.alias,
                self.kind,
                command,
                str(unit) if isinstance(unit, str) else self.usage_unit,
                float(limit) if isinstance(limit, (int, float)) else None,
            )
        reading = self.report_usage()
        plan = self.config.option("usage_limit")
        if reading.limit is None and isinstance(plan, (int, float)) and plan > 0:
            # A plan allowance the user configured, for a service that states only a balance.
            used = None if reading.remaining is None else max(float(plan) - reading.remaining, 0.0)
            reading = dataclasses.replace(reading, limit=float(plan), used=used)
        return reading

    def report_usage(self) -> Usage:
        """What the provider itself can answer, with no configured command in the way."""
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.alias} ({self.persistence})>"


__all__ = ["Persistence", "Provider"]
