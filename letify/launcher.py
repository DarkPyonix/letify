"""Launcher, the public entry point.

It loads the configuration, hands out providers, accepts declarations and owns the
pool. It is the thing that does the letting, so the conventional variable name is
``let``:

    let = letify.Launcher()

    @let.function(device=colab.G4)
    def train(lr, bs): ...

    train(lr=1e-4, bs=32)

There is no scope to open. A runtime lives for the invocation that needed it and is
released when that finishes, which is why a call needs no ceremony around it.

How long a session lives is declared, not commanded. ``lifetime="call"`` is the default
and ends the session with the call. ``lifetime="process"`` keeps it, because starting one
costs provider boot plus environment installation, which is minutes on Colab and worth
avoiding across a run of separate calls. Two declarations that agree on device and
environment share a session either way, since the pool keys by those rather than by which
function asked.

Nothing is torn down by hand. A call ends its own session, one left over is reaped once it
has been idle past the timeout, and everything goes at process exit.

Providers are reached by attribute on ``let.providers``. Three names there are
reserved: ``any`` for a request that does not name a provider, ``devices`` for the
registered accelerators, and ``active`` for the providers currently holding a
session, which is the quickest answer to what is costing money.
"""

from __future__ import annotations

import atexit
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from . import providers as provider_registry
from .config import Config, load
from .declare.env import Env
from .declare.function import Function
from .declare.instance import AnyInstance, Host, Instance, Lifetime
from .declare.sweep import grid, zip_
from .errors import LetifyError, UnknownInstance, UnknownProvider
from .runtime.pool import DEFAULT_IDLE_TIMEOUT, RuntimePool

if TYPE_CHECKING:
    from .providers.base import Provider
    from .store.volume import Volume

R = TypeVar("R")


class Providers:
    """Attribute access over the providers the configuration declared."""

    def __init__(self, launcher: Launcher):
        self._launcher = launcher

    def __getattr__(self, alias: str) -> Provider:
        if alias.startswith("_"):
            raise AttributeError(alias)
        if alias == "any":
            return AnyProvider()  # type: ignore[return-value]
        if alias == "devices":
            return self.devices  # type: ignore[return-value]
        if alias == "active":
            return self.active  # type: ignore[return-value]
        return self._launcher.provider(alias)

    def __getitem__(self, alias: str) -> Provider:
        return self._launcher.provider(alias)

    def __iter__(self) -> Iterator[Provider]:
        for alias in self._launcher.config.order:
            yield self._launcher.provider(alias)

    def __dir__(self) -> list[str]:
        return [*self._launcher.config.order, "any", "devices", "active"]

    @property
    def aliases(self) -> list[str]:
        return list(self._launcher.config.order)

    @property
    def devices(self) -> dict[str, list[str]]:
        """Accelerators each provider offers, keyed by alias.

        A provider that cannot be reached reports its reason instead of raising, so
        one broken entry does not hide the rest.
        """
        table: dict[str, list[str]] = {}
        for alias in self._launcher.config.order:
            try:
                table[alias] = sorted(self._launcher.provider(alias).instances)
            except Exception as exc:
                table[alias] = [f"unavailable: {exc}"]
        return table

    @property
    def active(self) -> dict[str, list[str]]:
        """Providers with a live runtime, and the runtime names."""
        table: dict[str, list[str]] = {}
        for runtime in self._launcher.pool.live:
            table.setdefault(runtime.provider.alias, []).append(runtime.name)
        return table

    def __repr__(self) -> str:
        return f"<Providers {', '.join(self._launcher.config.order)}>"


class AnyProvider:
    """Stands in for a provider when the declaration does not name one."""

    alias = "any"

    def __getattr__(self, name: str) -> AnyInstance:
        if name.startswith("_"):
            raise AttributeError(name)
        return AnyInstance(accelerator=name)

    def __repr__(self) -> str:
        return "<Providers.any>"


class Launcher:
    """Loads the configuration and turns declarations into running work."""

    def __init__(
        self,
        config: str | Path | None = None,
        *,
        name: str | None = None,
        max_runtimes: int = 3,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        stream_logs: bool = True,
        announce: bool = True,
        home: bool = True,
    ):
        self.config: Config = load(config, home=home)
        self.name = name or self.config.defaults.get("name") or _project_name()
        self.max_runtimes = int(self.config.defaults.get("max_runtimes", max_runtimes))
        self.idle_timeout = float(self.config.defaults.get("idle_timeout", idle_timeout))
        self.stream_logs = stream_logs
        self.announce = announce
        self.providers = Providers(self)
        self.functions: list[Function] = []
        self.pool = RuntimePool(
            max_runtimes=self.max_runtimes,
            idle_timeout=self.idle_timeout,
            on_start=self._announce_start,
        )
        self._cache: dict[str, Provider] = {}
        self._guard = threading.Lock()
        self._at_exit_registered = False

    # -- providers -----------------------------------------------------------

    def provider(self, alias: str) -> Provider:
        """Return the provider declared under this alias, building it once."""
        if alias in self._cache:
            return self._cache[alias]
        entry = self.config.providers.get(alias)
        if entry is None:
            known = ", ".join(self.config.order) or "none"
            raise UnknownProvider(
                f"no provider is declared under {alias!r}. Declared: {known}. "
                f"Add it to .letify, or to ~/.letify if it carries credentials."
            )
        provider = provider_registry.build(entry)
        self._cache[alias] = provider
        return provider

    def resolve(self, request: Instance | AnyInstance) -> Instance:
        """Turn a declaration into a concrete instance.

        An ``AnyInstance`` picks the first declared provider that registered a
        matching accelerator, so priority is the order in the configuration file.
        """
        if isinstance(request, Instance):
            return request
        for alias in self.config.order:
            try:
                instance = self.provider(alias).device(request.accelerator)
            except Exception:
                continue
            instance = instance.on_host(request.host)
            return instance
        raise UnknownInstance(
            f"no declared provider offers {request.accelerator!r}. Checked: "
            f"{', '.join(self.config.order)}"
        )

    # -- declaration ---------------------------------------------------------

    def function(
        self,
        *,
        device: Instance | AnyInstance,
        env: Env | None = None,
        host: Host | str | None = None,
        lifetime: Lifetime | str | None = None,
        volumes: Sequence[Volume] = (),
        concurrency: int = 1,
        timeout: float | None = 3600,
        retries: int = 1,
        keep_remote: bool = False,
    ) -> Callable[[Callable[..., R]], Function[R]]:
        """Declare where a function runs.

        ``device`` carries the provider, the account and the accelerator, and the core
        count and memory come with it rather than being asked for. ``host`` says where the
        host code runs: ``"local"``, the default, keeps Python here and forwards only CUDA
        calls, and ``"remote"`` ships this function to the machine with the GPU.
        ``concurrency`` is how many runtimes this declaration may use at once.

        ``lifetime`` says how long the session lives. ``"call"``, the default, ends it
        with the call. ``"process"`` keeps it so the next call skips session start, at the
        cost of an unused session billing until the idle reaper takes it.

        ``keep_remote`` returns a handle instead of the value, so a model stays in the
        runtime and later calls refer to it without copying it back.
        """

        def decorate(fn: Callable[..., R]) -> Function[R]:
            declared = Function(
                fn,
                self,
                device=device,
                env=env or Env(),
                host=host,
                lifetime=lifetime,
                volumes=volumes,
                concurrency=concurrency,
                timeout=timeout,
                retries=retries,
                keep_remote=keep_remote,
            )
            self.functions.append(declared)
            return declared

        return decorate

    # -- search spaces -------------------------------------------------------

    grid = staticmethod(grid)
    zip = staticmethod(zip_)

    # -- lifetime ------------------------------------------------------------

    @contextmanager
    def invocation(self) -> Iterator[Launcher]:
        """Bracket one top-level call, so its runtimes are released at the end.

        Used by the declaration machinery rather than by hand. A sweep is one
        invocation, so its runtimes start once and stop once.
        """
        self._register_at_exit()
        self.pool.hold()
        try:
            yield self
        finally:
            self.pool.unhold()

    def runtime(
        self,
        instance: Instance | AnyInstance,
        env: Env | None = None,
        *,
        volumes: Sequence[Volume] = (),
    ) -> Any:
        """Start one runtime now instead of on the first call.

        Useful when a session takes a while to come up and there is local work to do
        meanwhile, such as preparing data. It lives for the process, so the idle reaper or
        process exit is what ends it.

        The session is handed straight back, because the point is for the next call to
        find it warm. Holding it would make the pool skip it and start a second one,
        which is the opposite of what starting early is for.
        """
        self._register_at_exit()
        runtime = self.pool.acquire(
            self.resolve(instance), env or Env(), volumes, lifetime=Lifetime.process
        )
        self.pool.release(runtime)
        return runtime

    def reap_idle(self) -> list[str]:
        """Shut down runtimes idle past the timeout, without waiting for the reaper."""
        return self.pool.reap_idle()

    # -- reporting -----------------------------------------------------------

    def usage(self, alias: str | None = None) -> list[dict[str, Any]]:
        """What each account has left, or why the figure is not available.

        Every declared alias is listed, including one whose provider could not even be
        built, because an account missing from a table reads as an account with nothing
        left on it.
        """
        wanted = [alias] if alias else list(self.config.order)
        rows: list[dict[str, Any]] = []
        for name in wanted:
            try:
                provider = self.provider(name)
            except LetifyError as exc:
                rows.append({"alias": name, "unavailable": str(exc)})
                continue
            rows.append(provider.usage().to_dict())
        return rows

    def utilization(self, alias: str | None = None) -> list[dict[str, Any]]:
        """How hard each declared instance's accelerator is working right now.

        A local instance is read here. A remote one is read inside its live session, and
        an instance with no session reports no devices and says so, because starting one
        to measure its load would cost money and change the answer.
        """
        from .runtime.telemetry import parse_smi, read_smi

        wanted = [alias] if alias else list(self.config.order)
        live = {
            (runtime.provider.alias, runtime.instance.accelerator): runtime
            for runtime in self.pool.live
        }
        rows: list[dict[str, Any]] = []
        for name in wanted:
            try:
                provider = self.provider(name)
                instances = provider.instances
            except LetifyError as exc:
                rows.append({"alias": name, "unavailable": str(exc)})
                continue

            for accelerator, instance in instances.items():
                if instance.gpu is None:
                    continue
                row: dict[str, Any] = {
                    "alias": name,
                    "accelerator": accelerator,
                    "devices": [],
                    "reason": None,
                }
                runtime = live.get((name, accelerator))
                try:
                    if provider.kind == "local":
                        output = read_smi()
                    elif runtime is not None:
                        output, _logs = runtime.call(read_smi, (), {})
                    else:
                        row["reason"] = "no live session, so nothing to measure"
                        rows.append(row)
                        continue
                except LetifyError as exc:
                    row["reason"] = str(exc)
                    rows.append(row)
                    continue

                devices = parse_smi(output or "")
                row["devices"] = [device.to_dict() for device in devices]
                if not devices:
                    row["reason"] = "nvidia-smi reported nothing on that machine"
                rows.append(row)
        return rows

    def status(self) -> dict[str, Any]:
        """What is running right now, and what it is costing.

        Counts first, so a reader can see whether the ceiling is why a call is waiting.
        The pool's own bookkeeping is not here: whether the invocation guard is open is a
        fact about the pool rather than about what is running, and a boolean next to
        ``max_runtimes`` gets read as a count.

        This process only, because the pool lives in the process that owns it. What a
        machine itself is doing is what ``letify utilization`` answers.
        """
        live = list(self.pool.live)
        return {
            "name": self.name,
            "live": len(live),
            "busy": sum(1 for runtime in live if runtime.busy),
            "max_runtimes": self.max_runtimes,
            "runtimes": [
                {
                    "name": runtime.name,
                    "provider": runtime.provider.alias,
                    "accelerator": runtime.instance.accelerator,
                    "placement": str(runtime.instance.placement),
                    "busy": runtime.busy,
                    "lifetime": str(runtime.lifetime),
                    "persistent_channel": runtime.persistent_channel,
                    "idle_seconds": round(runtime.idle_for, 1),
                }
                for runtime in live
            ],
            "declared": [f.__name__ for f in self.functions],
            "config_sources": [str(p) for p in self.config.sources],
        }

    # -- internals -----------------------------------------------------------

    def _announce_start(self, instance: Instance, name: str) -> None:
        """Say when a session starts, because that is when money starts."""
        if not self.announce:
            return
        import sys

        print(
            f"letify: starting {name} on {instance.provider.alias} "
            f"{instance.accelerator} (host={instance.placement})",
            file=sys.stderr,
        )

    def _register_at_exit(self) -> None:
        """Make sure nothing survives this process.

        The lease already covers a crash, and this covers an ordinary exit that
        happens while a hold is still open.
        """
        with self._guard:
            if self._at_exit_registered:
                return
            self._at_exit_registered = True
        atexit.register(self.pool.shutdown)

    def __repr__(self) -> str:
        return f"<Launcher {self.name} providers={len(self.config.providers)}>"


def _project_name() -> str:
    """Read the project name from pyproject.toml, falling back to the directory.

    A name is needed because Modal wants an application name and Colab wants a
    session prefix. Taking it from the project means the user does not repeat it.
    """
    path = Path.cwd() / "pyproject.toml"
    if path.is_file():
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        name = data.get("project", {}).get("name")
        if isinstance(name, str) and name:
            return name
    return Path.cwd().name


__all__ = ["AnyProvider", "Launcher", "Providers"]
