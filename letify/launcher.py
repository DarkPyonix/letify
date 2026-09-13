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

Keeping a session alive between calls is declared, not commanded. A declaration made
with ``warm=True`` holds its runtime after the call returns, because starting a session
costs provider boot plus environment installation, which is minutes on Colab and worth
avoiding across a run of separate calls. Two warm declarations on the same device and
environment share one session, since the pool keys by those rather than by which
function asked.

Nothing is released by hand. A call releases its own runtime, a warm one is torn down by
the idle reaper once it stops being used, and everything goes at process exit.

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
from .declare.instance import AnyInstance, Host, Instance
from .declare.sweep import grid, zip_
from .errors import UnknownInstance, UnknownProvider
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
        volumes: Sequence[Volume] = (),
        concurrency: int = 1,
        timeout: float | None = 3600,
        retries: int = 1,
        warm: bool = False,
        keep_remote: bool = False,
    ) -> Callable[[Callable[..., R]], Function[R]]:
        """Declare where a function runs.

        ``device`` carries the provider, the account and the accelerator, and the core
        count and memory come with it rather than being asked for. ``cpu`` picks the
        execution mode: ``"local"``, the default, keeps Python here and forwards only
        CUDA calls, and ``"remote"`` ships this function to the machine with the GPU.
        ``concurrency`` is how many runtimes this declaration may use at once.

        ``warm`` keeps this declaration's runtime alive after a call returns, so the
        next call skips session start. The cost is that an unused session keeps
        billing until the idle timeout, which is why it is off by default.

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
                volumes=volumes,
                concurrency=concurrency,
                timeout=timeout,
                retries=retries,
                warm=warm,
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

    def shutdown(self) -> list[str]:
        """Stop everything, including runtimes still running a call."""
        return self.pool.shutdown()

    def runtime(
        self,
        instance: Instance | AnyInstance,
        env: Env | None = None,
        *,
        volumes: Sequence[Volume] = (),
    ) -> Any:
        """Start one runtime now instead of on the first call.

        Useful when a session takes a while to come up and there is local work to do
        meanwhile, such as preparing data. It is marked warm, so it survives until the
        idle timeout or process exit.
        """
        self._register_at_exit()
        return self.pool.acquire(self.resolve(instance), env or Env(), volumes, warm=True)

    def reap_idle(self) -> list[str]:
        """Shut down runtimes idle past the timeout, without waiting for the reaper."""
        return self.pool.reap_idle()

    # -- reporting -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """What is running right now, and what it is costing."""
        return {
            "name": self.name,
            "holding": self.pool.holding,
            "max_runtimes": self.max_runtimes,
            "runtimes": [
                {
                    "name": runtime.name,
                    "provider": runtime.provider.alias,
                    "accelerator": runtime.instance.accelerator,
                    "placement": str(runtime.instance.placement),
                    "busy": runtime.busy,
                    "persistent_channel": runtime.persistent_channel,
                    "idle_seconds": round(runtime.idle_for, 1),
                }
                for runtime in self.pool.live
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
            f"{instance.accelerator} (cpu={instance.placement})",
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
