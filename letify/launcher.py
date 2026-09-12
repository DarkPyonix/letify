"""Launcher, the public entry point.

The Launcher loads the configuration, hands out providers, takes function
declarations and owns the runtime pool. It is the thing that does the letting, so
the conventional variable name for it is ``let``:

    let = letify.Launcher()

    @let.function(gpu=colab.G4, env=env)
    def train(lr, bs): ...

    with let.run():
        train(lr=1e-4, bs=32)

Providers are reached by attribute on ``let.providers``, which returns a
``Provider``. Three names there are reserved: ``any`` for a request that does not
name a provider, ``gpus`` for the registered accelerator list, and ``active`` for
the providers that currently have a session running, which is the fastest way to
see what is costing money.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from . import providers as provider_registry
from .config import Config, load
from .env import Env
from .errors import UnknownInstance, UnknownProvider
from .function import Function
from .instance import AnyInstance, Instance
from .runtime import DEFAULT_IDLE_TIMEOUT, RuntimePool
from .sweep import Sweep, grid, zip_

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
            return _AnyProvider()  # type: ignore[return-value]
        if alias == "gpus":
            return self.gpus  # type: ignore[return-value]
        if alias == "active":
            return self.active  # type: ignore[return-value]
        return self._launcher.provider(alias)

    def __getitem__(self, alias: str) -> Provider:
        return self._launcher.provider(alias)

    def __iter__(self) -> Iterator[Provider]:
        for alias in self._launcher.config.order:
            yield self._launcher.provider(alias)

    def __dir__(self) -> list[str]:
        return [*self._launcher.config.order, "any", "gpus", "active"]

    @property
    def aliases(self) -> list[str]:
        return list(self._launcher.config.order)

    @property
    def gpus(self) -> dict[str, list[str]]:
        """Accelerators each provider offers, keyed by alias.

        A provider that cannot be reached reports its reason instead of raising,
        so one broken entry does not hide the rest.
        """
        table: dict[str, list[str]] = {}
        for alias in self._launcher.config.order:
            try:
                provider = self._launcher.provider(alias)
                table[alias] = sorted(provider.instances)
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


class _AnyProvider:
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
        home: bool = True,
    ):
        self.config: Config = load(config, home=home)
        self.name = name or self.config.defaults.get("name") or _project_name()
        self.max_runtimes = int(self.config.defaults.get("max_runtimes", max_runtimes))
        self.idle_timeout = float(self.config.defaults.get("idle_timeout", idle_timeout))
        self.stream_logs = stream_logs
        self.providers = Providers(self)
        self.functions: list[Function] = []
        self.pool = RuntimePool(max_runtimes=self.max_runtimes, idle_timeout=self.idle_timeout)
        self._cache: dict[str, Provider] = {}
        self._depth = 0
        self._guard = threading.Lock()

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
        matching accelerator, so priority is the order written in the
        configuration file.
        """
        if isinstance(request, Instance):
            return request
        for alias in self.config.order:
            try:
                provider = self.provider(alias)
                instance = provider.gpu(request.accelerator)
            except Exception:
                continue
            if request.cpu or request.cpus:
                instance = instance(cpu=request.cpu, cpus=request.cpus)
            return instance
        raise UnknownInstance(
            f"no declared provider offers {request.accelerator!r}. Checked: "
            f"{', '.join(self.config.order)}"
        )

    # -- declaration ---------------------------------------------------------

    def function(
        self,
        *,
        gpu: Instance | AnyInstance,
        env: Env | None = None,
        volumes: Sequence[Volume] = (),
        concurrency: int = 1,
        timeout: float | None = 3600,
        retries: int = 1,
        keep_remote: bool = False,
    ) -> Callable[[Callable[..., R]], Function[R]]:
        """Declare where a function runs.

        ``gpu`` carries the provider, the account, the accelerator and the CPU
        placement in one value, because they are one decision. ``concurrency`` is
        how many runtimes this declaration may use at once, which is a property of
        the declaration rather than of any single call.

        ``keep_remote`` returns a handle instead of the value, so a model stays in
        the runtime and later calls refer to it without copying it back.
        """

        def decorate(fn: Callable[..., R]) -> Function[R]:
            declared = Function(
                fn,
                self,
                gpu=gpu,
                env=env or Env(),
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

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._depth > 0

    @contextmanager
    def run(self) -> Iterator[Launcher]:
        """Open the scope in which sessions may exist.

        Leaving the scope shuts every runtime down. Nested scopes are allowed and
        only the outermost one tears down, so a helper can open a scope without
        ending its caller's session.
        """
        with self._guard:
            self._depth += 1
        try:
            yield self
        finally:
            with self._guard:
                self._depth -= 1
                closing = self._depth == 0
            if closing:
                self.pool.shutdown()

    def runtime(self, instance: Instance, env: Env | None = None, **kwargs: Any) -> Any:
        """Start one runtime now instead of waiting for the first call.

        Useful when the session takes a while to come up and there is local work
        to do meanwhile, such as preparing data.
        """
        resolved = self.resolve(instance)
        return self.pool.acquire(resolved, env or Env(), kwargs.get("volumes", ()))

    def reap_idle(self) -> list[str]:
        """Shut down runtimes that have been idle past the timeout."""
        return self.pool.reap_idle()

    def status(self) -> dict[str, Any]:
        """What is running right now, and what it is costing."""
        return {
            "name": self.name,
            "scope_open": self.is_running,
            "max_runtimes": self.max_runtimes,
            "runtimes": [
                {
                    "name": r.name,
                    "provider": r.provider.alias,
                    "accelerator": r.instance.accelerator,
                    "placement": r.instance.placement,
                    "idle_seconds": round(r.idle_for, 1),
                }
                for r in self.pool.live
            ],
            "declared": [f.__name__ for f in self.functions],
            "config_sources": [str(p) for p in self.config.sources],
        }

    def __repr__(self) -> str:
        return f"<Launcher {self.name} providers={len(self.config.providers)}>"


def _project_name() -> str:
    """Read the project name from pyproject.toml, falling back to the directory.

    A name is needed because Modal wants an application name and Colab wants a
    session prefix. Taking it from the project means the user does not have to
    repeat it.
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


__all__ = ["Launcher", "Providers", "Sweep"]
