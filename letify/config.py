"""Loading the ``.letify`` configuration.

Two files are read and merged. ``~/.letify`` holds accounts and credentials,
which belong to the machine, and the project's ``.letify`` holds defaults that
are safe to commit. The project file wins for defaults, the home file wins for
credentials, so a repository can be cloned by someone else and still run under
their own accounts.

Secrets are never read from the file itself. A field may name an environment
variable or an OS keyring entry, and the value is resolved at use time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib  # type: ignore[no-redef]

from .errors import ConfigError

CONFIG_NAME = ".letify"

#: Alias keys must be Python identifiers because providers are reached with
#: attribute access, as in ``let.providers.colab_a``.
RESERVED_ALIASES = frozenset({"any", "gpus", "active", "aliases", "kinds"})


@dataclass(slots=True)
class ProviderConfig:
    """One provider declaration, as written in the configuration file."""

    alias: str
    kind: str
    options: dict[str, Any] = field(default_factory=dict)
    order: int = 0

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)

    def secret(self, name: str, default: str | None = None) -> str | None:
        """Resolve a credential without ever storing it in a tracked file.

        Three forms are accepted, in this order of preference:
        ``<name>_env`` names an environment variable, ``<name>_keyring`` names a
        keyring entry as ``service/user``, and ``<name>`` is a literal value,
        which is only appropriate in the home configuration file.
        """
        env_key = self.options.get(f"{name}_env")
        if isinstance(env_key, str):
            value = os.environ.get(env_key)
            if value:
                return value

        keyring_key = self.options.get(f"{name}_keyring")
        if isinstance(keyring_key, str):
            value = _from_keyring(keyring_key)
            if value:
                return value

        literal = self.options.get(name)
        if isinstance(literal, str):
            return literal
        return default


@dataclass(slots=True)
class Config:
    """The merged configuration."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)

    @property
    def order(self) -> list[str]:
        """Aliases in declaration order, which sets ``any`` resolution priority."""
        return [c.alias for c in sorted(self.providers.values(), key=lambda c: c.order)]


def load(path: str | Path | None = None, *, home: bool = True) -> Config:
    """Read the configuration files and merge them.

    ``path`` overrides the project file. Set ``home`` to false to ignore
    ``~/.letify``, which tests use to stay isolated from the developer's accounts.
    """
    config = Config()
    files: list[Path] = []
    if home:
        files.append(Path.home() / CONFIG_NAME)
    files.append(Path(path) if path else Path.cwd() / CONFIG_NAME)

    counter = 0
    for file in files:
        if not file.is_file():
            continue
        config.sources.append(file)
        try:
            raw = tomllib.loads(file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{file} is not valid TOML: {exc}") from exc

        defaults = raw.pop("defaults", None)
        if isinstance(defaults, dict):
            config.defaults.update(defaults)

        for alias, body in raw.items():
            if not isinstance(body, dict):
                continue
            _check_alias(alias, file)
            kind = body.get("kind")
            if not isinstance(kind, str):
                raise ConfigError(f"{file}: provider {alias!r} has no 'kind' field")
            options = {k: v for k, v in body.items() if k != "kind"}
            existing = config.providers.get(alias)
            if existing is None:
                config.providers[alias] = ProviderConfig(alias, kind, options, counter)
                counter += 1
            else:
                # The project file refines what the home file declared.
                existing.kind = kind
                existing.options.update(options)

    # "local" is always available and needs no declaration.
    if "local" not in config.providers:
        config.providers["local"] = ProviderConfig("local", "local", {}, counter)
    return config


def _check_alias(alias: str, file: Path) -> None:
    if alias in RESERVED_ALIASES:
        raise ConfigError(
            f"{file}: {alias!r} is reserved. Pick another alias, because "
            f"let.providers.{alias} already means something else."
        )
    if not alias.isidentifier():
        hint = f" Try {alias.replace('-', '_')!r}." if "-" in alias else ""
        raise ConfigError(
            f"{file}: alias {alias!r} is not a Python identifier, so "
            f"let.providers.{alias} cannot work.{hint}"
        )


def _from_keyring(entry: str) -> str | None:
    try:
        import keyring
    except ImportError:
        return None
    service, _, user = entry.partition("/")
    if not user:
        return None
    return keyring.get_password(service, user)
