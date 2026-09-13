"""Reading ``.letify``.

Two files are merged. ``~/.letify`` holds accounts and connection details, which
belong to the machine, and the project's ``.letify`` holds defaults that are safe to
commit. The project file refines what the home file declared, so a repository can be
cloned by someone else and run under their own accounts.

Credentials are never read from the file itself. A field names an environment
variable or a keyring entry, and the value is resolved when it is used.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from . import inventory, login, writer
from .schema import RESERVED_ALIASES, Config, ProviderConfig
from .secrets import from_keyring, resolve_secret

CONFIG_NAME = ".letify"

#: A project entry carrying this and nothing else is a reference to an account in the home
#: file, written by ``letify login``. It is not a connection detail and never reaches a
#: provider.
REFERENCE_FIELD = "from_home"


def load(path: str | Path | None = None, *, home: bool = True) -> Config:
    """Read the configuration files and merge them.

    ``path`` overrides the project file. Set ``home`` to false to ignore ``~/.letify``,
    which is what tests do to stay isolated from the developer's own accounts.
    """
    config = Config()
    files: list[Path] = []
    if home:
        files.append(Path.home() / CONFIG_NAME)
    files.append(Path(path) if path else Path.cwd() / CONFIG_NAME)

    counter = 0
    # Aliases the project file expects to find in the home file, so a reference to an
    # account this machine does not have can name the command that fixes it.
    referenced: dict[str, tuple[str, Path]] = {}
    declared: set[str] = set()
    home_file = files[0] if home else None
    for file in files:
        if not file.is_file():
            continue
        config.sources.append(file)
        raw = _parse(file)

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
            options = {
                key: value for key, value in body.items() if key not in ("kind", REFERENCE_FIELD)
            }
            if body.get(REFERENCE_FIELD) is True:
                referenced.setdefault(alias, (kind, file))
            if file == home_file:
                declared.add(alias)
            existing = config.providers.get(alias)
            if existing is None:
                config.providers[alias] = ProviderConfig(alias, kind, options, counter)
                counter += 1
            else:
                # The project file refines what the home file declared.
                existing.kind = kind
                existing.options.update(options)

    for alias, (kind, file) in referenced.items():
        if alias in declared:
            continue
        raise ConfigError(
            f"{file}: {alias!r} refers to an account in ~/.letify that is not there. "
            f"Run 'letify login {kind} {alias}' to declare it on this machine."
        )

    # The local machine is always available and needs no declaration.
    if "local" not in config.providers:
        config.providers["local"] = ProviderConfig("local", "local", {}, counter)
    return config


def _parse(file: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{file} is not valid TOML: {exc}") from exc


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


__all__ = [
    "CONFIG_NAME",
    "REFERENCE_FIELD",
    "RESERVED_ALIASES",
    "Config",
    "ProviderConfig",
    "from_keyring",
    "inventory",
    "load",
    "login",
    "resolve_secret",
    "writer",
]
