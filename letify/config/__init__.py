"""Reading ``.letify``.

Two files are merged. ``~/.letify`` holds accounts and connection details, which
belong to the machine, and the project's ``.letify`` holds defaults that are safe to
commit. The project file refines what the home file declared, so a repository can be
cloned by someone else and run under their own accounts.

Credentials are never read from the file itself. A field names an environment
variable or a keyring entry, and the value is resolved when it is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import ConfigError
from .schema import RESERVED_ALIASES, Config, ProviderConfig
from .secrets import from_keyring, resolve_secret

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_NAME = ".letify"


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
            options = {key: value for key, value in body.items() if key != "kind"}
            existing = config.providers.get(alias)
            if existing is None:
                config.providers[alias] = ProviderConfig(alias, kind, options, counter)
                counter += 1
            else:
                # The project file refines what the home file declared.
                existing.kind = kind
                existing.options.update(options)

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
    "RESERVED_ALIASES",
    "Config",
    "ProviderConfig",
    "from_keyring",
    "load",
    "resolve_secret",
]
