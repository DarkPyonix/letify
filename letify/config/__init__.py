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

#: A home entry with this set to true is available in every project, including one with no
#: ``.letify``. Read from the home file only, because a repository must not decide what every
#: other repository on the machine can reach.
GLOBAL_FIELD = "global"

#: Fields that say where an account applies rather than how to connect to it.
_MARKERS = frozenset({"kind", GLOBAL_FIELD, "from_home"})


def load(path: str | Path | None = None, *, home: bool = True) -> Config:
    """Read the configuration files and decide which accounts this project can use.

    The home file is the set of accounts this machine has. The project file chooses from it:
    an account is available when the project names its alias, when the home entry says
    ``global = true``, or when it is ``local``. A named alias takes every home setting,
    ``kind`` included, and the project's own fields override them one by one.

    ``path`` overrides the project file. Set ``home`` to false to ignore ``~/.letify``,
    which is what tests do to stay isolated from the developer's own accounts.
    """
    config = Config()
    home_path = Path.home() / CONFIG_NAME
    project_path = Path(path) if path else Path.cwd() / CONFIG_NAME

    home_entries = _read_entries(home_path, config, require_kind=True) if home else {}
    project_entries = _read_entries(project_path, config, require_kind=False)

    counter = 0

    def add(alias: str, kind: str, options: dict[str, Any]) -> None:
        nonlocal counter
        config.providers[alias] = ProviderConfig(alias, kind, options, counter)
        counter += 1

    for alias, body in project_entries.items():
        base = home_entries.get(alias, {})
        kind = body.get("kind", base.get("kind"))
        if not isinstance(kind, str):
            raise ConfigError(
                f"{project_path}: {alias!r} names an account that ~/.letify does not have. "
                f"Run 'letify login <kind> {alias}' to declare it on this machine, or give "
                f"the table a 'kind' to declare it here."
            )
        add(alias, kind, {**_settings(base), **_settings(body)})

    for alias, body in home_entries.items():
        if alias not in config.providers and body.get(GLOBAL_FIELD) is True:
            add(alias, body["kind"], _settings(body))

    # The local machine is always available and needs no declaration.
    if "local" not in config.providers:
        add("local", "local", {})
    return config


def _read_entries(file: Path, config: Config, *, require_kind: bool) -> dict[str, dict[str, Any]]:
    """Read one file's provider tables in order, folding its defaults into the config."""
    if not file.is_file():
        return {}
    config.sources.append(file)
    raw = _parse(file)

    defaults = raw.pop("defaults", None)
    if isinstance(defaults, dict):
        config.defaults.update(defaults)

    entries: dict[str, dict[str, Any]] = {}
    for alias, body in raw.items():
        if not isinstance(body, dict):
            continue
        _check_alias(alias, file)
        if require_kind and not isinstance(body.get("kind"), str):
            raise ConfigError(f"{file}: provider {alias!r} has no 'kind' field")
        entries[alias] = body
    return entries


def _settings(body: dict[str, Any]) -> dict[str, Any]:
    """The fields that configure a provider, without the ones that only say where it applies.

    ``from_home`` is what letify 1.0.0 wrote into a project file, and it still reads as
    naming the alias, which is all it ever meant.
    """
    return {key: value for key, value in body.items() if key not in _MARKERS}


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
    "GLOBAL_FIELD",
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
