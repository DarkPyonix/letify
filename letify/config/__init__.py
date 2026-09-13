"""Reading the ``.letify`` configuration directories.

letify keeps its state in two directories. ``~/.letify/`` belongs to the machine: its
``config.toml`` lists every account, and ``accounts/<alias>/`` holds each account's
credentials. ``<project>/.letify/config.toml`` belongs to the repository and names which of
those accounts the project uses.

Credentials are never read from either ``config.toml``. A field names an environment variable,
or the credential lives in the account directory, and the value is resolved when it is used.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from . import inventory, login, writer
from .schema import RESERVED_ALIASES, Config, ProviderConfig
from .secrets import CONFIG_DIRECTORY, account_directory, resolve_secret

#: The configuration file inside each ``.letify`` directory.
CONFIG_FILE = "config.toml"

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

    ``path`` names another project ``.letify`` directory, or a ``config.toml`` directly.
    Set ``home`` to false to ignore ``~/.letify``,
    which is what tests do to stay isolated from the developer's own accounts.
    """
    config = Config()
    home_path = Path.home() / CONFIG_DIRECTORY / CONFIG_FILE
    project_path = _project_file(path)

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
                f"{project_path}: {alias!r} names an account that ~/.letify/config.toml does not "
                f"have. "
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


def _project_file(path: str | Path | None) -> Path:
    """The project's ``config.toml``, from a ``.letify`` directory, a file, or the default."""
    if path is None:
        return Path.cwd() / CONFIG_DIRECTORY / CONFIG_FILE
    given = Path(path)
    return given / CONFIG_FILE if given.is_dir() else given


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
    "CONFIG_DIRECTORY",
    "CONFIG_FILE",
    "GLOBAL_FIELD",
    "RESERVED_ALIASES",
    "Config",
    "ProviderConfig",
    "account_directory",
    "inventory",
    "load",
    "login",
    "resolve_secret",
    "writer",
]
