"""Resolving credentials without writing them into a configuration file.

A credential never appears in either ``config.toml``. A field such as ``access_token`` is
resolved in this order:

1. the environment variable named by ``access_token_env``,
2. the file ``access_token`` in ``~/.letify/accounts/<alias>/``,
3. a literal value, accepted only so a home entry can carry a non secret default.

The account directory is where every credential letify collects goes, one directory per
alias, readable by its owner only. Provider tools that keep their own login keep it there
too, which is what lets two accounts of one provider exist on one machine.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

#: The directory letify keeps its state in, at home and in a project.
CONFIG_DIRECTORY = ".letify"

#: Where one account's credentials and provider state live, under the home directory.
ACCOUNTS_DIRECTORY = "accounts"


def account_directory(alias: str) -> Path:
    """``~/.letify/accounts/<alias>/``, whether or not it exists yet."""
    return Path.home() / CONFIG_DIRECTORY / ACCOUNTS_DIRECTORY / alias


def resolve_secret(
    options: dict[str, Any], name: str, default: str | None = None, *, alias: str | None = None
) -> str | None:
    """Return the credential a configuration entry points at, or ``default``."""
    env_key = options.get(f"{name}_env")
    if isinstance(env_key, str):
        value = os.environ.get(env_key)
        if value:
            return value

    if alias:
        path = account_directory(alias) / name
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value

    literal = options.get(name)
    if isinstance(literal, str):
        return literal
    return default


def write_secret(alias: str, name: str, value: str) -> Path:
    """Store a credential in the account directory, readable by its owner only.

    The file is created with its permissions already set rather than changed afterwards,
    so there is no moment when another user could read it. Windows has no such mode bits,
    and there the file inherits the home directory's access list.
    """
    directory = account_directory(alias)
    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        try:
            directory.chmod(0o700)
        except OSError:
            pass
    path = directory / name
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
    if os.name != "nt":
        os.chmod(path, 0o600)
    return path


def forget_account(alias: str) -> bool:
    """Delete an account's directory and everything in it. Returns whether it existed."""
    directory = account_directory(alias)
    if not directory.exists():
        return False
    shutil.rmtree(directory)
    return True


__all__ = [
    "ACCOUNTS_DIRECTORY",
    "CONFIG_DIRECTORY",
    "account_directory",
    "forget_account",
    "resolve_secret",
    "write_secret",
]
