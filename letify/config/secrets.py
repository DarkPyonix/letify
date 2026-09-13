"""Resolving credentials without writing them down.

A configuration file in a repository is a bad place for a token, so a field names
where the value lives rather than holding it. Three forms are accepted, and the
order below is the order they are tried.

``<name>_env`` names an environment variable. ``<name>_keyring`` names an OS keyring
entry as ``service/user``. ``<name>`` is a literal, which is only appropriate in
``~/.letify``, outside any repository.
"""

from __future__ import annotations

import os
from typing import Any


def resolve_secret(options: dict[str, Any], name: str, default: str | None = None) -> str | None:
    """Return the credential a configuration entry points at."""
    env_key = options.get(f"{name}_env")
    if isinstance(env_key, str):
        value = os.environ.get(env_key)
        if value:
            return value

    keyring_entry = options.get(f"{name}_keyring")
    if isinstance(keyring_entry, str):
        value = from_keyring(keyring_entry)
        if value:
            return value

    literal = options.get(name)
    if isinstance(literal, str):
        return literal
    return default


def from_keyring(entry: str) -> str | None:
    """Read ``service/user`` from the OS keyring, if the package is installed."""
    try:
        import keyring
    except ImportError:
        return None
    service, _, user = entry.partition("/")
    if not user:
        return None
    return keyring.get_password(service, user)


__all__ = ["from_keyring", "resolve_secret"]
