"""Declared filesystem paths for letify.

Spec section "Declared filesystem paths":
letify writes exclusively to declared directories: locally under ~/.letify, and on
runtimes within the designated workspace root, with explicit documented exceptions
for system SSH and daemon runtime directories. The system temporary directory (/tmp)
is never used.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Root directory for letify state, tools and caches on the local machine.
CONFIG_DIRECTORY = ".letify"

#: Declared local directories and files directly under ~/.letify.
DECLARED_LOCAL_PATHS = frozenset(
    {
        "config.toml",
        "accounts",
        "tools",
        "cache",
        "tmp",
        "ssh",
        "runtime",
    }
)

#: Credential files allowed inside ~/.letify/accounts/<alias>/.
#: No subdirectories or CLI caches are permitted.
ALLOWED_ACCOUNT_CREDENTIAL_FILES = frozenset(
    {
        "token.json",
        "modal.toml",
        "eci.yaml",
        "cookie",
        "access_token",
        "username",
        "notebook_id",
        "known_hosts",
        "link.json",
        "password",
    }
)

#: Documented local exceptions outside ~/.letify.
DECLARED_LOCAL_EXCEPTIONS = frozenset(
    {
        "~/.ssh/id_letify",
        "~/.ssh/id_letify.pub",
        "~/.ssh/authorized_keys",
        "<project>/.venv/bin/<tool>",
        "<project>/typings/letify_providers.pyi",
    }
)

#: Declared subdirectories under a runtime workspace root.
DECLARED_REMOTE_PATHS = frozenset(
    {
        "tmp",
        "data/blobs",
        "data/calls",
        "project",
        "uv-cache",
        "bin/uv",
        "volumes",
        "blobs",
    }
)

#: Documented remote exceptions outside the workspace root.
DECLARED_REMOTE_EXCEPTIONS = frozenset(
    {
        "~/.ssh/authorized_keys",
        "/run/sshd",
        "~/.local/bin/uv",
    }
)


def letify_root() -> Path:
    """The local state root directory: ~/.letify."""
    return Path.home() / CONFIG_DIRECTORY


def config_path() -> Path:
    """The local machine configuration file: ~/.letify/config.toml."""
    return letify_root() / "config.toml"


def accounts_directory() -> Path:
    """The account credentials directory: ~/.letify/accounts."""
    return letify_root() / "accounts"


def account_directory(alias: str) -> Path:
    """The directory holding credentials for one account: ~/.letify/accounts/<alias>/."""
    return accounts_directory() / alias


def tools_directory() -> Path:
    """The root directory for installed external tool binaries: ~/.letify/tools."""
    return letify_root() / "tools"


def cache_directory() -> Path:
    """The cleanable local cache directory: ~/.letify/cache."""
    return letify_root() / "cache"


def digest_cache_path() -> Path:
    """The path to the client digest cache file.

    Returns ~/.letify/cache/digests.json, falling back to reading
    ~/.cache/letify/digests.json if it exists from an earlier install.
    """
    new_path = cache_directory() / "digests.json"
    old_path = Path.home() / ".cache" / "letify" / "digests.json"
    if not new_path.exists() and old_path.exists():
        return old_path
    return new_path


def storage_cache_directory(name: str) -> Path:
    """The root directory for local filesystem storage backend <name>.

    Returns ~/.letify/cache/storage/<name>, falling back to ~/.cache/letify/<name>
    if it pre-exists from an earlier install.
    """
    old_path = Path.home() / ".cache" / "letify" / name
    if old_path.exists():
        return old_path
    return cache_directory() / "storage" / name


def tool_cache_home(alias: str) -> Path:
    """The redirected HOME for provider CLIs: ~/.letify/cache/tools/<alias>/."""
    return cache_directory() / "tools" / alias


def local_tmp_directory() -> Path:
    """The dedicated local temporary directory: ~/.letify/tmp."""
    return letify_root() / "tmp"


def ssh_control_directory() -> Path:
    """The directory for OpenSSH multiplexing control sockets.

    Returns $XDG_RUNTIME_DIR/letify if set, otherwise ~/.letify/ssh.
    The system temporary directory (/tmp) is never used.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "letify"
    return letify_root() / "ssh"


def local_runtime_root() -> str:
    """The local provider workspace root string: ~/.letify/runtime."""
    return "~/.letify/runtime"


def local_runtime_directory() -> Path:
    """The local provider workspace directory as a Path."""
    return letify_root() / "runtime"

