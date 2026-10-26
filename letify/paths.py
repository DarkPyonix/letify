"""Declared filesystem paths for letify.

Spec section "Declared filesystem paths":
letify writes exclusively to declared directories: locally under ~/.letify (or the
directory LETIFY_HOME names), and on runtimes within the designated workspace root, with
explicit documented exceptions for system SSH and daemon runtime directories. The system
temporary directory (/tmp) is never used.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .errors import ConfigError

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
        "runs",
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


def home_is_moved() -> bool:
    """Whether ``LETIFY_HOME`` is set and non-empty."""
    return bool(os.environ.get("LETIFY_HOME"))


def letify_root() -> Path:
    """The local state root directory: ``LETIFY_HOME`` when set, else ~/.letify.

    ``~`` in ``LETIFY_HOME`` is expanded. A relative value raises ``ConfigError``, because a
    root that moves with the working directory would split one machine's state.
    """
    value = os.environ.get("LETIFY_HOME")
    if not value:
        return Path.home() / CONFIG_DIRECTORY
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ConfigError(f"LETIFY_HOME={value!r} must be an absolute path")
    return root


def config_path() -> Path:
    """The local machine configuration file: ~/.letify/config.toml.

    ``LETIFY_CONFIG`` names another file and wins over ``LETIFY_HOME`` for this file only.
    """
    named = os.environ.get("LETIFY_CONFIG")
    if named:
        return Path(named).expanduser()
    return letify_root() / "config.toml"


def config_label() -> str:
    """The machine configuration file as a message names it: the path in use."""
    if os.environ.get("LETIFY_CONFIG") or home_is_moved():
        return str(config_path())
    return "~/.letify/config.toml"


def tools_label() -> str:
    """The installed tools directory as a message names it: the path in use."""
    if home_is_moved():
        return str(tools_directory())
    return "~/.letify/tools"


def accounts_label() -> str:
    """The account credentials directory as a message names it: the path in use."""
    if home_is_moved():
        return str(accounts_directory())
    return "~/.letify/accounts"


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
    ~/.cache/letify/digests.json if it exists from an earlier install, unless LETIFY_HOME is set.
    """
    new_path = cache_directory() / "digests.json"
    if home_is_moved():
        return new_path
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
    if not home_is_moved() and old_path.exists():
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

    Returns $XDG_RUNTIME_DIR/letify if set, otherwise ~/.letify/ssh. With LETIFY_HOME set
    the directory is $XDG_RUNTIME_DIR/letify-<8 hex of the root's SHA-256>, so two roots
    never share connections and the socket path stays short.
    The system temporary directory (/tmp) is never used.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        if home_is_moved():
            digest = hashlib.sha256(str(letify_root().resolve()).encode("utf-8")).hexdigest()
            return Path(runtime) / f"letify-{digest[:8]}"
        return Path(runtime) / "letify"
    return letify_root() / "ssh"


def local_runtime_root() -> str:
    """The local provider workspace root string: ~/.letify/runtime, or under LETIFY_HOME."""
    if home_is_moved():
        return str(local_runtime_directory())
    return "~/.letify/runtime"


def local_runtime_directory() -> Path:
    """The local provider workspace directory as a Path."""
    return letify_root() / "runtime"

