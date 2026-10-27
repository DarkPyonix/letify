"""SSH options every letify command carries: connection multiplexing and cipher preference.

This module owns the options in spec "SSH authentication" that make repeated commands to
one machine share an authenticated connection, including where the control sockets live
and the checks on that directory. It does not own the address, the key or the host key
checking, which each command builder decides.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

from ..errors import ConfigError

#: Windows OpenSSH does not implement ControlMaster, so each command connects on its own.
WINDOWS = os.name == "nt"

#: AES-GCM and ChaCha20-Poly1305 first, then the client's own default list.
CIPHERS = "^aes128-gcm@openssh.com,chacha20-poly1305@openssh.com"

#: Seconds the shared connection stays open after its last command.
CONTROL_PERSIST = 60

#: Bytes in a Unix domain socket path, including the terminating byte.
SOCKET_LIMIT = 104 if sys.platform == "darwin" else 108

#: Bytes ``%C`` expands to, and bytes OpenSSH appends while it binds a temporary socket.
_HASH_BYTES = 40
_TEMPORARY_SUFFIX = 17


def control_directory() -> Path:
    """The per-user directory for control sockets, created and checked before use."""
    from ..paths import ssh_control_directory

    directory = ssh_control_directory()
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
    except FileNotFoundError:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except FileExistsError:
        # A dangling symbolic link or a file: the lstat check below says which.
        pass
    info = directory.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ConfigError(f"the SSH control directory {directory} is a symbolic link")
    if not stat.S_ISDIR(info.st_mode):
        raise ConfigError(f"the SSH control directory {directory} is not a directory")
    if info.st_uid != os.getuid():
        raise ConfigError(f"the SSH control directory {directory} is not owned by this user")
    if info.st_mode & 0o077:
        raise ConfigError(
            f"the SSH control directory {directory} has group or other permission bits"
            f" ({stat.S_IMODE(info.st_mode):o}), expected 700"
        )
    return directory


def control_path(alias: str) -> Path | None:
    """The ``ControlPath`` for one account, or None when it would not fit a socket path."""
    tag = hashlib.sha256(alias.encode("utf-8")).hexdigest()[:8]
    path = control_directory() / f"{tag}-%C"
    length = len(os.fsencode(path)) - len("%C") + _HASH_BYTES + _TEMPORARY_SUFFIX
    return path if length < SOCKET_LIMIT else None


def options(alias: str) -> list[str]:
    """The ``-o`` options for one account's SSH commands."""
    result = ["-o", f"Ciphers={CIPHERS}"]
    if WINDOWS:
        return result
    path = control_path(alias)
    if path is None:
        return result
    return [
        *result,
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPersist={CONTROL_PERSIST}",
        "-o",
        f"ControlPath={path}",
    ]


_SHARING = ("ControlMaster=", "ControlPath=", "ControlPersist=")


def without_sharing(command: list[str]) -> list[str]:
    """The command with the ControlMaster, ControlPath and ControlPersist options removed."""
    kept: list[str] = []
    index = 0
    while index < len(command):
        part = command[index]
        if part == "-o" and index + 1 < len(command) and command[index + 1].startswith(_SHARING):
            index += 2
            continue
        kept.append(part)
        index += 1
    return kept


def unshared(command: list[str]) -> list[str]:
    """The command as its own connection: no shared one is joined and none is left behind.

    The options go right after the program, so they are never taken for the remote command.
    """
    kept = without_sharing(command)
    if os.path.basename(kept[0]) not in ("ssh", "ssh.exe"):
        return kept
    return [kept[0], "-o", "ControlMaster=no", "-o", "ControlPath=none", *kept[1:]]


__all__ = [
    "CIPHERS",
    "CONTROL_PERSIST",
    "SOCKET_LIMIT",
    "WINDOWS",
    "control_directory",
    "control_path",
    "options",
    "unshared",
    "without_sharing",
]
