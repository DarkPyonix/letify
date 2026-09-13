"""SSH options every letify command carries: connection multiplexing and cipher preference.

This module owns the options in spec "SSH authentication" that make repeated commands to
one machine share an authenticated connection. It does not own the address, the key or
the host key checking, which each command builder decides.
"""

from __future__ import annotations

import os

from ..config.secrets import account_directory

#: Windows OpenSSH does not implement ControlMaster, so each command connects on its own.
WINDOWS = os.name == "nt"

#: AES-GCM and ChaCha20-Poly1305 first, then the client's own default list.
CIPHERS = "^aes128-gcm@openssh.com,chacha20-poly1305@openssh.com"

#: Seconds the shared connection stays open after its last command.
CONTROL_PERSIST = 60


def options(alias: str) -> list[str]:
    """The ``-o`` options for one account's SSH commands."""
    result = ["-o", f"Ciphers={CIPHERS}"]
    if WINDOWS:
        return result
    directory = account_directory(alias)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return [
        *result,
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPersist={CONTROL_PERSIST}",
        "-o",
        f"ControlPath={directory / 'ssh-%C'}",
    ]


__all__ = ["CIPHERS", "CONTROL_PERSIST", "WINDOWS", "options"]
