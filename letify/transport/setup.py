"""What setting up a Tunnel account needs on both machines, apart from SSH itself.

Owns the pinned Tailcat release and the install instructions for each platform, the check
that an SSH server answers on a port, and the token ``letify client shell connect`` prints
and ``letify login tunnel --connect`` reads. It does not own the agent, the key setup or
the account file; ``letify.transport.agent`` and ``letify.config.login`` do.
"""

from __future__ import annotations

import base64
import binascii
import getpass
import json
import platform
import re
import shutil
import socket
from typing import Any

#: The Tailcat release the install instructions name.
TAILCAT_VERSION = "0.6.0"

TAILCAT_RELEASES = "https://github.com/tailscale/tailcat/releases"

#: Seconds to wait for an SSH server's banner.
BANNER_TIMEOUT = 3.0

_LINUX_ARCH = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "armv7l": "armv7",
    "armv7": "armv7",
}
_WINDOWS_ARCH = {"amd64": "amd64", "x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}


def _asset(system: str, arch: str, extension: str) -> str:
    name = f"tailcat_{TAILCAT_VERSION}_{system}_{arch}.{extension}"
    return f"{TAILCAT_RELEASES}/download/v{TAILCAT_VERSION}/{name}"


def tailcat_install_instructions(system: str | None = None, machine: str | None = None) -> str:
    """How to install Tailcat on this operating system and CPU architecture."""
    system = system if system is not None else platform.system()
    machine = (machine if machine is not None else platform.machine()).lower()
    head = f"tailcat is not on PATH. Install tailcat {TAILCAT_VERSION}"
    if system == "Linux" and machine in _LINUX_ARCH:
        arch = _LINUX_ARCH[machine]
        return (
            f"{head} for linux {arch}:\n\n"
            f"  mkdir -p ~/.local/bin && curl -L {_asset('linux', arch, 'tar.gz')} "
            f"| tar xz -C ~/.local/bin tailcat\n\n"
            "Make sure ~/.local/bin is on PATH, then run this command again."
        )
    if system == "Darwin":
        return f"{head} with Homebrew:\n\n  brew install tailcat\n\nThen run this command again."
    if system == "Windows" and machine in _WINDOWS_ARCH:
        arch = _WINDOWS_ARCH[machine]
        return (
            f"{head} for windows {arch}:\n\n"
            f"  1. Download {_asset('windows', arch, 'zip')}\n"
            "  2. Extract tailcat.exe into a folder on PATH\n\n"
            "Then run this command again in a new terminal."
        )
    return (
        f"{head}. There is no prebuilt release named for {system} {machine}; "
        f"see {TAILCAT_RELEASES}. Then run this command again."
    )


def tailcat_on_path(binary: str = "tailcat") -> bool:
    return shutil.which(binary) is not None


SSHD_INSTRUCTIONS = (
    "Install and start an SSH server on this machine. In a Debian or Ubuntu container:\n\n"
    "  apt-get install -y openssh-server\n"
    "  mkdir -p /run/sshd\n"
    "  /usr/sbin/sshd\n\n"
    "Then run this command again. If the server listens on another port, pass --ssh-port."
)


def sshd_missing_message(port: int) -> str:
    return f"No SSH server answers on port {port}. {SSHD_INSTRUCTIONS}"


def ssh_answers(port: int, host: str = "127.0.0.1", timeout: float = BANNER_TIMEOUT) -> bool:
    """Whether a connection to ``host:port`` receives a line starting with ``SSH-``."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            received = b""
            while len(received) < 4:
                chunk = conn.recv(64)
                if not chunk:
                    break
                received += chunk
    except OSError:
        return False
    return received.startswith(b"SSH-")


# -- the token -----------------------------------------------------------------

#: Fields a token must carry. ``user`` and ``port`` are optional.
REQUIRED_FIELDS = ("tailcat", "tailcat_port")


def encode_token(fields: dict[str, Any]) -> str:
    """URL-safe base64 of compact JSON, without padding, so it is one copyable word."""
    raw = json.dumps(fields, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_token(token: str) -> dict[str, Any]:
    """Read a token back, raising ValueError when it is not one ``connect`` printed."""
    text = token.strip()
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        fields = json.loads(raw)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("the token is not the one 'letify client shell connect' printed") from exc
    if not isinstance(fields, dict):
        raise ValueError("the token is not the one 'letify client shell connect' printed")
    missing = [name for name in REQUIRED_FIELDS if not fields.get(name)]
    if missing:
        raise ValueError(f"the token has no {', '.join(missing)}")
    if not isinstance(fields["tailcat"], str):
        raise ValueError("the token's tailcat address is not text")
    try:
        fields["tailcat_port"] = int(fields["tailcat_port"])
        if fields.get("port") is not None:
            fields["port"] = int(fields["port"])
    except (TypeError, ValueError) as exc:
        raise ValueError("the token's ports are not numbers") from exc
    return fields


def local_user() -> str:
    return getpass.getuser()


def default_alias(hostname: str | None = None) -> str:
    """This machine's host name as a Python identifier, for the printed login command."""
    name = re.sub(r"\W", "_", hostname if hostname is not None else socket.gethostname())
    name = name.strip("_") or "machine"
    return f"machine_{name}" if name[0].isdigit() else name


__all__ = [
    "SSHD_INSTRUCTIONS",
    "TAILCAT_VERSION",
    "decode_token",
    "default_alias",
    "encode_token",
    "local_user",
    "ssh_answers",
    "sshd_missing_message",
    "tailcat_install_instructions",
    "tailcat_on_path",
]
