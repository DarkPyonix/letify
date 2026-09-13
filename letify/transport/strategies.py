"""Strategy, one way to connect, and the strategies the spec ranks.

Owns checking a strategy's needs and attempting it: forward SSH (rank 1), TCP hole
punching (rank 2), UDP hole punching with Tailcat (rank 3), reverse SSH when an account
opts in (rank 4), and the provider fallback (last). It does not own racing or choosing;
the pipeline does.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.secrets import account_directory
from . import nat
from .link import Link, PunchedLink, SSHLink

#: Seconds a single attempt may take before the strategy counts as failed.
CONNECT_TIMEOUT = 30.0
#: The comment every reverse SSH session key carries in authorized_keys.
SESSION_MARKER = "letify-session"
RESTRICTION = 'restrict,port-forwarding,command="/bin/false"'


@dataclass
class Target:
    """Everything a strategy needs to know about the machine it connects to."""

    alias: str
    address: str | None = None
    direct_ssh: Callable[[str | None], list[str]] | None = None
    user: str | None = None
    key: str | None = None
    remote_python: str = "python3"
    rendezvous: Any = None
    reverse_ssh: dict[str, Any] | None = None
    fallback: Callable[[], Link] | None = None
    ssh_port: int = 22
    stun: tuple[str, int] = nat.DEFAULT_STUN
    tailcat: str = "tailcat"
    host_key_alias: str | None = None
    #: The account's workspace root on the machine, before ``~`` is expanded there.
    workspace: str = "~/.letify-runtime"

    def _options(self) -> list[str]:
        directory = account_directory(self.alias)
        # ssh creates only ~/.ssh; without this directory it cannot record the host key
        # and warns "Failed to add the host to the list of known hosts" on every run.
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        known_hosts = directory / "known_hosts"
        options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            f"HostKeyAlias={self.host_key_alias or 'letify-' + self.alias}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
        ]
        if self.key:
            options += ["-i", self.key]
        return options

    def forwarded_ssh(self, port: int, remote_command: str | None = None) -> list[str]:
        """SSH to a local forwarding port."""
        login = f"{self.user}@127.0.0.1" if self.user else "127.0.0.1"
        tail = [remote_command] if remote_command else []
        return ["ssh", "-p", str(port), *self._options(), login, *tail]

    def proxied_ssh(self, proxy: str, remote_command: str | None = None) -> list[str]:
        """SSH whose transport is a ProxyCommand."""
        host = f"letify-{self.alias}"
        login = f"{self.user}@{host}" if self.user else host
        tail = [remote_command] if remote_command else []
        return ["ssh", *self._options(), "-o", f"ProxyCommand={proxy}", login, *tail]


class Strategy:
    """One way to connect."""

    name = ""
    rank = 0

    def needs(self, target: Target) -> str | None:
        """What is missing for this strategy, or None when it can be attempted."""
        raise NotImplementedError

    def attempt(self, target: Target) -> Link:
        """Connect, or raise."""
        raise NotImplementedError

    def assume(self, target: Target) -> Link:
        """The link to use when this is the only applicable strategy."""
        return self.attempt(target)


def _rendezvous_unmet(target: Target) -> str | None:
    if target.rendezvous is None:
        return "no rendezvous"
    return target.rendezvous.unavailable()


def _verify(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=CONNECT_TIMEOUT)
    if result.returncode != 0:
        raise OSError((result.stderr or "").strip() or f"ssh exited {result.returncode}")


class DirectSSH(Strategy):
    """Forward SSH to the machine's address."""

    name = "direct_ssh"
    rank = 1

    def needs(self, target: Target) -> str | None:
        return None if target.address and target.direct_ssh else "no address"

    def assume(self, target: Target) -> Link:
        # Alone there is nothing to race, so the first real command is the check, and its
        # error says what went wrong.
        return SSHLink(self.name, self.rank, target.direct_ssh, remote_python=target.remote_python)  # type: ignore[arg-type]

    def attempt(self, target: Target) -> Link:
        _verify(target.direct_ssh("exit 0"))  # type: ignore[misc]
        return self.assume(target)


class TCPPunch(Strategy):
    """TCP hole punching: STUN over TCP 443, then a simultaneous open at an agreed time."""

    name = "tcp_punch"
    rank = 2

    def needs(self, target: Target) -> str | None:
        return _rendezvous_unmet(target)

    def attempt(self, target: Target) -> Link:
        return PunchedLink(
            self.name,
            self.rank,
            self._punch(target),
            target.forwarded_ssh,
            lambda: self._punch(target),
        )

    def _punch(self, target: Target):
        holder = nat.reusable_socket(0)
        port = holder.getsockname()[1]
        try:
            mapping = nat.stun_mapping(port, tuple(target.stun))
            token = secrets.token_bytes(nat.TOKEN_BYTES)
            start_at = time.time() + float(getattr(target.rendezvous, "lead_seconds", 10.0))
            answer = target.rendezvous.exchange(
                {
                    "kind": "tcp_punch",
                    "mapping": list(mapping),
                    "token": token.hex(),
                    "start_at": start_at,
                    "ssh_port": target.ssh_port,
                    "stun": list(target.stun),
                },
                CONNECT_TIMEOUT,
            )
            return nat.punch(
                port, tuple(answer["mapping"]), token, initiator=True, start_at=start_at
            )
        finally:
            holder.close()


class TailcatUDP(Strategy):
    """UDP hole punching with Tailcat, then SSH over it."""

    name = "tailcat"
    rank = 3

    def needs(self, target: Target) -> str | None:
        if shutil.which(target.tailcat) is None:
            return f"{target.tailcat} is not on PATH"
        return _rendezvous_unmet(target)

    def attempt(self, target: Target) -> Link:
        address, port = target.rendezvous.tailcat_endpoint(target.ssh_port, CONNECT_TIMEOUT)
        proxy = f"{target.tailcat} {address} {port}"

        def command(remote_command: str | None = None) -> list[str]:
            return target.proxied_ssh(proxy, remote_command)

        _verify(command("exit 0"))
        return SSHLink(self.name, self.rank, command, remote_python=target.remote_python)


class AuthorizedKeys:
    """The user's ``authorized_keys``, holding one restricted line per live reverse session."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path.home() / ".ssh" / "authorized_keys"

    def _lines(self) -> list[str]:
        return self.path.read_text().splitlines() if self.path.is_file() else []

    def _write(self, lines: list[str]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.write_text("".join(line + "\n" for line in lines))
        if os.name != "nt":
            self.path.chmod(0o600)

    def purge(self, alias: str) -> None:
        """Remove every session line for ``alias``, so a crashed session leaves no key."""
        kept = [line for line in self._lines() if line.split()[-2:] != [SESSION_MARKER, alias]]
        if kept != self._lines():
            self._write(kept)

    def install(self, alias: str, public_key: str) -> str:
        kind, body = public_key.split()[:2]
        line = f"{RESTRICTION} {kind} {body} {SESSION_MARKER} {alias}"
        self._write([*self._lines(), line])
        return line

    def remove(self, line: str) -> None:
        self._write([existing for existing in self._lines() if existing != line])


class ReverseSSH(Strategy):
    """The remote side forwards its SSH server back to the user's machine with ``ssh -R 0:``."""

    name = "reverse_ssh"
    rank = 4

    def __init__(self, keys: AuthorizedKeys | None = None):
        self.keys = keys

    def needs(self, target: Target) -> str | None:
        if not target.reverse_ssh:
            return "no reverse_ssh entry"
        return _rendezvous_unmet(target)

    def attempt(self, target: Target) -> Link:
        keys = self.keys or AuthorizedKeys()
        keys.purge(target.alias)
        directory = Path(tempfile.mkdtemp(prefix="letify-reverse-"))
        private = directory / "id_session"
        result = subprocess.run(
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-N",
                "",
                "-q",
                "-C",
                f"{SESSION_MARKER} {target.alias}",
                "-f",
                str(private),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            shutil.rmtree(directory, ignore_errors=True)
            raise OSError(f"ssh-keygen failed: {result.stderr.strip()}")
        line = keys.install(target.alias, private.with_suffix(".pub").read_text())

        def cleanup() -> None:
            keys.remove(line)
            shutil.rmtree(directory, ignore_errors=True)

        spec = target.reverse_ssh or {}
        try:
            answer = target.rendezvous.exchange(
                {
                    "kind": "reverse_ssh",
                    "address": spec["address"],
                    "port": int(spec.get("port", 22)),
                    "user": spec["user"],
                    "private_key": private.read_text(),
                    "key_directory": (
                        f"{target.workspace.rstrip('/')}/tmp/letify-{secrets.token_hex(8)}"
                    ),
                    "ssh_port": target.ssh_port,
                },
                CONNECT_TIMEOUT,
            )
            port = int(answer["port"])
        except BaseException:
            cleanup()
            raise

        def command(remote_command: str | None = None) -> list[str]:
            return target.forwarded_ssh(port, remote_command)

        return SSHLink(
            self.name, self.rank, command, remote_python=target.remote_python, on_close=cleanup
        )


class ProviderFallback(Strategy):
    """The provider's own path, such as ``colab exec``. Last, because it is the slowest."""

    name = "fallback"
    #: The fallback cannot carry the probe, so it does not start the grace period.
    probed = False

    def __init__(self, rank: int = 4):
        self.rank = rank

    def needs(self, target: Target) -> str | None:
        return None if target.fallback is not None else "no provider fallback"

    def attempt(self, target: Target) -> Link:
        link = target.fallback()  # type: ignore[misc]
        link.rank = self.rank
        return link


__all__ = [
    "AuthorizedKeys",
    "DirectSSH",
    "ProviderFallback",
    "ReverseSSH",
    "Strategy",
    "TCPPunch",
    "TailcatUDP",
    "Target",
]
