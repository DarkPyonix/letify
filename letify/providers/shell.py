"""Shell, a machine where letify can run commands.

The shared ability is command execution, not any one transport. Every Shell reaches its
machine through the connection pipeline in ``letify.transport``, and a subclass differs
only in its rendezvous and its strategy list: ``Colab`` runs its remote half over
``colab exec``, ``Elice`` over forward SSH to a machine its API allocated, and a plain
machine behind NAT through the remote agent over Tailcat.

Persistence defaults to ephemeral. Guessing wrong in that direction only costs
time, because letify rebuilds the environment each runtime and the work still
succeeds. Guessing persistent when the disk is actually wiped fails outright, so
the safe default is the pessimistic one. Declare ``persistent = true`` once you
know the machine keeps its disk.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..declare.instance import Instance
from ..errors import ProviderUnavailable, RuntimeFailure
from ..transport import sshopts
from .base import Provider
from .naming import gib_from_mib, normalize_gpu

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime
    from ..transport.link import Link
    from ..transport.rendezvous import Rendezvous
    from ..transport.strategies import Strategy, Target


#: Markers the check command prints for the workspace probe, replaced by a readable line.
WORKSPACE_OK = "letify-workspace-ok"
WORKSPACE_FAILED = "letify-workspace-failed"


class Shell(Provider):
    """A remote machine reached through the connection pipeline."""

    kind = "shell"
    extra = "shell"
    default_persistence = "ephemeral"

    #: A machine reached directly has a short round trip, so forwarding PyTorch operators
    #: is a real option here.
    has_fast_path = True

    #: A machine letify only runs commands on has no account behind it to meter.
    usage_source = "a machine reached by SSH has no account behind it"

    #: Whether connection decisions are printed. The Launcher sets its own announce flag here.
    announce = True

    #: SSH keeps a process alive behind pipes, so handles and blob reuse work.
    persistent_channel = True

    # -- connection ----------------------------------------------------------

    @property
    def address(self) -> str | None:
        """The address forward SSH uses. An account naming ``tailcat`` needs none."""
        value = self.config.option("address")
        if isinstance(value, str) and value:
            return value
        if isinstance(self.config.option("tailcat"), str):
            return None
        raise ProviderUnavailable(self.kind, f"{self.alias} has no 'address' field")

    @property
    def user(self) -> str | None:
        value = self.config.option("user")
        return value if isinstance(value, str) else None

    @property
    def port(self) -> int:
        """SSH port.

        Some hosts hand out a fresh port every time the machine starts. Set
        ``port_command`` to a command that prints the current port and it is read at
        connection time instead of being fixed in the configuration.
        """
        command = self.config.option("port_command")
        if isinstance(command, str):
            result = subprocess.run(
                shlex.split(command), capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return int(result.stdout.strip())
        value = self.config.option("port", 22)
        return int(value) if isinstance(value, (int, str)) else 22

    @property
    def key_path(self) -> str | None:
        value = self.config.option("key")
        return value if isinstance(value, str) else None

    @property
    def remote_python(self) -> str:
        value = self.config.option("python", "python3")
        return str(value)

    @property
    def reverse_ssh(self) -> dict[str, Any] | None:
        value = self.config.option("reverse_ssh")
        return dict(value) if isinstance(value, Mapping) else None

    @property
    def tailcat_binary(self) -> str:
        return str(self.config.option("tailcat_binary", "tailcat"))

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        """Build the OpenSSH command line that reaches this machine's address directly."""
        target = f"{self.user}@{self.address}" if self.user else self.address
        command = [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            *sshopts.options(self.alias),
        ]
        if self.key_path:
            command += ["-i", self.key_path]
        jump = self.config.option("jump")
        if isinstance(jump, str):
            command += ["-J", jump]
        command.append(target)
        if remote_command:
            command.append(remote_command)
        return command

    # -- the pipeline ----------------------------------------------------------

    def rendezvous(self, runtime: Runtime | None = None) -> Rendezvous | None:
        """The remote agent over Tailcat, when the account names its address."""
        address = self.config.option("tailcat")
        if not isinstance(address, str):
            return None
        port = self.config.option("tailcat_port")
        if not isinstance(port, (int, str)):
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} names a tailcat address but no tailcat_port. "
                f"'letify client shell connect' prints both",
            )
        from ..transport.rendezvous import TailcatRendezvous

        return TailcatRendezvous(address, int(port), self.tailcat_binary)

    def strategies(self) -> list[Strategy]:
        """The ranked strategy list. Reverse SSH joins only when the account sets it."""
        from ..transport.strategies import (
            DirectSSH,
            ProviderFallback,
            ReverseSSH,
            TailcatUDP,
            TCPPunch,
        )

        chosen: list[Strategy] = [DirectSSH(), TCPPunch(), TailcatUDP()]
        if self.reverse_ssh:
            chosen.append(ReverseSSH())
        chosen.append(ProviderFallback(rank=5 if self.reverse_ssh else 4))
        return chosen

    def fallback(self, runtime: Runtime | None = None) -> Callable[[], Link] | None:
        """The provider's own path. A plain machine has none."""
        return None

    def target(self, runtime: Runtime | None = None) -> Target:
        from ..transport import nat
        from ..transport.strategies import Target

        address = self.config.option("address")
        stun: tuple[str, int] = nat.DEFAULT_STUN
        configured = self.config.option("stun")
        if isinstance(configured, str) and ":" in configured:
            host, _, number = configured.rpartition(":")
            stun = (host, int(number))
        return Target(
            alias=self.alias,
            address=address if isinstance(address, str) else None,
            direct_ssh=self.ssh_command if isinstance(address, str) else None,
            user=self.user,
            key=self.key_path,
            remote_python=self.remote_python,
            rendezvous=self.rendezvous(runtime),
            reverse_ssh=self.reverse_ssh,
            fallback=self.fallback(runtime),
            stun=stun,
            tailcat=self.tailcat_binary,
            workspace=self.workspace_root,
        )

    def _link_key(self, runtime: Runtime | None) -> str:
        """Links are per machine here; a provider whose runtimes are machines keys by runtime."""
        return ""

    def link(self, runtime: Runtime | None = None) -> Link:
        """The chosen link, connecting through the pipeline the first time it is asked for."""
        links: dict[str, Link] = self.__dict__.setdefault("_links", {})
        key = self._link_key(runtime)
        if key not in links:
            from ..transport.announce import printer
            from ..transport.pipeline import LinkCache, Pipeline, network_fingerprint

            target = self.target(runtime)
            links[key] = Pipeline(
                self.strategies(),
                target=target,
                alias=self.alias,
                cache=LinkCache(self.alias),
                fingerprint=lambda: network_fingerprint(target.stun),
                say=printer(self.announce),
                previous=self.__dict__.get("_closed_links", {}).pop(key, None),
            ).connect()
        return links[key]

    def close_link(self, runtime: Runtime | None = None) -> None:
        key = self._link_key(runtime)
        link = self.__dict__.get("_links", {}).pop(key, None)
        if link is not None:
            # Remembered so a later connection to the same key is printed as re-established.
            self.__dict__.setdefault("_closed_links", {})[key] = link.strategy
            link.close()

    def connect(self) -> None:
        """Choose the link now rather than at the first command."""
        self.link()

    def check(self) -> str:
        """Run one command to confirm the machine answers and its workspace root is writable."""
        from ..runtime.bootstrap import workspace_check

        root = self.workspace_root
        remote = (
            "uname -a; nvidia-smi --query-gpu=name --format=csv,noheader; "
            f"if letify_out=$({workspace_check(root)} 2>&1); then echo {WORKSPACE_OK}; "
            f'else echo "{WORKSPACE_FAILED}: $letify_out"; fi'
        )
        link = self.link()
        result = subprocess.run(
            link.ssh_command(remote),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeFailure(
                f"{self.alias} did not answer over SSH",
                command=" ".join(link.ssh_command("...")),
                stderr=result.stderr.strip(),
            )
        lines = []
        for line in result.stdout.splitlines():
            if line.strip() == WORKSPACE_OK:
                lines.append(f"workspace {root}: writable")
            elif line.startswith(f"{WORKSPACE_FAILED}:"):
                reason = line[len(WORKSPACE_FAILED) + 1 :].strip()
                lines.append(f"workspace {root}: not writable: {reason}")
            else:
                lines.append(line)
        return "\n".join(lines) + "\n"

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """Ask the machine which GPUs it has.

        A configuration entry may declare them instead, which avoids connecting during
        import. A declaration is trusted without checking, because an entry that names its
        cards has said what it has.
        """
        from ..config.inventory import read_table

        declared = read_table(self.config.options)
        if declared:
            return {name: Instance(self, gpu=name) for name in declared}

        result = subprocess.run(
            self.link().ssh_command(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
            ),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise ProviderUnavailable(
                self.kind,
                f"could not list GPUs on {self.alias}: {result.stderr.strip() or 'ssh failed'}",
                self.extra,
            )
        table: dict[str, Instance] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            name, _, memory = line.partition(",")
            label = normalize_gpu(name)
            table[label] = Instance(self, gpu=label, vram_gb=gib_from_mib(memory))
        return table

    def busy(self) -> tuple[int, ...]:
        """Ask the machine over its link which cards another user is computing on.

        Read at every reservation, never cached. A query that cannot run raises rather than
        reading as free, because free would put a run on a card someone else is using.
        """
        from ..runtime import telemetry

        link = self.link()

        def run(command: tuple[str, ...]) -> str:
            remote = shlex.join(command)
            result = subprocess.run(
                link.ssh_command(remote), capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                raise RuntimeFailure(
                    f"{self.alias}: the busy check could not run, so which cards are free "
                    f"is unknown and nothing was reserved",
                    command=remote,
                    stderr=result.stderr.strip(),
                )
            return result.stdout

        owners: dict[int, tuple[str, ...]] = {}
        busy = telemetry.busy_indices(exclude_pids=self.worker_pids(), run=run, owners_out=owners)
        self.last_busy_owners = owners
        return busy

    def store_backend(self) -> str:
        backend = self.config.option("store")
        return str(backend) if isinstance(backend, str) else "shell"

    # -- sessions ------------------------------------------------------------

    def open_channel(self, runtime: Runtime) -> Channel:
        """One remote Python reading framed requests, or one program per call on a fallback."""
        from ..protocol.worker import BOOTSTRAP
        from ..runtime.channel import OneShotChannel, PersistentChannel

        link = self.link(runtime)
        if not link.persistent:
            return OneShotChannel(
                link.runner,  # type: ignore[attr-defined]
                name=runtime.name,
                files=getattr(link, "files", None),
            )
        return PersistentChannel(
            link.ssh_command(f"{self.remote_python} -u -c {shlex.quote(BOOTSTRAP)}"),
            name=runtime.name,
        )

    def device_channel(self, runtime: Runtime) -> Channel:
        """The session's call channel, which hosts the PyTorch device executor."""
        from ..errors import UnsupportedMode

        channel = runtime.channel
        if channel is None or not channel.persistent:
            raise UnsupportedMode(
                f"{self.alias} is reached over a link that runs one command per call, so a "
                f"PyTorch device worker cannot stay alive there. Use host='remote'."
            )
        return channel


__all__ = ["Shell"]
