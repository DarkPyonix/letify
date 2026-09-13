"""Shell, a machine where letify can run commands.

The shared ability is command execution, not any one transport. SSH is the default,
and the subclasses change only how the connection is obtained: ``Colab`` creates
the session with a CLI, ``Tunnel`` builds a network path first, and ``Elice``
allocates the machine through an API.

Persistence defaults to ephemeral. Guessing wrong in that direction only costs
time, because letify rebuilds the environment each runtime and the work still
succeeds. Guessing persistent when the disk is actually wiped fails outright, so
the safe default is the pessimistic one. Declare ``persistent = true`` once you
know the machine keeps its disk.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..declare.instance import Instance
from ..errors import ProviderUnavailable, RuntimeFailure
from .base import Provider
from .naming import gib_from_mib, normalize_gpu

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime


class Shell(Provider):
    """A remote machine reached over SSH."""

    kind = "shell"
    extra = "shell"
    default_persistence = "ephemeral"

    #: A machine reached directly has a short round trip, so forwarding CUDA calls
    #: is a real option here.
    has_fast_path = True

    #: A machine letify only runs commands on has no account behind it to meter.
    usage_source = "a machine reached by SSH has no account behind it"

    #: SSH keeps a process alive behind pipes, so handles and blob reuse work.
    persistent_channel = True

    # -- connection ----------------------------------------------------------

    @property
    def address(self) -> str:
        value = self.config.option("address")
        if not isinstance(value, str):
            raise ProviderUnavailable(self.kind, f"{self.alias} has no 'address' field")
        return value

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

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        """Build the OpenSSH command line that reaches this machine."""
        target = f"{self.user}@{self.address}" if self.user else self.address
        command = [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
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

    def connect(self) -> None:
        """Open whatever path this provider needs. Plain SSH needs nothing."""
        return None

    def check(self) -> str:
        """Run one command to confirm the machine answers."""
        self.connect()
        result = subprocess.run(
            self.ssh_command("uname -a; nvidia-smi --query-gpu=name --format=csv,noheader"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeFailure(
                f"{self.alias} did not answer over SSH",
                command=" ".join(self.ssh_command("...")),
                stderr=result.stderr.strip(),
            )
        return result.stdout

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """Ask the machine which GPUs it has.

        A configuration entry may list them instead, which avoids connecting during
        import. A declared list is trusted without checking.
        """
        declared = self.config.option("gpus")
        if isinstance(declared, list) and declared:
            return {str(name): Instance(self, gpu=str(name)) for name in declared}

        self.connect()
        result = subprocess.run(
            self.ssh_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
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

    def store_backend(self) -> str:
        backend = self.config.option("store")
        return str(backend) if isinstance(backend, str) else "shell"

    # -- sessions ------------------------------------------------------------

    def open_channel(self, runtime: Runtime) -> Channel:
        """One remote Python reading framed requests from standard input."""
        from ..runtime.channel import PersistentChannel

        self.connect()
        from ..protocol.worker import BOOTSTRAP

        return PersistentChannel(
            self.ssh_command(f"{self.remote_python} -u -c {shlex.quote(BOOTSTRAP)}"),
            name=runtime.name,
        )


__all__ = ["Shell"]
