"""Shell, a machine where letify can run commands.

The shared ability is command execution, not any one transport. SSH is the
default and the subclasses change how the connection is made: ``Colab`` creates
the session with the Colab CLI, ``Tunnel`` builds a network path first, and
``Elice`` allocates the machine through the Elice Cloud API.

Persistence defaults to ephemeral. Guessing wrong in that direction only costs
time, because letify rebuilds the environment each runtime and the work still
succeeds. Guessing persistent when the disk is actually wiped fails outright, so
the safe default is the pessimistic one. Declare ``persistent = true`` in the
configuration once you know the machine keeps its disk.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..errors import ProviderUnavailable, RuntimeFailure
from ..instance import Instance
from .base import Provider

if TYPE_CHECKING:
    from ..env import Env
    from ..runtime import Runtime


class Shell(Provider):
    """A remote machine reached over SSH."""

    kind = "shell"
    extra = "shell"
    default_persistence = "ephemeral"

    #: A machine reached directly has a short round trip, so forwarding CUDA
    #: calls is a real option here.
    has_fast_path = True

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
        ``port_command`` in the configuration to a shell command that prints the
        current port, and it is read at connection time instead.
        """
        command = self.config.option("port_command")
        if isinstance(command, str):
            out = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=60)
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return int(out.stdout.strip())
        value = self.config.option("port", 22)
        return int(value) if isinstance(value, (int, str)) else 22

    @property
    def key_path(self) -> str | None:
        value = self.config.option("key")
        return value if isinstance(value, str) else None

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        """Build the OpenSSH command line used to reach this machine."""
        target = f"{self.user}@{self.address}" if self.user else self.address
        command = ["ssh", "-p", str(self.port), "-o", "BatchMode=yes"]
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
        """Open whatever path this provider needs. SSH needs nothing extra."""
        return None

    def check(self) -> str:
        """Run one command to confirm the machine answers."""
        out = subprocess.run(
            self.ssh_command("uname -a && nvidia-smi --query-gpu=name --format=csv,noheader"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode != 0:
            raise RuntimeFailure(
                f"{self.alias} did not answer over SSH",
                command=" ".join(self.ssh_command("...")),
                stderr=out.stderr.strip(),
            )
        return out.stdout

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """Ask the machine which GPUs it has.

        The configuration may list them instead, which avoids connecting during
        import. A declared list is trusted without checking.
        """
        declared = self.config.option("gpus")
        if isinstance(declared, list) and declared:
            return {str(name): Instance(self, gpu=str(name)) for name in declared}

        self.connect()
        out = subprocess.run(
            self.ssh_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode != 0:
            raise ProviderUnavailable(
                self.kind,
                f"could not list GPUs on {self.alias}: {out.stderr.strip() or 'ssh failed'}",
                self.extra,
            )
        table: dict[str, Instance] = {}
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            name, _, memory = line.partition(",")
            label = _normalize_gpu_name(name)
            vram = _parse_mib(memory)
            table[label] = Instance(self, gpu=label, vram_gb=vram)
        return table

    def store_backend(self) -> str:
        backend = self.config.option("store")
        return str(backend) if isinstance(backend, str) else "shell"

    # -- runtimes ------------------------------------------------------------

    def start(self, instance: Instance, env: Env, *, name: str) -> Runtime:
        from ..runtime import Runtime

        self.connect()
        runtime = Runtime(name=name, provider=self, instance=instance, env=env)
        runtime.boot()
        return runtime


def _normalize_gpu_name(raw: str) -> str:
    """Turn an nvidia-smi product name into a short attribute-friendly label.

    ``NVIDIA RTX PRO 6000 Blackwell`` becomes ``RTX_PRO_6000``, and
    ``NVIDIA A100-SXM4-80GB`` becomes ``A100``.
    """
    text = raw.strip().removeprefix("NVIDIA").strip()
    for marker in ("-SXM", "-PCIE", " SXM", " PCIe", " Blackwell", " Laptop"):
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    return text.strip().replace(" ", "_").replace("-", "_")


def _parse_mib(raw: str) -> int | None:
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return round(int(digits) / 1024)
