"""Colab, a Google Colab runtime driven by the official CLI.

Sessions are created and destroyed with ``colab new`` and ``colab stop``. For
running work there are two paths, and which one is used decides what letify can do.

``colab ssh --proxy-mode`` is an OpenSSH ProxyCommand bridge over a WebSocket, so a
worker process can be kept alive behind pipes exactly as on any other machine. That
is the preferred channel, because it is what makes handles resolvable and lets a
large argument be sent once.

``colab exec`` runs one command and returns its output. It always works, needs
nothing beyond the CLI, and keeps nothing between calls. It is the fallback, and it
is what ``channel = "exec"`` in the configuration selects.

Two facts here are settled by measurement rather than preference.

Storage does not survive a runtime. Changing the accelerator type gives a new
virtual machine with an empty disk, which was checked by writing a marker file and
looking for it after the switch.

CUDA call forwarding works but costs. The control path crosses a Google frontend, so
a round trip from Korea is 150 ms to 200 ms. With a NVFP4 micro step near 0.5 s and
about three host synchronizations per step that leaves roughly half the throughput,
and token by token decoding falls to a few percent. letify warns with those numbers
and then does what the declaration asked for.

One account detail matters: accelerators need a Pro or Pro plus entitlement, and the
remote control features are permitted while the compute unit balance is positive. An
exhausted balance reverts the account to the free tier policy, which disallows them.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..declare.instance import Instance
from ..errors import ProviderUnavailable, RuntimeFailure
from .shell import Shell

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: Accelerators the CLI accepts. G4 is the RTX PRO 6000 Blackwell part, which is the
#: only Colab option with NVFP4 tensor cores.
GPUS = {
    "T4": {"vram_gb": 16},
    "L4": {"vram_gb": 22},
    "G4": {"vram_gb": 96},
    "A100": {"vram_gb": 40},
    "H100": {"vram_gb": 80},
}
TPUS = ("v5e1", "v6e1")

#: Names users reach for that the CLI does not know.
ALIASES = {
    "RTX_PRO_6000": "G4",
    "RTXPRO6000": "G4",
    "A100_80GB": "A100",
    "H100_80GB": "H100",
}


class Colab(Shell):
    """A Colab runtime for one Google account."""

    kind = "colab"
    extra = "colab"
    default_persistence = "ephemeral"

    #: The control path crosses a Google frontend rather than reaching the machine
    #: directly, so forwarding pays a long round trip per synchronization. Possible,
    #: since ``colab ssh --proxy-mode`` carries arbitrary TCP, but slow.
    has_fast_path = False

    #: Measured from Seoul to a Colab runtime in the United States.
    expected_round_trip_ms = 175.0

    @property
    def binary(self) -> str:
        return str(self.config.option("binary", "colab"))

    @property
    def account(self) -> str | None:
        value = self.config.option("account")
        return value if isinstance(value, str) else None

    @property
    def channel_kind(self) -> str:
        """``ssh`` for a persistent worker, ``exec`` for one command per call."""
        return str(self.config.option("channel", "ssh"))

    @property
    def persistent_channel(self) -> bool:  # type: ignore[override]
        return self.channel_kind == "ssh"

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def _require_cli(self) -> None:
        if not self.available():
            raise ProviderUnavailable(
                self.kind, f"the {self.binary!r} command is not on PATH", self.extra
            )

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """Return the fixed Colab accelerator list.

        No connection is needed. Whether an accelerator is free right now is decided
        when a runtime starts, because Colab does not promise availability.
        """
        table = {
            name: Instance(self, gpu=name, vram_gb=spec.get("vram_gb"))
            for name, spec in GPUS.items()
        }
        table.update({name: Instance(self, tpu=name) for name in TPUS})
        for alias, target in ALIASES.items():
            table[alias] = table[target]
        return table

    def store_backend(self) -> str:
        """Google Cloud Storage, which sits inside the same infrastructure.

        A Colab runtime is a Google Compute Engine virtual machine, so reading a
        bucket is an internal transfer. Google Drive is a consumer file service
        reached one file at a time, which is why it is not the default here.
        """
        backend = self.config.option("store")
        return str(backend) if isinstance(backend, str) else "gcs"

    # -- the CLI -------------------------------------------------------------

    def _cli(self, *args: str, timeout: float | None = None, stdin: str | None = None) -> str:
        self._require_cli()
        command = [self.binary, *args]
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env(),
        )
        if result.returncode != 0:
            raise RuntimeFailure(
                f"`{' '.join(command)}` exited {result.returncode}",
                command=" ".join(command),
                stderr=(result.stderr or "").strip(),
            )
        return result.stdout

    def _env(self) -> dict[str, str] | None:
        if not self.account:
            return None
        return {**os.environ, "COLAB_ACCOUNT": self.account}

    def sessions(self) -> list[str]:
        """Names of the sessions this account currently holds."""
        names = []
        for line in self._cli("sessions", timeout=120).splitlines():
            token = line.strip().split()[:1]
            if token and not token[0].lower().startswith(("name", "session", "-")):
                names.append(token[0])
        return names

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        """Reach the runtime through the CLI's WebSocket SSH bridge."""
        self._require_cli()
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ProxyCommand={self.binary} ssh --proxy-mode",
            "colab",
        ]
        if remote_command:
            command.append(remote_command)
        return command

    # -- sessions ------------------------------------------------------------

    def create_session(self, instance: Instance, name: str) -> None:
        args = ["new", "-s", name]
        if instance.gpu:
            args += ["--gpu", ALIASES.get(instance.gpu, instance.gpu)]
        elif instance.tpu:
            args += ["--tpu", instance.tpu]
        self._cli(*args, timeout=900)

    def stop(self, runtime: Runtime) -> None:
        try:
            self._cli("stop", "-s", runtime.name, timeout=180)
        except (RuntimeFailure, ProviderUnavailable):
            # Stopping is best effort. A session that is already gone is fine.
            pass

    def open_channel(self, runtime: Runtime) -> Channel:
        from ..runtime.channel import OneShotChannel, PersistentChannel

        if self.channel_kind == "ssh":
            from ..protocol.worker import BOOTSTRAP

            return PersistentChannel(
                self.ssh_command(f"{self.remote_python} -u -c {shlex.quote(BOOTSTRAP)}"),
                name=runtime.name,
            )

        def run(source: str, timeout: float | None) -> str:
            return self._cli("exec", "-s", runtime.name, stdin=source, timeout=timeout)

        return OneShotChannel(run, name=runtime.name)


__all__ = ["ALIASES", "GPUS", "TPUS", "Colab"]
