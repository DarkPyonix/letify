"""Colab, a Google Colab runtime driven by the official Colab CLI.

Sessions are created and destroyed with ``colab new`` and ``colab stop``, and
commands run through ``colab exec``. Because the CLI also offers
``colab ssh --proxy-mode``, which is an OpenSSH ProxyCommand bridge over a
WebSocket, this is a Shell like any other once the session exists.

Two things are settled by measurement and are not configurable.

Storage does not survive a runtime, so this provider is ephemeral. Changing the
accelerator type gives a new virtual machine and an empty disk, which was
verified by writing a marker file and looking for it after the switch.

CUDA call forwarding is not offered here. The control path goes through a Google
frontend, so a round trip from Korea is on the order of 150 ms to 200 ms. With a
NVFP4 micro step near 0.5 s and about three host synchronizations per step, that
leaves roughly half the throughput of a local run, and token by token decoding
falls to a few percent. Shipping the loop keeps both near a local run, so that is
the only mode this provider exposes.

One account matters: accelerators need a Colab Pro or Pro+ entitlement, and the
remote control features are allowed on paid plans while the compute unit balance
is positive. A balance that runs out reverts the account to the free tier policy,
which disallows them.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..errors import ProviderUnavailable, RuntimeFailure, UnsupportedMode
from ..instance import Instance
from .shell import Shell

if TYPE_CHECKING:
    from ..env import Env
    from ..runtime import Runtime

#: Accelerators the CLI accepts. G4 is the RTX PRO 6000 Blackwell part, which is
#: the only Colab option with NVFP4 tensor cores.
GPUS = {
    "T4": dict(vram_gb=16),
    "L4": dict(vram_gb=22),
    "G4": dict(vram_gb=96),
    "A100": dict(vram_gb=40),
    "H100": dict(vram_gb=80),
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

    #: The control path crosses a Google frontend, so the round trip is too long
    #: for CUDA call forwarding to pay off.
    has_fast_path = False

    @property
    def binary(self) -> str:
        value = self.config.option("binary", "colab")
        return str(value)

    @property
    def account(self) -> str | None:
        value = self.config.option("account")
        return value if isinstance(value, str) else None

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def _require_cli(self) -> None:
        if not self.available():
            raise ProviderUnavailable(
                self.kind,
                f"the {self.binary!r} command is not on PATH",
                self.extra,
            )

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """Return the fixed Colab accelerator list.

        No connection is needed. Whether an accelerator is actually free at this
        moment is decided when a runtime starts, because Colab does not promise
        availability.
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

    # -- sessions ------------------------------------------------------------

    def _cli(self, *args: str, timeout: float | None = None, stdin: str | None = None) -> str:
        self._require_cli()
        command = [self.binary, *args]
        env_overrides = {}
        if self.account:
            env_overrides["COLAB_ACCOUNT"] = self.account
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_merged_env(env_overrides),
        )
        if result.returncode != 0:
            raise RuntimeFailure(
                f"`{' '.join(command)}` exited {result.returncode}",
                command=" ".join(command),
                stderr=(result.stderr or "").strip(),
            )
        return result.stdout

    def session_name(self, runtime_name: str) -> str:
        return runtime_name

    def create_session(self, instance: Instance, name: str) -> None:
        args = ["new", "-s", name]
        if instance.gpu:
            args += ["--gpu", ALIASES.get(instance.gpu, instance.gpu)]
        elif instance.tpu:
            args += ["--tpu", instance.tpu]
        self._cli(*args, timeout=900)

    def stop_session(self, name: str) -> None:
        try:
            self._cli("stop", "-s", name, timeout=180)
        except RuntimeFailure:
            # Stopping is best effort. A session that is already gone is fine.
            pass

    def exec(self, name: str, code: str, *, timeout: float | None = None) -> str:
        return self._cli("exec", "-s", name, stdin=code, timeout=timeout)

    def sessions(self) -> list[str]:
        output = self._cli("sessions", timeout=120)
        names = []
        for line in output.splitlines():
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
        ]
        command.append("colab")
        if remote_command:
            command.append(remote_command)
        return command

    # -- runtimes ------------------------------------------------------------

    def start(self, instance: Instance, env: Env, *, name: str | None = None) -> Runtime:
        from ..runtime import Runtime

        if instance.placement == "local":
            raise UnsupportedMode(
                "Colab does not support cpu='local'. Forwarding CUDA calls over the "
                "Colab control path costs one round trip of about 150 ms per host "
                "synchronization, which leaves roughly half the throughput for "
                "fine-tuning and a few percent for token by token decoding. Use "
                "cpu='remote' so the loop runs inside the runtime."
            )
        runtime_name = name or f"letify-{instance.accelerator.lower()}-{uuid.uuid4().hex[:6]}"
        self.create_session(instance, runtime_name)
        runtime = Runtime(name=runtime_name, provider=self, instance=instance, env=env)
        runtime.boot()
        return runtime


def _merged_env(overrides: dict[str, str]) -> dict[str, str] | None:
    import os

    if not overrides:
        return None
    return {**os.environ, **overrides}
