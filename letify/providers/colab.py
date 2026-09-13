"""Colab, a Google Colab runtime driven by the official CLI.

Sessions are created and destroyed with ``colab new`` and ``colab stop``. Colab is a
``Shell`` whose rendezvous is its provider layer: ``colab exec`` runs letify's remote half
on the runtime, which installs and starts an SSH server (a Colab VM has none) and answers
the punch or Tailcat request. The connection pipeline then races TCP hole punching and
Tailcat, with no forward SSH, and falls back to ``colab exec`` itself, one program per
call. ``channel = "exec"`` in the configuration skips the pipeline and uses that fallback
directly.

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

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .. import tools
from ..declare.instance import Instance
from ..errors import ProviderUnavailable, RuntimeFailure
from .shell import Shell

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime
    from ..transport.link import Link
    from ..transport.rendezvous import Rendezvous
    from ..transport.strategies import Strategy, Target

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
    #: directly, so forwarding pays a long round trip per synchronization.
    has_fast_path = False

    #: Measured from Seoul to a Colab runtime in the United States.
    expected_round_trip_ms = 175.0

    #: Colab meters in compute units and keeps the balance in the web console; the CLI has
    #: no command that prints it. A configuration entry can name one.
    usage_unit = "compute units"
    usage_source = "the Colab CLI has no balance command; the figure is in the web console"

    @property
    def account(self) -> str | None:
        value = self.config.option("account")
        return value if isinstance(value, str) else None

    @property
    def channel_kind(self) -> str:
        """``ssh`` for a persistent worker over the pipeline, ``exec`` for one command per call."""
        return str(self.config.option("channel", "ssh"))

    @property
    def persistent_channel(self) -> bool:  # type: ignore[override]
        return self.channel_kind == "ssh"

    def available(self) -> bool:
        return tools.find_uv() is not None

    def _colab(self) -> list[str]:
        """The Colab CLI, run through uv so it never enters the user's environment."""
        uv = tools.find_uv()
        if uv is None:
            raise ProviderUnavailable(self.kind, tools.missing_uv_message())
        return tools.command(tools.COLAB, uv)

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
        command = [*self._colab(), *args]
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

    def _env(self) -> dict[str, str]:
        env = tools.environment(self.alias)
        if self.account:
            env["COLAB_ACCOUNT"] = self.account
        return env

    def _exec(self, session: str, source: str, timeout: float | None) -> str:
        return self._cli("exec", "-s", session, stdin=source, timeout=timeout)

    def sessions(self) -> list[str]:
        """Names of the sessions this account currently holds."""
        names = []
        for line in self._cli("sessions", timeout=120).splitlines():
            token = line.strip().split()[:1]
            if token and not token[0].lower().startswith(("name", "session", "-")):
                names.append(token[0])
        return names

    # -- the pipeline ----------------------------------------------------------

    def strategies(self) -> list[Strategy]:
        """The Shell list without forward SSH, which a Colab VM does not accept."""
        return [strategy for strategy in super().strategies() if strategy.name != "direct_ssh"]

    def rendezvous(self, runtime: Runtime | None = None) -> Rendezvous | None:
        if runtime is None:
            return None
        from ..transport.rendezvous import ColabRendezvous

        name = runtime.name
        return ColabRendezvous(
            lambda source, timeout: self._exec(name, source, timeout), self._public_key()
        )

    def fallback(self, runtime: Runtime | None = None) -> Callable[[], Link] | None:
        if runtime is None:
            return None
        from ..transport.link import OneShotLink

        name = runtime.name
        return lambda: OneShotLink(
            "fallback", 4, lambda source, timeout: self._exec(name, source, timeout)
        )

    def target(self, runtime: Runtime | None = None) -> Target:
        target = super().target(runtime)
        target.user = self.user or "root"
        if runtime is not None:
            # Each runtime is a new VM with a new host key.
            target.host_key_alias = f"letify-{self.alias}-{runtime.name}"
        return target

    def _link_key(self, runtime: Runtime | None) -> str:
        return runtime.name if runtime is not None else ""

    def _public_key(self) -> str | None:
        if not self.key_path:
            return None
        path = Path(self.key_path + ".pub").expanduser()
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    # -- sessions ------------------------------------------------------------

    def create_session(self, instance: Instance, name: str) -> None:
        args = ["new", "-s", name]
        if instance.gpu:
            args += ["--gpu", ALIASES.get(instance.gpu, instance.gpu)]
        elif instance.tpu:
            args += ["--tpu", instance.tpu]
        self._cli(*args, timeout=900)

    def stop(self, runtime: Runtime) -> None:
        self.close_link(runtime)
        try:
            self._cli("stop", "-s", runtime.name, timeout=180)
        except (RuntimeFailure, ProviderUnavailable):
            # Stopping is best effort. A session that is already gone is fine.
            pass

    def open_channel(self, runtime: Runtime) -> Channel:
        from ..runtime.channel import OneShotChannel

        if self.channel_kind == "exec":
            name = runtime.name

            def run(source: str, timeout: float | None) -> str:
                return self._exec(name, source, timeout)

            return OneShotChannel(run, name=runtime.name)
        return super().open_channel(runtime)


__all__ = ["ALIASES", "GPUS", "TPUS", "Colab"]
