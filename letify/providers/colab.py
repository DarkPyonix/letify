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

import datetime
import json
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import tools
from ..config.secrets import account_directory
from ..declare.instance import Instance
from ..errors import ProviderUnavailable, RuntimeFailure
from .shell import Shell
from .usage import Usage

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

#: Where a Colab VM keeps its GPU driver libraries. Only the notebook kernel's own
#: environment names it, so a worker started over SSH has to be given it or it finds no
#: libnvidia-ml.so and no libcuda.so on a session that holds a GPU. Spec "Colab".
DRIVER_LIBRARY_PATH = "/usr/lib64-nvidia"

#: The compute unit balance, as the Colab web page asks for it.
CCU_INFO_URL = "https://colab.research.google.com/tun/m/ccu-info?authuser=0"

#: Colab prefixes JSON answers with this line to defeat cross-site script inclusion.
XSSI_PREFIX = ")]}'"

#: A balance answers in well under a second, and a person is waiting for the table.
USAGE_HTTP_TIMEOUT = 15.0


def _access_token(stored: dict[str, Any]) -> str:
    """A usable access token from the Colab CLI's token file, refreshed in memory if stale.

    A token with more than a minute left is used as it is. Otherwise the refresh token is
    exchanged at ``token_uri``, and the answer is not written back.
    """
    token = stored.get("token")
    expiry = stored.get("expiry")
    if isinstance(token, str) and isinstance(expiry, str):
        try:
            ends = datetime.datetime.fromisoformat(expiry.removesuffix("Z"))
        except ValueError:
            ends = None
        if ends is not None:
            if ends.tzinfo is None:
                ends = ends.replace(tzinfo=datetime.UTC)
            left = ends - datetime.datetime.now(datetime.UTC)
            if left.total_seconds() > 60:
                return token
    form = urllib.parse.urlencode(
        {
            "client_id": stored["client_id"],
            "client_secret": stored["client_secret"],
            "refresh_token": stored["refresh_token"],
            "grant_type": "refresh_token",
        }
    ).encode()
    request = urllib.request.Request(str(stored["token_uri"]), data=form, method="POST")
    with urllib.request.urlopen(request, timeout=USAGE_HTTP_TIMEOUT) as response:
        return str(json.loads(response.read())["access_token"])


class Colab(Shell):
    """A Colab runtime for one Google account."""

    kind = "colab"
    extra = "colab"
    default_persistence = "ephemeral"
    #: Its machine lives only as long as a session, so utilization is read inside one.
    reads_machine = False

    #: The control path crosses a Google frontend rather than reaching the machine
    #: directly, so forwarding pays a long round trip per synchronization.
    has_fast_path = False

    #: Measured from Seoul to a Colab runtime in the United States.
    expected_round_trip_ms = 175.0

    #: Colab meters in compute units. The CLI has no balance command, so the balance is
    #: read from the endpoint the Colab web page reads, with the CLI's own OAuth token.
    usage_unit = "compute units"
    usage_source = "Colab ccu-info, the balance the Colab web page shows"

    def report_usage(self) -> Usage:
        """Read the compute unit balance and the hourly rate from ``ccu-info``.

        Read-only: the token file is never rewritten, and a refreshed token lives only in
        memory for this one request.
        """
        note: str | None = None
        remaining: float | None = None
        rate: float | None = None
        token_file = account_directory(self.alias) / ".config" / "colab-cli" / "token.json"
        try:
            stored = json.loads(token_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = None
        if not isinstance(stored, dict):
            note = f"not signed in to Colab. Run 'letify login colab {self.alias}'"
        else:
            try:
                token = _access_token(stored)
                request = urllib.request.Request(
                    CCU_INFO_URL,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                        "X-Colab-Client-Agent": "colab-cli",
                        "X-Colab-Tunnel": "Google",
                    },
                )
                with urllib.request.urlopen(request, timeout=USAGE_HTTP_TIMEOUT) as response:
                    text = response.read().decode(errors="replace")
                body = json.loads(text.removeprefix(XSSI_PREFIX))
                balance = body.get("currentBalance")
                hourly = body.get("consumptionRateHourly")
                remaining = float(balance) if isinstance(balance, (int, float)) else None
                rate = float(hourly) if isinstance(hourly, (int, float)) else None
                if remaining is None:
                    note = "ccu-info answered without currentBalance"
            except (OSError, ValueError, KeyError, AttributeError) as exc:
                note = f"ccu-info could not be read: {exc}"
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            remaining=remaining,
            rate_per_hour=rate,
            as_of=time.time(),
            note=note,
        )

    #: The VM is discarded with the session, so the root sits beside Colab's own files.
    default_workspace = "/content/letify"

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
        """Return the fixed Colab accelerator list, plus a CPU instance with no accelerator.

        No connection is needed. Whether an accelerator is free right now is decided
        when a runtime starts, because Colab does not promise availability.
        """
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None)}
        table.update(
            {
                name: Instance(self, gpu=name, vram_gb=spec.get("vram_gb"))
                for name, spec in GPUS.items()
            }
        )
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
            # A line starting with "[colab]" is the CLI's own message, not a session.
            if token and not token[0].lower().startswith(("name", "session", "-", "[colab]")):
                # The CLI prints a session as "[<name>] <id> | Hardware: ...".
                word = token[0]
                if word.startswith("[") and word.endswith("]"):
                    word = word[1:-1]
                names.append(word)
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
        from .colab_files import ColabFiles

        name = runtime.name

        def run(source: str, timeout: float | None) -> str:
            return self._exec(name, source, timeout)

        files = ColabFiles(self.alias, name, run, workspace=self.workspace_root)
        return lambda: OneShotLink("fallback", 4, run, files=files)

    def target(self, runtime: Runtime | None = None) -> Target:
        target = super().target(runtime)
        target.user = self.user or "root"
        if runtime is not None:
            # Each runtime is a new VM with a new host key.
            target.host_key_alias = f"letify-{self.alias}-{runtime.name}"
        return target

    def remote_command(self, command: str) -> str:
        """Name the driver library directory the notebook kernel's environment names.

        A login shell on the VM does not, so without this the worker sees no CUDA device.
        A path the machine already set is kept after it.
        """
        return (
            f"LD_LIBRARY_PATH={DRIVER_LIBRARY_PATH}"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} " + command
        )

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
        from .colab_files import ColabFiles

        if self.channel_kind == "exec":
            name = runtime.name

            def run(source: str, timeout: float | None) -> str:
                return self._exec(name, source, timeout)

            files = ColabFiles(self.alias, name, run, workspace=self.workspace_root)
            return OneShotChannel(run, name=name, files=files)
        return super().open_channel(runtime)


__all__ = ["ALIASES", "DRIVER_LIBRARY_PATH", "GPUS", "TPUS", "Colab"]
