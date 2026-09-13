"""Tunnel, a Shell that builds a network path before connecting.

For a machine behind NAT or a firewall that cannot accept an inbound connection,
which is the normal situation for a lab server on a campus network. The tunnel only
creates the path; commands still run over SSH through the parent class.

Tailscale is the default. It is the only candidate that needs no server of your
own, it authenticates from an auth key with no prompt, it carries any TCP port
because it is a layer 3 tunnel, and it relays over TCP 443 when UDP is blocked
instead of failing.

frp is the fallback for the case that relay creates. A relayed Tailscale path keeps
working but slowly: 2.2 Mbit/s has been measured across continents where a direct
path expected 30 to 40 Mbit/s, because the relay servers limit throughput for
fairness. frp runs over TLS on port 443 and is the easiest of the candidates to
self-host.

Two settings are requirements learned from the alternatives rather than preferences.
MTU is held at 1280 to 1400, because every mesh VPN in this class shows the same
failure otherwise: the connection works, small commands work, and bulk transfers
stall silently. And the tunnel is the last resort, after a direct address and a jump
host, because those need no setup at all.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Literal

from ..errors import ProviderUnavailable
from .shell import Shell

Transport = Literal["tailscale", "frp", "none"]

#: Safe ceiling for a path that crosses the public internet. Raising it is only
#: worth doing on a link you have measured.
DEFAULT_MTU = 1280


class Tunnel(Shell):
    """A machine reached through a tunnel, then over SSH."""

    kind = "tunnel"
    extra = "shell"
    default_persistence = "ephemeral"
    has_fast_path = True

    _connected: bool = False

    @property
    def transport(self) -> Transport:
        return str(self.config.option("transport", "tailscale"))  # type: ignore[return-value]

    @property
    def mtu(self) -> int:
        value = self.config.option("mtu", DEFAULT_MTU)
        return int(value) if isinstance(value, (int, str)) else DEFAULT_MTU

    def connect(self) -> None:
        """Bring the path up once per process."""
        if self._connected or self.transport == "none":
            self._connected = True
            return
        if self.transport == "tailscale":
            self._connect_tailscale()
        elif self.transport == "frp":
            self._connect_frp()
        else:
            raise ProviderUnavailable(self.kind, f"unknown transport {self.transport!r}")
        self._connected = True

    # -- tailscale -----------------------------------------------------------

    def _connect_tailscale(self) -> None:
        binary = str(self.config.option("tailscale_binary", "tailscale"))
        if not shutil.which(binary):
            raise ProviderUnavailable(
                self.kind,
                f"the {binary!r} command is not on PATH. Install Tailscale on this "
                f"machine and on {self.alias}",
            )
        if self._tailscale_is_up(binary):
            return
        auth_key = self.config.secret("auth_key")
        if not auth_key:
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} needs a Tailscale auth key. Set auth_key_env or "
                f"auth_key_keyring in the configuration and keep the key itself out "
                f"of tracked files",
            )
        command = [binary, "up", f"--auth-key={auth_key}"]
        login_server = self.config.option("login_server")
        if isinstance(login_server, str):
            # A self-hosted control plane such as Headscale, which removes the
            # ephemeral-minute limit on Tailscale's own free plan.
            command.append(f"--login-server={login_server}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise ProviderUnavailable(self.kind, f"tailscale up failed: {result.stderr.strip()}")

    def _tailscale_is_up(self, binary: str) -> bool:
        try:
            result = subprocess.run(
                [binary, "status", "--json"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        compact = result.stdout.replace(" ", "")
        return result.returncode == 0 and '"BackendState":"Running"' in compact

    # -- frp -----------------------------------------------------------------

    def _connect_frp(self) -> None:
        binary = str(self.config.option("frp_binary", "frpc"))
        config_path = self.config.option("frp_config")
        if not shutil.which(binary):
            raise ProviderUnavailable(
                self.kind, f"the {binary!r} command is not on PATH", self.extra
            )
        if not isinstance(config_path, str):
            raise ProviderUnavailable(
                self.kind, f"{self.alias} needs an frp_config path in the configuration"
            )
        # frpc stays alive for the life of the path, so it runs detached and the local
        # port it publishes is what SSH then connects to.
        subprocess.Popen(
            [binary, "-c", config_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # -- diagnosis -----------------------------------------------------------

    def diagnose(self) -> dict[str, object]:
        """Report what the path looks like right now.

        Run this when bulk transfers stall. A relayed path and an MTU above 1400 are
        the two usual causes, and this names which one applies.
        """
        report: dict[str, object] = {"transport": self.transport, "mtu": self.mtu}
        if self.transport != "tailscale":
            return report
        binary = str(self.config.option("tailscale_binary", "tailscale"))
        try:
            result = subprocess.run([binary, "status"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["error"] = str(exc)
            return report
        report["relayed"] = "relay" in result.stdout.lower()
        report["status"] = result.stdout.strip()[:2000]
        return report


__all__ = ["DEFAULT_MTU", "Transport", "Tunnel"]
