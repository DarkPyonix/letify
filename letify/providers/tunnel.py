"""Tunnel, a plain machine behind NAT with no provider API.

The machine's user runs ``letify agent serve`` on it once. An account that names an
``agent`` (spec "Agent and pairing" item 11) is reached through the signalling service and
a letify-link session, with no ssh. The agent machine's sshd is reachable through
``ssh_forward`` when the user wants it. An account that names ``tailcat`` is refused when
the provider is built (spec "Agent and pairing" item 14).
"""

from __future__ import annotations

import subprocess
from typing import Any

from ..errors import ProviderUnavailable, RuntimeFailure
from .shell import Shell


class Tunnel(Shell):
    """A machine behind NAT, reached through the remote agent."""

    kind = "tunnel"
    extra = "shell"
    default_persistence = "ephemeral"
    has_fast_path = True
    #: Listing its workers would run the connection pipeline. Spec "Workers on a shell host".
    discovers_sessions = False

    @property
    def agent(self) -> str | None:
        """The node id of the agent this account is paired with, or None."""
        value = self.config.option("agent")
        return value if isinstance(value, str) and value else None

    @property
    def address(self) -> str | None:
        return None if self.agent else super().address

    @property
    def uses_letify_link(self) -> bool:
        """An agent account is always a letify-link account; ``link = "ssh"`` is refused."""
        if self.agent is None:
            return super().uses_letify_link
        if self.config.option("link", "letify-link") not in (None, "letify-link"):
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} names an agent and has no ssh server to reach; "
                f"remove link = {self.config.option('link')!r}",
            )
        return True

    def link(self, runtime: Any = None) -> Any:
        """The session's measurements with no ssh command, on an agent account."""
        if self.agent is None:
            return super().link(runtime)
        from ..transport.linkchannel import PlainSshLink

        links: dict[str, Any] = self.__dict__.setdefault("_links", {})
        key = self._link_key(runtime)
        if key not in links:
            remotes = self.__dict__.setdefault("_link_remotes", {})
            links[key] = PlainSshLink(None, lambda: remotes.get(key))
        return links[key]

    def connect_link(self, runtime: Any = None) -> Any:
        """A letify-link session bootstrapped through the agent and the signalling service."""
        if self.agent is None:
            return super().connect_link(runtime)
        import letify

        from ..config.secrets import account_directory
        from ..protocol.worker import BOOTSTRAP
        from ..transport import linkchannel
        from ..transport.announce import printer

        signal = self.config.option("signal")
        if not (isinstance(signal, str) and signal):
            raise ProviderUnavailable(
                self.kind, f"{self.alias} names an agent but no signal URL; log in again"
            )
        directory = account_directory(self.alias)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        floor = self.link_floor
        return linkchannel.connect_signal(
            self.alias,
            agent=self.agent,
            signal=signal,
            python=self.remote_python,
            bootstrap=BOOTSTRAP,
            key_path=directory / "link.key",
            version=letify.__version__,
            max_rtt_ms=floor.max_rtt_ms,
            min_bps=floor.min_bps,
            say=printer(self.announce),
        )

    def ssh_forward(self, runtime: Any = None, port: int = 0) -> Any:
        """A loopback port that reaches the agent machine's sshd, on an agent account.

        Spec "Sidecar interface" item 11. Connect with ``ssh -p <forward.port> <user>@127.0.0.1``.
        """
        if self.agent is None:
            raise ProviderUnavailable(
                self.kind, f"{self.alias} names no agent, so it has no session to forward over"
            )
        return self.link_remote(runtime).forward_ssh(port)

    def run_command(self, command: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        """Run one shell command on the machine, over the session on an agent account.

        Login uses it for the workspace check and the GPU query, which need no ssh.
        """
        return self._run_helper(["sh", "-c", command], timeout)

    def _run_helper(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        if self.agent is not None:
            self.link_remote(None)
        return super()._run_helper(argv, timeout)

    def check(self) -> str:
        """Confirm the machine answers over the session and its workspace root is writable."""
        if self.agent is None:
            return super().check()
        from ..runtime.bootstrap import workspace_check

        root = self.workspace_root
        script = (
            "uname -a; nvidia-smi --query-gpu=name --format=csv,noheader; "
            f"if letify_out=$({workspace_check(root)} 2>&1); then echo letify-workspace-ok; "
            'else echo "letify-workspace-failed: $letify_out"; fi'
        )
        done = self._run_helper(["sh", "-c", script], 120)
        if done.returncode != 0:
            raise RuntimeFailure(
                f"{self.alias} did not answer over letify-link", stderr=done.stderr.strip()
            )
        lines = []
        for line in done.stdout.splitlines():
            if line.strip() == "letify-workspace-ok":
                lines.append(f"workspace {root}: writable")
            elif line.startswith("letify-workspace-failed:"):
                lines.append(f"workspace {root}: not writable: {line.split(':', 1)[1].strip()}")
            else:
                lines.append(line)
        return "\n".join(lines) + "\n"


__all__ = ["Tunnel"]
