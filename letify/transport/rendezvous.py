"""Rendezvous, the way to start letify's remote half and swap addresses with it.

Owns sending a request to the remote side and reading its answer. Colab and Elice fill
this role with their provider layer, which runs the remote half as a program; a plain
machine behind NAT answers through the remote agent over Tailcat. It does not own what a
request asks for; the strategies do.
"""

from __future__ import annotations

import abc
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import nat


def remote_script(request: dict[str, Any]) -> str:
    """The standard library module plus one call that answers ``request`` and stays running."""
    source = Path(nat.__file__).read_text(encoding="utf-8")
    return source + f"\nrun_detached({json.dumps(request)!r}, {source!r})\n"


class Rendezvous(abc.ABC):
    """Reaches the remote half of a strategy."""

    #: Seconds between sending a punch request and the agreed start time.
    lead_seconds: float = 10.0

    def unavailable(self) -> str | None:
        """Why this rendezvous cannot be used right now, or None when it can."""
        return None

    @abc.abstractmethod
    def exchange(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send one request and return the remote side's answer."""

    def tailcat_endpoint(self, ssh_port: int, timeout: float) -> tuple[str, int]:
        """A Tailcat address and port that reach the machine's SSH server."""
        answer = self.exchange({"kind": "tailcat", "ssh_port": ssh_port}, timeout)
        return str(answer["address"]), ssh_port


def _answer(output: str) -> dict[str, Any]:
    for line in output.splitlines():
        if line.startswith(nat.ANSWER_MARKER):
            answer = json.loads(line[len(nat.ANSWER_MARKER) :])
            if "error" in answer:
                raise OSError(answer["error"])
            return answer
    raise OSError(f"the remote half gave no answer: {output.strip()[-500:]}")


class CommandRendezvous(Rendezvous):
    """A rendezvous that can run a Python program on the remote machine."""

    def extras(self) -> dict[str, Any]:
        """Fields this provider adds to every request."""
        return {}

    @abc.abstractmethod
    def run_python(self, source: str, timeout: float) -> str:
        """Run ``source`` remotely and return what it printed."""

    def exchange(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        return _answer(self.run_python(remote_script({**self.extras(), **request}), timeout))


class ColabRendezvous(CommandRendezvous):
    """``colab exec`` on one runtime. A Colab VM has no SSH server, so each request starts one."""

    def __init__(self, run: Callable[[str, float], str], public_key: str | None = None):
        self._run = run
        self.public_key = public_key

    def unavailable(self) -> str | None:
        # The VM authorizes the account's public key; without one SSH cannot log in.
        return None if self.public_key else "no key"

    def extras(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"start_sshd": True}
        if self.public_key:
            extra["authorized_key"] = self.public_key
        return extra

    def run_python(self, source: str, timeout: float) -> str:
        return self._run(source, timeout)


class ShellCommandRendezvous(CommandRendezvous):
    """A Python program over forward SSH, for a machine a provider API has already opened."""

    def __init__(self, ssh: Callable[[str | None], list[str]], python: str = "python3"):
        self.ssh = ssh
        self.python = python

    def run_python(self, source: str, timeout: float) -> str:  # pragma: no cover - live ssh
        result = subprocess.run(
            self.ssh(f"{self.python} -"),
            input=source,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout + result.stderr


class TailcatRendezvous(Rendezvous):
    """The remote agent started by ``letify client shell connect``, reached over Tailcat."""

    lead_seconds = 5.0

    def __init__(self, address: str, port: int, binary: str = "tailcat"):
        self.address = address
        self.port = port
        self.binary = binary

    def unavailable(self) -> str | None:
        return None if shutil.which(self.binary) else f"{self.binary} is not on PATH"

    def tailcat_endpoint(self, ssh_port: int, timeout: float) -> tuple[str, int]:
        # The agent splices a connection that opens with SSH- to the SSH server itself.
        return self.address, self.port

    def exchange(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        process = subprocess.Popen(
            [self.binary, self.address, str(self.port)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            process.stdin.write("LETIFY-RDV " + json.dumps(request) + "\n")  # type: ignore[union-attr]
            process.stdin.flush()  # type: ignore[union-attr]
            line = process.stdout.readline()  # type: ignore[union-attr]
        finally:
            process.terminate()
        if not line.strip():
            raise OSError(f"the agent at {self.address} did not answer")
        answer = json.loads(line)
        if "error" in answer:
            raise OSError(answer["error"])
        return answer


__all__ = [
    "ColabRendezvous",
    "CommandRendezvous",
    "Rendezvous",
    "ShellCommandRendezvous",
    "TailcatRendezvous",
    "remote_script",
]
