"""Rendezvous, the way to start letify's remote half and swap addresses with it.

Owns sending a request to the remote side and reading its answer. Colab, Kaggle and Elice
fill this role with their provider layer, which runs the remote half as a program; a plain
machine has none. It does not own what a request asks for; the strategies do.
"""

from __future__ import annotations

import abc
import json
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


def _answer(output: str) -> dict[str, Any]:
    for line in output.splitlines():
        if line.startswith(nat.ANSWER_MARKER):
            answer = json.loads(line[len(nat.ANSWER_MARKER) :])
            if "error" in answer:
                raise OSError(answer["error"])
            return answer
    raise OSError(f"the remote half gave no answer: {output.strip()[-500:]}")


class CommandRendezvous(Rendezvous):
    """A rendezvous that can run a Python program on the remote machine.

    Because it can, ``letify.transport.linkchannel`` can also take the letify-link sidecar
    out of the wheel on the machine through it.
    """

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


__all__ = [
    "ColabRendezvous",
    "CommandRendezvous",
    "Rendezvous",
    "ShellCommandRendezvous",
    "remote_script",
]
