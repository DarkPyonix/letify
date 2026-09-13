"""Link, an established connection to a remote machine.

Owns what a connection offers once a strategy has made it: an SSH command line for the
worker channel and for bulk transfer, a stream the probe can run over, and closing. It
does not own how the connection was made or whether it is the one to use.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from ..errors import RuntimeFailure
from . import nat


def probe_source() -> str:
    """The probe responder as a program, run on the remote side over an SSH command."""
    return Path(nat.__file__).read_text(encoding="utf-8") + "\nserve_probe(PipeStream())\n"


class Link:
    """A connection made by one strategy."""

    persistent = True

    def __init__(self, strategy: str, rank: int):
        self.strategy = strategy
        self.rank = rank

    def probe_stream(self) -> object | None:
        """A stream the probe can run over, or None when this link cannot carry it."""
        return None

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        raise RuntimeFailure(f"the {self.strategy} link carries no SSH session")

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.strategy} rank {self.rank}>"


class ProcessStream:
    """A child process's pipes as a stream, for the probe over an SSH command."""

    def __init__(self, command: list[str]):
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )

    def recv(self, size: int) -> bytes:
        return os.read(self.process.stdout.fileno(), size)  # type: ignore[union-attr]

    def sendall(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = self.process.stdin.write(view)  # type: ignore[union-attr]
            view = view[written or 0 :]

    def close(self) -> None:
        try:
            self.process.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass
        self.process.terminate()


class SSHLink(Link):
    """A link that is an SSH command line: forward SSH, SSH over Tailcat, or a reverse forward."""

    def __init__(
        self,
        strategy: str,
        rank: int,
        command: Callable[[str | None], list[str]],
        *,
        remote_python: str = "python3",
        on_close: Callable[[], None] | None = None,
    ):
        super().__init__(strategy, rank)
        self._command = command
        self.remote_python = remote_python
        self._on_close = on_close
        self._streams: list[ProcessStream] = []

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        return self._command(remote_command)

    def probe_stream(self) -> ProcessStream:
        program = f"{self.remote_python} -u -c {shlex.quote(probe_source())}"
        stream = ProcessStream(self._command(program))
        self._streams.append(stream)
        return stream

    def close(self) -> None:
        for stream in self._streams:
            stream.close()
        self._streams.clear()
        if self._on_close is not None:
            callback, self._on_close = self._on_close, None
            callback()


class PunchedLink(Link):
    """A punched TCP connection. It carries the probe, then SSH through a local forwarding port.

    The first SSH command uses the punched connection. Each later one punches again, because
    one TCP connection carries one SSH session.
    """

    def __init__(
        self,
        strategy: str,
        rank: int,
        sock: socket.socket,
        command_for_port: Callable[[int, str | None], list[str]],
        redial: Callable[[], socket.socket],
    ):
        super().__init__(strategy, rank)
        self._sock: socket.socket | None = sock
        self._command_for_port = command_for_port
        self._redial = redial
        self._lock = threading.Lock()
        self._listeners: list[socket.socket] = []

    def probe_stream(self) -> socket.socket | None:
        return self._sock

    def _take(self) -> socket.socket:
        with self._lock:
            sock, self._sock = self._sock, None
        return sock if sock is not None else self._redial()

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Port 0: the operating system chooses, because Windows reserves ranges per machine.
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self._listeners.append(listener)
        port = listener.getsockname()[1]

        def forward() -> None:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            finally:
                listener.close()
            remote = self._take()
            remote.sendall(b"B")
            nat.splice(conn, remote)

        threading.Thread(target=forward, daemon=True).start()
        return self._command_for_port(port, remote_command)

    def close(self) -> None:
        for listener in self._listeners:
            listener.close()
        with self._lock:
            sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()


class OneShotLink(Link):
    """A provider's own command path, which runs one program per call and keeps nothing."""

    persistent = False

    def __init__(
        self,
        strategy: str,
        rank: int,
        runner: Callable[[str, float | None], str],
        *,
        files: object | None = None,
    ):
        super().__init__(strategy, rank)
        self.runner = runner
        #: The provider's bulk transfer path, when it has one beside the command path.
        self.files = files


__all__ = ["Link", "OneShotLink", "ProcessStream", "PunchedLink", "SSHLink", "probe_source"]
