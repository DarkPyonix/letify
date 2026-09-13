"""The remote agent that ``letify client shell connect`` starts on a plain machine behind NAT.

Owns one listening port, chosen by the operating system, with ``tailcat serve`` in front
of it. A connection that opens with ``SSH-`` is spliced to the machine's SSH server, and
one that opens with ``LETIFY-RDV `` carries a single rendezvous request answered by
``nat.begin``. It does not own the user's side of the pipeline.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import threading
import time

from . import nat

REQUEST_PREFIX = b"LETIFY-RDV "
SSH_PREFIX = b"SSH-"


class Agent:
    """Answers rendezvous requests and carries SSH for one machine."""

    def __init__(self, ssh: tuple[str, int] = ("127.0.0.1", 22), tailcat: str = "tailcat"):
        self.ssh = tuple(ssh)
        self.tailcat = tailcat
        self.port = 0
        self._sock: socket.socket | None = None
        self._process: subprocess.Popen | None = None

    def bind(self) -> int:
        """Listen on a port the operating system chooses, and return it."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        self._sock = sock
        self.port = sock.getsockname()[1]
        return self.port

    def start_tailcat(self) -> str:
        """Start ``tailcat serve`` in front of the agent and return the address it prints."""
        if shutil.which(self.tailcat) is None:
            raise FileNotFoundError(f"{self.tailcat} is not on PATH; install Tailcat first")
        self._process = subprocess.Popen(
            [self.tailcat, "serve", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in self._process.stdout:  # type: ignore[union-attr]
            found = re.search(r"address:\s*(tc\S+)", line)
            if found:
                return found.group(1)
        raise RuntimeError("tailcat serve exited without printing its address")

    def serve_forever(self) -> None:
        while self._sock is not None:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn: socket.socket) -> None:
        deadline = time.monotonic() + 10.0
        while True:
            try:
                head = conn.recv(len(REQUEST_PREFIX), socket.MSG_PEEK)
            except OSError:
                head = b""
            if not head:
                conn.close()
                return
            if head.startswith(SSH_PREFIX):
                nat.splice(conn, socket.create_connection(self.ssh))
                return
            if head == REQUEST_PREFIX:
                break
            partial = REQUEST_PREFIX.startswith(head) or SSH_PREFIX.startswith(head)
            if not partial or time.monotonic() > deadline:
                conn.close()
                return
            time.sleep(0.01)

        reader = conn.makefile("rb")
        reader.read(len(REQUEST_PREFIX))
        try:
            answer, continuation = nat.begin(json.loads(reader.readline()))
            threading.Thread(target=continuation, daemon=True).start()
        except Exception as exc:
            answer = {"error": f"{type(exc).__name__}: {exc}"}
        conn.sendall((json.dumps(answer) + "\n").encode())
        reader.close()
        conn.close()

    def close(self) -> None:
        if self._sock is not None:
            sock, self._sock = self._sock, None
            sock.close()
        if self._process is not None:
            self._process.terminate()


__all__ = ["Agent"]
