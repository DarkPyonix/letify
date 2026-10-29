"""The worker channel over a letify-link session, with ssh as the bootstrap only.

Owns connecting to a machine through one ssh command, the remote shim of ``linkshim.py``
and a ``letify-link`` session, and the channel class that speaks the framed worker protocol on
a stream of that session. It does not own the sidecar protocol (``linkd.py``), the choice of a
transport (the provider reads the account option) or the worker program. Spec "Sidecar
interface" item 7.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import ProviderUnavailable, RuntimeFailure
from ..protocol import wire
from ..runtime.channel import FramedChannel
from . import linkd

#: Seconds the remote gets to download the wheel and take ``letify-link`` out of it.
PREPARE_TIMEOUT = 600.0
#: Seconds the shim gets to start the remote sidecar and print its ready line.
SHIM_TIMEOUT = 60.0
#: Seconds the remote shim waits for a session after the hand-over before it leaves.
WAIT_SESSION_S = 120
#: Seconds the remote shim keeps a session with no live path before it leaves.
RESUME_S = 900


def shim_source() -> str:
    """The text of ``linkshim.py``, which the machine runs with its own Python."""
    return (Path(__file__).with_name("linkshim.py")).read_text(encoding="utf-8")


class LinkRemote:
    """One connected machine: the local sidecar, the session, and the ssh command that ended."""

    def __init__(self, daemon: linkd.LinkDaemon, session: linkd.LinkSession, alias: str):
        self.daemon = daemon
        self.session = session
        self.alias = alias

    def open_stream(self) -> socket.socket:
        """One more stream of the session. No punch and no ssh connection."""
        try:
            return self.session.open_stream()
        except linkd.LinkError as error:
            raise RuntimeFailure(
                f"{self.alias}: cannot open a letify-link stream: {error}"
            ) from error

    @property
    def rtt_ms(self) -> float | None:
        """The round trip the session measured on its best path, or None before one is known."""
        try:
            rtts = [p["srtt_ms"] for p in self.session.paths() if p.get("srtt_ms")]
        except linkd.LinkError:
            return None
        return min(rtts) if rtts else None

    @property
    def alive(self) -> bool:
        return not self.session.closed.is_set()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.session.close()
        with contextlib.suppress(Exception):
            self.daemon.close()


class _StdioRendezvous:
    """Carries the sidecar's rendezvous messages to the shim as JSON lines on ssh's stdin."""

    def __init__(self, daemon_id: Callable[[], str], write: Callable[[dict[str, Any]], None]):
        self._daemon_id = daemon_id
        self._write = write

    def send(self, to: str, message: bytes) -> None:
        self._write({"op": "rendezvous", "from": self._daemon_id(), "message": message.hex()})


def _fail(alias: str, reason: str) -> ProviderUnavailable:
    return ProviderUnavailable("shell", f"{alias} cannot connect over letify-link: {reason}")


def prepare_binary(
    alias: str, ssh: Callable[[str], list[str]], python: str, version: str, given: str | None
) -> str:
    """The path of ``letify-link`` on the machine: the account's, or taken from the wheel."""
    if given:
        return given
    from .. import install

    source = install.remote_link_source(version)
    command = ssh(f"{python} -c {shlex.quote(source)}")
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=PREPARE_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _fail(alias, f"the ssh bootstrap could not prepare letify-link: {error}") from error
    for line in done.stdout.splitlines():
        if line.startswith("LETIFY-LINK "):
            return str(json.loads(line[len("LETIFY-LINK ") :]))
    raise _fail(
        alias,
        "letify-link is not on the machine and the wheel could not supply it: "
        + (done.stderr or done.stdout).strip()[-400:],
    )


def connect(
    alias: str,
    *,
    ssh: Callable[[str], list[str]],
    address: str,
    python: str,
    bootstrap: str,
    key_path: str | os.PathLike[str],
    version: str,
    binary: str | None = None,
    punch: bool = False,
    max_rtt_ms: float | None = None,
    min_bps: float | None = None,
    start_timeout_ms: int = 60_000,
) -> LinkRemote:
    """Bootstrap over ssh once, start a session, and return the connected machine.

    Spec "Sidecar interface" item 7. Every failure raises ``ProviderUnavailable`` with the
    reason; nothing falls back to the ssh data path.
    """
    try:
        daemon = linkd.LinkDaemon(key_path)
    except linkd.LinkMissing as error:
        raise _fail(alias, str(error)) from error
    except linkd.LinkError as error:
        raise _fail(alias, f"the local sidecar failed: {error}") from error
    proc: subprocess.Popen[bytes] | None = None
    try:
        remote_binary = prepare_binary(alias, ssh, python, version, binary)
        proc = subprocess.Popen(
            ssh(f"{python} -u -c {shlex.quote(shim_source())}"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        stdin_lock = threading.Lock()

        def write(message: dict[str, Any]) -> None:
            with stdin_lock, contextlib.suppress(OSError, ValueError):
                assert proc is not None and proc.stdin is not None
                proc.stdin.write(json.dumps(message).encode() + b"\n")
                proc.stdin.flush()

        write(
            {
                "binary": remote_binary,
                "allow": daemon.node_id,
                "bind": "0.0.0.0:0",
                "python": python,
                "bootstrap": bootstrap,
                "wait_s": WAIT_SESSION_S,
                "resume_s": RESUME_S,
            }
        )
        lines = _Lines(proc)
        ready = lines.wait("ready", SHIM_TIMEOUT, alias, proc)
        remote_id = str(ready["node_id"])
        daemon.rendezvous = _StdioRendezvous(lambda: daemon.node_id, write)
        lines.on_rendezvous = lambda to, message: daemon.deliver(remote_id, message)
        candidates: list[Any] = [f"{address}:{ready['port']}"]
        if punch:
            write({"op": "punch", "peer": daemon.node_id, "sockets": 4, "public_ip": address})
            names = daemon.punch(remote_id, "initiator", sockets=4, public_ip=address)
            candidates += [{"name": f"punch{i}", "punched": n} for i, n in enumerate(names)]
        session = daemon.open_session(
            remote_id, candidates, config={"start_timeout_ms": start_timeout_ms}
        )
        remote = LinkRemote(daemon, session, alias)
        _hold_to_floor(remote, max_rtt_ms, min_bps)
        write({"op": "handover"})
        lines.wait("handed_over", SHIM_TIMEOUT, alias, proc)
        # The shim now runs without ssh, so the ssh command is ended rather than waited for.
        with contextlib.suppress(OSError, ValueError):
            proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=10)
        return remote
    except linkd.LinkError as error:
        daemon.close()
        raise _fail(alias, str(error)) from error
    except BaseException:
        daemon.close()
        raise
    finally:
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()


def _hold_to_floor(remote: LinkRemote, max_rtt_ms: float | None, min_bps: float | None) -> None:
    """Spec "Link floor": the session's measured path is held to the account's floor."""
    found: list[str] = []
    rtt = remote.rtt_ms
    if max_rtt_ms is not None and rtt is not None and rtt > max_rtt_ms:
        found.append(f"round trip {rtt:.1f} ms above the floor {max_rtt_ms:.0f} ms")
    if min_bps is not None:
        while not remote.session.events.empty():
            event = remote.session.events.get_nowait()
            slow = (
                event.get("event") == "path_probe"
                and (event.get("throughput") or min_bps) < min_bps
            )
            if slow:
                found.append(
                    f"throughput {event['throughput'] / (1 << 20):.1f} MiB/s below the "
                    f"floor {min_bps / (1 << 20):.1f} MiB/s"
                )
    if found:
        remote.close()
        raise _fail(
            remote.alias,
            "the session's path is below the link floor: "
            + "; ".join(found)
            + ". Set min_mib_per_s and max_rtt_ms on the account to change the floor",
        )


class _Lines:
    """Reads the shim's JSON lines on a thread, so a waiting caller can time out."""

    def __init__(self, proc: subprocess.Popen[bytes]):
        import queue

        self.on_rendezvous: Callable[[str, bytes], None] = lambda to, message: None
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._proc = proc
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if event.get("event") == "rendezvous":
                with contextlib.suppress(Exception):
                    self.on_rendezvous(event["to"], bytes.fromhex(event["message"]))
            else:
                self._queue.put(event)
        self._queue.put(None)

    def wait(
        self, name: str, timeout: float, alias: str, proc: subprocess.Popen[bytes]
    ) -> dict[str, Any]:
        import queue

        deadline = time.monotonic() + timeout
        while True:
            try:
                event = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise _fail(alias, f"the remote shim said nothing in {timeout:.0f} s") from None
            if event is None:
                stderr = b""
                if proc.stderr is not None:
                    with contextlib.suppress(OSError, ValueError):
                        stderr = proc.stderr.read()
                raise _fail(
                    alias,
                    "the remote shim exited before it was ready: "
                    + stderr.decode("utf-8", "replace").strip()[-400:],
                )
            if event.get("event") == "error":
                raise _fail(alias, str(event.get("error")))
            if event.get("event") == name:
                return event


class LinkChannel(FramedChannel):
    """The worker channel: framed requests on one letify-link stream.

    The same interface as ``PersistentChannel``. The remote end of the stream is the standard
    input and output of a worker process the shim started, so the first bytes are the
    bootstrap source and everything after is frames.
    """

    def __init__(self, remote: LinkRemote, *, name: str = "runtime"):
        self.remote = remote
        self.name = name
        self._socket: socket.socket | None = None
        self._start_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._socket is not None:
                return
            sock = self.remote.open_stream()
            self._socket = sock
            self._connection = None
            from ..runtime.channel import Connection

            self._connection = Connection(
                self.name, lambda view: sock.send(view), sock.recv_into, self._emit
            )
            self._send_worker()
            self._await_ready()

    def _kill(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)

    def _startup_failure(self, expired: bool, cause: Exception) -> Exception:
        if expired:
            return super()._startup_failure(expired, cause)
        return RuntimeFailure(
            f"{self.name}: the worker exited before it was ready, or the letify-link "
            f"session closed ({self.remote.session.close_reason or 'no reason given'})"
        )

    def close(self) -> None:
        sock, self._socket = self._socket, None
        connection, self._connection = self._connection, None
        if sock is None:
            return
        try:
            if connection is not None:
                with contextlib.suppress(OSError):
                    connection.sender.frame(wire.SHUTDOWN, 0)
        finally:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()

    def detach(self) -> None:
        """Close the stream without a shutdown frame, which the worker would take as an end."""
        sock, self._socket = self._socket, None
        self._connection = None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()

    @property
    def alive(self) -> bool:
        return self._socket is not None and self.remote.alive


__all__ = ["LinkChannel", "LinkRemote", "connect", "prepare_binary", "shim_source"]
