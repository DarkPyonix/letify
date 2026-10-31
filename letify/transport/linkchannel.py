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
import math
import os
import secrets
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ..errors import ProviderUnavailable, RuntimeFailure
from ..protocol import wire
from ..runtime.channel import FramedChannel, relay_program
from ..runtime.lease import GRACE
from . import linkd

#: Seconds the remote gets to download the wheel and take ``letify-link`` out of it.
PREPARE_TIMEOUT = 600.0
#: Seconds the shim gets to start the remote sidecar and print its ready line.
SHIM_TIMEOUT = 60.0
#: TCP and UDP sockets a punch opens, and the policing check each punched path passes: a floor of
#: 1 MiB/s and bursts of 500 ms. Spec "Sidecar interface" item 3.
PUNCH_TCP = 4
PUNCH_UDP = 1
POLICE = {"floor_bps": 1 << 20, "burst_ms": 500}
#: The punch window of a first attempt over a command channel, in milliseconds, and the factor
#: the retry's window is longer by. Spec "Sidecar interface" item 8.
PUNCH_WINDOW_MS = 20_000
RETRY_WINDOW_FACTOR = 3
#: The least lead a punch gets, in milliseconds, and how many bootstrap round trips it is.
PUNCH_LEAD_FLOOR_MS = 2000
PUNCH_LEAD_ROUND_TRIPS = 3
#: Seconds the remote shim waits for a session after the hand-over before it leaves.
WAIT_SESSION_S = 120
#: Seconds a session keeps its place with no live path. It is the lease of spec "Sessions",
#: so a session never outlives what the lease allows. The shim leaves after the same time.
RESUME_S = int(GRACE)
#: Lines of a worker's standard error the shim returns on request.
STDERR_LINES = 40
#: Seconds to wait for the shim to answer a request for a worker's standard error.
STDERR_WAIT_S = 5.0


def shim_path() -> Path:
    """The file of ``linkshim.py``."""
    return Path(__file__).with_name("linkshim.py")


def shim_source() -> str:
    """The text of ``linkshim.py``, which the machine runs with its own Python."""
    return shim_path().read_text(encoding="utf-8")


def stream_header(kind: str, token: str, **fields: Any) -> bytes:
    """The one line of JSON that opens a stream. Spec "Sidecar interface" item 7."""
    return json.dumps({"kind": kind, "token": token, **fields}).encode() + b"\n"


def punch_lead_ms(rtt_s: float) -> int:
    """The lead of a punch over a command channel: the larger of 2 s and 3 round trips.

    Spec "Sidecar interface" item 8. A punch offer and its answer cross the channel once each,
    and each crossing takes at most one round trip, so 3 leave a round trip of margin.
    """
    return max(PUNCH_LEAD_FLOOR_MS, math.ceil(round(PUNCH_LEAD_ROUND_TRIPS * rtt_s * 1000, 6)))


class Bootstrap(Protocol):
    """The line stream between the local side and the shim until the hand-over.

    An ssh command is one implementation (``PopenBootstrap``) and a provider's command channel is
    the other (``linkmailbox.MailboxBootstrap``). ``rtt_s`` is None for a stream that is not a
    rendezvous cost, and the measured round trip in seconds for one that is.
    """

    rtt_s: float | None

    def send_line(self, line: bytes) -> None: ...

    def recv_line(self) -> bytes | None:
        """The next line, blocking, or None once the stream has ended."""
        ...

    def stderr_text(self) -> str: ...

    def close(self) -> None: ...


class PopenBootstrap:
    """The shim's standard input and output as an ssh command's."""

    rtt_s = None

    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._lock = threading.Lock()

    def send_line(self, line: bytes) -> None:
        with self._lock, contextlib.suppress(OSError, ValueError):
            assert self.proc.stdin is not None
            self.proc.stdin.write(line.rstrip(b"\n") + b"\n")
            self.proc.stdin.flush()

    def recv_line(self) -> bytes | None:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        return line or None

    def stderr_text(self) -> str:
        stderr = b""
        if self.proc.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                stderr = self.proc.stderr.read()
        return stderr.decode("utf-8", "replace")

    def close(self) -> None:
        """End the command: the shim runs without it after the hand-over."""
        if self.proc.poll() is not None:
            return
        with contextlib.suppress(OSError, ValueError):
            assert self.proc.stdin is not None
            self.proc.stdin.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self.proc.kill()


def letify_link_enabled(provider: Any) -> bool:
    """Whether the account's ``link`` option is ``"letify-link"``; ``"ssh"`` or absent is not.

    Spec "Sidecar interface" item 7. Any other value is refused by name.
    """
    value = provider.config.option("link", "ssh")
    if value in (None, "ssh"):
        return False
    if value == "letify-link":
        return True
    raise ProviderUnavailable(
        provider.kind, f'{provider.alias} sets link = {value!r}; use "ssh" or "letify-link"'
    )


class PlainSshLink:
    """The ssh command of an account that carries its channel over letify-link, if it has one.

    Owns the ssh command for the commands that are not the data path (a card query, a worker
    stop) and the numbers the transfer decision reads, which are the session's. It runs no
    strategy race and no probe. Spec "Sidecar interface" item 7.
    """

    persistent = True
    strategy = "direct_ssh"
    rank = 0
    note: str | None = None

    def __init__(
        self, command: Callable[[str | None], list[str]] | None, remote: Callable[[], Any]
    ):
        self._command = command
        self._remote = remote

    def ssh_command(self, remote_command: str | None = None) -> list[str]:
        if self._command is None:
            raise RuntimeFailure("the letify-link session carries no SSH session")
        return self._command(remote_command)

    @property
    def rtt_ms(self) -> float | None:
        remote = self._remote()
        return remote.rtt_ms if remote is not None else None

    @property
    def upload_bps(self) -> float | None:
        remote = self._remote()
        return remote.upload_bps if remote is not None else None

    def spare_connections(self) -> int:
        return 0

    def probe_stream(self) -> object | None:
        return None

    def close(self) -> None:
        return None


class LinkRemote:
    """One connected machine: the local sidecar, the session, and the ssh command that ended."""

    def __init__(
        self,
        daemon: linkd.LinkDaemon,
        session: linkd.LinkSession,
        alias: str,
        *,
        remote_id: str = "",
        endpoint: str = "",
    ):
        self.daemon = daemon
        self.session = session
        self.alias = alias
        #: The shim's node id and the ``address:port`` it listens on, which a run record keeps.
        self.remote_id = remote_id
        self.endpoint = endpoint

    def record(self) -> dict[str, str]:
        """What a second process needs to open its own session to this machine."""
        return {
            "node_id": self.remote_id,
            "endpoint": self.endpoint,
            "local_node": self.daemon.node_id,
        }

    def open_stream(self, priority: str = "interactive") -> socket.socket:
        """One more stream of the session. No punch and no ssh connection.

        ``priority`` is the stream's class, ``"interactive"`` or ``"bulk"``.
        """
        try:
            return self.session.open_stream(priority)
        except linkd.LinkError as error:
            raise RuntimeFailure(
                f"{self.alias}: cannot open a letify-link stream: {error}"
            ) from error

    def _paths(self) -> list[dict[str, Any]]:
        try:
            return list(self.session.paths())
        except linkd.LinkError:
            return []

    @property
    def rtt_ms(self) -> float | None:
        """The round trip the session measured on its best path, or None before one is known."""
        rtts = [p["srtt_ms"] for p in self._paths() if p.get("srtt_ms")]
        return min(rtts) if rtts else None

    @property
    def upload_bps(self) -> float | None:
        """The rate the session measured on its best path, or None before one is known."""
        rates = [p["probe_rate"] for p in self._paths() if p.get("probe_rate")]
        return max(rates) if rates else None

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
    return _binary_path(alias, done.stdout, done.stderr)


def _upload_over_ssh(
    alias: str, ssh: Callable[[str], list[str]], python: str, upload: str, workspace: str
) -> str:
    """Upload the development binary of Spec item 7 over the ssh command's standard input."""
    from . import linkupload

    def run(source: str, timeout: float) -> str:
        command = ssh(f"{python} -c {shlex.quote(source)}")
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise _fail(alias, f"the ssh upload step failed: {error}") from error
        return done.stdout + done.stderr

    def push(part: str, local: Path) -> None:
        command = ssh(f"cat > {shlex.quote(part)}")
        try:
            with open(local, "rb") as handle:
                done = subprocess.run(
                    command, stdin=handle, capture_output=True, timeout=PREPARE_TIMEOUT
                )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise _fail(alias, f"the ssh upload failed: {error}") from error
        if done.returncode != 0:
            tail = done.stderr.decode(errors="replace")[-400:]
            raise _fail(alias, f"the ssh upload failed: {tail}")

    return linkupload.ensure(alias, upload, workspace=workspace, run=run, push=push)


def _binary_path(alias: str, stdout: str, stderr: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("LETIFY-LINK "):
            return str(json.loads(line[len("LETIFY-LINK ") :]))
    raise _fail(
        alias,
        "letify-link is not on the machine and the wheel could not supply it: "
        + (stderr or stdout).strip()[-400:],
    )


def connect(
    alias: str,
    *,
    ssh: Callable[[str], list[str]],
    address: str | None,
    python: str,
    bootstrap: str,
    key_path: str | os.PathLike[str],
    version: str,
    binary: str | None = None,
    upload: str | None = None,
    workspace: str = "~",
    punch: bool = False,
    max_rtt_ms: float | None = None,
    min_bps: float | None = None,
    start_timeout_ms: int = 60_000,
) -> LinkRemote:
    """Bootstrap over ssh once, start a session, and return the connected machine.

    Spec "Sidecar interface" item 7. Every failure raises ``ProviderUnavailable`` with the
    reason; nothing falls back to the ssh data path.
    """
    def prepare() -> str:
        if upload:
            return _upload_over_ssh(alias, ssh, python, upload, workspace)
        return prepare_binary(alias, ssh, python, version, binary)

    return connect_over(
        alias,
        prepare=prepare,
        start=lambda: PopenBootstrap(ssh(f"{python} -u -c {shlex.quote(shim_source())}")),
        address=address,
        python=python,
        bootstrap=bootstrap,
        key_path=key_path,
        punch=punch,
        max_rtt_ms=max_rtt_ms,
        min_bps=min_bps,
        start_timeout_ms=start_timeout_ms,
    )


def connect_commands(
    alias: str,
    run: Callable[[str, float], str],
    *,
    python: str,
    bootstrap: str,
    key_path: str | os.PathLike[str],
    version: str,
    binary: str | None = None,
    upload: str | None = None,
    workspace: str = "~",
    wrap: Callable[[str], str] | None = None,
    max_rtt_ms: float | None = None,
    min_bps: float | None = None,
    start_timeout_ms: int = 60_000,
) -> LinkRemote:
    """Bootstrap over a provider's command channel once, and return the connected runtime.

    Spec "Sidecar interface" item 8. ``run(source, timeout)`` runs one Python program on the
    runtime and returns what it printed. The runtime has no address, so the session starts from
    punched candidates only. ``wrap`` is the provider's ``remote_command``, applied to the
    shim's workers.
    """
    from . import linkmailbox

    def prepare() -> str:
        if upload:
            from . import linkupload

            return linkupload.ensure(alias, upload, workspace=workspace, run=run)
        if binary:
            return binary
        from .. import install

        try:
            output = run(install.remote_link_source(version), PREPARE_TIMEOUT)
        except Exception as error:
            raise _fail(
                alias, f"the command channel could not prepare letify-link: {error}"
            ) from error
        return _binary_path(alias, output, "")

    return connect_over(
        alias,
        prepare=prepare,
        start=lambda: _started(linkmailbox.MailboxBootstrap(run, python=python)),
        address=None,
        python=python,
        bootstrap=bootstrap,
        key_path=key_path,
        punch=True,
        wrap=wrap,
        max_rtt_ms=max_rtt_ms,
        min_bps=min_bps,
        start_timeout_ms=start_timeout_ms,
    )


def _started(mailbox: Any) -> Any:
    mailbox.start()
    return mailbox


def connect_over(
    alias: str,
    *,
    prepare: Callable[[], str],
    start: Callable[[], Bootstrap],
    address: str | None,
    python: str,
    bootstrap: str,
    key_path: str | os.PathLike[str],
    punch: bool = False,
    wrap: Callable[[str], str] | None = None,
    max_rtt_ms: float | None = None,
    min_bps: float | None = None,
    start_timeout_ms: int = 60_000,
) -> LinkRemote:
    """The connect of items 7 and 8: ``prepare`` returns the remote binary's path, ``start`` the
    line stream to the shim, and everything after is the same for both."""
    try:
        daemon = linkd.LinkDaemon(key_path)
    except linkd.LinkMissing as error:
        raise _fail(alias, str(error)) from error
    except linkd.LinkError as error:
        raise _fail(alias, f"the local sidecar failed: {error}") from error
    stream: Bootstrap | None = None
    try:
        remote_binary = prepare()
        stream = start()
        rtt = stream.rtt_s
        shim_wait = SHIM_TIMEOUT + (10 * rtt if rtt else 0.0)

        def write(message: dict[str, Any]) -> None:
            assert stream is not None
            stream.send_line(json.dumps(message).encode())

        config: dict[str, Any] = {
            "binary": remote_binary,
            "allow": daemon.node_id,
            "bind": "0.0.0.0:0",
            "python": python,
            "bootstrap": bootstrap,
            "relay": relay_program(),
            "wait_s": WAIT_SESSION_S,
            "resume_s": RESUME_S,
        }
        if wrap is not None:
            # The provider's wrapper (driver library path, environment) applies to every worker
            # and relay the shim starts, as it applies to the worker's ssh command.
            config["worker_command"] = wrap(f"{python} -u -c {shlex.quote(bootstrap)}")
            config["relay_command"] = wrap(f"{python} -c {shlex.quote(relay_program())}")
        write(config)
        lines = _Lines(stream)
        ready = lines.wait("ready", shim_wait, alias, stream)
        remote_id = str(ready["node_id"])
        daemon.rendezvous = _StdioRendezvous(lambda: daemon.node_id, write)
        lines.on_rendezvous = lambda to, message: daemon.deliver(remote_id, message)
        candidates: list[Any] = [] if address is None else [f"{address}:{ready['port']}"]
        if punch or address is None:
            # An account with an address is reachable, so its mappings are that address. One
            # without is behind a NAT, and the sidecars ask the address observers instead.
            where: dict[str, Any] = (
                {"observers": "default"} if address is None else {"public_ip": address}
            )
            names = _punch(alias, daemon, remote_id, stream, write, lines, address, where)
            candidates += [{"name": f"punch{i}", "punched": n} for i, n in enumerate(names)]
        session = daemon.open_session(
            remote_id,
            candidates,
            config={"start_timeout_ms": start_timeout_ms, "resume_window_s": RESUME_S},
        )
        remote = LinkRemote(
            daemon,
            session,
            alias,
            remote_id=remote_id,
            endpoint="" if address is None else f"{address}:{ready['port']}",
        )
        _hold_to_floor(remote, max_rtt_ms, min_bps)
        write({"op": "handover"})
        lines.wait("handed_over", shim_wait, alias, stream)
        # The shim now runs without the bootstrap, so that is ended rather than waited for.
        stream.close()
        return remote
    except linkd.LinkError as error:
        daemon.close()
        raise _fail(alias, str(error)) from error
    except BaseException:
        daemon.close()
        raise
    finally:
        if stream is not None:
            stream.close()


def _punch(
    alias: str,
    daemon: linkd.LinkDaemon,
    remote_id: str,
    stream: Bootstrap,
    write: Callable[[dict[str, Any]], None],
    lines: _Lines,
    address: str | None,
    where: dict[str, Any],
) -> list[str]:
    """Punch with the shim as responder and return the usable path names.

    A stream with no measured round trip (ssh) makes one attempt with the sidecar's own lead and
    window. A command channel makes up to two, each with the lead of ``punch_lead_ms`` from the
    freshest round trip, the second with a window ``RETRY_WINDOW_FACTOR`` times as long. Spec
    "Sidecar interface" item 8.
    """
    windows: list[int | None] = [None]
    if stream.rtt_s is not None:
        windows = [PUNCH_WINDOW_MS, PUNCH_WINDOW_MS * RETRY_WINDOW_FACTOR]
    failures: list[str] = []
    for number, window in enumerate(windows, 1):
        timing: dict[str, Any] = {}
        if window is not None:
            timing = {"lead_ms": punch_lead_ms(stream.rtt_s or 0.0), "window_ms": window}
        write(
            {
                "op": "punch",
                "peer": daemon.node_id,
                "sockets": PUNCH_TCP,
                "udp": PUNCH_UDP,
                **where,
                **timing,
            }
        )
        try:
            return daemon.punch(
                remote_id,
                "initiator",
                sockets=PUNCH_TCP,
                udp=PUNCH_UDP,
                police=POLICE,
                **where,
                **timing,
            )
        except linkd.LinkError as error:
            if address is not None:
                raise
            if window is None:
                raise _fail(
                    alias, f"{alias} has no address and no punched path connected: {error}"
                ) from error
            failures.append(
                f"attempt {number} (lead {timing['lead_ms']} ms, window {window} ms): {error}"
            )
            # Let the responder finish this attempt before the next one starts.
            settle_s = PUNCH_LEAD_ROUND_TRIPS * (stream.rtt_s or 0.0) + 2.0
            lines.settle(("punched", "punch_failed"), settle_s)
    raise _fail(
        alias, f"{alias} has no address and no punched path connected: " + "; ".join(failures)
    )


def attach(
    alias: str,
    record: dict[str, Any],
    *,
    key_path: str | os.PathLike[str],
    start_timeout_ms: int = 60_000,
) -> LinkRemote:
    """A new session to a shim that already serves a worker, for a second owner.

    Spec "Sidecar interface" item 7, taking over a call. The shim's ``allow`` list holds the
    node id of the first owner, so the new owner has to be that node: the same key. Another
    node is refused here rather than at a handshake that would fail with no reason given.
    """
    from ..errors import UnsupportedMode

    try:
        daemon = linkd.LinkDaemon(key_path)
    except linkd.LinkMissing as error:
        raise _fail(alias, str(error)) from error
    except linkd.LinkError as error:
        raise _fail(alias, f"the local sidecar failed: {error}") from error
    try:
        if not record.get("endpoint"):
            raise UnsupportedMode(
                f"{alias} has no address, so its letify-link shim has no endpoint a second "
                f"process can dial, and attaching to it would need a punch through a new "
                f"bootstrap, which is not built yet"
            )
        if daemon.node_id != record.get("local_node"):
            raise UnsupportedMode(
                f"{alias} runs this call behind a letify-link shim whose allow list holds "
                f"node {str(record.get('local_node'))[:12]}, and this process is node "
                f"{daemon.node_id[:12]}. A takeover from another machine or key needs that node "
                f"added to the allow list, which is not built yet"
            )
        session = daemon.open_session(
            str(record["node_id"]),
            [str(record["endpoint"])],
            config={"start_timeout_ms": start_timeout_ms, "resume_window_s": RESUME_S},
        )
    except linkd.LinkError as error:
        daemon.close()
        raise _fail(alias, str(error)) from error
    except BaseException:
        daemon.close()
        raise
    return LinkRemote(
        daemon,
        session,
        alias,
        remote_id=str(record["node_id"]),
        endpoint=str(record["endpoint"]),
    )


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

    def __init__(self, stream: Bootstrap):
        import queue

        self.on_rendezvous: Callable[[str, bytes], None] = lambda to, message: None
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stream = stream
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while True:
            raw = self._stream.recv_line()
            if raw is None:
                break
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

    def settle(self, names: tuple[str, ...], timeout: float) -> None:
        """Wait for one of the events ``names``, or the timeout. Never raises."""
        import queue

        deadline = time.monotonic() + timeout
        while True:
            try:
                event = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return
            if event is None or event.get("event") in names:
                if event is None:
                    self._queue.put(None)
                return

    def wait(
        self, name: str, timeout: float, alias: str, stream: Bootstrap
    ) -> dict[str, Any]:
        import queue

        deadline = time.monotonic() + timeout
        while True:
            try:
                event = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise _fail(alias, f"the remote shim said nothing in {timeout:.0f} s") from None
            if event is None:
                raise _fail(
                    alias,
                    "the remote shim exited before it was ready: "
                    + stream.stderr_text().strip()[-400:],
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

    def __init__(
        self,
        remote: LinkRemote,
        *,
        name: str = "runtime",
        door: str | None = None,
        owns_remote: bool = False,
        priority: str = "interactive",
    ):
        self.remote = remote
        #: The class of the stream's sends: "interactive" for the worker channel, "bulk" for a
        #: transfer.
        self.priority = priority
        self.name = name
        #: The path of a running worker's second door, for a channel that attaches to it.
        self.door = door
        #: Whether closing the channel ends the session, as for an attached channel.
        self.owns_remote = owns_remote
        self.token = secrets.token_hex(8)
        self._socket: socket.socket | None = None
        self._start_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._socket is not None:
                return
            sock = self.remote.open_stream(self.priority)
            self._socket = sock
            self._connection = None
            from ..runtime.channel import Connection

            self._connection = Connection(
                self.name, lambda view: sock.send(view), sock.recv_into, self._emit
            )
            if self.door is None:
                header = stream_header("worker", self.token)
            else:
                header = stream_header("door", self.token, path=self.door)
            self._connection.sender.raw(header)
            if self.door is None:
                self._send_worker()
            self._await_ready()

    def takeover_record(self) -> dict[str, str]:
        """What a run record keeps so a second process can reach this machine's shim."""
        return self.remote.record()

    def tail(self) -> str:
        """The last lines of the standard error of this stream's worker, or "" when unknown."""
        try:
            sock = self.remote.session.open_stream()
        except Exception:
            return ""
        chunks: list[bytes] = []
        try:
            sock.settimeout(STDERR_WAIT_S)
            sock.sendall(stream_header("stderr", self.token))
            while len(chunks) < 64:
                data = sock.recv(65536)
                if not data:
                    break
                chunks.append(data)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                sock.close()
        return b"".join(chunks).decode("utf-8", "replace")

    def _kill(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)

    def _startup_failure(self, expired: bool, cause: Exception) -> Exception:
        stderr = self.tail().strip()
        if expired:
            return RuntimeFailure(f"{super()._startup_failure(expired, cause)}", stderr=stderr)
        return RuntimeFailure(
            f"{self.name}: the worker exited before it was ready, or the letify-link "
            f"session closed ({self.remote.session.close_reason or 'no reason given'})",
            stderr=stderr,
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
            if self.owns_remote:
                self.remote.close()

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


__all__ = [
    "Bootstrap",
    "LinkChannel",
    "LinkRemote",
    "PlainSshLink",
    "PopenBootstrap",
    "attach",
    "connect",
    "connect_commands",
    "connect_over",
    "letify_link_enabled",
    "prepare_binary",
    "punch_lead_ms",
    "shim_path",
    "shim_source",
    "stream_header",
]
