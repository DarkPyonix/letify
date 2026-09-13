"""How letify talks to a runtime.

Two kinds of channel exist, and which one a provider offers decides what letify
can do there.

A ``PersistentChannel`` keeps one worker process alive behind a pipe. Requests are
framed lines, so the worker process, the blob table and anything written to disk all
survive between calls. That is what keeps a session cache alive, a large argument
sendable once, and a materialized volume readable by a later call.

A ``OneShotChannel`` can only run a command and collect its output. Every call
starts a fresh process, so nothing persists and a session cache lasts one call.
It exists because some transports offer nothing more.

Both hand back the user's own stdout separately from the outcome, because they
share one stream.
"""

from __future__ import annotations

import abc
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

from .. import protocol
from ..errors import ProtocolError, RuntimeFailure, RuntimeLost
from ..protocol.framing import ready_version

#: Marks the line a one-shot ``eval`` prints its value on.
EVAL_MARKER = "__LETIFY_EVAL__"

if TYPE_CHECKING:
    pass

#: How long to wait for the worker to announce itself before giving up.
STARTUP_TIMEOUT = 120.0


class Channel(abc.ABC):
    """A way to run work inside a runtime."""

    #: Whether state survives between calls on this channel.
    persistent: bool = False

    #: The major.minor of the worker's interpreter, once the channel has learned it.
    python_version: str | None = None

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def switch_interpreter(self, python: str, *, timeout: float | None = None) -> None:
        """Run everything after this with another interpreter on the runtime."""

    def eval(self, source: str, *, timeout: float | None = None) -> Any:
        """Run source inside the runtime and return what it left in ``__letify_value__``."""
        value, _logs = self.request({"op": "eval", "source": source}, timeout=timeout)
        return value

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, str]:
        """Send one request and return ``(value, logs)``."""

    def call(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        *,
        timeout: float | None = None,
    ) -> tuple[Any, str]:
        """Run one function call inside the runtime."""
        import base64

        payload = {
            "op": "call",
            "payload": base64.b64encode(protocol.dumps_call(fn, args, kwargs)).decode(),
        }
        return self.request(payload, timeout=timeout)


class PersistentChannel(Channel):
    """One worker process, kept alive behind a pipe.

    The command is whatever starts a Python reading from standard input on the
    target machine: a local interpreter, an SSH invocation, or a CLI bridge. The
    worker source is written once, and every call after that is a framed line.
    """

    persistent = True

    def __init__(
        self, command: list[str], *, name: str = "runtime", env: dict[str, str] | None = None
    ):
        self.command = command
        self.name = name
        self.env = env
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self.env,
            )
        except OSError as exc:
            raise RuntimeFailure(
                f"{self.name}: could not start the worker: {exc}",
                command=" ".join(self.command),
            ) from exc

        self._send_worker()
        self._await_ready()

    def _send_worker(self) -> None:
        """Hand the worker source over as a length-prefixed base64 blob.

        The bootstrap stub on the far side reads the length, reads that many
        characters, decodes and executes them, and leaves standard input open for the
        framed requests that follow.
        """
        import base64

        from ..protocol.worker import SOURCE

        process = self._process
        assert process is not None and process.stdin is not None
        payload = base64.b64encode(SOURCE.replace("\r\n", "\n").encode()).decode()
        process.stdin.write(f"{len(payload)}\n{payload}")
        process.stdin.flush()

    def _await_ready(self) -> None:
        """Read until the worker says it is serving, so a bad start fails here."""
        process = self._process
        assert process is not None and process.stdout is not None
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                stderr = process.stderr.read() if process.stderr else ""
                raise RuntimeFailure(
                    f"{self.name}: the worker exited before it was ready",
                    command=" ".join(self.command),
                    stderr=stderr.strip(),
                )
            if protocol.is_ready(line):
                self.python_version = ready_version(line)
                return
        raise RuntimeFailure(f"{self.name}: the worker did not become ready in time")

    def switch_interpreter(self, python: str, *, timeout: float | None = 120) -> None:
        """Have the worker exec ``python`` on the same pipes, then start the worker again."""
        from ..protocol.worker import BOOTSTRAP

        self.request({"op": "reexec", "python": python, "bootstrap": BOOTSTRAP}, timeout=timeout)
        with self._lock:
            self._send_worker()
            self._await_ready()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(protocol.SHUTDOWN + "\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=5)
        finally:
            # Closing the read ends too, because leaving them to the garbage collector
            # raises an ignored OSError on Windows when the process is already gone.
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # -- requests ------------------------------------------------------------

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, str]:
        with self._lock:
            if self._process is None:
                self.start()
            process = self._process
            assert process is not None
            if process.poll() is not None:
                raise RuntimeLost(f"{self.name}: the worker exited with code {process.returncode}")
            assert process.stdin is not None and process.stdout is not None

            try:
                process.stdin.write(protocol.encode_request(payload) + "\n")
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise RuntimeLost(f"{self.name}: the worker pipe is closed") from exc

            logs: list[str] = []
            watchdog = self._arm_watchdog(timeout)
            while True:
                line = process.stdout.readline()
                if watchdog is not None and watchdog.expired:
                    raise RuntimeFailure(f"{self.name}: the call exceeded {timeout}s")
                if not line:
                    stderr = process.stderr.read() if process.stderr else ""
                    raise ProtocolError(
                        f"{self.name}: the worker stopped without replying, so the "
                        f"process died before it finished. The usual causes are an out "
                        f"of memory kill, a preempted session, or a crash below "
                        f"Python.\n--- last remote output ---\n"
                        f"{''.join(logs)[-2000:]}{stderr[-2000:]}"
                    )
                if protocol.is_reply(line):
                    if watchdog is not None:
                        watchdog.cancel()
                    outcome = protocol.decode_reply(line)
                    return protocol.unwrap(outcome, runtime_key=self.name), "".join(logs)
                logs.append(line)

    def _arm_watchdog(self, timeout: float | None) -> _Watchdog | None:
        """Start the timer that makes a silent call give up.

        Checking a deadline between output lines is not enough: a call that prints
        nothing sits in ``readline`` for as long as it likes, and the timeout the
        declaration asked for never arrives. So the worker's pipe is closed from a timer
        thread, which is what wakes the read up.
        """
        if timeout is None or self._process is None:
            return None
        return _Watchdog(self._process, timeout)


class _Watchdog:
    """Closes a worker's output pipe when a call runs past its timeout.

    A blocking read does not notice a deadline on its own, so something has to make the
    read return. Closing the pipe does that, and the session is discarded afterwards
    anyway, so nothing is lost by breaking it.
    """

    def __init__(self, process: subprocess.Popen[str], timeout: float):
        self.expired = False
        self._process = process
        self._timer = threading.Timer(timeout, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.expired = True
        try:
            self._process.kill()
        except OSError:
            pass

    def cancel(self) -> None:
        self._timer.cancel()


class OneShotChannel(Channel):
    """Runs each call as its own command, with nothing kept between them.

    ``runner`` takes the source to execute and a timeout, and returns the stdout it
    produced. Everything else follows from there.
    """

    persistent = False

    def __init__(self, runner: Any, *, name: str = "runtime", files: Any = None):
        self.runner = runner
        self.name = name
        #: A provider's own file transfer, which serves ``put_file``, ``get_file`` and
        #: ``pack_dir`` where a program per call cannot carry the bytes.
        self.files = files
        #: The interpreter every program runs with once set, as a child of the runner's own.
        self.interpreter: str | None = None

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _run(self, source: str, timeout: float | None) -> str:
        if self.interpreter is not None:
            source = _child_source(self.interpreter, source)
        return self.runner(source, timeout)

    def switch_interpreter(self, python: str, *, timeout: float | None = 120) -> None:
        from .bootstrap import VERSION_SOURCE

        self.interpreter = python
        self.python_version = self.eval(VERSION_SOURCE, timeout=timeout)

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, str]:
        op = payload.get("op")
        if op == "exec":
            self._run(payload["source"], timeout)
            return None, ""
        if op == "eval":
            return _parse_eval(self._run(_eval_source(payload["source"]), timeout), self.name)
        if op == "have":
            # Nothing persists, so the runtime holds nothing.
            return [], ""
        if op == "lease":
            self._run(_lease_source(payload["grace"]), timeout)
            return None, ""
        if self.files is not None and op in ("put_file", "get_file", "pack_dir"):
            return self.files.serve(payload, timeout), ""
        raise RuntimeFailure(
            f"{self.name}: this provider runs one-shot commands, so it cannot serve "
            f"{op!r}. Persistent state and blob reuse need a channel that "
            f"keeps a process alive."
        )

    def call(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        *,
        timeout: float | None = None,
    ) -> tuple[Any, str]:
        from ..protocol import driver

        source = driver.build(fn, args, kwargs)
        stdout = self._run(source, timeout)
        logs, value = protocol.parse(stdout, runtime_key=self.name)
        return value, logs


def _child_source(python: str, source: str) -> str:
    """Wrap a program so it runs as a child of ``python``, its output passed through."""
    return (
        "import subprocess as _letify_s, sys as _letify_y\n"
        f"_letify_r = _letify_s.run([{python!r}, '-c', {source!r}],"
        " capture_output=True, text=True)\n"
        "_letify_y.stdout.write(_letify_r.stdout)\n"
        "_letify_y.stderr.write(_letify_r.stderr)\n"
        "_letify_y.stdout.flush()\n"
    )


def _eval_source(source: str) -> str:
    """Append the line that prints ``__letify_value__`` behind the eval marker."""
    return (
        f"{source}\n"
        "import base64 as _letify_b, pickle as _letify_p\n"
        f"print({EVAL_MARKER!r} + _letify_b.b64encode("
        "_letify_p.dumps(globals().get('__letify_value__'), protocol=4)).decode(), flush=True)\n"
    )


def _parse_eval(stdout: str, name: str) -> tuple[Any, str]:
    import base64
    import pickle

    logs: list[str] = []
    found = None
    for line in stdout.splitlines(keepends=True):
        if line.startswith(EVAL_MARKER):
            found = line[len(EVAL_MARKER) :].strip()
        else:
            logs.append(line)
    if found is None:
        raise ProtocolError(
            f"{name}: the program ended without reporting its value, so it failed or died."
            f"\n--- last remote output ---\n{stdout[-2000:]}"
        )
    return pickle.loads(base64.b64decode(found)), "".join(logs)


def _lease_source(grace: float) -> str:
    """Arm the self termination timer in a process that will not outlive the call.

    On a one-shot channel this is mostly symbolic, because the process ends with
    the command. The provider's own session timeout is what bounds the cost there.
    """
    return (
        "import os, threading, time\n"
        "_state = globals().setdefault('__letify_lease__', {})\n"
        f"_state['deadline'] = time.time() + {float(grace)!r}\n"
        "print('letify: lease armed')\n"
    )


__all__ = ["STARTUP_TIMEOUT", "Channel", "OneShotChannel", "PersistentChannel"]
