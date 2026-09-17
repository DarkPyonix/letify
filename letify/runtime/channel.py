"""How letify talks to a runtime.

Two kinds of channel exist, and which one a provider offers decides what letify
can do there.

A ``PersistentChannel`` keeps one worker process alive behind a pipe pair. Messages are
binary frames, as spec "Frames" describes, so the worker process, the blob table and
anything written to disk all survive between calls. That is what keeps a session cache
alive, a large argument sendable once, and a materialized volume readable by a later call.

A ``OneShotChannel`` can only run a command and collect its output. Every call
starts a fresh process, so nothing persists and a session cache lasts one call.
It exists because some transports offer nothing more.

``Connection`` owns the frames of one persistent channel: overlapping requests, the
thread that happens to be reading, and the worker's output, which is written live to
this process's own streams. It does not own how the bytes move, which each
``FramedChannel`` subclass supplies.
"""

from __future__ import annotations

import abc
import collections
import contextlib
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import IO, Any

from .. import protocol
from ..errors import ProtocolError, RuntimeFailure, RuntimeLost
from ..protocol import wire

#: Marks the line a one-shot ``eval`` prints its value on.
EVAL_MARKER = "__LETIFY_EVAL__"

#: How long to wait for the worker to announce itself before giving up.
STARTUP_TIMEOUT = 120.0

#: Output kept per request, and per channel for error messages.
TAIL_BYTES = 64 << 10

#: Seconds a thread waiting for a device reply polls the read descriptor before blocking.
DEVICE_POLL_S = 0.002


class Channel(abc.ABC):
    """A way to run work inside a runtime."""

    #: Whether state survives between calls on this channel.
    persistent: bool = False

    #: The major.minor of the worker's interpreter, once the channel has learned it.
    python_version: str | None = None

    #: Whether worker output is written to this process's stdout and stderr as it arrives.
    echo: bool = True

    #: Called with ``("stdout" or "stderr", bytes)`` for each piece of output instead of echo.
    on_output: Callable[[str, bytes], None] | None = None

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

    def stream(self, payload: dict[str, Any], *, timeout: float | None = None) -> Iterator[Any]:
        """Send one request answered by several replies, yielding each value in order.

        Spec "Waiting for a reply". A channel that cannot carry such an answer refuses it,
        so nothing silently falls back to one request per piece.
        """
        raise RuntimeFailure(
            f"{getattr(self, 'name', 'runtime')}: this channel cannot carry a streamed "
            f"reply, so {payload.get('op')!r} is not available here."
        )

    def pipeline(self, *, window: int, timeout: float | None = None) -> Pipeline:
        """Open one stream carrying several requests in order, each answered in order.

        Spec "Waiting for a reply". A channel that cannot carry it refuses, so nothing
        silently falls back to one request per piece.
        """
        raise RuntimeFailure(
            f"{getattr(self, 'name', 'runtime')}: this channel cannot carry pipelined "
            f"requests, so streaming project data is not available here."
        )

    def call(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        *,
        timeout: float | None = None,
    ) -> tuple[Any, str]:
        """Run one function call inside the runtime."""
        head, buffers = protocol.dumps_call_parts(fn, args, kwargs)
        return self.request({"op": "call", "payload": head, "buffers": buffers}, timeout=timeout)

    def _emit(self, stream: str, data: bytes) -> None:
        """Hand one piece of worker output to ``on_output``, or write it to the local stream."""
        if self.on_output is not None:
            self.on_output(stream, data)
            return
        if not self.echo:
            return
        target = sys.stdout if stream == "stdout" else sys.stderr
        try:
            binary = getattr(target, "buffer", None)
            if binary is None:
                target.write(data.decode("utf-8", "replace"))
            else:
                target.flush()
                binary.write(data)
                binary.flush()
            target.flush()
        except (OSError, ValueError):
            pass


class _Tail:
    """The last ``limit`` bytes written to it."""

    def __init__(self, limit: int = TAIL_BYTES):
        self._data = bytearray()
        self._limit = limit

    def add(self, chunk: bytes) -> None:
        self._data += chunk
        if len(self._data) > 2 * self._limit:
            del self._data[: -self._limit]

    def text(self) -> str:
        return bytes(self._data[-self._limit :]).decode("utf-8", "replace")


class _Gate:
    """Many requests may send at once; moving the worker to another interpreter excludes them."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sending = 0
        self._exclusive = False

    @contextlib.contextmanager
    def shared(self) -> Iterator[None]:
        with self._condition:
            while self._exclusive:
                self._condition.wait()
            self._sending += 1
        try:
            yield
        finally:
            with self._condition:
                self._sending -= 1
                self._condition.notify_all()

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._condition:
            while self._exclusive:
                self._condition.wait()
            self._exclusive = True
            while self._sending:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._exclusive = False
                self._condition.notify_all()


class _Slot:
    """One open request: where its reply, its failure and its output end up."""

    __slots__ = ("error", "event", "pieces", "reply", "stream", "streaming", "tail")

    def __init__(self, stream: int, *, streaming: bool = False):
        self.stream = stream
        self.event = threading.Event()
        self.reply: tuple[Any, list] | None = None
        self.error: ProtocolError | None = None
        self.tail = _Tail()
        #: Whether more than one reply is expected, so the slot stays open until the last.
        self.streaming = streaming
        #: Replies of a streamed answer, in arrival order.
        self.pieces: collections.deque = collections.deque()


class _Watchdog:
    """Kills a worker when a request runs past its timeout.

    A blocking read does not notice a deadline on its own, so something has to make the
    read return. Killing the process does that, and the session is discarded afterwards
    anyway, so nothing is lost by breaking it.
    """

    def __init__(self, kill: Callable[[], None], timeout: float):
        self.expired = False
        self._kill = kill
        self._timer = threading.Timer(timeout, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.expired = True
        try:
            self._kill()
        except (OSError, RuntimeFailure):
            pass

    def cancel(self) -> None:
        self._timer.cancel()


class Connection:
    """Frames over one pair of byte streams, shared by any number of waiting requests.

    Spec "Waiting for a reply": one waiting thread at a time reads frames and hands each
    to the request it belongs to. The others sleep on a condition until their reply has been
    delivered or nobody is reading, so a short request is never starved by a long one.
    """

    def __init__(
        self,
        name: str,
        write: Callable[[memoryview], int],
        readinto: Callable[[memoryview], int],
        emit: Callable[[str, bytes], None],
        *,
        death_detail: Callable[[], str] = lambda: "",
        poll_fd: int | None = None,
    ):
        self.name = name
        self.sender = wire.Sender(write)
        #: How long a device reply is polled for before a blocking read, in seconds.
        self.poll_s = DEVICE_POLL_S
        self._poller: Any = None
        if poll_fd is not None and hasattr(select, "poll"):
            self._poller = select.poll()
            self._poller.register(poll_fd, select.POLLIN)
        self.gate = _Gate()
        self.hello: str | None = None
        self.failure: ProtocolError | None = None
        self._receiver = wire.Receiver(readinto)
        self._emit = emit
        self._death_detail = death_detail
        #: Guards ``_reading`` and wakes waiters when a reply, a hello or a failure arrives.
        self._turn = threading.Condition()
        self._reading = False
        self._slots: dict[int, _Slot] = {}
        self._slots_lock = threading.Lock()
        self._next_stream = 1
        self._tail = _Tail()
        #: Device executor messages that arrived on ``wire.DEVICE_STREAM``, or None when closed.
        self._device: collections.deque | None = None
        #: ``data_want`` payloads that arrived on ``wire.DATA_STREAM``, for the data sender.
        self._wants: collections.deque = collections.deque()

    # -- reading ---------------------------------------------------------------

    def _pump(self) -> None:
        """Read one frame and deliver it. Only the thread whose turn it is to read calls this."""
        try:
            event = self._receiver.next_event()
        except ProtocolError as exc:
            self._fail(f"{self.name}: {exc}")
            return
        except (EOFError, OSError, ValueError):
            self._fail(None)
            return
        if event is None:
            return
        kind, stream, value = event
        if kind in (wire.STDOUT, wire.STDERR):
            self._tail.add(value)
            with self._slots_lock:
                open_slots = list(self._slots.values())
            for slot in open_slots:
                slot.tail.add(value)
            self._emit("stdout" if kind == wire.STDOUT else "stderr", value)
        elif kind == wire.HELLO:
            with self._turn:
                self.hello = value
                self._turn.notify_all()
        elif kind == wire.REQUEST and stream == wire.DATA_STREAM:
            # The worker asking for blobs it is about to read. Nothing answers it: the
            # background sender picks the request up and reorders its queue.
            try:
                self._wants.append(wire.loads(value[0], value[1]))
            except Exception:
                pass
            with self._turn:
                self._turn.notify_all()
        elif kind == wire.REPLY and stream == wire.DEVICE_STREAM:
            with self._turn:
                if self._device is not None:
                    self._device.append(value)
                self._turn.notify_all()
        elif kind == wire.REPLY:
            with self._slots_lock:
                slot = self._slots.get(stream)
                # A streamed answer keeps its slot until its last reply has been read.
                if slot is not None and not slot.streaming:
                    del self._slots[stream]
            if slot is not None:
                with self._turn:
                    if slot.streaming:
                        slot.pieces.append(value)
                    else:
                        slot.reply = value
                    slot.event.set()
                    self._turn.notify_all()

    def _fail(self, message: str | None) -> None:
        if message is None:
            detail = self._death_detail()
            message = (
                f"{self.name}: the worker stopped without replying, so the process died "
                f"before it finished. The usual causes are an out of memory kill, a "
                f"preempted session, or a crash below Python.\n--- last remote output ---\n"
                f"{self._tail.text()[-2000:]}{detail[-2000:]}"
            )
        with self._turn:
            self.failure = ProtocolError(message)
            with self._slots_lock:
                slots = list(self._slots.values())
                self._slots.clear()
            for slot in slots:
                slot.error = self.failure
                slot.event.set()
            self._turn.notify_all()

    def _wait(self, done: Callable[[], bool], *, poll: bool = False) -> None:
        """Return once ``done()`` holds or the connection failed, reading frames when it is
        this thread's turn. With ``poll``, each read is preceded by polling for up to
        ``poll_s``."""
        with self._turn:
            while True:
                if done() or self.failure is not None:
                    return
                if not self._reading:
                    self._reading = True
                    break
                self._turn.wait()
        try:
            while True:
                if poll:
                    self._poll_ready()
                self._pump()
                with self._turn:
                    if done() or self.failure is not None:
                        return
        finally:
            with self._turn:
                self._reading = False
                self._turn.notify_all()

    def wait_hello(self) -> str:
        self._wait(lambda: self.hello is not None)
        if self.hello is None:
            raise ProtocolError(str(self.failure))
        return self.hello

    # -- device stream ---------------------------------------------------------

    def open_device(self) -> None:
        """Keep the messages that arrive on the device stream for ``device_reply``."""
        with self._turn:
            self._device = collections.deque()

    def close_device(self) -> None:
        with self._turn:
            self._device = None
            self._turn.notify_all()

    def device_reply(self) -> tuple | None:
        """The next device stream message, reading frames when it is this thread's turn.

        None when the connection failed or the device stream was closed.
        """
        self._wait(lambda: not self._device_open() or bool(self._device), poll=True)
        with self._turn:
            if self._device:
                return self._device.popleft()
        return None

    def _device_open(self) -> bool:
        return self._device is not None

    # -- data stream -----------------------------------------------------------

    def take_wants(self) -> list[dict[str, Any]]:
        """Every ``data_want`` that has arrived so far, oldest first, and forget them."""
        found = []
        while True:
            try:
                found.append(self._wants.popleft())
            except IndexError:
                return found

    def _poll_ready(self) -> None:
        """Poll the read descriptor without blocking until it is readable or ``poll_s`` ends."""
        poller = self._poller
        if poller is None:
            return
        end = time.perf_counter() + self.poll_s
        while not poller.poll(0):
            if time.perf_counter() >= end:
                return

    # -- requests --------------------------------------------------------------

    def request(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None,
        kill: Callable[[], None],
        shared: bool = True,
    ) -> tuple[Any, str]:
        watchdog = _Watchdog(kill, timeout) if timeout is not None else None
        try:
            with self.gate.shared() if shared else contextlib.nullcontext():
                slot = self._open()
                try:
                    self.sender.message(wire.REQUEST, slot.stream, payload)
                except OSError as exc:
                    self._forget(slot)
                    if watchdog is not None and watchdog.expired:
                        raise RuntimeFailure(f"{self.name}: the call exceeded {timeout}s") from exc
                    raise RuntimeLost(f"{self.name}: the worker pipe is closed") from exc
                except BaseException:
                    self._forget(slot)
                    raise
            self._wait(slot.event.is_set)
        finally:
            if watchdog is not None:
                watchdog.cancel()
        if slot.reply is None:
            if watchdog is not None and watchdog.expired:
                raise RuntimeFailure(f"{self.name}: the call exceeded {timeout}s")
            raise ProtocolError(str(slot.error or self.failure))
        head, buffers = slot.reply
        slot.reply = None
        outcome = wire.loads(head, buffers)
        return protocol.unwrap(outcome, runtime_key=self.name), slot.tail.text()

    def stream(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None,
        kill: Callable[[], None],
    ) -> Iterator[Any]:
        """Send one request and yield each reply's value until the one marked last.

        Spec "Waiting for a reply". The slot stays open across the whole sequence, so the
        worker sends the next piece without waiting for this process to read the one before.
        """
        watchdog = _Watchdog(kill, timeout) if timeout is not None else None
        slot: _Slot | None = None
        try:
            with self.gate.shared():
                slot = self._open(streaming=True)
                try:
                    self.sender.message(wire.REQUEST, slot.stream, payload)
                except OSError as exc:
                    if watchdog is not None and watchdog.expired:
                        raise RuntimeFailure(f"{self.name}: the call exceeded {timeout}s") from exc
                    raise RuntimeLost(f"{self.name}: the worker pipe is closed") from exc
            while True:
                self._wait(lambda: bool(slot.pieces))  # type: ignore[union-attr]
                with self._turn:
                    piece = slot.pieces.popleft() if slot.pieces else None
                    if not slot.pieces:
                        slot.event.clear()
                if piece is None:
                    if watchdog is not None and watchdog.expired:
                        raise RuntimeFailure(f"{self.name}: the call exceeded {timeout}s")
                    raise ProtocolError(str(slot.error or self.failure))
                head, buffers = piece
                outcome = wire.loads(head, buffers)
                last = not isinstance(outcome, dict) or bool(outcome.get("last"))
                yield protocol.unwrap(outcome, runtime_key=self.name)
                if last:
                    return
        finally:
            if slot is not None:
                self._forget(slot)
            if watchdog is not None:
                watchdog.cancel()

    def pipeline(self, *, window: int, timeout: float | None, kill: Callable[[], None]) -> Pipeline:
        """Open one stream that carries several requests, each answered in order."""
        return Pipeline(self, window=window, timeout=timeout, kill=kill)

    def _open(self, *, streaming: bool = False) -> _Slot:
        with self._slots_lock:
            if self.failure is not None:
                raise ProtocolError(str(self.failure))
            slot = _Slot(self._next_stream, streaming=streaming)
            self._next_stream += 2
            self._slots[slot.stream] = slot
        return slot

    def _forget(self, slot: _Slot) -> None:
        with self._slots_lock:
            self._slots.pop(slot.stream, None)


class Pipeline:
    """Several requests sent in order on one stream, each answered by one reply in order.

    Spec "Waiting for a reply". ``send`` does not wait for the reply of the request before
    it: at most ``window`` requests are unanswered at once, so the link carries the next
    piece while the worker serves the one before, and the worker holds at most ``window``
    it has not served. The replies are read whenever a send has to wait for room and by
    ``drain``, and the first failed reply raises. One thread uses a pipeline at a time.
    """

    def __init__(
        self,
        connection: Connection,
        *,
        window: int,
        timeout: float | None,
        kill: Callable[[], None],
    ):
        self._connection = connection
        self._window = max(1, window)
        self._timeout = timeout
        self._kill = kill
        self._slot = connection._open(streaming=True)
        #: Requests sent, and replies read, since the stream was opened.
        self.sent = 0
        self.answered = 0

    def send(self, payload: dict[str, Any]) -> None:
        """Send one request, first reading replies until fewer than ``window`` are open."""
        self._settle(self.sent - self._window + 1)
        connection = self._connection
        with connection.gate.shared():
            try:
                connection.sender.message(wire.REQUEST, self._slot.stream, payload)
            except OSError as exc:
                raise RuntimeLost(f"{connection.name}: the worker pipe is closed") from exc
        self.sent += 1

    def drain(self) -> None:
        """Read every outstanding reply, raising the first failure among them."""
        self._settle(self.sent)

    def close(self) -> None:
        """Release the stream id. A reply that arrives after this is dropped."""
        self._connection._forget(self._slot)

    def _settle(self, count: int) -> None:
        """Read replies until at least ``count`` requests are answered."""
        self._take()
        if self.answered >= count:
            return
        connection = self._connection
        slot = self._slot
        watchdog = _Watchdog(self._kill, self._timeout) if self._timeout is not None else None
        try:
            while self.answered < count:
                connection._wait(lambda: bool(slot.pieces) or slot.error is not None)
                if not slot.pieces:
                    if watchdog is not None and watchdog.expired:
                        raise RuntimeFailure(
                            f"{connection.name}: the call exceeded {self._timeout}s"
                        )
                    raise ProtocolError(str(slot.error or connection.failure))
                self._take()
        finally:
            if watchdog is not None:
                watchdog.cancel()

    def _take(self) -> None:
        """Count and check the replies that have arrived, in order."""
        slot = self._slot
        with self._connection._turn:
            pieces = list(slot.pieces)
            slot.pieces.clear()
            slot.event.clear()
        for head, buffers in pieces:
            self.answered += 1
            outcome = wire.loads(head, buffers)
            protocol.unwrap(outcome, runtime_key=self._connection.name)


class FramedChannel(Channel):
    """A persistent channel whose messages are frames, whatever carries the bytes."""

    persistent = True

    #: Whether the worker writes each frame as a base64 line, for a text-only transport.
    text_frames = False

    name: str
    _connection: Connection | None = None

    def _kill(self) -> None:
        """End the worker, so every blocked read returns."""

    def _startup_failure(self, expired: bool, cause: Exception) -> Exception:
        """What a worker that never said hello raises. ``cause`` is the connection failure."""
        if expired:
            return RuntimeFailure(f"{self.name}: the worker did not become ready in time")
        return RuntimeFailure(f"{self.name}: the worker exited before it was ready")

    def _send_worker(self) -> None:
        """Hand the worker source to the bootstrap stub: a byte count line, then the source."""
        from ..protocol.worker import source

        connection = self._connection
        assert connection is not None
        data = source(text_frames=self.text_frames).encode("utf-8")
        connection.hello = None
        try:
            connection.sender.raw(b"%d\n" % len(data) + data)
        except OSError:
            # The worker is already gone. Waiting for its hello reports why.
            pass

    def _await_ready(self) -> None:
        """Read until the worker says hello, so a bad start fails here."""
        connection = self._connection
        assert connection is not None
        watchdog = _Watchdog(self._kill, STARTUP_TIMEOUT)
        try:
            self.python_version = connection.wait_hello()
        except ProtocolError as exc:
            raise self._startup_failure(watchdog.expired, exc) from exc
        finally:
            watchdog.cancel()

    def switch_interpreter(self, python: str, *, timeout: float | None = 120) -> None:
        """Have the worker exec ``python`` on the same pipes, then start the worker again."""
        from ..protocol.worker import BOOTSTRAP

        connection = self._require()
        with connection.gate.exclusive():
            connection.request(
                {"op": "reexec", "python": python, "bootstrap": BOOTSTRAP},
                timeout=timeout,
                kill=self._kill,
                shared=False,
            )
            self._send_worker()
            self._await_ready()

    @property
    def connection(self) -> Connection:
        """The open connection, starting the worker first if it is not running."""
        return self._require()

    def _require(self) -> Connection:
        if self._connection is None:
            self.start()
        if self._connection is None:
            raise RuntimeLost(f"{self.name}: the channel is closed")
        return self._connection

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, str]:
        connection = self._require()
        return connection.request(payload, timeout=timeout, kill=self._kill)

    def stream(self, payload: dict[str, Any], *, timeout: float | None = None) -> Iterator[Any]:
        connection = self._require()
        return connection.stream(payload, timeout=timeout, kill=self._kill)

    def pipeline(self, *, window: int, timeout: float | None = None) -> Pipeline:
        connection = self._require()
        return connection.pipeline(window=window, timeout=timeout, kill=self._kill)


class PersistentChannel(FramedChannel):
    """One worker process, kept alive behind a pipe pair.

    The command is whatever starts a Python reading from standard input on the
    target machine: a local interpreter, an SSH invocation, or a CLI bridge. The
    worker source is written once, and every message after that is frames.
    """

    def __init__(
        self, command: list[str], *, name: str = "runtime", env: dict[str, str] | None = None
    ):
        self.command = command
        self.name = name
        self.env = env
        self._process: subprocess.Popen[bytes] | None = None
        self._connection = None
        self._start_lock = threading.Lock()
        self._stderr = _Tail()
        self._stderr_reader: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        with self._start_lock:
            if self._process is not None:
                return
            try:
                process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    env=self.env,
                )
            except OSError as exc:
                raise RuntimeFailure(
                    f"{self.name}: could not start the worker: {exc}",
                    command=" ".join(self.command),
                ) from exc
            assert process.stdin is not None and process.stdout is not None
            self._process = process
            wire.widen_pipe(process.stdin.fileno())
            wire.widen_pipe(process.stdout.fileno())
            # Read on a thread from the start, so SSH or the stub can never fill this pipe.
            self._stderr = _Tail()
            self._stderr_reader = threading.Thread(
                target=_drain, args=(process.stderr, self._stderr.add), daemon=True
            )
            self._stderr_reader.start()
            self._connection = Connection(
                self.name,
                wire.fd_writer(process.stdin.fileno()),
                process.stdout.readinto,  # type: ignore[attr-defined]
                self._emit,
                death_detail=self._stderr_text,
                poll_fd=process.stdout.fileno(),
            )
            self._send_worker()
            self._await_ready()

    def _kill(self) -> None:
        if self._process is not None:
            self._process.kill()

    def _stderr_text(self) -> str:
        """The process's own standard error, complete once the process has exited."""
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_reader is not None:
            self._stderr_reader.join(1)
        return self._stderr.text()

    def _startup_failure(self, expired: bool, cause: Exception) -> Exception:
        if expired:
            return super()._startup_failure(expired, cause)
        return RuntimeFailure(
            f"{self.name}: the worker exited before it was ready",
            command=" ".join(self.command),
            stderr=self._stderr_text().strip(),
        )

    def close(self) -> None:
        process, self._process = self._process, None
        connection, self._connection = self._connection, None
        if process is None:
            return
        # Discard what the worker still writes while it shuts down, so it cannot block.
        threading.Thread(target=_drain, args=(process.stdout, None), daemon=True).start()
        try:
            if process.stdin is not None and not process.stdin.closed:
                if connection is not None:
                    with contextlib.suppress(OSError):
                        connection.sender.frame(wire.SHUTDOWN, 0)
                process.stdin.close()
            process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=5)
        finally:
            if self._stderr_reader is not None:
                self._stderr_reader.join(1)

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # -- requests ------------------------------------------------------------

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, str]:
        if self._process is None:
            self.start()
        process, connection = self._process, self._connection
        if process is None or connection is None:
            raise RuntimeLost(f"{self.name}: the channel is closed")
        if process.poll() is not None:
            raise RuntimeLost(f"{self.name}: the worker exited with code {process.returncode}")
        return connection.request(payload, timeout=timeout, kill=self._kill)


def _drain(stream: IO[bytes] | None, sink: Callable[[bytes], None] | None) -> None:
    """Read a pipe to its end, handing each chunk to ``sink``, and close it."""
    if stream is None:
        return
    try:
        while True:
            chunk = os.read(stream.fileno(), 1 << 16)
            if not chunk:
                break
            if sink is not None:
                sink(chunk)
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()


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
            raw = payload.get("payload")
            if isinstance(raw, (bytes, bytearray, memoryview)):
                # The provider's file transfer carries text, so the bytes are encoded here.
                import base64

                payload = {**payload, "payload": base64.b64encode(raw).decode()}
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


__all__ = [
    "STARTUP_TIMEOUT",
    "Channel",
    "Connection",
    "FramedChannel",
    "OneShotChannel",
    "PersistentChannel",
]
