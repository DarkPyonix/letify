"""The PyTorch forwarding client: the operator queue, handles and synchronization.

This module owns one connection to a device worker: starting it, queueing operators and
sending them in batches, assigning and releasing handles, answering synchronizations and
turning a runtime failure into ``RemoteError``, as spec "Batching and synchronization",
"Handles" and "Failure semantics" describe. It does not own how an operator's metadata is
computed, which is ``tensor``, or which stream the bytes cross, which is a ``Transport``.
"""

from __future__ import annotations

import collections
import contextlib
import os
import pickle
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from ...errors import RemoteError, RuntimeLost
from .frames import StreamTransport, Transport, TransportClosed
from .guard import check_torch_version, check_worker_version

#: A queue this long is sent without waiting for more.
BATCH_OPS = 256

#: A queue whose oldest operator has waited this long, in seconds, is sent by the next dispatch.
LINGER_S = 0.002

#: A queue nothing has been added to for this long, in seconds, is sent by the background thread.
IDLE_S = 0.05

#: A release list this long is sent without waiting for an operator.
BATCH_RELEASES = 4096

#: Handles reserved for the outputs of an operator the runtime ran at once.
DESCRIBED_OUTPUTS = 64

#: Passed to ``python -c`` to start the worker: read a length-prefixed source, execute it.
BOOTSTRAP = (
    "import sys;n=int(sys.stdin.buffer.readline());"
    "exec(compile(sys.stdin.buffer.read(n),'letify-device','exec'))"
)


def worker_source(device: str) -> bytes:
    """The worker program: ``frames``, then ``executor``, then the serve call."""
    here = Path(__file__).parent
    parts = ["from __future__ import annotations\n"]
    for name in ("frames.py", "executor.py"):
        text = (here / name).read_text(encoding="utf-8")
        parts.append(text.replace("from __future__ import annotations\n", ""))
    parts.append(f"\nserve({device!r})\n")
    return "".join(parts).encode()


def worker_command(python: str) -> list[str]:
    """The command that starts a device worker with this interpreter on this machine."""
    return [python, "-u", "-c", BOOTSTRAP]


@dataclass
class Stats:
    """What one client has done, counted."""

    #: Operators and requests dispatched, counted when queued rather than when sent.
    ops: int = 0
    batches: int = 0
    round_trips: int = 0
    released: int = 0
    sent_bytes: int = 0
    #: Operators whose output metadata came from the cache instead of a meta kernel.
    cached: int = 0

    def snapshot(self) -> Stats:
        return Stats(**{field.name: getattr(self, field.name) for field in fields(self)})

    def __sub__(self, other: Stats) -> Stats:
        names = [field.name for field in fields(self)]
        return Stats(**{name: getattr(self, name) - getattr(other, name) for name in names})


class Client:
    """One connection to a device worker."""

    def __init__(
        self,
        transport: Transport,
        *,
        process: subprocess.Popen | None = None,
        name: str = "device",
    ):
        self.transport = transport
        self.process = process
        self.name = name
        self.stats = Stats()
        #: Handles whose last RemoteTensor was collected, appended from ``Ref.__del__``.
        self.released: list[int] = []
        self._next_handle = 1
        self._queue: list[tuple] = []
        self._buffers: list[Any] = []
        self._keep: list[Any] = []
        self._first_at = 0.0
        self._last_at = 0.0
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._closed = False
        self._lost: str | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=40)
        self._mapping: list[Any] = []
        self.hello: dict[str, Any] = {}

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        """Read the worker's first message, check its PyTorch, and start the linger thread."""
        head, _buffers = self._recv()
        self.hello = pickle.loads(head)
        check_worker_version(local=torch.__version__, remote=str(self.hello["torch"]))
        threading.Thread(target=self._linger, name=f"{self.name}-linger", daemon=True).start()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._lost is None:
                try:
                    self._flush(reply=False)
                    self.transport.send(pickle.dumps({"close": True}, protocol=5), [])
                except (RuntimeLost, TransportClosed):
                    pass
            self._closed = True
            self._wake.notify_all()
        self.transport.close()
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - a worker stuck in a kernel
                self.process.kill()
                self.process.wait()

    def _drain_stderr(self, stream: Any) -> None:
        for raw in iter(stream.readline, b""):
            self._stderr.append(raw.decode(errors="replace").rstrip())

    # -- handles --------------------------------------------------------------------

    def new_handle(self) -> int:
        handle = self._next_handle
        self._next_handle += 1
        return handle

    # -- queue ----------------------------------------------------------------------

    def enqueue(self, entry: tuple, state: Any = None) -> None:
        """Queue one operator. Sent when the batch is full, aged, or carries a large buffer."""
        with self._lock:
            self._check_open()
            base = len(self._buffers)
            if state is not None and state.buffers:
                self._buffers.extend(state.buffers)
                self._keep.extend(state.keep)
            now = time.monotonic()
            if not self._queue:
                self._first_at = now
                self._wake.notify()
            self._last_at = now
            self._queue.append((*entry, base))
            self.stats.ops += 1
            if (
                len(self._queue) >= BATCH_OPS
                or now - self._first_at >= LINGER_S
                or (state is not None and state.big)
                or len(self.released) >= BATCH_RELEASES
            ):
                self._flush(reply=False)

    def _linger(self) -> None:
        """Send a queue that nothing has been added to for ``IDLE_S``.

        The dispatching thread sends every other aged queue itself, so this thread wakes
        only when dispatch has paused and does not take the GIL from a running step.
        """
        while True:
            with self._lock:
                while not self._queue and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                wait = self._last_at + IDLE_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            with self._lock:
                if self._closed or self._lost is not None:
                    return
                if self._queue and time.monotonic() - self._last_at >= IDLE_S:
                    try:
                        self._flush(reply=False)
                    except RuntimeLost:
                        return

    def _flush(self, *, reply: bool) -> None:
        if not self._queue and not self.released and not reply:
            return
        released = self.released[:]
        del self.released[: len(released)]
        ops, buffers, keep = self._queue, self._buffers, self._keep
        self._queue, self._buffers, self._keep = [], [], []
        head = pickle.dumps({"ops": ops, "release": released, "reply": reply}, protocol=5)
        try:
            self.transport.send(head, buffers)
        except TransportClosed as exc:
            raise self._lose(str(exc)) from exc
        del keep
        self.stats.batches += 1
        self.stats.released += len(released)
        self.stats.sent_bytes += len(head) + sum(memoryview(b).nbytes for b in buffers)

    # -- synchronization ---------------------------------------------------------------

    def request(self, entries: Sequence[tuple], state: Any = None) -> tuple[list, list]:
        """Send the queue with these entries and wait for the reply. One round trip."""
        with self._lock:
            self._check_open()
            base = len(self._buffers)
            if state is not None and state.buffers:
                self._buffers.extend(state.buffers)
                self._keep.extend(state.keep)
            for entry in entries:
                self._queue.append((*entry, base))
            self.stats.ops += len(entries)
            self._flush(reply=True)
            head, buffers = self._recv()
            self.stats.round_trips += 1
        reply = pickle.loads(head)
        failure = reply.get("failure")
        if failure is not None:
            op, message, remote_traceback = failure
            raise RemoteError(f"{op} failed on the runtime: {message}", remote_traceback)
        return reply["results"], buffers

    def synchronize(self) -> None:
        """Wait for every queued operator, raising the first failure among them."""
        self.request([])

    def call(self, name: str, *args: Any, reply: bool = True) -> Any:
        """A letify request to the worker, such as a seed or a memory query."""
        if not reply:
            self.enqueue((name, args, {}, None, None))
            return None
        results, _buffers = self.request([(name, args, {}, None, "value")])
        return results[-1]

    def live_handles(self) -> int:
        """How many tensors the worker holds, after sending pending releases."""
        with self._lock:
            self._check_open()
            # Releases apply after their batch's operators, so they go in a batch of their own.
            self._flush(reply=False)
        return int(self.call("letify.live"))

    def fetch(self, tensor: Any, dtype: str | None) -> torch.Tensor:
        """Copy a RemoteTensor's values into a new CPU tensor."""
        handle = tensor._ref.handle
        results, buffers = self.request([("letify.fetch", (handle, dtype), {}, None, "fetch")])
        index, shape, got = results[-1]
        data = buffers[index]
        kind = getattr(torch, got)
        if not len(data):
            return torch.empty(shape, dtype=kind)
        return torch.frombuffer(data, dtype=kind).reshape(shape)

    def execute_now(self, name: str, args: Any, kwargs: dict, state: Any) -> Any:
        """Run an operator whose metadata could not be computed here, and describe it."""
        from .tensor import from_description

        first = self._next_handle
        self._next_handle += DESCRIBED_OUTPUTS
        results, _buffers = self.request([(name, args, kwargs, first, "describe")], state)
        return from_description(self, first, results[-1])

    # -- errors -------------------------------------------------------------------------

    def _recv(self) -> tuple[bytes, list[bytearray]]:
        try:
            return self.transport.recv()
        except TransportClosed as exc:
            raise self._lose(str(exc)) from exc

    def _lose(self, reason: str) -> RuntimeLost:
        self._lost = reason
        if self.process is not None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        tail = "\n".join(self._stderr)
        detail = f"\n--- device worker stderr ---\n{tail}" if tail else ""
        return RuntimeLost(f"{self.name}: the device worker was lost: {reason}{detail}")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeLost(f"{self.name}: the device worker is closed")
        if self._lost is not None:
            raise RuntimeLost(f"{self.name}: the device worker was lost: {self._lost}")

    # -- cuda mapping ---------------------------------------------------------------------

    @contextlib.contextmanager
    def activate(self) -> Iterator[Client]:
        """Map ``"cuda"`` and ``torch.cuda`` onto this client for the block."""
        from .cuda import mapped

        with mapped(self):
            yield self

    @contextlib.contextmanager
    def suspended(self) -> Iterator[None]:
        """Undo the CUDA mapping inside an ``activate`` block, for a plain local run."""
        from .cuda import unmapped

        with unmapped():
            yield


def connect(
    command: list[str],
    *,
    device: str,
    env: dict[str, str] | None = None,
    name: str = "device",
) -> Client:
    """Start a device worker with this command and return a connected client."""
    check_torch_version(torch.__version__)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=None if env is None else {**os.environ, **env},
        )
    except OSError as exc:
        raise RuntimeLost(f"{name}: could not start the device worker: {exc}") from exc
    assert process.stdin is not None and process.stdout is not None
    source = worker_source(device)
    transport = StreamTransport(
        read_fd=process.stdout.fileno(),
        write_fd=process.stdin.fileno(),
        readinto=process.stdout.readinto,
        owns_fds=False,
    )
    client = Client(transport, process=process, name=name)
    threading.Thread(
        target=client._drain_stderr, args=(process.stderr,), name=f"{name}-stderr", daemon=True
    ).start()
    try:
        transport._write_all([memoryview(b"%d\n" % len(source)), memoryview(source)])
        client.start()
    except (TransportClosed, RuntimeLost) as exc:
        process.kill()
        process.wait()
        if isinstance(exc, RuntimeLost):
            raise
        raise client._lose(str(exc)) from exc
    return client


__all__ = ["BOOTSTRAP", "Client", "Stats", "connect", "worker_command", "worker_source"]
