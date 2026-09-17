"""The PyTorch forwarding client: templates, the entry queue, handles and synchronization.

This module owns one connection to a device executor: starting it, numbering templates,
queueing entries and handing batches to a sender thread, replaying captured steps,
assigning and releasing handles, answering reads with only what they depend on, and turning
a runtime failure into ``RemoteError``, as spec "Operator templates", "Batching and
synchronization", "Step capture", "Handles" and "Failure semantics" describe. It does not
own how an operator's metadata is computed, which is ``tensor``, how a repeated sequence is
detected, which is ``trace``, or which stream the bytes cross, which is a ``Transport``.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import pickle
import queue
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from blake3 import blake3

from ...errors import RemoteError, RuntimeLost
from .executor import E_DEFINE, E_OP, E_REQUEST, E_STEP, E_STEP_DEFINE, UPLOAD_CACHE_BYTES
from .frames import ChannelTransport, StreamTransport, Transport, TransportClosed
from .guard import check_torch_version, check_worker_version
from .tensor import _CACHE, _NAMES, MISS, REPLAY, Big, layout, outputs, reader
from .trace import Tracer

#: A queue this long is sent without waiting for more.
BATCH_OPS = 256

#: A queue whose oldest entry has waited this long, in seconds, is sent by the next dispatch.
LINGER_S = 0.002

#: A queue nothing has been added to for this long, in seconds, is sent by the background thread.
IDLE_S = 0.05

#: A release list this long is sent without waiting for an operator.
BATCH_RELEASES = 4096

#: Handles reserved for the outputs of an operator the runtime ran at once.
DESCRIBED_OUTPUTS = 64

#: CPU tensors at least this large are hashed and travel as their digest when the executor
#: already holds their bytes.
CACHE_MIN_BYTES = 64 << 10

#: CPU tensors at least this large are hashed with BLAKE3's automatic multithreading.
HASH_THREADS_BYTES = 1 << 20

#: Unread replies at which the next non-blocking read first reads the oldest.
UNREAD_LIMIT = 1024


class Waiter:
    """The synchronization a reply-carrying batch ends with."""

    __slots__ = ("buffers", "done", "failure", "results")

    def __init__(self) -> None:
        self.done = False
        self.results: list = []
        self.buffers: list = []
        self.failure: tuple | None = None


class Pending:
    """A non-blocking read: the CPU tensor it fills and the fetch entry that fills it."""

    __slots__ = ("done", "entry", "error", "target")

    def __init__(self, target: torch.Tensor, entry: tuple):
        self.target = target
        self.entry = entry
        self.done = False
        self.error: RemoteError | None = None

    def apply(self, value: Any, buffers: list, failure: tuple | None) -> None:
        if value is None:
            op, message, remote_traceback = failure or ("letify.fetch", "skipped", "")
            self.error = RemoteError(f"{op} failed on the runtime: {message}", remote_traceback)
        else:
            index, shape, got = value
            data = buffers[index]
            if len(data):
                source = torch.frombuffer(data, dtype=getattr(torch, got)).reshape(shape)
                with torch._C.DisableTorchFunction():
                    self.target.copy_(source)
        self.done = True


#: Passed to ``python -c`` to start the worker: read a length-prefixed source, execute it.
BOOTSTRAP = (
    "import sys;n=int(sys.stdin.buffer.readline());"
    "exec(compile(sys.stdin.buffer.read(n),'letify-device','exec'))"
)


def worker_source(device: str | None) -> bytes:
    """The executor program: ``wire``, ``frames``, ``executor``, then the serve call if any."""
    here = Path(__file__).parent
    paths = [here.parent.parent / "protocol" / "wire.py", here / "frames.py", here / "executor.py"]
    parts = ["from __future__ import annotations\n"]
    for path in paths:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        parts.append(text.replace("from __future__ import annotations\n", ""))
    if device is not None:
        parts.append(f"\nserve({device!r})\n")
    return "".join(parts).encode()


def worker_command(python: str) -> list[str]:
    """The command that starts a device executor process with this interpreter."""
    return [python, "-u", "-c", BOOTSTRAP]


@dataclass
class Stats:
    """What one client has done, counted."""

    #: Operators and requests dispatched, counted when dispatched rather than when sent.
    ops: int = 0
    #: Eager operator, step and request entries queued. Definitions are not counted.
    entries: int = 0
    batches: int = 0
    round_trips: int = 0
    released: int = 0
    sent_bytes: int = 0
    #: Operators whose output metadata came from the cache instead of a meta kernel.
    cached: int = 0
    templates: int = 0
    #: Steps registered with the worker.
    steps: int = 0
    #: Operators dispatched inside a repetition of a captured step.
    replayed: int = 0
    #: Repetitions ended by an operator that did not match.
    fallbacks: int = 0
    #: Uploads sent as a digest because the executor held their bytes.
    cache_hits: int = 0
    #: Bytes those uploads did not send.
    cache_saved_bytes: int = 0
    #: Non-blocking reads queued.
    deferred: int = 0
    #: Value reads deferred by the automatic replacement of ``item`` and its neighbours.
    auto_deferred: int = 0
    #: Deferred values asked for before their reply had arrived, so each one cost a wait.
    resolved_early: int = 0

    def snapshot(self) -> Stats:
        return Stats(**{field.name: getattr(self, field.name) for field in fields(self)})

    def __sub__(self, other: Stats) -> Stats:
        names = [field.name for field in fields(self)]
        return Stats(**{name: getattr(self, name) - getattr(other, name) for name in names})


#: Collected handles remembered for early release before the set is trimmed.
DEAD_LIMIT = 1 << 17


def _reads(entry: tuple, into: set[int]) -> None:
    """Add the handles a queued entry reads from outside itself to ``into``."""
    kind = entry[0]
    if kind == E_OP:
        into.update(entry[2])
    elif kind == E_STEP:
        into.update(entry[5])
    elif kind == E_REQUEST and entry[1] == "letify.fetch":
        into.add(entry[2][0])


def _mutates(func: Any, name: str, kwargs: dict) -> bool:
    """Whether an operator writes to an argument, from its schema where PyTorch has one."""
    try:
        return any(
            argument.alias_info is not None and argument.alias_info.is_write
            for argument in func._schema.arguments
        )
    except AttributeError:  # pragma: no cover - a PyTorch without alias information
        return name.split(".")[1].endswith("_") or "out" in kwargs


class Client:
    """One connection to a device executor."""

    def __init__(
        self,
        transport: Transport,
        *,
        process: subprocess.Popen | None = None,
        name: str = "device",
        auto_fetch: bool | None = None,
    ):
        from .value import auto_fetch_enabled

        self.transport = transport
        self.process = process
        self.name = name
        self.stats = Stats()
        #: Whether a single value read returns a deferred value instead of synchronizing.
        self.auto_fetch = auto_fetch_enabled(auto_fetch)
        #: The process that connected, the only one that may write to the transport.
        self._pid = os.getpid()
        #: Handles whose last RemoteTensor was collected, appended from ``Ref.__del__``.
        self.released: list[int] = []
        self._next_handle = 1
        self._queue: collections.deque[tuple] = collections.deque()
        self._first_at = 0.0
        self._last_at = 0.0
        #: Held while a batch is taken from the queue, so batches leave in queue order.
        self._flush_lock = threading.Lock()
        #: Held for a round trip, so replies are read in the order requests were sent.
        self._request_lock = threading.RLock()
        self._outbox: queue.SimpleQueue = queue.SimpleQueue()
        #: Batches handed to the sender thread and not yet written, guarded by ``_flush_lock``.
        self._unsent = 0
        self._waiting = threading.Event()
        #: Handles below this were created by an entry already sent, so they may be released.
        self._sent_upto = 1
        self._closed = False
        self._lost: str | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=40)
        self._templates: dict[tuple, int] = {}
        self._mutating: dict[int, bool] = {}
        self._steps: dict[int, Any] = {}
        self.tracer = Tracer()
        #: The executor's upload table as keys and lengths, updated in the executor's order.
        self._uploads: collections.OrderedDict[tuple, int] = collections.OrderedDict()
        self._upload_bytes = 0
        self._upload_budget = UPLOAD_CACHE_BYTES
        #: A budget set here and not yet sent, guarded by ``_flush_lock``.
        self._budget_unsent: int | None = None
        #: The consumers of each reply-carrying batch sent and not yet answered, in send order.
        self._replies: collections.deque[list] = collections.deque()
        #: Non-blocking reads whose fetch entry is still queued, by ``id`` of the entry.
        self._later: dict[int, Pending] = {}
        #: Non-blocking reads not yet applied, by ``id`` of the CPU tensor they fill.
        self.unfilled: dict[int, Pending] = {}
        #: A failure carried by a reply no synchronization waited for.
        self._deferred: tuple | None = None
        self.hello: dict[str, Any] = {}
        self._sender: threading.Thread | None = None
        #: The runtime's kernel choice for each signature it was asked about.
        self._kernels: dict[tuple, str] = {}
        #: Handles whose last RemoteTensor was collected, as taken by a batch. Updated only
        #: under ``_flush_lock``, so a batch reads a consistent set.
        self._dead: set[int] = set()

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        """Read the executor's first message, check its PyTorch, and start the threads."""
        head, _buffers = self._recv()
        self.hello = pickle.loads(head)
        check_worker_version(local=torch.__version__, remote=str(self.hello["torch"]))
        self._sender = threading.Thread(
            target=self._send_loop, name=f"{self.name}-send", daemon=True
        )
        self._sender.start()
        threading.Thread(target=self._linger, name=f"{self.name}-linger", daemon=True).start()

    def close(self) -> None:
        with self._request_lock:
            if self._closed:
                return
            if self._lost is None:
                try:
                    self._cut()
                    self._flush()
                    with self._flush_lock:
                        self._unsent += 1
                        self._outbox.put((pickle.dumps({"close": True}, protocol=5), [], None))
                except (RuntimeLost, TransportClosed):
                    pass
            self._closed = True
            if REPLAY[0] is self:
                REPLAY[0] = None
            self._waiting.set()
            self._outbox.put(None)
        if self._sender is not None:
            self._sender.join(timeout=30)
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

    @property
    def queued(self) -> int:
        """Entries queued and not yet sent."""
        return len(self._queue)

    # -- upload cache -----------------------------------------------------------------

    def set_upload_cache_bytes(self, budget: int) -> None:
        """Bound the executor's upload table to ``budget`` bytes, 0 for no cache."""
        with self._flush_lock:
            self._upload_budget = budget
            self._evict_uploads()
            self._budget_unsent = budget

    def _evict_uploads(self) -> None:
        uploads = self._uploads
        while self._upload_bytes > self._upload_budget:
            _key, size = uploads.popitem(last=False)
            self._upload_bytes -= size

    # -- templates ------------------------------------------------------------------

    def _template(
        self, structure: tuple, name: str, args: tuple, kwargs: dict, defer: list | None = None
    ) -> int:
        """The structure's template number, defining it first when it is new.

        A definition is queued, or appended to ``defer`` for a request that carries it.
        """
        tid = self._templates.get(structure)
        if tid is None:
            tid = len(self._templates) + 1
            self._templates[structure] = tid
            self._mutating[tid] = _mutates(structure[0], name, kwargs)
            definition = (E_DEFINE, tid, name, layout(args, kwargs))
            if defer is not None:
                defer.append(definition)
            else:
                self._enqueue(definition, check=False, counted=False)
            self.stats.templates += 1
        return tid

    # -- dispatch -------------------------------------------------------------------

    def put(
        self,
        structure: tuple,
        name: str,
        args: tuple,
        kwargs: dict,
        plan: Any,
        tensors: list,
        scalars: list,
        blobs: list,
        outs: list,
        big: bool,
        key: tuple | None = None,
    ) -> None:
        """Queue or replay one operator whose outputs ``tensor.dispatch`` already built.

        ``key`` is the operator's metadata key, from which a matched position's reader is
        recorded, or None when the arguments cannot be a key.
        """
        if self._lost is not None or self._closed:
            self._check_open()
        tid = self._templates.get(structure)
        if tid is None:
            tid = self._template(structure, name, args, kwargs)
        self.stats.ops += 1
        handles = tuple([tensor._ref.handle for tensor in tensors])
        tracer = self.tracer
        if tracer.active is not None:
            if tracer.match(tid, plan.shape_id, handles, scalars, blobs):
                self.stats.replayed += 1
                readers = tracer.active.readers
                if readers[tracer.pos - 1] is None:
                    readers[tracer.pos - 1] = self._reader(structure, name, args, kwargs, plan, key)
                if tracer.pos == tracer.size:
                    self._enqueue((E_STEP, *tracer.take()))  # type: ignore[misc]
                    tracer.begin(tracer.active, self._next_handle)
                    if big:
                        self._flush()
                elif big:
                    self._enqueue((E_STEP, *tracer.take()))  # type: ignore[misc]
                    self._flush()
                return
            self._fall_back()
        self._enqueue((E_OP, tid, handles, tuple(scalars), tuple(blobs), tuple(outs), None))
        found = tracer.record(tid, plan.shape_id, handles, outs)
        if found is not None:
            step, new = found
            if new:
                step.mutates = any(self._mutating[op[0]] for op in step.ops)
                self._steps[step.sid] = step
                ops = tuple((op[0], op[2], op[3]) for op in step.ops)
                self._enqueue((E_STEP_DEFINE, step.sid, ops), check=False, counted=False)
                self.stats.steps += 1
            tracer.begin(step, self._next_handle)
            REPLAY[0] = self
        if big:
            self._flush()

    @staticmethod
    def _reader(
        structure: tuple, name: str, args: tuple, kwargs: dict, plan: Any, key: tuple | None
    ) -> tuple:
        """The reader entry of a step position, as spec "Step capture" describes."""
        if key is None or len(key) < 1 or key[0] is not structure:
            return (None,)
        _name, floats_keyed, offsets_keyed = _NAMES[structure[0]]
        read = reader(args, kwargs, floats_keyed, offsets_keyed)
        if read is None:
            return (None,)
        return (structure[0], read, plan, key[1:], structure)

    def replay(self, func: Any, args: tuple, kwargs: dict, given: Any) -> Any:
        """Replay one operator through its position's reader, or ``MISS`` for the full path."""
        tracer = self.tracer
        step = tracer.active
        if step is None:
            return MISS
        pos = tracer.pos
        entry = step.readers[pos]
        if entry is None or entry[0] is not func:
            return MISS
        got = entry[1](args, kwargs)
        if got is None:
            return MISS
        tensors, scalars, tail = got
        if tensors:
            if tensors[0]._ref.client is not self or (given is not None and given is not self):
                return MISS
        elif given is not self:
            return MISS
        if tail == entry[3]:
            plan = entry[2]
        else:
            try:
                plan = _CACHE.get((entry[4], *tail))
            except TypeError:  # pragma: no cover - a key value that cannot be hashed
                return MISS
            if plan is None:
                return MISS
        expected = step.ops[pos]
        if plan.shape_id != expected[1]:
            return MISS
        if self._lost is not None or self._closed:
            self._check_open()
        handles = tuple([tensor._ref.handle for tensor in tensors])
        if not tracer.match(expected[0], expected[1], handles, scalars, ()):
            return MISS
        stats = self.stats
        stats.ops += 1
        stats.cached += 1
        stats.replayed += 1
        result, _outs = outputs(plan, self, tensors)
        if tracer.pos == tracer.size:
            self._enqueue((E_STEP, *tracer.take()))  # type: ignore[misc]
            tracer.begin(step, self._next_handle)
        return result

    def _fall_back(self) -> None:
        tracer = self.tracer
        if tracer.active is None:
            return
        entry = tracer.take()
        if entry is not None:
            self._enqueue((E_STEP, *entry))
        if tracer.pos:
            self.stats.fallbacks += 1
        tracer.active = None
        if REPLAY[0] is self:
            REPLAY[0] = None
        tracer.reset()

    def _cut(self) -> None:
        """Queue the matched part of an unfinished repetition, which then continues."""
        entry = self.tracer.take()
        if entry is not None:
            self._enqueue((E_STEP, *entry), check=False)

    # -- queue ----------------------------------------------------------------------

    def _enqueue(self, entry: tuple, *, check: bool = True, counted: bool = True) -> None:
        """Queue one entry. The queue is sent when it is full or aged."""
        pending = self._queue
        now = time.monotonic()
        if not pending:
            self._first_at = now
            self._waiting.set()
        pending.append(entry)
        self._last_at = now
        if counted:
            self.stats.entries += 1
        if check and (
            len(pending) >= BATCH_OPS
            or now - self._first_at >= LINGER_S
            or len(self.released) >= BATCH_RELEASES
        ):
            self._flush()

    def _linger(self) -> None:
        """Send a queue that nothing has been added to for ``IDLE_S``.

        The dispatching thread sends every other aged queue itself, so this thread wakes
        only when dispatch has paused and does not take the GIL from a running step.
        """
        while True:
            self._waiting.wait()
            if self._closed or self._lost is not None:
                return
            wait = self._last_at + IDLE_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
                continue
            self._waiting.clear()
            if self._queue:
                try:
                    self._flush()
                except RuntimeLost:
                    return
                if self._queue:
                    self._waiting.set()

    def _flush(
        self,
        *,
        waiter: Waiter | None = None,
        count: int | None = None,
        extra: Sequence[tuple] = (),
        through: tuple | None = None,
    ) -> None:
        """Take ``count`` entries, or all, plus ``extra``, and hand them to the sender thread.

        ``through`` takes the entries up to and including that queued entry, and none when it
        has already left the queue.
        """
        with self._flush_lock:
            # Releases are taken before entries, so every entry dispatched before one of
            # these handles was collected is taken below or was sent by an earlier batch.
            taken_releases = self.released[:]
            if taken_releases:
                del self.released[: len(taken_releases)]
                self._dead.update(taken_releases)
            pending = self._queue
            if through is not None:
                count = 0
                for index, queued in enumerate(pending):
                    if queued is through:
                        count = index + 1
                        break
                if not count:
                    # Put back the releases taken above so a later batch still sends them.
                    self.released[:0] = taken_releases
                    return
            taken = len(pending) if count is None else count
            entries = [pending.popleft() for _ in range(taken)]
            entries.extend(extra)
            buffers: list = []
            keep: list = []
            # Placed in entry order, which is the order the executor updates its upload table.
            for index, entry in enumerate(entries):
                kind = entry[0]
                if kind == E_OP:
                    if entry[4] and any(type(blob) is Big for blob in entry[4]):
                        placed = self._place(entry[4], buffers, keep)
                        entries[index] = (*entry[:4], placed, *entry[5:])
                elif kind == E_STEP and entry[7] and any(type(blob) is Big for blob in entry[7]):
                    placed = self._place(entry[7], buffers, keep)
                    entries[index] = (*entry[:7], placed, entry[8])
            upto = self._sent_upto
            located = False
            for index in range(len(entries) - 1, -1, -1):
                entry = entries[index]
                kind = entry[0]
                if kind == E_OP:
                    if not located:
                        outs = entry[5]
                        if type(outs) is int:
                            upto, located = max(upto, outs + DESCRIBED_OUTPUTS), True
                        else:
                            made = [handle for handle in outs if handle is not None]
                            if made:
                                upto, located = max(upto, made[-1] + 1), True
                elif kind == E_STEP and not located:
                    step = self._steps[entry[1]]
                    made = entry[2] + step.news_before[entry[4]]
                    upto, located = max(upto, made), True
            self._keep(entries, pending)
            self._sent_upto = upto
            released: list[int] = []
            if taken_releases:
                tracer = self.tracer
                if pending or (tracer.active is not None and tracer.pos > tracer.start):
                    # An entry left queued, or matched operators not yet queued, may still
                    # use a released handle, so every release waits for a later batch.
                    self.released.extend(taken_releases)
                else:
                    released = [handle for handle in taken_releases if handle < upto]
                    if len(released) < len(taken_releases):
                        self.released.extend(h for h in taken_releases if h >= upto)
            consumers: list = []
            later = self._later
            if later:
                for entry in entries:
                    if entry[0] == E_REQUEST:
                        read = later.pop(id(entry), None)
                        if read is not None:
                            consumers.append(read)
            if waiter is not None:
                consumers.append(waiter)
            reply = bool(consumers)
            budget = self._budget_unsent
            if not entries and not released and not reply and budget is None:
                return
            message: dict[str, Any] = {"entries": entries, "release": released, "reply": reply}
            if budget is not None:
                message["cache_bytes"] = budget
                self._budget_unsent = None
            head = pickle.dumps(message, protocol=5)
            direct = waiter is not None and self._unsent == 0 and not self._replies
            if reply:
                self._replies.append(consumers)
            if direct:
                # The waiting thread writes its own synchronization, in order, because no
                # earlier batch is queued for or being written by the sender thread and no
                # earlier reply is unread.
                try:
                    self.transport.send(head, buffers)
                except TransportClosed as exc:
                    if self._lost is None:
                        self._lost = str(exc)
            else:
                self._unsent += 1
                self._outbox.put((head, buffers, keep))
            self.stats.batches += 1
            self.stats.released += len(released)
            self.stats.sent_bytes += len(head) + sum(view.nbytes for view in buffers)
            if pending:
                self._first_at = time.monotonic()

    def _keep(self, entries: list, pending: collections.deque) -> None:
        """Fill ``keep`` in the batch's step entries, as spec "Step capture" describes.

        An offset released at a position the entry runs is kept when its handle is not in
        ``_dead``, or when an entry after it in this batch, a queued entry or the unfinished
        repetition reads the handle. Called under ``_flush_lock``.
        """
        if not any(entry[0] == E_STEP and entry[8] is None for entry in entries):
            return
        dead = self._dead
        read: set[int] = set(self.tracer.externals)
        for queued in pending:
            _reads(queued, read)
        lowest = self.tracer.first if self.tracer.active is not None else self._next_handle
        for index in range(len(entries) - 1, -1, -1):
            entry = entries[index]
            if entry[0] == E_STEP:
                first = entry[2]
                lowest = min(lowest, first)
                if entry[8] is None:
                    frees = self._steps[entry[1]].frees
                    kept = tuple(
                        offset
                        for position in range(entry[3], entry[4])
                        for offset in frees[position]
                        if first + offset not in dead or first + offset in read
                    )
                    entries[index] = (*entry[:8], kept)
            _reads(entry, read)
        if len(dead) > DEAD_LIMIT:
            for queued in pending:
                if queued[0] == E_STEP:
                    lowest = min(lowest, queued[2])
            self._dead = {handle for handle in dead if handle >= lowest}

    def _place(self, blobs: tuple, buffers: list, keep: list) -> tuple:
        """Each large blob as a buffer index, ``("p", index, key)`` or ``("h", key)``.

        ``("p", index, key)`` sends the bytes and inserts them into the upload table;
        ``("h", key)`` names bytes the table holds, as spec "Upload cache" describes.
        """
        placed: list = []
        uploads = self._uploads
        for blob in blobs:
            if type(blob) is not Big:
                placed.append(blob)
                continue
            view = blob.view
            size = view.nbytes
            if CACHE_MIN_BYTES <= size <= self._upload_budget:
                threads = blake3.AUTO if size >= HASH_THREADS_BYTES else 1
                key = (blake3(view, max_threads=threads).digest(), size)
                if key in uploads:
                    uploads.move_to_end(key)
                    self.stats.cache_hits += 1
                    self.stats.cache_saved_bytes += size
                    placed.append(("h", key))
                    continue
                uploads[key] = size
                self._upload_bytes += size
                self._evict_uploads()
                buffers.append(view)
                keep.append(blob.keep)
                placed.append(("p", len(buffers) - 1, key))
                continue
            buffers.append(view)
            keep.append(blob.keep)
            placed.append(len(buffers) - 1)
        return tuple(placed)

    def _send_loop(self) -> None:
        """Write batches in the order they were handed over."""
        while True:
            item = self._outbox.get()
            if item is None:
                return
            head, buffers, keep = item
            try:
                self.transport.send(head, buffers)
            except TransportClosed as exc:
                if self._lost is None:
                    self._lost = str(exc)
                return
            finally:
                with self._flush_lock:
                    self._unsent -= 1
            del item, keep, buffers

    # -- synchronization ---------------------------------------------------------------

    def _dependency_cut(self, handle: int) -> int | None:
        """How many queued entries a read of ``handle`` needs, or None for all of them."""
        entries = list(self._queue)
        for index in range(len(entries) - 1, -1, -1):
            entry = entries[index]
            kind = entry[0]
            if kind == E_OP:
                if self._mutating[entry[1]]:
                    return None
                outs = entry[5]
                if type(outs) is not int and handle in outs:
                    return index + 1
            elif kind == E_STEP:
                step = self._steps[entry[1]]
                if step.mutates:
                    return None
                if entry[2] <= handle < entry[2] + step.news_before[entry[4]]:
                    return index + 1
        return 0

    def _request(
        self, entry: tuple, *, handle: int | None = None, defines: Sequence[tuple] = ()
    ) -> tuple[list, list]:
        """Send what ``entry`` needs with it, and wait for the reply. One round trip.

        ``defines`` are template definitions ``entry`` uses, sent just ahead of it.
        """
        with self._request_lock:
            self._check_open()
            count = None
            if handle is not None and not self.tracer.unfinished:
                count = self._dependency_cut(handle)
            if count is None:
                self._cut()
            self.stats.ops += 1
            self.stats.entries += 1
            waiter = Waiter()
            self._flush(waiter=waiter, count=count, extra=[*defines, entry])
            while not waiter.done:
                self._receive()
            self.stats.round_trips += 1
            failure, self._deferred = self._deferred or waiter.failure, None
        if failure is not None:
            op, message, remote_traceback = failure
            raise RemoteError(f"{op} failed on the runtime: {message}", remote_traceback)
        return waiter.results, waiter.buffers

    def _receive(self) -> None:
        """Read the oldest unread reply and hand its results to their consumers."""
        head, buffers = self._recv()
        consumers = self._replies.popleft()
        reply = pickle.loads(head)
        results = reply["results"]
        failure = reply.get("failure")
        waiter = None
        reported = False
        # Reads come first in a batch, so the i-th read owns the i-th result. A waiter's
        # entry may ask for no result, and it takes the whole list.
        for position, consumer in enumerate(consumers):
            if type(consumer) is Waiter:
                waiter = consumer
                continue
            consumer.apply(results[position], buffers, failure)
            reported = reported or consumer.error is not None
            self.unfilled.pop(id(consumer.target), None)
        if waiter is not None:
            waiter.results, waiter.buffers, waiter.failure = results, buffers, failure
            waiter.done = True
        elif failure is not None and not reported and self._deferred is None:
            self._deferred = failure

    def read_later(
        self, tensor: Any, dtype: torch.dtype | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Queue a fetch of ``tensor`` and return the CPU tensor it fills, without waiting."""
        self._check_open()
        if len(self._replies) >= UNREAD_LIMIT:
            with self._request_lock:
                if self._replies:
                    self._receive()
        if target is None:
            with torch._C.DisableTorchFunction():
                target = torch.empty(tuple(tensor.shape), dtype=dtype or tensor.dtype)
        name = None if dtype is None else str(dtype).split(".")[-1]
        entry = (E_REQUEST, "letify.fetch", (tensor._ref.handle, name), "fetch")
        read = Pending(target, entry)
        self._cut()
        self.stats.ops += 1
        self.stats.deferred += 1
        self._later[id(entry)] = read
        self.unfilled[id(target)] = read
        self._enqueue(entry)
        return target

    def read_pending(self, tensor: Any, dtype: torch.dtype | None = None) -> Pending:
        """Queue a fetch of ``tensor`` and return its pending read.

        The CPU tensor it fills is not registered in ``unfilled``, because only the deferred
        value of spec "Deferred value reads" holds it and no torch function ever sees it.
        """
        self._check_open()
        if len(self._replies) >= UNREAD_LIMIT:
            with self._request_lock:
                if self._replies:
                    self._receive()
        with torch._C.DisableTorchFunction():
            target = torch.empty(tuple(tensor.shape), dtype=dtype or tensor.dtype)
        entry = (E_REQUEST, "letify.fetch", (tensor._ref.handle, None), "fetch")
        read = Pending(target, entry)
        self._cut()
        self.stats.ops += 1
        self.stats.auto_deferred += 1
        self._later[id(entry)] = read
        self._enqueue(entry)
        return read

    def wait(self, read: Pending) -> torch.Tensor:
        """Send what ``read`` needs and read replies until it is applied."""
        with self._request_lock:
            if not read.done:
                self._check_open()
                self._flush(through=read.entry)
                while not read.done:
                    self._receive()
        if read.error is not None:
            raise read.error
        return read.target

    def wait_used(self, values: Any) -> None:
        """Wait for every unfilled tensor among ``values``, or one list level down."""
        unfilled = self.unfilled
        for value in values:
            if type(value) in (list, tuple):
                self.wait_used(value)
                continue
            read = unfilled.get(id(value))
            if read is not None and read.target is value:
                self.wait(read)

    def synchronize(self) -> None:
        """Wait for every queued operator, raising the first failure among them."""
        self._request((E_REQUEST, "letify.live", (), None))

    def call(self, name: str, *args: Any, reply: bool = True) -> Any:
        """A letify request to the executor, such as a seed or a memory query."""
        if not reply:
            self._check_open()
            self._cut()
            self.stats.ops += 1
            self._enqueue((E_REQUEST, name, args, None))
            return None
        results, _buffers = self._request((E_REQUEST, name, args, "value"))
        return results[-1]

    def kernel(self, function: str, tensors: Sequence[Any], flags: dict) -> str:
        """The backend the runtime's own dispatch picks for ``function``, asked once per signature.

        ``tensors`` may hold None for an absent argument. The answer is a backend name, such
        as ``Cudnn`` or ``Native`` for batch norm and ``FLASH_ATTENTION`` or ``MATH`` for
        attention.
        """
        described = tuple(
            None
            if tensor is None
            else (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype).split(".")[-1])
            for tensor in tensors
        )
        key = (function, described, tuple(sorted(flags.items())))
        answer = self._kernels.get(key)
        if answer is None:
            answer = self.call("letify.kernel", function, described, dict(flags))
            self._kernels[key] = answer
        return answer

    def live_handles(self) -> int:
        """How many tensors the executor holds, after sending pending releases."""
        with self._request_lock:
            self._check_open()
            self._cut()
            # Releases apply after their batch's entries, so they go in a batch of their own.
            self._flush()
        return int(self.call("letify.live"))

    def read_value(self, tensor: Any) -> Any:
        """``Tensor.item()`` and the reads built on it."""
        from .tensor import _SCALAR

        structure = (_SCALAR, tensor._form)
        defines: list = []
        name = "aten._local_scalar_dense.default"
        tid = self._template(structure, name, (tensor,), {}, defines)
        handle = tensor._ref.handle
        entry = (E_OP, tid, (handle,), (), (), (), "value")
        results, _buffers = self._request(entry, handle=handle, defines=defines)
        return results[-1]

    def fetch(self, tensor: Any, dtype: str | None) -> torch.Tensor:
        """Copy a RemoteTensor's values into a new CPU tensor."""
        handle = tensor._ref.handle
        results, buffers = self._request(
            (E_REQUEST, "letify.fetch", (handle, dtype), "fetch"), handle=handle
        )
        index, shape, got = results[-1]
        data = buffers[index]
        kind = getattr(torch, got)
        if not len(data):
            return torch.empty(shape, dtype=kind)
        return torch.frombuffer(data, dtype=kind).reshape(shape)

    def execute_now(
        self,
        structure: tuple,
        name: str,
        args: tuple,
        kwargs: dict,
        tensors: list,
        scalars: list,
        blobs: list,
    ) -> Any:
        """Run an operator whose metadata could not be computed here, and describe it."""
        from .tensor import from_description

        self._check_open()
        with self._request_lock:
            self._fall_back()
            self.tracer.reset()
            defines: list = []
            tid = self._template(structure, name, args, kwargs, defines)
            first = self._next_handle
            self._next_handle += DESCRIBED_OUTPUTS
            handles = tuple(tensor._ref.handle for tensor in tensors)
            entry = (E_OP, tid, handles, tuple(scalars), tuple(blobs), first, "describe")
            results, _buffers = self._request(entry, defines=defines)
        return from_description(self, first, results[-1])

    # -- errors -------------------------------------------------------------------------

    def _recv(self) -> tuple[Any, list]:
        try:
            return self.transport.recv()
        except TransportClosed as exc:
            raise self._lose(str(exc)) from exc

    def _lose(self, reason: str) -> RuntimeLost:
        self._lost = reason
        self._waiting.set()
        if self.process is not None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        tail = "\n".join(self._stderr)
        detail = f"\n--- device worker stderr ---\n{tail}" if tail else ""
        return RuntimeLost(f"{self.name}: the device worker was lost: {reason}{detail}")

    def _check_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeLost(
                f"{self.name}: this process was forked from process {self._pid}, which owns "
                f"the device worker's channel, so it cannot send on it"
            )
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


class Fetch:
    """What ``letify.fetch`` returns: an awaitable resolving to a CPU tensor."""

    __slots__ = ("client", "read", "value")

    def __init__(self, client: Client | None, read: Pending | None, value: torch.Tensor):
        self.client = client
        self.read = read
        self.value = value

    def __await__(self) -> Iterator[Any]:
        if self.client is not None and self.read is not None and not self.read.done:
            yield from asyncio.to_thread(self.client.wait, self.read).__await__()
        if self.read is not None and self.read.error is not None:
            raise self.read.error
        return self.value


def fetch(tensor: torch.Tensor) -> Fetch:
    """Queue a read of ``tensor`` now and return an awaitable for its CPU copy."""
    from .tensor import RemoteTensor

    if type(tensor) is RemoteTensor:
        client = tensor._ref.client
        target = client.read_later(tensor)
        return Fetch(client, client.unfilled.get(id(target)), target)
    return Fetch(None, None, tensor.detach().cpu())


def connect(
    command: list[str],
    *,
    device: str,
    env: dict[str, str] | None = None,
    name: str = "device",
    auto_fetch: bool | None = None,
) -> Client:
    """Start a device executor as its own process with this command and return a client."""
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
    client = Client(transport, process=process, name=name, auto_fetch=auto_fetch)
    threading.Thread(
        target=client._drain_stderr, args=(process.stderr,), name=f"{name}-stderr", daemon=True
    ).start()
    try:
        transport.raw(b"%d\n" % len(source) + source)
        client.start()
    except (TransportClosed, RuntimeLost) as exc:
        process.kill()
        process.wait()
        if isinstance(exc, RuntimeLost):
            raise
        raise client._lose(str(exc)) from exc
    return client


def attach(
    channel: Any, *, device: str, name: str = "device", auto_fetch: bool | None = None
) -> Client:
    """Start a device executor inside a persistent channel's call worker and return a client."""
    check_torch_version(torch.__version__)
    transport = ChannelTransport(channel.connection)
    client = Client(transport, name=name, auto_fetch=auto_fetch)
    channel.request(
        {"op": "device", "device": device, "source": worker_source(None).decode()}, timeout=600
    )
    client.start()
    return client


__all__ = [
    "BOOTSTRAP",
    "Client",
    "Stats",
    "attach",
    "connect",
    "fetch",
    "worker_command",
    "worker_source",
]
