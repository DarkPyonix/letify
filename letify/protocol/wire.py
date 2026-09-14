"""Binary frames, the wire of a persistent channel.

This module owns the frame header, turning one message into frames and frames back into
messages, as spec "Frames" describes. It does not own which requests exist or what a
channel does with a reply: that is ``runtime.channel`` on the client and the worker source
on the runtime.

It imports the standard library only, because its source is also sent to the runtime ahead
of the worker body, where letify may not be installed. One implementation therefore frames
both ends.
"""

from __future__ import annotations

import base64
import functools
import io
import os
import pickle
import struct
import sys
import threading

#: First two bytes of every frame.
MAGIC = b"LF"

#: magic, type, flags, stream id, payload length.
HEADER = struct.Struct("<2sBBIQ")

HELLO = 1
REQUEST = 2
REPLY = 3
DATA = 4
STDOUT = 5
STDERR = 6
SHUTDOWN = 7

#: The stream that carries the PyTorch device executor's messages. Calls take odd ids.
DEVICE_STREAM = 2

#: The largest payload of one ``DATA`` frame, so other streams interleave between chunks.
CHUNK = 8 << 20

#: A ``bytes`` value this large or larger travels as an out-of-band buffer.
OUT_OF_BAND = 1 << 20

#: A tensor this many bytes or larger travels as an out-of-band buffer.
TENSOR_OUT_OF_BAND = 64 << 10

#: Frames at most this large are written with their header in one call.
_JOIN_LIMIT = 1 << 16

#: Containers with more items than this are not searched for large ``bytes`` values.
_WALK_LIMIT = 64

_COUNT = struct.Struct("<I")

#: High bit of a buffer length in a head: the buffer unpickles as ``bytes``, filled in place.
_AS_BYTES = 1 << 63

#: The pipe size both ends ask for on Linux.
PIPE_SIZE = 1 << 20

if __name__.startswith("letify."):
    from ..errors import ProtocolError as FrameError
else:  # pragma: no cover - the copy that runs inside a worker

    class FrameError(Exception):
        """A frame that does not follow the layout."""


class _OutOfBand:
    """A ``bytes`` value that pickles as an out-of-band buffer and unpickles as ``bytes``."""

    __slots__ = ("value",)

    def __init__(self, value: bytes):
        self.value = value

    def __reduce_ex__(self, protocol: object) -> tuple:
        return bytes, (pickle.PickleBuffer(self.value),)


def _wrap(value: object, depth: int = 0) -> object:
    """Mark large ``bytes`` values, at the top level or in small lists, tuples and dicts."""
    kind = type(value)
    if kind is bytes:
        return _OutOfBand(value) if len(value) >= OUT_OF_BAND else value  # type: ignore[arg-type]
    if depth >= 3:
        return value
    if kind is dict and len(value) <= _WALK_LIMIT:  # type: ignore[arg-type]
        return {k: _wrap(v, depth + 1) for k, v in value.items()}  # type: ignore[attr-defined]
    if kind in (list, tuple) and len(value) <= _WALK_LIMIT:  # type: ignore[arg-type]
        return kind(_wrap(v, depth + 1) for v in value)  # type: ignore[attr-defined,operator]
    return value


class _Reduced:
    """An argument of a reduction that pickles as a reduction of its own."""

    __slots__ = ("reduction",)

    def __init__(self, reduction: tuple):
        self.reduction = reduction

    def __reduce_ex__(self, protocol: object) -> tuple:
        return self.reduction


def reduce_tensor(obj: object) -> object:
    """A reduction that sends a CPU tensor's bytes as a buffer, or ``NotImplemented``.

    ``Tensor.__reduce_ex__`` copies the whole storage into the pickle. This one rebuilds the
    tensor with ``torch.frombuffer``, ``reshape`` and ``requires_grad_``, which a runtime
    without letify can unpickle, as spec "Frames" describes.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return NotImplemented
    kind = type(obj)
    if kind is not torch.Tensor and kind is not torch.nn.Parameter:
        return NotImplemented
    tensor = obj  # type: ignore[assignment]
    if (
        tensor.device.type != "cpu"
        or tensor.layout is not torch.strided
        or not tensor.is_leaf
        or tensor.is_quantized
        or tensor.is_conj()
        or tensor.is_neg()
        or tensor.numel() == 0
    ):
        return NotImplemented
    try:
        import ctypes
    except ImportError:  # pragma: no cover - CPython always has ctypes
        return NotImplemented
    data = tensor.detach()
    if not data.is_contiguous():
        data = data.contiguous()
    size = data.numel() * data.element_size()
    array = (ctypes.c_char * size).from_address(data.data_ptr())
    # The array does not own the memory, so it holds the tensor for as long as a view does.
    array.tensor = data
    view = memoryview(array).cast("B")
    payload = pickle.PickleBuffer(view) if size >= TENSOR_OUT_OF_BAND else bytearray(view)
    flat = _Reduced((functools.partial(torch.frombuffer, dtype=data.dtype), (payload,)))
    shaped = _Reduced((torch.Tensor.reshape, (flat, tuple(data.shape))))
    if kind is torch.nn.Parameter:
        return torch.nn.Parameter, (shaped, tensor.requires_grad)
    if tensor.requires_grad:
        return torch.Tensor.requires_grad_, (shaped,)
    return shaped.reduction


class _Pickler(pickle.Pickler):
    def reducer_override(self, obj: object) -> object:
        return reduce_tensor(obj)


def pickle_parts(obj: object) -> tuple[bytes, list[pickle.PickleBuffer]]:
    """Pickle ``obj`` with protocol 5 and the tensor reducer, keeping buffers apart."""
    buffers: list[pickle.PickleBuffer] = []
    if "torch" not in sys.modules:
        return pickle.dumps(obj, protocol=5, buffer_callback=buffers.append), buffers
    file = io.BytesIO()
    _Pickler(file, protocol=5, buffer_callback=buffers.append).dump(obj)
    return file.getvalue(), buffers


def dumps(obj: object) -> tuple[bytes, list[memoryview]]:
    """Pickle ``obj`` with protocol 5, returning the pickle and its out-of-band buffers."""
    head, buffers = pickle_parts(_wrap(obj))
    return head, [buffer.raw() for buffer in buffers]


def loads(head: bytes | bytearray | memoryview, buffers: list) -> object:
    return pickle.loads(head, buffers=buffers)


def _bytes_allocator():
    """A function returning ``(bytes object, writable view of its contents)``, or None.

    The object comes from ``PyBytes_FromStringAndSize(NULL, n)``, so reading into the view
    fills the value a buffer unpickles as, with no zeroing and no copy afterwards. It is
    checked once on a small object, and None where ctypes or that layout is not available.
    """
    try:
        import ctypes

        new = ctypes.pythonapi.PyBytes_FromStringAndSize
        new.restype = ctypes.py_object
        new.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t)
        offset = bytes.__basicsize__ - 1

        def allocate(size: int):
            obj = new(None, size)
            array = (ctypes.c_char * size).from_address(id(obj) + offset)
            return obj, memoryview(array).cast("B")

        probe, view = allocate(8)
        view[:] = b"letify!!"
        del view
        if probe != b"letify!!" or hash(probe) != hash(b"letify!!"):
            return None
        return allocate
    except Exception:
        return None


_new_bytes = _bytes_allocator()


def widen_pipe(fd: int) -> None:
    """Ask for a 1 MiB pipe on Linux, or the largest the system allows. Anything else is kept."""
    try:
        import fcntl

        setting = getattr(fcntl, "F_SETPIPE_SZ", 1031)
        try:
            fcntl.fcntl(fd, setting, PIPE_SIZE)
        except OSError:
            with open("/proc/sys/fs/pipe-max-size") as limit:
                fcntl.fcntl(fd, setting, min(PIPE_SIZE, int(limit.read())))
    except (ImportError, OSError, ValueError):
        pass


class Sender:
    """Writes frames through ``write``, one frame at a time under a lock."""

    def __init__(self, write):
        self._write = write
        self.lock = threading.Lock()

    def _all(self, data) -> None:
        view = memoryview(data)
        while view:
            view = view[self._write(view) :]

    def raw(self, data: bytes) -> None:
        """Write bytes that are not a frame, such as the worker source for the bootstrap stub."""
        with self.lock:
            self._all(data)

    def frame(self, kind: int, stream: int, payload=b"") -> None:
        view = memoryview(payload).cast("B")
        header = HEADER.pack(MAGIC, kind, 0, stream, view.nbytes)
        with self.lock:
            if view.nbytes <= _JOIN_LIMIT:
                self._all(header + view.tobytes())
            else:
                self._all(header)
                self._all(view)

    def message(self, kind: int, stream: int, obj: object) -> None:
        """Send one object: a head frame, then its buffers as ``DATA`` frames."""
        head, buffers = dumps(obj)
        views = [memoryview(buffer).cast("B") for buffer in buffers]
        marks = [
            views[i].nbytes | _AS_BYTES if type(buffers[i].obj) is bytes else views[i].nbytes
            for i in range(len(views))
        ]
        self.parts(kind, stream, head, views, marks)

    def parts(self, kind: int, stream: int, head, views: list, marks: list | None = None) -> None:
        """Send an already pickled head and its buffers, written from the caller's memory."""
        if marks is None:
            marks = [view.nbytes for view in views]
        lengths = struct.pack(f"<{len(views)}Q", *marks)
        self.frame(kind, stream, _COUNT.pack(len(views)) + lengths + head)
        for view in views:
            for offset in range(0, view.nbytes, CHUNK):
                self.frame(DATA, stream, view[offset : offset + CHUNK])


class TextSender(Sender):
    """Writes each frame as one line of base64, for a transport that carries text only."""

    def frame(self, kind: int, stream: int, payload=b"") -> None:
        view = memoryview(payload).cast("B")
        header = HEADER.pack(MAGIC, kind, 0, stream, view.nbytes)
        line = base64.b64encode(header + view.tobytes()) + b"\n"
        with self.lock:
            self._all(line)


class _Partial:
    """A message whose head arrived and whose buffers are still being filled."""

    __slots__ = ("buffers", "head", "index", "kind", "offset", "views")

    def __init__(self, kind: int, head: memoryview, lengths: tuple):
        self.kind = kind
        self.head = head
        #: What ``loads`` receives, and the writable view each is filled through.
        self.buffers: list = []
        self.views: list = []
        for length in lengths:
            size = length & ~_AS_BYTES
            if length & _AS_BYTES and size and _new_bytes is not None:
                obj, view = _new_bytes(size)
            else:
                obj = bytearray(size)
                view = memoryview(obj)
            self.buffers.append(obj)
            self.views.append(view)
        self.index = 0
        self.offset = 0
        self._skip_empty()

    def _skip_empty(self) -> None:
        while self.index < len(self.views) and self.offset == self.views[self.index].nbytes:
            self.index += 1
            self.offset = 0

    @property
    def complete(self) -> bool:
        return self.index == len(self.views)


class Receiver:
    """Reads frames through ``readinto`` and reassembles messages per stream.

    ``readinto`` fills a memoryview and returns how many bytes it wrote, 0 at end of stream.
    ``DATA`` payloads are read straight into the buffer they belong to.
    """

    def __init__(self, readinto):
        self._readinto = readinto
        self._open: dict[int, _Partial] = {}
        self._header = bytearray(HEADER.size)

    def _exact(self, view: memoryview) -> None:
        got = 0
        size = view.nbytes
        while got < size:
            count = self._readinto(view[got:])
            if not count:
                raise EOFError("the stream ended inside a frame" if got else "the stream ended")
            got += count

    def next_event(self) -> tuple | None:
        """Read one frame.

        Returns ``(HELLO, 0, version)``, ``(STDOUT or STDERR, 0, bytes)``, ``(SHUTDOWN, 0,
        None)``, ``(REQUEST or REPLY, stream, (head, buffers))`` once a message is whole, or
        None for a frame that only advanced a message. Raises ``EOFError`` at end of stream.
        """
        self._exact(memoryview(self._header))
        magic, kind, _flags, stream, length = HEADER.unpack(self._header)
        if magic != MAGIC:
            raise FrameError(f"a frame arrived with magic {magic!r} instead of {MAGIC!r}")
        if kind == DATA:
            return self._data(stream, length)
        payload = bytearray(length)
        self._exact(memoryview(payload))
        if kind in (REQUEST, REPLY):
            return self._head(kind, stream, payload)
        if kind in (STDOUT, STDERR):
            return kind, 0, bytes(payload)
        if kind == HELLO:
            return kind, 0, payload.decode("ascii")
        if kind == SHUTDOWN:
            return kind, 0, None
        raise FrameError(f"a frame arrived with unknown type {kind}")

    def _head(self, kind: int, stream: int, payload: bytearray) -> tuple | None:
        view = memoryview(payload)
        (count,) = _COUNT.unpack_from(view)
        lengths = struct.unpack_from(f"<{count}Q", view, _COUNT.size)
        partial = _Partial(kind, view[_COUNT.size + 8 * count :], lengths)
        if partial.complete:
            partial.views = []
            return kind, stream, (partial.head, partial.buffers)
        self._open[stream] = partial
        return None

    def _data(self, stream: int, length: int) -> tuple | None:
        partial = self._open.get(stream)
        if partial is None:
            raise FrameError(f"data arrived for stream {stream}, which has no open message")
        while length:
            if partial.complete:
                raise FrameError(f"stream {stream} sent more data than its message declared")
            view = partial.views[partial.index]
            take = min(view.nbytes - partial.offset, length)
            self._exact(view[partial.offset : partial.offset + take])
            partial.offset += take
            length -= take
            partial._skip_empty()
        if not partial.complete:
            return None
        del self._open[stream]
        partial.views = []
        return partial.kind, stream, (partial.head, partial.buffers)


def chunks_readinto(read_chunks):
    """A ``readinto`` over a source of byte chunks, such as decoded base64 frame lines.

    ``read_chunks`` returns the next list of chunks, empty at end of stream.
    """
    pending = bytearray()

    def readinto(view: memoryview) -> int:
        while not pending:
            chunks = read_chunks()
            if not chunks:
                return 0
            for chunk in chunks:
                pending.extend(chunk)
        count = min(len(pending), view.nbytes)
        view[:count] = pending[:count]
        del pending[:count]
        return count

    return readinto


def fd_writer(fd: int):
    """A ``write`` for ``Sender`` over a file descriptor."""
    return lambda view: os.write(fd, view)


__all__ = [
    "CHUNK",
    "DATA",
    "DEVICE_STREAM",
    "HEADER",
    "HELLO",
    "MAGIC",
    "OUT_OF_BAND",
    "PIPE_SIZE",
    "REPLY",
    "REQUEST",
    "SHUTDOWN",
    "STDERR",
    "STDOUT",
    "FrameError",
    "Receiver",
    "Sender",
    "TextSender",
    "chunks_readinto",
    "dumps",
    "fd_writer",
    "loads",
    "widen_pipe",
]
