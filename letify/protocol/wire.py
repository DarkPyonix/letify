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
import os
import pickle
import struct
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

#: The largest payload of one ``DATA`` frame, so other streams interleave between chunks.
CHUNK = 8 << 20

#: A ``bytes`` value this large or larger travels as an out-of-band buffer.
OUT_OF_BAND = 1 << 20

#: Frames at most this large are written with their header in one call.
_JOIN_LIMIT = 1 << 16

#: Containers with more items than this are not searched for large ``bytes`` values.
_WALK_LIMIT = 64

_COUNT = struct.Struct("<I")

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


def dumps(obj: object) -> tuple[bytes, list[memoryview]]:
    """Pickle ``obj`` with protocol 5, returning the pickle and its out-of-band buffers."""
    buffers: list[pickle.PickleBuffer] = []
    head = pickle.dumps(_wrap(obj), protocol=5, buffer_callback=buffers.append)
    return head, [buffer.raw() for buffer in buffers]


def loads(head: bytes | bytearray | memoryview, buffers: list) -> object:
    return pickle.loads(head, buffers=buffers)


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
        lengths = struct.pack(f"<{len(views)}Q", *[view.nbytes for view in views])
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

    __slots__ = ("buffers", "head", "index", "kind", "offset")

    def __init__(self, kind: int, head: memoryview, lengths: tuple):
        self.kind = kind
        self.head = head
        self.buffers = [bytearray(length) for length in lengths]
        self.index = 0
        self.offset = 0
        self._skip_empty()

    def _skip_empty(self) -> None:
        while self.index < len(self.buffers) and self.offset == len(self.buffers[self.index]):
            self.index += 1
            self.offset = 0

    @property
    def complete(self) -> bool:
        return self.index == len(self.buffers)


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
            buffer = partial.buffers[partial.index]
            take = min(len(buffer) - partial.offset, length)
            self._exact(memoryview(buffer)[partial.offset : partial.offset + take])
            partial.offset += take
            length -= take
            partial._skip_empty()
        if not partial.complete:
            return None
        del self._open[stream]
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
    "HEADER",
    "HELLO",
    "MAGIC",
    "OUT_OF_BAND",
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
]
