"""The Transport between a PyTorch forwarding client and its device worker.

This module owns the ``Transport`` interface and ``StreamTransport``, which frames one
message as a head plus out-of-band buffers over a pair of file descriptors, as spec
"Transport" describes. It does not own what a message means: that is ``client`` and
``executor``.

It imports the standard library only, because its source is sent to the runtime ahead of
the executor, where letify may be absent. ``tensor_view`` takes a tensor but uses only
``ctypes`` on it.
"""

from __future__ import annotations

import ctypes
import os
import struct
from typing import Any, Protocol

#: Head length and buffer count.
_PREFIX = struct.Struct("<QI")

#: At most this many views are passed to one ``os.writev`` call.
_IOV_MAX = 512


class TransportClosed(Exception):
    """The other end closed the stream, or it ended in the middle of a message."""


class Transport(Protocol):
    """A message pipe carrying a head and out-of-band buffers."""

    def send(self, head: bytes, buffers: list[Any]) -> None: ...

    def recv(self) -> tuple[bytes, list[bytearray]]: ...

    def close(self) -> None: ...


class StreamTransport:
    """One message stream over a readable and a writable file descriptor.

    A message is the head length (8 bytes), the buffer count (4 bytes), one 8 byte length
    per buffer, the head and then each buffer, all little-endian. Buffers are written from
    the caller's memory with ``os.writev`` and read into ``bytearray``s sized from the
    lengths, so nothing is joined or copied in between.
    """

    def __init__(
        self,
        read_fd: int | None,
        write_fd: int | None,
        *,
        readinto: Any = None,
        owns_fds: bool = True,
    ):
        self.owns_fds = owns_fds
        self.read_fd = read_fd
        self.write_fd = write_fd
        if readinto is not None:
            self._readinto = readinto
        elif read_fd is not None:
            fd = read_fd
            self._readinto = lambda view: os.readv(fd, [view])
        else:
            self._readinto = None

    def send(self, head: bytes, buffers: list[Any]) -> None:
        if self.write_fd is None:
            raise TransportClosed("this transport has no writable end")
        views = [memoryview(buffer).cast("B") for buffer in buffers]
        prefix = _PREFIX.pack(len(head), len(views))
        if views:
            prefix += struct.pack(f"<{len(views)}Q", *(view.nbytes for view in views))
        self._write_all([memoryview(prefix), memoryview(head), *views])

    def _write_all(self, views: list[memoryview]) -> None:
        pending = [view for view in views if view.nbytes]
        try:
            while pending:
                written = os.writev(self.write_fd, pending[:_IOV_MAX])
                while written and pending:
                    first = pending[0]
                    if written >= first.nbytes:
                        written -= first.nbytes
                        pending.pop(0)
                    else:
                        pending[0] = first[written:]
                        written = 0
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            raise TransportClosed(f"the stream closed while writing: {exc}") from exc

    def _exact(self, view: memoryview) -> None:
        while view.nbytes:
            try:
                count = self._readinto(view)
            except OSError as exc:
                raise TransportClosed(f"the stream closed while reading: {exc}") from exc
            if not count:
                raise TransportClosed("the stream ended")
            view = view[count:]

    def recv(self) -> tuple[bytes, list[bytearray]]:
        if self._readinto is None:
            raise TransportClosed("this transport has no readable end")
        fixed = bytearray(_PREFIX.size)
        self._exact(memoryview(fixed))
        head_length, count = _PREFIX.unpack(fixed)
        lengths: tuple[int, ...] = ()
        if count:
            raw = bytearray(8 * count)
            self._exact(memoryview(raw))
            lengths = struct.unpack(f"<{count}Q", raw)
        head = bytearray(head_length)
        self._exact(memoryview(head))
        buffers = []
        for length in lengths:
            buffer = bytearray(length)
            self._exact(memoryview(buffer))
            buffers.append(buffer)
        return bytes(head), buffers

    def close(self) -> None:
        if not self.owns_fds:
            self.read_fd = self.write_fd = None
            return
        for fd in {self.read_fd, self.write_fd} - {None}:
            try:
                os.close(fd)  # type: ignore[arg-type]
            except OSError:
                pass
        self.read_fd = self.write_fd = None


def tensor_view(tensor: Any) -> memoryview:
    """A byte view of a contiguous CPU tensor's own memory, with no copy.

    The view does not keep the tensor alive, so the caller holds the tensor until the view
    has been written.
    """
    if not tensor.is_contiguous():
        raise ValueError("tensor_view needs a contiguous tensor")
    size = tensor.numel() * tensor.element_size()
    if size == 0:
        return memoryview(b"")
    array = (ctypes.c_char * size).from_address(tensor.data_ptr())
    return memoryview(array).cast("B")
