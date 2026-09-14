"""The Transport between a PyTorch forwarding client and its device executor.

This module owns the ``Transport`` interface and its implementations, as spec "Transport"
describes: ``StreamTransport`` over a pair of file descriptors, ``ChannelTransport`` on a
session's persistent channel, and ``QueueTransport``, the executor's end inside the call
worker. All of them put the frames of ``protocol.wire`` on the wire, on stream 2. It does
not own what a message means: that is ``client`` and ``executor``.

It imports the standard library only, beside ``wire``, because its source is sent to the
runtime after ``wire`` and ahead of the executor, where letify may be absent.
``tensor_view`` takes a tensor but uses only ``ctypes`` on it.
"""

from __future__ import annotations

import ctypes
from typing import Any, Protocol

try:
    from ...protocol.wire import DEVICE_STREAM, REPLY, REQUEST, Receiver, Sender
except ImportError:  # pragma: no cover - the concatenated copy inside a worker
    pass


class TransportClosed(Exception):
    """The other end closed the stream, or it ended in the middle of a message."""


class Transport(Protocol):
    """A message pipe carrying a head and out-of-band buffers."""

    def send(self, head: bytes, buffers: list[Any]) -> None: ...

    def recv(self) -> tuple[Any, list]: ...

    def close(self) -> None: ...


def _views(buffers: list[Any]) -> list[memoryview]:
    return [memoryview(buffer).cast("B") for buffer in buffers]


class StreamTransport:
    """Device messages as frames over a readable and a writable file descriptor.

    The client end sends ``REQUEST`` frames and reads ``REPLY`` frames; the executor's end
    is made with ``outgoing=REPLY`` and ``incoming=REQUEST``.
    """

    def __init__(
        self,
        read_fd: int | None,
        write_fd: int | None,
        *,
        readinto: Any = None,
        owns_fds: bool = True,
        outgoing: int | None = None,
        incoming: int | None = None,
    ):
        import os

        self._os = os
        self.owns_fds = owns_fds
        self.read_fd = read_fd
        self.write_fd = write_fd
        self.outgoing = REQUEST if outgoing is None else outgoing
        self.incoming = REPLY if incoming is None else incoming
        if readinto is None and read_fd is not None:
            fd = read_fd
            readinto = lambda view: os.readv(fd, [view])  # noqa: E731
        self._receiver = Receiver(readinto) if readinto is not None else None
        fd_out = write_fd
        self._sender = Sender(lambda view: os.write(fd_out, view)) if write_fd is not None else None

    def raw(self, data: bytes) -> None:
        """Write bytes that are not a frame, such as the source for the bootstrap stub."""
        if self._sender is None:
            raise TransportClosed("this transport has no writable end")
        try:
            self._sender.raw(data)
        except OSError as exc:
            raise TransportClosed(f"the stream closed while writing: {exc}") from exc

    def send(self, head: bytes, buffers: list[Any]) -> None:
        if self._sender is None or self.write_fd is None:
            raise TransportClosed("this transport has no writable end")
        try:
            self._sender.parts(self.outgoing, DEVICE_STREAM, head, _views(buffers))
        except OSError as exc:
            raise TransportClosed(f"the stream closed while writing: {exc}") from exc

    def recv(self) -> tuple[Any, list]:
        if self._receiver is None:
            raise TransportClosed("this transport has no readable end")
        while True:
            try:
                event = self._receiver.next_event()
            except EOFError as exc:
                raise TransportClosed(str(exc)) from exc
            except (OSError, ValueError, Exception) as exc:
                raise TransportClosed(f"the stream closed while reading: {exc}") from exc
            if event is not None and event[0] == self.incoming:
                return event[2]

    def close(self) -> None:
        if not self.owns_fds:
            self.read_fd = self.write_fd = None
            return
        for fd in {self.read_fd, self.write_fd} - {None}:
            try:
                self._os.close(fd)  # type: ignore[arg-type]
            except OSError:
                pass
        self.read_fd = self.write_fd = None


class ChannelTransport:
    """Device messages on stream 2 of a session's persistent channel, beside the calls."""

    def __init__(self, connection: Any):
        self.connection = connection
        connection.open_device()

    def send(self, head: bytes, buffers: list[Any]) -> None:
        try:
            self.connection.sender.parts(REQUEST, DEVICE_STREAM, head, _views(buffers))
        except OSError as exc:
            raise TransportClosed(f"the channel closed while writing: {exc}") from exc

    def recv(self) -> tuple[Any, list]:
        message = self.connection.device_reply()
        if message is None:
            raise TransportClosed(f"the channel closed: {self.connection.failure}")
        return message

    def close(self) -> None:
        self.connection.close_device()


class QueueTransport:
    """The executor's end inside the call worker: requests from a queue, replies as frames."""

    def __init__(self, sender: Any, inbox: Any):
        self.sender = sender
        self.inbox = inbox

    def send(self, head: bytes, buffers: list[Any]) -> None:
        try:
            self.sender.parts(REPLY, DEVICE_STREAM, head, _views(buffers))
        except OSError as exc:
            raise TransportClosed(f"the channel closed while writing: {exc}") from exc

    def recv(self) -> tuple[Any, list]:
        message = self.inbox.get()
        if message is None:
            raise TransportClosed("the channel closed")
        return message

    def close(self) -> None:
        pass


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
