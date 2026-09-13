"""The wire between this process and a runtime.

Five pieces, each with one job.

``handle`` holds the reference types ``Blob`` and ``RemoteFile``, so a large
argument can be named rather than resent and a file inside a runtime can be pointed
at. ``codec`` serializes a call and reads an outcome back. ``framing`` wraps a
message as one line, which is what survives SSH and WebSocket channels. ``worker``
is the source of the process that lives inside a runtime. ``driver`` is the fallback
for a channel that can only run a command and collect its output.
"""

from __future__ import annotations

from . import codec
from .codec import (
    BEGIN,
    END,
    INLINE_LIMIT,
    PROTOCOL_VERSION,
    digest_of,
    dumps_call,
    parse,
    ship_by_value,
    split_output,
    unwrap,
)
from .framing import (
    READY,
    REPLY,
    SHUTDOWN,
    decode_reply,
    encode_request,
    is_ready,
    is_reply,
)
from .handle import Blob, RemoteFile

__all__ = [
    "BEGIN",
    "END",
    "INLINE_LIMIT",
    "PROTOCOL_VERSION",
    "READY",
    "REPLY",
    "SHUTDOWN",
    "Blob",
    "RemoteFile",
    "codec",
    "decode_reply",
    "digest_of",
    "dumps_call",
    "encode_request",
    "is_ready",
    "is_reply",
    "parse",
    "ship_by_value",
    "split_output",
    "unwrap",
]
