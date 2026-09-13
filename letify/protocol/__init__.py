"""The wire between this process and a runtime.

Five pieces, each with one job.

``handle`` holds the reference types, so a value can stay in a runtime and a large
argument can be named rather than resent. ``codec`` serializes a call and reads an
outcome back. ``framing`` wraps a message as one line, which is what survives SSH
and WebSocket channels. ``worker`` is the source of the process that lives inside
a runtime. ``driver`` is the fallback for a channel that can only run a command
and collect its output.
"""

from __future__ import annotations

from .codec import (
    BEGIN,
    END,
    INLINE_LIMIT,
    PROTOCOL_VERSION,
    digest_of,
    dumps_call,
    parse,
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
from .guards import check_handles, walk
from .handle import Blob, Handle, RemoteFile

__all__ = [
    "BEGIN",
    "END",
    "INLINE_LIMIT",
    "PROTOCOL_VERSION",
    "READY",
    "REPLY",
    "SHUTDOWN",
    "Blob",
    "Handle",
    "RemoteFile",
    "check_handles",
    "decode_reply",
    "digest_of",
    "dumps_call",
    "encode_request",
    "is_ready",
    "is_reply",
    "parse",
    "split_output",
    "unwrap",
    "walk",
]
