"""The wire between this process and a runtime.

Five pieces, each with one job.

``handle`` holds the reference types ``Blob`` and ``RemoteFile``, so a large
argument can be named rather than resent and a file inside a runtime can be pointed
at. ``codec`` serializes a call, hashes arguments and reads an outcome back. ``wire``
turns messages into binary frames and back. ``worker`` is the source of the process
that lives inside a runtime. ``driver`` is the fallback for a channel that can only run
a command and collect its output.
"""

from __future__ import annotations

from . import codec, wire
from .codec import (
    BEGIN,
    END,
    INLINE_LIMIT,
    PROTOCOL_VERSION,
    DigestCache,
    digest_of,
    digest_parts,
    dumps_call,
    dumps_call_parts,
    parse,
    ship_by_value,
    split_output,
    unwrap,
)
from .handle import Blob, RemoteFile

__all__ = [
    "BEGIN",
    "END",
    "INLINE_LIMIT",
    "PROTOCOL_VERSION",
    "Blob",
    "DigestCache",
    "RemoteFile",
    "codec",
    "digest_of",
    "digest_parts",
    "dumps_call",
    "dumps_call_parts",
    "parse",
    "ship_by_value",
    "split_output",
    "unwrap",
    "wire",
]
