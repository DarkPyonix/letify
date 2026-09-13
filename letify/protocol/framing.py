"""Message framing for a persistent channel.

One base64 line per message, with responses prefixed by a marker. That survives
an SSH channel, a WebSocket bridge and a plain pipe without any of them mangling
the bytes, which raw binary framing does not.

The marker also separates letify's replies from the user's own prints, which share
the same stream. Anything on stdout without the prefix is the function's output
and is passed through as logs.
"""

from __future__ import annotations

import base64
import pickle
from typing import Any

from .worker import READY, REPLY

SHUTDOWN = "__LETIFY_SHUTDOWN__"


def encode_request(request: dict[str, Any]) -> str:
    """Turn a request into the single line that is written to the worker."""
    return base64.b64encode(pickle.dumps(request, protocol=5)).decode()


def is_reply(line: str) -> bool:
    return line.startswith(REPLY)


def decode_reply(line: str) -> dict[str, Any]:
    """Decode a reply line into the worker's outcome dictionary."""
    payload = line[len(REPLY) :].strip()
    return pickle.loads(base64.b64decode(payload))


def is_ready(line: str) -> bool:
    return line.strip() == READY


__all__ = [
    "READY",
    "REPLY",
    "SHUTDOWN",
    "decode_reply",
    "encode_request",
    "is_ready",
    "is_reply",
]
