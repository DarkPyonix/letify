"""References to things that live in a runtime.

A ``Blob`` names a payload by the hash of its contents, so the same tensor passed
to ten calls is sent on the first one only. A ``RemoteFile`` names a file that
exists inside a runtime.

``Blob`` is deliberately plain. The worker source recognizes it by a
``__letify_kind__`` marker read by attribute rather than by an isinstance check.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Blob:
    """A content addressed payload the runtime may already hold.

    Sent in place of a large argument. The runtime resolves it from its own blob
    table, and the bytes only cross the network on the first call that uses them.
    """

    digest: str
    size: int

    #: Read by the remote worker to recognize a blob without importing letify.
    __letify_kind__ = "blob"

    def __repr__(self) -> str:
        return f"<Blob {self.digest[:8]} {self.size} bytes>"


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file that exists inside the runtime rather than in this process.

    Returned by a volume materialization so that a declared function can be given
    a path without the bytes passing through the caller again.
    """

    path: str
    digest: str
    size: int

    def __repr__(self) -> str:
        return f"<RemoteFile {self.path} {self.size} bytes>"
