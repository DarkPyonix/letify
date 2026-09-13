"""References to things that live in a runtime.

A ``Handle`` points at an object in a runtime's object table and a ``Blob`` names
a payload by the hash of its contents. Both exist so that data stops travelling
more than once: a model stays where it was built, and the same tensor passed to
ten calls is sent on the first one only.

Both types are deliberately plain. The remote side has to recognize them without
importing letify, so each carries a ``__letify_kind__`` marker that the worker
reads by attribute.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Handle:
    """A reference to an object that lives in a runtime.

    The handle carries the key of the runtime that owns it. Passing it to a call
    on a different runtime raises ``HandleScopeError`` rather than silently
    copying the object across the network, because a handle is a pointer into one
    process and one CUDA context.
    """

    runtime: str
    object_id: str
    type_name: str
    summary: str = ""

    #: Read by the remote worker to recognize a handle without importing letify.
    __letify_kind__ = "handle"

    def __repr__(self) -> str:
        return f"<Handle {self.type_name} {self.object_id[:8]} on {self.runtime}>"


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
