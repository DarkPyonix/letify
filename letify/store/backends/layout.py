"""Where a blob and a ref live inside a store.

Every backend keeps the same layout, so a blob written through one is readable through
another pointed at the same bucket:

    blobs/<first two hex characters>/<digest>
    refs/<name>

The two character prefix keeps any one directory from holding every blob, which matters
on a filesystem backend and costs nothing on an object store.
"""

from __future__ import annotations

BLOB_PREFIX = "blobs"
REF_PREFIX = "refs"


def blob_key(digest: str) -> str:
    return f"{BLOB_PREFIX}/{digest[:2]}/{digest}"


def ref_key(name: str) -> str:
    return f"{REF_PREFIX}/{name}"


__all__ = ["BLOB_PREFIX", "REF_PREFIX", "blob_key", "ref_key"]
