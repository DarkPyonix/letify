"""A store in a directory.

Used by the local provider, and by any remote machine whose own disk is the store.
Because the local machine's storage outlives every runtime, this is also the origin a
remote runtime pulls from when data is already here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..cas import Backend
from .layout import BLOB_PREFIX, blob_key, ref_key


class FilesystemBackend(Backend):
    """Blobs and refs as files under one root."""

    name = "filesystem"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def has(self, digest: str) -> bool:
        return self._path(blob_key(digest)).is_file()

    def put(self, digest: str, payload: bytes) -> None:
        target = self._path(blob_key(digest))
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so a reader never sees a partial blob.
        temporary = target.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(target)

    def get(self, digest: str) -> bytes:
        return self._path(blob_key(digest)).read_bytes()

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        base = self.root / BLOB_PREFIX
        if not base.is_dir():
            return
        for path in base.rglob("*"):
            if path.is_file() and path.name.startswith(prefix):
                yield path.name

    def read_ref(self, name: str) -> str | None:
        path = self._path(ref_key(name))
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    def write_ref(self, name: str, digest: str) -> None:
        path = self._path(ref_key(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(digest, encoding="utf-8")


__all__ = ["FilesystemBackend"]
