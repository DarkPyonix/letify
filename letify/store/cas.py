"""The content addressed store.

Blobs are immutable and named by the hash of their contents. Two consequences
follow, and both were the reason for choosing this over a two way file sync.

Concurrent writers cannot conflict. Different contents get different names, so
two runtimes uploading at the same time never overwrite each other's work. A two
way sync has no such guarantee: the last writer wins and the other runtime's
changes disappear.

Nothing needs to be verified twice. A sync has to compare sizes and timestamps to
decide whether a file changed. Here, holding the digest is proof of holding the
contents, so a transfer that already happened is skipped by name alone.

Mutable state lives in a separate, tiny namespace of refs, in the same way Git
keeps branch names apart from objects. A ref is a few dozen bytes, so writing one
is atomic in practice and a last writer wins race on it is harmless.

The unit of a blob is a decision, not a detail. A model shard is already large,
so one file is one blob. An environment is tens of thousands of small files, so
the whole tree is packed into one archive keyed by the hash of its lock file.
That turns tens of thousands of round trips into one, which is where the real
speedup comes from.
"""

from __future__ import annotations

import abc
import io
import os
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..wire import digest_of

#: Files smaller than this are packed into an archive rather than stored alone.
PACK_THRESHOLD = 1 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BlobInfo:
    """What the store knows about one blob without reading it."""

    digest: str
    size: int


class Backend(abc.ABC):
    """Where blobs and refs are actually kept."""

    name: str = ""

    @abc.abstractmethod
    def has(self, digest: str) -> bool: ...

    @abc.abstractmethod
    def put(self, digest: str, payload: bytes) -> None: ...

    @abc.abstractmethod
    def get(self, digest: str) -> bytes: ...

    @abc.abstractmethod
    def list_digests(self, prefix: str = "") -> Iterator[str]: ...

    @abc.abstractmethod
    def read_ref(self, name: str) -> str | None: ...

    @abc.abstractmethod
    def write_ref(self, name: str, digest: str) -> None: ...

    def missing(self, digests: list[str]) -> list[str]:
        """Which of these digests the store does not hold.

        A backend that can answer in one request should override this. The
        default asks once per digest, which is exactly the per-object round trip
        this design exists to avoid.
        """
        return [d for d in digests if not self.has(d)]


class Store:
    """A content addressed store on top of a backend."""

    def __init__(self, backend: Backend):
        self.backend = backend

    # -- blobs ---------------------------------------------------------------

    def put_bytes(self, payload: bytes) -> BlobInfo:
        digest = digest_of(payload)
        if not self.backend.has(digest):
            self.backend.put(digest, payload)
        return BlobInfo(digest, len(payload))

    def put_file(self, path: str | Path) -> BlobInfo:
        return self.put_bytes(Path(path).read_bytes())

    def get_bytes(self, digest: str) -> bytes:
        return self.backend.get(digest)

    def fetch_file(self, digest: str, target: str | Path) -> Path:
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.backend.get(digest))
        return destination

    # -- archives ------------------------------------------------------------

    def put_tree(self, root: str | Path, *, key: str | None = None) -> BlobInfo:
        """Pack a directory into one blob.

        Use this for anything made of many small files, such as a virtual
        environment or a package cache. Packing first is what turns a transfer
        that is dominated by per-file latency into one that is dominated by
        bandwidth.
        """
        source = Path(root)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(source, arcname=source.name, recursive=True)
        payload = buffer.getvalue()
        info = self.put_bytes(payload)
        if key:
            self.backend.write_ref(key, info.digest)
        return info

    def fetch_tree(self, digest: str, target: str | Path) -> Path:
        """Unpack a blob written by ``put_tree`` into a directory."""
        destination = Path(target)
        destination.mkdir(parents=True, exist_ok=True)
        payload = self.backend.get(digest)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            _safe_extract(archive, destination)
        return destination

    # -- refs ----------------------------------------------------------------

    def resolve(self, ref: str) -> str | None:
        """Read the digest a name points at."""
        return self.backend.read_ref(ref)

    def point(self, ref: str, digest: str) -> None:
        """Point a name at a digest."""
        self.backend.write_ref(ref, digest)

    # -- planning ------------------------------------------------------------

    def plan_upload(self, paths: list[Path]) -> tuple[list[Path], list[str]]:
        """Split files into the ones that must be uploaded and the ones already held.

        The digests are computed locally, which is cheap: hashing runs at gigabytes
        per second while the network runs at megabytes per second.
        """
        digests = {path: digest_of(path.read_bytes()) for path in paths}
        missing = set(self.backend.missing(list(digests.values())))
        upload = [path for path, digest in digests.items() if digest in missing]
        held = [digest for digest in digests.values() if digest not in missing]
        return upload, held


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract without letting a member escape the destination directory.

    The explicit path check covers Python versions whose default extraction is
    still permissive, and the data filter covers the rest.
    """
    root = destination.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root) + os.sep) and target != root:
            raise ValueError(f"archive member {member.name!r} would escape {root}")
    try:
        archive.extractall(destination, filter="data")
    except TypeError:
        archive.extractall(destination)
