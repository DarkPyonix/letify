"""Project data: local paths a call reaches, sent as content addressed file blobs.

Owns detection of local paths while a call is pickled, file digests and the local digest
cache, choosing where missing blobs come from, and the data log line, as spec "Project data"
describes. It does not own the worker side, which is in ``protocol/worker.py``, nor the
bucket client, which is the ``gcs`` backend.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import queue
import sys
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import RemoteError, RuntimeFailure

if TYPE_CHECKING:
    from ..providers.base import Provider
    from ..runtime.session import Runtime
    from .backends.objects import GCSBackend

#: The largest piece of a file one ``data_put`` request carries.
CHUNK = 64 << 20

#: Pieces of a streamed file blob held before the receiving thread waits for the writer.
WRITE_QUEUE = 4

#: Files at least this large are hashed through a memory map with blake3's threads.
LARGE_FILE = 64 << 20

#: Uploads at least this large show a progress line.
PROGRESS_THRESHOLD = 64 << 20

#: Directory entries a directory walk never enters.
SKIPPED = frozenset({".git", ".venv", "__pycache__"})

_CACHE_LOCK = threading.Lock()


# -- which paths are data ------------------------------------------------------


def project_root(start: Path | None = None) -> Path:
    """The nearest directory upward holding a ``pyproject.toml``, or the working directory."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return here


def allowed_roots(root: Path) -> list[Path]:
    """The project root and each ``[tool.letify] data_roots`` entry, resolved."""
    roots = [root]
    try:
        settings = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return roots
    named = settings.get("tool", {}).get("letify", {}).get("data_roots", [])
    if isinstance(named, list):
        roots.extend((root / str(entry)).resolve() for entry in named)
    return roots


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


# -- digests -------------------------------------------------------------------


def hash_file(path: Path, size: int) -> str:
    """The Argument addressing digest of a file's contents, read without holding it whole."""
    from ..protocol.codec import _hasher

    hasher = _hasher()
    with open(path, "rb") as handle:
        if size >= LARGE_FILE and hasher.name != "blake2b":
            import mmap

            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                hasher.update(mapped)
        else:
            while chunk := handle.read(1 << 20):
                hasher.update(chunk)
    if hasher.name == "blake2b":
        return hasher.hexdigest()
    return hasher.hexdigest(length=16)


class DigestCache:
    """File digests keyed by resolved path, reused while size, mtime and inode match."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".cache" / "letify" / "digests.json"
        self._dirty = False
        try:
            self._entries: dict[str, list[Any]] = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            self._entries = {}

    def digest(self, file: Path, info: os.stat_result | None = None) -> str:
        info = info or os.stat(file)
        key = str(file)
        entry = self._entries.get(key)
        if entry and entry[:3] == [info.st_size, info.st_mtime_ns, info.st_ino]:
            return str(entry[3])
        digest = hash_file(file, info.st_size)
        self._entries[key] = [info.st_size, info.st_mtime_ns, info.st_ino, digest]
        self._dirty = True
        return digest

    def __len__(self) -> int:
        return len(self._entries)

    def prune(self) -> int:
        """Drop entries whose file no longer exists, and answer how many were dropped."""
        gone = [key for key in self._entries if not os.path.exists(key)]
        for key in gone:
            del self._entries[key]
        if gone:
            self._dirty = True
        return len(gone)

    def record(self, file: Path, digest: str) -> None:
        """Remember the digest of a file this process has just written."""
        info = os.stat(file)
        self._entries[str(file)] = [info.st_size, info.st_mtime_ns, info.st_ino, digest]
        self._dirty = True

    def save(self) -> None:
        """Replace the cache file atomically when an entry was added, changed or pruned."""
        self.prune()
        if not self._dirty:
            return
        with _CACHE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}")
            temporary.write_text(json.dumps(self._entries), encoding="utf-8")
            os.replace(temporary, self.path)
        self._dirty = False


# -- detection during pickling -------------------------------------------------


@dataclass
class Placed:
    """One detected local path and where the runtime places it."""

    local: Path
    runtime: str
    directory: bool
    #: Relative POSIX path, digest, size and local file per file, sorted by path.
    entries: list[tuple[str, str, int, str]] = field(default_factory=list)
    #: An output location: a directory, or a path that did not exist.
    output: bool = False


class Collector:
    """Replaces local data paths in a pickled call with their runtime paths."""

    def __init__(self, call_dir: str) -> None:
        self.call_dir = call_dir
        self.placed: dict[str, Placed] = {}
        self._rejected: set[str] = set()
        self._roots: tuple[Path, list[Path]] | None = None
        self._cache: DigestCache | None = None

    @property
    def cache(self) -> DigestCache:
        if self._cache is None:
            self._cache = DigestCache()
        return self._cache

    def reduce(self, obj: Any) -> Any:
        """A reduction to ``pathlib.Path(<runtime path>)``, or NotImplemented."""
        try:
            text = os.fspath(obj)
        except TypeError:
            return NotImplemented
        if not isinstance(text, str) or text in self._rejected:
            return NotImplemented
        placed = self._place(text)
        if placed is None:
            self._rejected.add(text)
            return NotImplemented
        return pathlib.Path, (placed.runtime,)

    def _place(self, text: str) -> Placed | None:
        try:
            resolved = Path(text).resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        known = self.placed.get(str(resolved))
        if known is not None:
            return known
        import stat

        try:
            info: os.stat_result | None = os.stat(resolved)
        except FileNotFoundError:
            info = None
        except (OSError, ValueError):
            return None
        is_dir = info is not None and stat.S_ISDIR(info.st_mode)
        if info is not None and not (is_dir or stat.S_ISREG(info.st_mode)):
            return None
        if self._roots is None:
            root = project_root()
            self._roots = (root, allowed_roots(root))
        project, roots = self._roots
        if resolved == project or resolved in project.parents:
            return None
        if not any(_inside(resolved, root) for root in roots):
            return None
        index = len(self.placed)
        placed = Placed(resolved, f"{self.call_dir}/{index}/{resolved.name}", is_dir)
        placed.output = info is None or is_dir
        if info is None:
            pass
        elif is_dir:
            placed.entries = self._walk(resolved)
        else:
            digest = self.cache.digest(resolved, info)
            placed.entries = [(resolved.name, digest, info.st_size, str(resolved))]
        self.placed[str(resolved)] = placed
        return placed

    def _walk(self, top: Path) -> list[tuple[str, str, int, str]]:
        entries = []
        for directory, names, files in os.walk(top, followlinks=False):
            names[:] = [name for name in names if name not in SKIPPED]
            for name in files:
                full = os.path.join(directory, name)
                try:
                    info = os.stat(full)
                except OSError:
                    continue
                if not os.path.isfile(full):
                    continue
                relative = Path(full).relative_to(top).as_posix()
                digest = self.cache.digest(Path(full), info)
                entries.append((relative, digest, info.st_size, full))
        entries.sort()
        return entries

    def request(self, blobs: str) -> dict[str, Any]:
        """The ``data`` field of the call request: directories to make and links to place."""
        links = []
        dirs = []
        outputs = []
        for placed in self.placed.values():
            if placed.directory:
                dirs.append(placed.runtime)
                # An output location gets writable copies, so the body can rewrite a file.
                links.extend(
                    [f"{placed.runtime}/{rel}", digest, placed.output]
                    for rel, digest, _s, _f in placed.entries
                )
            elif placed.entries:
                links.append([placed.runtime, placed.entries[0][1], False])
            else:
                dirs.append(placed.runtime.rsplit("/", 1)[0])
            if placed.output:
                outputs.append(placed.runtime)
        return {
            "dir": self.call_dir,
            "blobs": blobs,
            "dirs": dirs,
            "links": links,
            "outputs": outputs,
        }

    @property
    def inputs(self) -> bool:
        """Whether any detected path carries a file to send."""
        return any(placed.entries for placed in self.placed.values())

    @property
    def outputs(self) -> list[Placed]:
        return [placed for placed in self.placed.values() if placed.output]


# -- where the bytes come from -------------------------------------------------


def data_bucket(provider: Provider) -> GCSBackend | None:
    """The account's data bucket on an ephemeral provider, or None."""
    if provider.persistent:
        return None
    name = provider.config.option("bucket")
    if not name:
        return None
    cached = getattr(provider, "_letify_data_bucket", None)
    if cached is not None:
        return cached
    from .backends.objects import GCS_ENDPOINT, STS_ENDPOINT, GCSBackend

    option = provider.config.option
    bucket = GCSBackend(
        str(name),
        prefix=str(option("bucket_prefix", "letify")),
        endpoint=str(option("bucket_endpoint", GCS_ENDPOINT)),
        sts_endpoint=str(option("sts_endpoint", STS_ENDPOINT)),
    )
    provider._letify_data_bucket = bucket  # type: ignore[attr-defined]
    return bucket


def _mib(size: int) -> str:
    return f"{size / (1 << 20):.1f} MiB"


class _Uploaded:
    """Bytes sent so far, drawn as a progress line when the upload is large."""

    def __init__(self, files: int, total: int) -> None:
        self.base = 0
        self.progress = None
        if total >= PROGRESS_THRESHOLD:
            from ..install import _Progress

            self.progress = _Progress(f"{files} files", total, sys.stderr, verb="uploading")

    def update(self, done: int) -> None:
        if self.progress is not None:
            self.progress.update(self.base + done)

    def advance(self, size: int) -> None:
        self.base += size
        self.update(0)

    def finish(self) -> None:
        if self.progress is not None:
            self.progress.finish(self.base)


def budget_bytes(provider: Provider) -> int | None:
    """The account's ``data_cache_gib`` in bytes, or None for the worker's default."""
    gib = provider.config.option("data_cache_gib")
    if gib is None:
        return None
    return int(float(gib) * (1 << 30))


def evict(runtime: Runtime, blobs: str) -> None:
    """Keep the runtime's file blob cache within its budget, as spec describes."""
    from .. import install

    result = _worker(
        runtime, {"op": "data_evict", "dir": blobs, "budget": budget_bytes(runtime.provider)}
    )
    if result and result["files"]:
        install.log(
            f"data cache evicted {result['files']} files {_mib(result['bytes'])} in "
            f"{result['seconds']:.1f} s, {_mib(result['total'])} of "
            f"{result['budget'] / (1 << 30):.1f} GiB in use"
        )


def send(runtime: Runtime, collector: Collector, blobs: str) -> int:
    """Make sure the runtime's file blob cache holds every digest the call needs.

    Answers how many blobs the runtime's cache received.
    """
    from .. import install

    collector.cache.save()
    sources: dict[str, tuple[int, str]] = {}
    detected_files = detected_bytes = 0
    for placed in collector.placed.values():
        for _rel, digest, size, local in placed.entries:
            sources.setdefault(digest, (size, local))
            detected_files += 1
            detected_bytes += size
    started = time.monotonic()
    held = set(
        _worker(
            runtime,
            {"op": "data_have", "dir": blobs, "digests": [[d, s[0]] for d, s in sources.items()]},
        )
        or ()
    )
    missing = [digest for digest in sources if digest not in held]
    bucket = data_bucket(runtime.provider)
    if bucket is not None:
        where = "the bucket"
        absent = set(bucket.missing(missing)) if missing else set()
        upload = [digest for digest in missing if digest in absent]
    else:
        where = "the runtime"
        upload = missing
    total = sum(sources[digest][0] for digest in upload)
    meter = _Uploaded(len(upload), total)
    for digest in upload:
        size, local = sources[digest]
        if bucket is not None:
            bucket.put_file(digest, local, size, progress=meter.update)
        else:
            _put(runtime, blobs, digest, local, size, meter)
        meter.advance(size)
    meter.finish()
    if bucket is not None and missing:
        token = bucket.read_token()
        items = [[digest, bucket.blob_url(digest)] for digest in missing]
        _worker(
            runtime,
            {
                "op": "data_pull",
                "dir": blobs,
                "items": items,
                "headers": {"Authorization": f"Bearer {token}"},
            },
        )
        token = ""
    elapsed = time.monotonic() - started
    uploaded = set(upload)
    up_files = up_bytes = 0
    for placed in collector.placed.values():
        for _rel, digest, size, _local in placed.entries:
            if digest in uploaded:
                up_files += 1
                up_bytes += size
    rate = up_bytes / (1 << 20) / elapsed if elapsed > 0 else 0.0
    install.log(
        f"data {detected_files} files {_mib(detected_bytes)} detected, "
        f"{detected_files - up_files} files {_mib(detected_bytes - up_bytes)} already on {where}, "
        f"uploaded {up_files} files {_mib(up_bytes)} in {elapsed:.1f} s ({rate:.1f} MiB/s)"
    )
    return len(missing)


def _put(
    runtime: Runtime, blobs: str, digest: str, local: str, size: int, meter: _Uploaded
) -> None:
    """Send one file over the channel in pieces of at most ``CHUNK`` bytes."""
    buffer = bytearray(max(1, min(CHUNK, size)))
    view = memoryview(buffer)
    offset = 0
    with open(local, "rb") as handle:
        while True:
            count = 0
            while count < len(buffer):
                read = handle.readinto(view[count:])
                if not read:
                    break
                count += read
            last = count < len(buffer) or offset + count >= size
            _worker(
                runtime,
                {
                    "op": "data_put",
                    "dir": blobs,
                    "digest": digest,
                    "offset": offset,
                    "chunk": pickle.PickleBuffer(view[:count]),
                    "last": last,
                },
            )
            offset += count
            meter.update(offset)
            if last:
                return


# -- writing back ----------------------------------------------------------------

_OUTPUT_LOCKS: dict[str, threading.Lock] = {}
_OUTPUT_LOCKS_GUARD = threading.Lock()


def _output_lock(path: Path) -> threading.Lock:
    with _OUTPUT_LOCKS_GUARD:
        return _OUTPUT_LOCKS.setdefault(str(path), threading.Lock())


class _Downloaded(_Uploaded):
    """Bytes received so far, drawn as the download progress line when large."""

    def __init__(self, files: int, total: int) -> None:
        self.base = 0
        self.progress = None
        if total >= PROGRESS_THRESHOLD:
            from ..install import _Progress

            self.progress = _Progress(f"{files} files", total, sys.stderr)


def write_back(runtime: Runtime, collector: Collector, blobs: str) -> int:
    """Copy what a returned call created or changed at its output locations to the client.

    Answers how many files the runtime listed, each of which is now in its blob cache.

    Spec "Writing back".
    """
    from .. import install

    started = time.monotonic()
    written = _worker(runtime, {"op": "data_written", "dir": collector.call_dir}) or {}
    cache = collector.cache
    plan: list[tuple[Placed, list[tuple[Path, str, int]]]] = []
    for placed in collector.outputs:
        files = []
        for rel, digest, size in written.get(placed.runtime, ()):
            target = placed.local / rel if rel else placed.local
            files.append((target, str(digest), int(size)))
        plan.append((placed, files))
    total = sum(size for _placed, files in plan for _t, _d, size in files)
    meter = _Downloaded(sum(len(files) for _p, files in plan), total)
    sent_files = sent_bytes = kept_files = kept_bytes = 0
    for placed, files in plan:
        with _output_lock(placed.local):
            for target, digest, size in files:
                if _same(cache, target, digest):
                    kept_files += 1
                    kept_bytes += size
                    meter.advance(size)
                    continue
                _get(runtime, blobs, digest, target, size, meter)
                cache.record(target, digest)
                meter.advance(size)
                sent_files += 1
                sent_bytes += size
    meter.finish()
    cache.save()
    elapsed = time.monotonic() - started
    rate = sent_bytes / (1 << 20) / elapsed if elapsed > 0 else 0.0
    install.log(
        f"data wrote back {sent_files} files {_mib(sent_bytes)} in {elapsed:.1f} s "
        f"({rate:.1f} MiB/s), {kept_files} files {_mib(kept_bytes)} already on the client"
    )
    return sum(len(files) for _placed, files in plan)


def _same(cache: DigestCache, target: Path, digest: str) -> bool:
    try:
        info = os.stat(target)
    except OSError:
        return False
    import stat

    return stat.S_ISREG(info.st_mode) and cache.digest(target, info) == digest


def _get(
    runtime: Runtime, blobs: str, digest: str, target: Path, size: int, meter: _Uploaded
) -> None:
    """Receive one file blob as one stream, writing and hashing it as the pieces arrive.

    Spec "Writing back". One writing thread takes the pieces, so receiving the next one
    overlaps writing and hashing the one before it and the link does not idle.
    """
    from ..protocol.codec import _hasher

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(
        f".{target.name}.letify-partial.{os.getpid()}.{threading.get_ident()}"
    )
    hasher = _hasher()
    pieces: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE)
    failure: list[BaseException] = []

    def write_pieces() -> None:
        """Append each piece to the partial file and hash it, until the queue ends."""
        try:
            with open(partial, "wb") as handle:
                while True:
                    piece = pieces.get()
                    if piece is None:
                        return
                    handle.write(piece)
                    hasher.update(piece)
        except BaseException as exc:  # pragma: no cover - a local disk failure
            failure.append(exc)
            # Drained, so the receiving thread is never left waiting on a full queue.
            while pieces.get() is not None:
                pass

    writer = threading.Thread(target=write_pieces, daemon=True)
    writer.start()
    received = 0
    try:
        for piece in _stream(runtime, {"op": "data_stream", "dir": blobs, "digest": digest}):
            view = memoryview(piece).cast("B")
            if failure:
                break
            received += view.nbytes
            pieces.put(view)
            meter.update(received)
        pieces.put(None)
        writer.join()
        if failure:
            raise failure[0]
        if received != size:
            raise RuntimeFailure(
                f"{runtime.name}: file blob {digest} arrived with {received} bytes, not {size}"
            )
        found = hasher.hexdigest() if hasher.name == "blake2b" else hasher.hexdigest(length=16)
        if found != digest:
            raise RuntimeFailure(f"{runtime.name}: file blob {digest} arrived with digest {found}")
        os.replace(partial, target)
    finally:
        if writer.is_alive():
            pieces.put(None)
            writer.join(5)
        partial.unlink(missing_ok=True)


def _worker(runtime: Runtime, payload: dict[str, Any]) -> Any:
    try:
        return runtime.request(payload, timeout=3600)
    except RemoteError as exc:
        raise RuntimeFailure(f"{runtime.name}: {payload['op']} failed: {exc}") from exc


def _stream(runtime: Runtime, payload: dict[str, Any]) -> Any:
    """Each piece of a request the worker answers with several replies."""
    try:
        yield from runtime.stream(payload, timeout=3600)
    except RemoteError as exc:
        raise RuntimeFailure(f"{runtime.name}: {payload['op']} failed: {exc}") from exc


__all__ = [
    "CHUNK",
    "Collector",
    "DigestCache",
    "allowed_roots",
    "data_bucket",
    "hash_file",
    "project_root",
    "send",
]
