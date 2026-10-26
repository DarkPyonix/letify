"""Named handoffs on the machine's Drive store: ``stash`` and ``unstash``.

Owns packing a file or directory into an uncompressed tar, naming the blob by its digest,
the ref ``stash/<name>`` that points at it, and unpacking a blob back to a path. It also owns
the two places this runs: the local process, where this module moves the bytes with the
machine's ``DriveBackend``, and a shipped function, where the worker moves them and this
process only grants access (``grant`` and ``GrantWatcher``). It does not own the Drive calls
themselves, which ``letify.store.backends.drive`` owns, or the wire stream a worker asks on,
which ``letify.protocol.wire`` and the channel own.

Spec "Named handoffs on the Drive store". A handoff never falls back to another store or to
the local disk: a machine with no Drive raises ``ConfigError``.
"""

from __future__ import annotations

import contextlib
import http.client
import itertools
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, overload

from ..errors import ConfigError, LetifyError, RuntimeFailure, UnsupportedMode
from ..protocol import codec

#: The environment variable the one-shot driver script sets, holding the provider's name,
#: so a body can tell it runs on a channel with no stream back to the local process.
ONE_SHOT_MARKER = "LETIFY_ONE_SHOT"

#: The environment variable the worker sets to its process id, so a child process of the
#: body, which inherits it but lacks the worker's grant hook, can say why it cannot ask.
WORKER_MARKER = "LETIFY_WORKER_PID"

#: The prefix of a handoff's ref, so handoffs share the folder with other refs.
REF_SPACE = "stash/"

_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")

#: Seconds a worker waits for the local process to answer a request for a grant.
GRANT_TIMEOUT = 600.0

_BACKENDS: dict[tuple[str, str, str], Any] = {}
_BACKENDS_LOCK = threading.Lock()


def check_name(name: object) -> str:
    """The name, or ``ValueError`` naming the rule."""
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError(
            "a handoff name is 1 to 128 characters of letters, digits, '.', '_' and '-', "
            f"and does not start with '.', got {name!r}"
        )
    return name


def drive_backend() -> Any:
    """The machine's Drive store, or ``ConfigError`` naming the two settings that declare it."""
    from ..config import drive_settings
    from .backends.drive import DRIVE_ENDPOINT, DriveBackend

    settings = drive_settings()
    if settings is None:
        raise ConfigError(
            "letify.stash and letify.unstash use the machine's Google Drive store, and this "
            "machine declares none. Set drive_login = '<alias of a Google login>' and "
            "drive_project = '<Google Cloud project id>' in the machine configuration."
        )
    endpoint = settings.endpoint or DRIVE_ENDPOINT
    key = (settings.login, settings.project, endpoint)
    with _BACKENDS_LOCK:
        if key not in _BACKENDS:
            _BACKENDS[key] = DriveBackend(settings.login, settings.project, endpoint=endpoint)
        return _BACKENDS[key]


# -- packing and unpacking -------------------------------------------------------


class _Hashing:
    """A write-only file that hashes every byte it passes on, so no second read is needed."""

    def __init__(self, handle: Any, hasher: Any) -> None:
        self._handle = handle
        self._hasher = hasher
        self.size = 0

    def write(self, data: bytes) -> int:
        self._handle.write(data)
        self._hasher.update(data)
        self.size += len(data)
        return len(data)

    def flush(self) -> None:
        self._handle.flush()

    def tell(self) -> int:
        return self.size


@contextlib.contextmanager
def _scratch(near: Path) -> Iterator[Path]:
    """A hidden directory beside ``near``, on its filesystem so a rename never copies."""
    parent = near.parent
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".letify-handoff-", dir=parent))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def pack(source: Path, archive: Path) -> tuple[str, int]:
    """Write ``source`` as an uncompressed tar of the one entry named by its base name.

    Returns ``(digest, size)``, the digest hashed as the archive is written.
    """
    hasher = codec.new_hasher()
    with open(archive, "wb") as handle:
        hashed = _Hashing(handle, hasher)
        with tarfile.open(fileobj=hashed, mode="w") as tar:  # type: ignore[call-overload]
            tar.add(source, arcname=Path(os.path.abspath(source)).name, recursive=True)
    return codec.finish_digest(hasher), hashed.size


def digest_of_file(path: Path) -> str:
    hasher = codec.new_hasher()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return codec.finish_digest(hasher)


def unpack(archive: Path, path: Path, scratch: Path) -> None:
    """Make the tar's one entry the thing at ``path``, replacing what ``path`` held."""
    staging = scratch / "new"
    staging.mkdir()
    with tarfile.open(archive, "r:") as tar:
        try:
            tar.extractall(staging, filter="data")
        except TypeError:  # pragma: no cover - interpreters before the extraction filters
            root = os.path.realpath(staging)
            for member in tar.getmembers():
                target = os.path.realpath(os.path.join(root, member.name))
                if os.path.commonpath([root, target]) != root:
                    raise RuntimeFailure(
                        "a handoff archive member would escape its destination"
                    ) from None
            tar.extractall(staging)
    entries = os.listdir(staging)
    if len(entries) != 1:
        raise RuntimeFailure(
            f"a handoff archive holds {len(entries)} top level entries, and a handoff holds one"
        )
    if os.path.lexists(path):
        shutil.move(str(path), str(scratch / "old"))
    os.replace(staging / entries[0], path)


# -- the refs ----------------------------------------------------------------------


def commit(backend: Any, name: str, digest: str) -> None:
    """Point ``stash/<name>`` at ``digest``, and delete the blob it named before.

    The previous blob stays when another ``stash/`` ref still names it. A ref that already
    holds ``digest`` is not written again.
    """
    ref = REF_SPACE + name
    previous = backend.read_ref(ref)
    if previous == digest:
        return
    backend.write_ref(ref, digest)
    if previous and previous not in backend.refs_with_prefix(REF_SPACE).values():
        backend.delete_blob(previous)


def _named(name: str, exc: Exception) -> RuntimeFailure:
    return RuntimeFailure(f"handoff {name!r}: {exc}")


# -- the local process ---------------------------------------------------------------


def stash(name: str, path: str | os.PathLike[str]) -> str:
    """Put a file or directory on the machine's Drive store under ``name``.

    Returns the digest once the upload has finished and the ref is written. Spec "Named
    handoffs on the Drive store".
    """
    check_name(name)
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"cannot stash {str(source)!r}: it does not exist")
    main = _worker_main(name)
    if main is not None:
        return _stash_in_worker(main, name, source)
    backend = drive_backend()
    with _scratch(source.absolute()) as scratch:
        archive = scratch / "stash.tar"
        digest, _size = pack(source, archive)
        try:
            if not backend.has(digest):
                backend.put_file(digest, archive)
            commit(backend, name, digest)
        except RuntimeFailure as exc:
            raise _named(name, exc) from exc
    return digest


@overload
def unstash(name: str) -> dict[str, bytes] | None: ...
@overload
def unstash(name: str, path: str | os.PathLike[str]) -> Path | None: ...
def unstash(
    name: str, path: str | os.PathLike[str] | None = None
) -> Path | dict[str, bytes] | None:
    """Put the handoff ``name`` at ``path``, replacing what was there.

    Returns ``path``, or None when no handoff has that name, in which case nothing changed.
    With no ``path`` nothing is written to disk: the result is a dict from the path of each
    regular file in the handoff, as stored, to its bytes, or None when no handoff has that
    name. Spec "Named handoffs on the Drive store".
    """
    check_name(name)
    main = _worker_main(name)
    if main is not None:
        if path is None:
            return _unstash_memory_in_worker(main, name)
        return _unstash_in_worker(main, name, Path(path))
    backend = drive_backend()
    try:
        digest = backend.read_ref(REF_SPACE + name)
    except RuntimeFailure as exc:
        raise _named(name, exc) from exc
    if digest is None:
        return None
    if path is None:
        try:
            source = backend.pull_source(digest)
        except RuntimeFailure as exc:
            raise _named(name, exc) from exc
        return _members_from(
            name, digest, str(source["url"]), dict(source.get("headers") or {})  # type: ignore[call-overload]
        )
    target = Path(path)
    with _scratch(target.absolute()) as scratch:
        archive = scratch / "unstash.tar"
        try:
            backend.get_file(digest, archive)
        except RuntimeFailure as exc:
            raise _named(name, exc) from exc
        _verify(name, archive, digest)
        unpack(archive, target, scratch)
    return target


# -- reading a handoff into memory ----------------------------------------------------

_BLOCK = 512
_REGULAR = (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.CONTTYPE)


def _truncated() -> RuntimeFailure:
    return RuntimeFailure("a handoff archive ended before its last member was complete")


def _take(stream: Any, size: int, hasher: Any) -> bytes:
    """Exactly ``size`` bytes, hashed, from one ``read(size)`` call unless a read comes short.

    A single call lets the buffered reader allocate the result once. Only a short read, which
    a chunked response can give, joins pieces.
    """
    data = stream.read(size)
    if len(data) != size:
        pieces = [data]
        have = len(data)
        while have < size:
            more = stream.read(size - have)
            if not more:
                raise _truncated()
            pieces.append(more)
            have += len(more)
        data = b"".join(pieces)
    if hasher is not None:
        hasher.update(data)
    return data


def _skip(stream: Any, size: int, hasher: Any) -> None:
    """Read and hash ``size`` bytes without keeping them."""
    while size > 0:
        _take(stream, min(size, 1 << 20), hasher)
        size -= min(size, 1 << 20)


def _pax_fields(data: bytes) -> dict[str, str]:
    """The ``key=value`` records of a pax extended header."""
    fields: dict[str, str] = {}
    position = 0
    while position < len(data):
        space = data.find(b" ", position)
        if space < 0:
            break
        length = int(data[position:space])
        record = data[space + 1 : position + length - 1]
        key, _, value = record.partition(b"=")
        fields[key.decode("utf-8")] = value.decode("utf-8", "surrogateescape")
        position += length
    return fields


def read_archive(stream: Any, hasher: Any) -> dict[str, bytes]:
    """Read a tar from ``stream`` in one pass into a dict of regular files, hashing every byte.

    Headers are parsed as they arrive, so a member's bytes are never held twice. GNU long
    name records and pax records are applied to the member that follows. The stream is read
    to its end so the hash covers the whole blob. Digest checking is the caller's.
    """
    members: dict[str, bytes] = {}
    name: str | None = None
    size_override: int | None = None
    while True:
        header = stream.read(_BLOCK)
        if len(header) < _BLOCK:
            more = header
            while len(more) < _BLOCK:
                piece = stream.read(_BLOCK - len(more))
                if not piece:
                    break
                more += piece
            header = more
            if len(header) < _BLOCK:
                if hasher is not None:
                    hasher.update(header)
                if header.strip(b"\0"):
                    raise _truncated()
                return members
        if hasher is not None:
            hasher.update(header)
        if header == b"\0" * _BLOCK:
            while chunk := stream.read(1 << 20):
                if hasher is not None:
                    hasher.update(chunk)
            return members
        try:
            info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        except tarfile.TarError as exc:
            raise RuntimeFailure(f"a handoff archive has an unreadable header: {exc}") from exc
        size = info.size
        padding = -size % _BLOCK
        if info.type in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
            text = _take(stream, size, hasher).split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
            _skip(stream, padding, hasher)
            if info.type == tarfile.GNUTYPE_LONGNAME:
                name = text
            continue
        if info.type == tarfile.XHDTYPE:
            fields = _pax_fields(_take(stream, size, hasher))
            _skip(stream, padding, hasher)
            name = fields.get("path", name)
            if "size" in fields:
                size_override = int(fields["size"])
            continue
        if info.type in (tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE):
            _skip(stream, size + padding, hasher)
            continue
        if info.type in _REGULAR:
            if size_override is not None:
                size = size_override
                padding = -size % _BLOCK
            members[name if name is not None else info.name] = _take(stream, size, hasher)
            _skip(stream, padding, hasher)
        else:
            _skip(stream, size + padding, hasher)
        name = None
        size_override = None


def _members_from(name: str, digest: str, url: str, headers: dict[str, str]) -> dict[str, bytes]:
    """Download the blob at ``url`` straight into a dict of its files, then check the digest."""
    from .backends.drive import open_stream

    hasher = codec.hasher_for(digest)
    try:
        response = open_stream(url, headers)
        try:
            members = read_archive(response, hasher)
        finally:
            response.close()
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeFailure(f"handoff {name!r}: Drive download failed: {exc}") from exc
    except RuntimeFailure as exc:
        raise _named(name, exc) from exc
    if hasher is not None:
        found = codec.finish_digest(hasher)
        if found != digest:
            raise RuntimeFailure(
                f"handoff {name!r}: the blob downloaded from Drive has digest {found}, "
                f"and the ref names {digest}"
            )
    return members


def _verify(name: str, archive: Path, digest: str) -> None:
    hasher = codec.hasher_for(digest)
    if hasher is None:
        return
    with open(archive, "rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    found = codec.finish_digest(hasher)
    if found != digest:
        raise RuntimeFailure(
            f"handoff {name!r}: the blob downloaded from Drive has digest {found}, "
            f"and the ref names {digest}"
        )


# -- inside a shipped function ---------------------------------------------------------


def _worker_main(name: str) -> Any:
    """The worker's ``__main__`` module when this runs inside a shipped function, else None.

    The same test ``all_reduce`` makes: only the worker defines the hook that sends a request
    to the local process.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "_drive_ask"):
        return main
    worker_pid = os.environ.get(WORKER_MARKER)
    if worker_pid is not None and worker_pid != str(os.getpid()):
        raise UnsupportedMode(
            f"handoff {name!r}: stash and unstash must be called from the process letify "
            f"runs the body in, not from a child process it started (this is a child of "
            f"worker process {worker_pid}). Call it in the body and pass the path to the child."
        )
    one_shot = os.environ.get(ONE_SHOT_MARKER)
    if one_shot is not None:
        raise UnsupportedMode(
            f"stash and unstash need a stream back to the local process, and the body is "
            f"running on a one-shot channel of {one_shot}, which has none. Use a provider "
            f"with a persistent channel, or move the bytes from the local process."
        )
    return None


_IDS = itertools.count(1)


def _ask(main: Any, message: dict[str, Any]) -> dict[str, Any]:
    """Send one request to the local process and wait for its answer."""
    ident = next(_IDS)
    main._drive_ask({"op": "drive_ask", "id": ident, **message})
    deadline = time.monotonic() + GRANT_TIMEOUT
    cond = main._DRIVE_LOCK
    answers = main._DRIVE_ANSWERS
    with cond:
        while ident not in answers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeFailure(
                    f"handoff {message.get('name')!r}: the local process gave no answer "
                    f"within {GRANT_TIMEOUT:.0f} s"
                )
            cond.wait(min(remaining, 1.0))
        answer = answers.pop(ident)
    failure = answer.get("error")
    if failure:
        raised = {"ConfigError": ConfigError, "ValueError": ValueError}.get(
            failure["type"], RuntimeFailure
        )
        raise raised(failure["message"])
    return dict(answer)


def _stash_in_worker(main: Any, name: str, source: Path) -> str:
    from .backends.drive import upload_pieces

    with _scratch(source.absolute()) as scratch:
        archive = scratch / "stash.tar"
        digest, size = pack(source, archive)
        answer = _ask(main, {"kind": "stash", "name": name, "digest": digest, "size": size})
        if answer.get("done"):
            return digest
        target = answer["upload"]
        try:
            file_id = upload_pieces(
                archive, target["url"], dict(target.get("headers") or {}), int(target["piece"])
            )
        except RuntimeFailure as exc:
            raise _named(name, exc) from exc
        finally:
            target = None
        _ask(main, {"kind": "stashed", "name": name, "digest": digest, "file_id": file_id})
    return digest


def _unstash_memory_in_worker(main: Any, name: str) -> dict[str, bytes] | None:
    answer = _ask(main, {"kind": "unstash", "name": name})
    if answer.get("missing"):
        return None
    try:
        return _members_from(
            name, str(answer["digest"]), str(answer["url"]), dict(answer.get("headers") or {})
        )
    finally:
        answer = None  # type: ignore[assignment]


def _unstash_in_worker(main: Any, name: str, target: Path) -> Path | None:
    from .backends.drive import download_to

    answer = _ask(main, {"kind": "unstash", "name": name})
    if answer.get("missing"):
        return None
    digest = str(answer["digest"])
    with _scratch(target.absolute()) as scratch:
        archive = scratch / "unstash.tar"
        try:
            download_to(str(answer["url"]), dict(answer.get("headers") or {}), archive)
        except RuntimeFailure as exc:
            raise _named(name, exc) from exc
        finally:
            answer = None  # type: ignore[assignment]
        _verify(name, archive, digest)
        unpack(archive, target, scratch)
    return target


# -- the local process answering a worker -----------------------------------------------


def grant(message: dict[str, Any]) -> dict[str, Any]:
    """Answer one worker request with what it needs to move the bytes itself.

    ``stash`` answers ``done`` when Drive already holds the digest, after writing the ref,
    and otherwise a resumable upload session URL. ``stashed`` records the uploaded file and
    writes the ref. ``unstash`` answers the blob's media URL with the short-lived headers, or
    ``missing``. A failure is returned as ``error`` for the worker to raise.
    """
    try:
        name = check_name(message.get("name"))
        backend = drive_backend()
        kind = message["kind"]
        try:
            if kind == "stash":
                digest, size = str(message["digest"]), int(message["size"])
                if backend.has(digest):
                    commit(backend, name, digest)
                    return {"done": True}
                return {"upload": backend.begin_upload(digest, size)}
            if kind == "stashed":
                digest = str(message["digest"])
                backend.record_upload(digest, str(message["file_id"]))
                commit(backend, name, digest)
                return {"done": True}
            if kind == "unstash":
                digest = backend.read_ref(REF_SPACE + name)
                if digest is None:
                    return {"missing": True}
                source = backend.pull_source(digest)
                return {
                    "digest": digest,
                    "url": source["url"],
                    "headers": dict(source.get("headers") or {}),
                }
        except RuntimeFailure as exc:
            raise _named(name, exc) from exc
        raise ValueError(f"unknown handoff request {kind!r}")
    except (LetifyError, ValueError, OSError) as exc:
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}


class GrantWatcher:
    """Answers a running call's requests for Drive grants, on a thread of its own.

    The same shape as the data sender and the all_reduce broker: it polls the channel's
    connection for requests the worker made on the Drive stream, and answers each with a
    ``drive_grant`` request. It carries URLs and short-lived headers, never the bytes.
    """

    def __init__(self, channel: Any, poll_s: float = 0.02) -> None:
        self._channel = channel
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="letify-drive", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def _run(self) -> None:
        while not self._stop.is_set():
            connection = getattr(self._channel, "connection", None)
            asks = connection.take_grants() if connection is not None else []
            for ask in asks:
                reply = grant(ask)
                try:
                    self._channel.request(
                        {"op": "drive_grant", "id": ask.get("id"), **reply}, timeout=300
                    )
                except Exception:
                    # The worker is gone or its call already failed; nobody is waiting.
                    pass
            self._stop.wait(self._poll_s)


__all__ = ["GrantWatcher", "grant", "stash", "unstash"]
