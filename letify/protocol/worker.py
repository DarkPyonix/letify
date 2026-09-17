"""The worker that runs inside a runtime.

This module holds source code as a string rather than code to import, because it has to
run on the remote side where letify is not installed. The channel sends it once, the
remote Python executes it, and from then on every message is binary frames over the same
pipes, as spec "Frames" describes. The frame code itself is ``wire.py``, sent ahead of the
body, so both ends run one implementation.

Keeping one process alive is what makes three features work.

The process persists, so a value a declared body stored with ``letify.session_cache``
is still there for a later call. The store lives in the letify module, which the
declared function imports by reference, so the runtime's environment has to include
letify.

A blob table persists, so a large argument is sent once and reused by name. The
worker answers which digests it already holds before the caller sends anything.

Files written into the runtime survive between calls, so a volume can materialize
an environment archive or a checkpoint and a later call can read it from disk.

The worker's file descriptors 1 and 2 are pipes drained by threads into ``STDOUT`` and
``STDERR`` frames, as spec "Worker output" describes, so nothing the body prints can fill a
pipe or mix with a reply.
"""

from __future__ import annotations

from pathlib import Path

from . import wire as _wire

#: Passed to ``python -c`` to get the worker running.
#:
#: ``python -`` cannot be used: it reads all of standard input to EOF before it
#: compiles anything, and the pipe has to stay open for requests. So a stub small
#: enough to survive shell quoting reads a byte count line and that many bytes of source,
#: execs them, and leaves standard input where it was.
BOOTSTRAP = (
    "import sys;"
    "b=sys.stdin.buffer;"
    "n=int(b.readline());"
    "exec(compile(b.read(n).decode(),'letify-worker','exec'))"
)

_BODY = r'''
import hashlib, io, queue, sys, tarfile, time, traceback

_BLOBS = {}
_SIZES = {}
# Set by blob_dir on a persistent provider: where argument blobs are also written, and the
# total size the directory is kept under.
_DISK = {}
_PICKLE_MAGIC = b"LTFYPKL1"
_READ_SIZE = 1 << 16

# Frames go to a private copy of the pipe this process started on. Descriptors 1 and 2 are
# pointed at pipes of their own before anything else can print, so output from C extensions
# and from child processes arrives as frames too.
_FRAME_FD = os.dup(1)
_ERR_FD = os.dup(2)
widen_pipe(0)
widen_pipe(_FRAME_FD)
_SENDER = (TextSender if _LETIFY_TEXT_FRAMES else Sender)(fd_writer(_FRAME_FD))


class _Pump(threading.Thread):
    """Reads one captured descriptor and sends what it reads as output frames."""

    def __init__(self, fd, kind):
        threading.Thread.__init__(self, daemon=True)
        self.kind = kind
        self.busy = False
        self.read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)

    def run(self):
        while True:
            try:
                data = os.read(self.read_fd, _READ_SIZE)
            except OSError:
                break
            if not data:
                break
            self.busy = True
            try:
                _SENDER.frame(self.kind, 0, data)
            except OSError:
                break
            finally:
                self.busy = False
        os.close(self.read_fd)


_PUMPS = [_Pump(1, STDOUT), _Pump(2, STDERR)]
for _pump in _PUMPS:
    _pump.start()


def _forked_child():
    """Detach a process the body forked from the worker's channel and tie it to the worker.

    The frame reader thread may hold the lock of ``sys.stdin`` at the fork, and
    multiprocessing closes ``sys.stdin`` in its children, so the child gets its own.
    """
    try:
        sys.stdin = open(os.devnull)
    except OSError:
        pass
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal
            ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0)
        except (OSError, AttributeError):
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forked_child)

def _readable(fd):
    try:
        import select
        return bool(select.select([fd], [], [], 0)[0])
    except (OSError, ValueError):
        return False


def _settle():
    """Wait until what the body wrote has been sent, so it precedes the reply."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if os.name == "nt":
        time.sleep(0.001)
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _pending():
            # A pump that has just read may not have marked itself busy yet. Yielding once
            # lets it run, so a second idle reading means nothing is left to send.
            time.sleep(0)
            if not _pending():
                return
        else:
            time.sleep(0.0001)


def _pending():
    return any(pump.busy or _readable(pump.read_fd) for pump in _PUMPS)


def _release_output():
    """Give descriptors 1 and 2 back to the original pipes and let the pumps finish."""
    _settle()
    os.dup2(_FRAME_FD, 1)
    os.dup2(_ERR_FD, 2)
    for pump in _PUMPS:
        pump.join(5)


def _digest(payload):
    try:
        import blake3
        return blake3.blake3(payload).hexdigest(length=16)
    except ImportError:
        return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _resolve(value):
    """Replace blob references with the values they name, recursively."""
    kind = getattr(value, "__letify_kind__", None)
    if kind == "blob":
        entry = _BLOBS.get(value.digest) or _disk_load(value.digest)
        if entry is None:
            raise KeyError("blob %s was never sent to this runtime" % value.digest)
        if entry[0] == "value":
            return entry[1]
        # A value that may be mutated is unpickled from a fresh copy for every call.
        return pickle.loads(entry[1], buffers=[bytearray(b) for b in entry[2]])
    if isinstance(value, (list, tuple)):
        return type(value)(_resolve(v) for v in value)
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    return value


_NO_LETIFY = (
    "letify is not installed in this runtime's environment, so the call, which refers "
    "to letify (for example through letify.session_cache), cannot be loaded. Add it to "
    "the project with 'uv add letify' so uv.lock carries it into the runtime."
)


def _ship_main_to_children(cloudpickle):
    """Let a child process the body spawns rebuild what the caller's __main__ defined.

    Spec "Child processes of a call". Set once, on the pickler multiprocessing uses.
    """
    try:
        from multiprocessing.reduction import ForkingPickler
    except ImportError:
        return
    if getattr(ForkingPickler, "_letify_ships_main", False):
        return
    import types

    previous = getattr(ForkingPickler, "reducer_override", None)

    def reducer_override(self, obj):
        defined = isinstance(obj, (types.FunctionType, type))
        if defined and getattr(obj, "__module__", None) == "__main__":
            found = sys.modules.get("__main__")
            for part in getattr(obj, "__qualname__", "").split("."):
                found = getattr(found, part, None)
            if found is not obj:
                return cloudpickle.loads, (cloudpickle.dumps(obj),)
        if previous is not None:
            return previous(self, obj)
        return NotImplemented

    ForkingPickler.reducer_override = reducer_override
    ForkingPickler._letify_ships_main = True


def _load_call(request):
    # Imported here, not at start: until the worker moves to the project's interpreter it
    # runs on whatever python3 the machine has, and needs the standard library only.
    import cloudpickle

    _ship_main_to_children(cloudpickle)
    try:
        return cloudpickle.loads(request["payload"], buffers=request.get("buffers") or ())
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "letify":
            raise ModuleNotFoundError(_NO_LETIFY, name=exc.name) from exc
        raise


def _reply(stream, outcome):
    try:
        _SENDER.message(REPLY, stream, outcome)
    except OSError:
        os._exit(0)
    except BaseException:
        _SENDER.message(REPLY, stream, {
            "ok": False,
            "error": "the return value could not be serialized",
            "traceback": traceback.format_exc(),
        })


def _op_call(request):
    data = request.pop("data", None)
    if data is None:
        return _call(request)
    state = _data_register(data)
    try:
        outcome = _call(request)
        # Unregistered first, so the write-back walks the real file system rather than the
        # manifest, and a file that never arrived is simply not there.
        _data_unregister(state)
        if data.get("outputs"):
            _DATA_WRITTEN[data["dir"]] = _data_collect(data, state["placed"])
        return outcome
    finally:
        import shutil
        _data_unregister(state)
        _data_check_cache(data, state["placed"])
        shutil.rmtree(data["dir"], ignore_errors=True)


# Write-back lists of returned calls, by call directory, until data_written takes them.
_DATA_WRITTEN = {}


def _data_hash_file(path):
    """The file digest of a file on the runtime: blake3 when present, else blake2b."""
    hasher = _data_hasher()
    if hasher is None:
        hasher = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(1 << 22)
            if not piece:
                break
            hasher.update(piece)
    if hasher.name == "blake2b":
        return hasher.hexdigest()
    return hasher.hexdigest(length=16)


def _data_collect(data, placed):
    """Commit each file a returned call created or changed at an output into the cache."""
    import stat
    skipped = (".git", ".venv", "__pycache__")
    written = {}
    for output in data["outputs"]:
        files = []
        if os.path.isfile(output) and not os.path.islink(output):
            candidates = [("", output)]
        elif os.path.isdir(output) and not os.path.islink(output):
            candidates = []
            for directory, names, found in os.walk(output, followlinks=False):
                names[:] = [name for name in names if name not in skipped]
                for name in found:
                    full = os.path.join(directory, name)
                    rel = os.path.relpath(full, output).replace(os.sep, "/")
                    candidates.append((rel, full))
        else:
            candidates = []
        for rel, full in sorted(candidates):
            try:
                info = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            known = placed.get(full)
            if known is not None and known[1:] == (info.st_ino, info.st_size, info.st_mtime_ns):
                continue
            digest = _data_hash_file(full)
            if known is not None and known[0] == digest:
                continue
            final = _data_file(data["blobs"], digest)
            try:
                held = os.stat(final).st_size == info.st_size
            except OSError:
                held = False
            if not held:
                os.makedirs(os.path.dirname(final), exist_ok=True)
                if info.st_nlink > 1:
                    # Still a link to another cache file, written in place: copy it out.
                    import shutil
                    partial = "%s.partial.%d" % (final, os.getpid())
                    shutil.copyfile(full, partial)
                    full = partial
                try:
                    os.chmod(full, 0o444)
                except OSError:
                    pass
                os.replace(full, final)
            files.append([rel, digest, info.st_size])
        written[output] = files
    return written


def _data_check_cache(data, placed):
    """Remove a cache file a body wrote into through its hard link."""
    for _path, (digest, _ino, size, mtime, cached) in placed.items():
        if cached is None:
            continue
        source = _data_file(data["blobs"], digest)
        try:
            info = os.stat(source)
        except OSError:
            continue
        if (info.st_size, info.st_mtime_ns) != cached:
            try:
                os.remove(source)
            except OSError:
                pass


def _op_data_written(request):
    """The write-back list of a returned call, removed as it is answered."""
    return {"ok": True, "value": _DATA_WRITTEN.pop(request["dir"], {})}


#: Bytes of one piece of a streamed file blob, one ``DATA`` frame each.
_DATA_PIECE = 8 << 20


def _op_data_stream(stream, request):
    """Send a whole file blob as replies on one stream, without waiting between pieces.

    Spec "Writing back". The pipe's own capacity paces this, so the link stays busy while
    the caller writes and hashes the piece before.
    """
    path = _data_file(request["dir"], request["digest"])
    with open(path, "rb") as handle:
        while True:
            buffer = bytearray(_DATA_PIECE)
            count = handle.readinto(buffer)
            del buffer[count:]
            last = count < _DATA_PIECE
            _SENDER.message(REPLY, stream, {
                "ok": True, "value": pickle.PickleBuffer(buffer), "last": last,
            })
            buffer = None
            if last:
                return


def _data_file(root, digest):
    return os.path.join(root, digest[:2], digest)


def _data_hasher():
    try:
        import blake3
    except ImportError:
        return None
    return blake3.blake3(max_threads=blake3.blake3.AUTO)


def _data_commit(partial, final, digest, hasher):
    """Rename a received file into the cache once its digest matches, read-only."""
    if hasher is not None:
        found = hasher.hexdigest(length=16)
        if found != digest:
            os.remove(partial)
            raise ValueError("file blob %s arrived with digest %s" % (digest, found))
    try:
        os.chmod(partial, 0o444)
    except OSError:
        pass
    os.replace(partial, final)


def _op_data_have(request):
    """Which file blobs the cache holds with the expected size."""
    held = []
    for digest, size in request["digests"]:
        path = _data_file(request["dir"], digest)
        try:
            if os.stat(path).st_size == size:
                held.append(digest)
                # Marks it used, so eviction leaves it alone until the call links it.
                os.utime(path)
        except OSError:
            pass
    return {"ok": True, "value": held}


# Spec "Runtime data cache budget".
_DATA_DEFAULT_BUDGET = 50 << 30
_DATA_RECENT_S = 600


def _data_scan(root):
    """Committed blobs as (last use, path, size, link count), and their total size."""
    files = []
    total = 0
    if not os.path.isdir(root):
        return files, total
    for shard in os.scandir(root):
        if not shard.is_dir(follow_symlinks=False):
            continue
        for entry in os.scandir(shard.path):
            if ".partial." in entry.name:
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            files.append((info.st_mtime, entry.path, info.st_size, info.st_nlink))
            total += info.st_size
    return files, total


def _data_budget(root, total, budget):
    if budget is not None:
        return int(budget)
    probe = root
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        info = os.statvfs(probe)
        free = info.f_bavail * info.f_frsize
    except (OSError, AttributeError):
        return _DATA_DEFAULT_BUDGET
    return min(_DATA_DEFAULT_BUDGET, (total + free) // 2)


def _op_data_evict(request):
    """Remove least recently used blobs no call links until the cache is within budget."""
    started = time.monotonic()
    root = os.path.expanduser(request["dir"])
    files, total = _data_scan(root)
    budget = _data_budget(root, total, request.get("budget"))
    removed = freed = 0
    if total > budget:
        now = time.time()
        files.sort()
        for used, path, size, links in files:
            if total <= budget:
                break
            if links > 1 or now - used < _DATA_RECENT_S:
                continue
            try:
                os.remove(path)
            except OSError:
                continue
            total -= size
            removed += 1
            freed += size
    return {"ok": True, "value": {
        "files": removed, "bytes": freed, "total": total, "budget": budget,
        "seconds": time.monotonic() - started,
    }}


def _op_data_cache(request):
    """The cache's blob count, size and budget, after removing unlinked blobs when clearing."""
    root = os.path.expanduser(request["dir"])
    files, total = _data_scan(root)
    removed = freed = 0
    if request.get("clear"):
        for _used, path, size, links in files:
            if links > 1:
                continue
            try:
                os.remove(path)
            except OSError:
                continue
            removed += 1
            freed += size
        files, total = _data_scan(root)
    return {"ok": True, "value": {
        "files": len(files), "bytes": total,
        "budget": _data_budget(root, total, request.get("budget")),
        "removed": removed, "removed_bytes": freed,
    }}


_DATA_OPEN = {}


def _op_data_put(request):
    """Append one piece of a file blob, and commit it after the last piece."""
    digest = request["digest"]
    final = _data_file(request["dir"], digest)
    partial = "%s.partial.%d" % (final, os.getpid())
    chunk = request.pop("chunk")
    if request["offset"] == 0:
        os.makedirs(os.path.dirname(final), exist_ok=True)
        stale = _DATA_OPEN.pop(digest, None)
        if stale is not None:
            stale[0].close()
        _DATA_OPEN[digest] = (open(partial, "wb"), _data_hasher())
    handle, hasher = _DATA_OPEN[digest]
    handle.write(chunk)
    if hasher is not None:
        hasher.update(chunk)
    chunk = None
    if request.get("last"):
        del _DATA_OPEN[digest]
        handle.close()
        _data_commit(partial, final, digest, hasher)
        _data_arrived(request["dir"], digest)
    return {"ok": True, "value": None}


def _op_data_pull(request):
    """Download file blobs from the bucket, eight at a time, then forget the headers."""
    import urllib.request

    headers = request.pop("headers", None) or {}
    root = request["dir"]
    items = list(request.pop("items"))
    failures = []
    lock = threading.Lock()

    def fetch():
        while True:
            with lock:
                if not items or failures:
                    return
                digest, url = items.pop()
            final = _data_file(root, digest)
            partial = "%s.partial.%d.%d" % (final, os.getpid(), threading.get_ident())
            try:
                os.makedirs(os.path.dirname(final), exist_ok=True)
                hasher = _data_hasher()
                wanted = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(wanted, timeout=3600) as response, \
                        open(partial, "wb") as out:
                    while True:
                        piece = response.read(1 << 20)
                        if not piece:
                            break
                        out.write(piece)
                        if hasher is not None:
                            hasher.update(piece)
                _data_commit(partial, final, digest, hasher)
                _data_arrived(root, digest)
            except BaseException as exc:
                with lock:
                    failures.append("%s: %s" % (digest, exc))
                return

    threads = [threading.Thread(target=fetch, daemon=True) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    headers = None
    if failures:
        raise RuntimeError("pulling file blobs failed: " + "; ".join(failures))
    return {"ok": True, "value": None}


class _DataFailure(Exception):
    """A blob that never arrived. Infrastructure, so the caller may retry the call."""


# Calls whose data is still arriving, by call directory, and what each is waiting for.
_DATA_CALLS = {}
_DATA_STATS = {}
_DATA_READY = threading.Condition()
_DATA_PATCHED = []

#: Set while this thread is placing a file. The patch below stands aside for it, because
#: placing one opens and stats the very paths the patch answers from the manifest.
_DATA_PLACING = threading.local()

#: How far behind the send order an observed read may be before it counts as a miss.
_DATA_MISS_AHEAD = 32

#: Marks a file being copied into place. It is never listed and never read.
_DATA_PARTIAL = ".letify-placing."


def _data_register(data):
    """Build the call's layout from its manifest, place what has arrived, and patch reads.

    Spec "What the body sees before a file arrives": every directory exists before the call
    starts, a file appears only once its blob is complete, and everything else is answered
    from the manifest.
    """
    state = {
        "dir": data["dir"],
        "blobs": data["blobs"],
        "placed": {},
        "pending": {},
        "by_digest": {},
        "copy": {},
        "order": {},
        "arrived": 0,
        "wait": float(data.get("wait") or 600.0),
        "observe": bool(data.get("observe", True)),
        "prefetch": int(data.get("prefetch") or 64),
        "files": 0,
        "bytes": 0,
        "seconds": 0.0,
        "waiting": 0,
        "since": None,
        "first": None,
        "misses": 0,
    }
    for index, digest in enumerate(data.get("order") or ()):
        state["order"][digest] = index
    for directory in data.get("dirs", ()):
        os.makedirs(directory, exist_ok=True)
    for path, digest, copy in data.get("links", ()):
        state["copy"][path] = bool(copy)
    manifest = data.get("manifest") or {}
    for path, entry in manifest.items():
        digest, size = entry[0], int(entry[1])
        state["pending"][path] = (digest, size)
        state["by_digest"].setdefault(digest, []).append(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not manifest:
        # No manifest travelled, so every link the call carries is already in the cache.
        for path, digest, _copy in data.get("links", ()):
            state["pending"][path] = (digest, 0)
            state["by_digest"].setdefault(digest, []).append(path)
    with _DATA_READY:
        _DATA_CALLS[state["dir"]] = state
        for path in list(state["pending"]):
            _data_place(state, path)
        if state["pending"]:
            # Only a call still waiting for bytes needs the patch, the manifest on disk and
            # the wrappers. A call whose files are all placed reads the real file system.
            # Inside the lock: the data thread moves a path out of pending and into placed
            # as each blob completes, and the manifest is written from both maps.
            _data_install_patch()
            _data_write_manifest(state)
            _data_observe(state)
    return state


def _data_unregister(state):
    """Stop answering this call's reads from the manifest, and keep its wait counters."""
    with _DATA_READY:
        if _DATA_CALLS.pop(state["dir"], None) is None:
            return
        _DATA_STATS[state["dir"]] = {
            "files": state["files"],
            "bytes": state["bytes"],
            "seconds": state["seconds"],
            "first": state["first"],
            "misses": state["misses"],
        }
        _DATA_READY.notify_all()


def _data_place(state, path):
    """Put one file at its runtime path when its blob is complete. Answers whether it is."""
    entry = state["pending"].get(path)
    if entry is None:
        return path in state["placed"]
    _DATA_PLACING.on = True
    try:
        return _data_place_now(state, path, entry)
    finally:
        _DATA_PLACING.on = False


def _data_place_now(state, path, entry):
    """Place one file, with this thread's reads going to the real file system."""
    digest, _size = entry
    source = _data_file(state["blobs"], digest)
    if not os.path.isfile(source):
        return False
    import shutil
    try:
        os.utime(source)
    except OSError:
        pass
    linked = False
    if not state["copy"].get(path, False):
        try:
            os.link(source, path)
            linked = True
        except OSError:
            pass
    if not linked:
        # Copied to a name of its own and renamed into place, so a reader never stats a
        # file that is half written. Spec "What the body sees before a file arrives".
        partial = "%s%s%d" % (path, _DATA_PARTIAL, os.getpid())
        try:
            shutil.copyfile(source, partial)
            os.replace(partial, path)
        except OSError:
            try:
                os.remove(partial)
            except OSError:
                pass
            return False
    info = os.stat(path)
    cached = (info.st_size, info.st_mtime_ns) if linked else None
    state["placed"][path] = (digest, info.st_ino, info.st_size, info.st_mtime_ns, cached)
    del state["pending"][path]
    return True


def _data_arrived(blobs, digest):
    """A blob finished arriving: place every file of every running call that wants it."""
    with _DATA_READY:
        for state in _DATA_CALLS.values():
            if state["blobs"] != blobs:
                continue
            state["arrived"] += 1
            for path in list(state["by_digest"].get(digest, ())):
                if path in state["pending"]:
                    _data_place(state, path)
        _DATA_READY.notify_all()


def _data_want(digests, blocking=False):
    """Ask the local process for these blobs, in this order. Nothing replies to it."""
    if not digests:
        return
    message = {"op": "data_want", "digests": list(digests), "blocking": blocking}
    try:
        _SENDER.message(REQUEST, DATA_STREAM, message)
    except OSError:
        pass


def _data_state_for(path):
    """The running call a path belongs to, or None."""
    for state in _DATA_CALLS.values():
        if path.startswith(state["dir"]):
            return state
    return None


def _data_wait(path):
    """Block this caller until the file at ``path`` holds all of its bytes.

    Spec "Streaming the rest while the call runs": only the thread that opened the file
    waits, and the call, its other readers and the background sender keep running.
    """
    with _DATA_READY:
        state = _data_state_for(path)
        if state is None or path in state["placed"]:
            return
        entry = state["pending"].get(path)
        if entry is None:
            return
        digest, size = entry
        index = state["order"].get(digest)
        if index is not None and index > state["arrived"] + _DATA_MISS_AHEAD:
            state["misses"] += 1
        started = time.monotonic()
        if state["waiting"] == 0:
            state["since"] = started
        state["waiting"] += 1
        state["files"] += 1
        state["bytes"] += size
        if state["first"] is None:
            state["first"] = os.path.basename(path)
        deadline = started + state["wait"]
    _data_want([digest], True)
    try:
        with _DATA_READY:
            while path in state["pending"] and state["dir"] in _DATA_CALLS:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise _DataFailure(
                        "the file blob %s of %s did not arrive within %.0f s"
                        % (digest, os.path.basename(path), state["wait"])
                    )
                _DATA_READY.wait(min(left, 0.5))
    finally:
        with _DATA_READY:
            state["waiting"] -= 1
            if state["waiting"] == 0 and state["since"] is not None:
                state["seconds"] += time.monotonic() - state["since"]
                state["since"] = None


def _op_data_stats(request):
    """What a finished call waited for, as spec "Data log line" reports it."""
    return {"ok": True, "value": _DATA_STATS.pop(request["dir"], {})}


def _data_pending_entry(state, path):
    """The manifest entry of a file that has not arrived, or None."""
    return state["pending"].get(path)


class _PendingEntry:
    """A directory entry of a file the manifest names but the runtime does not hold yet."""

    __slots__ = ("name", "path", "_size")

    def __init__(self, name, path, size):
        self.name = name
        self.path = path
        self._size = size

    def inode(self):
        return 0

    def is_dir(self, follow_symlinks=True):
        return False

    def is_file(self, follow_symlinks=True):
        return True

    def is_symlink(self):
        return False

    def stat(self, follow_symlinks=True):
        return _data_fake_stat(self._size)


class _Scandir:
    """What the patched ``os.scandir`` answers: real entries, then the pending ones."""

    def __init__(self, entries, close):
        self._entries = entries
        self._close = close

    def __iter__(self):
        return iter(self._entries)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
        return False

    def close(self):
        if self._close is not None:
            self._close.close()
            self._close = None


def _data_fake_stat(size):
    """A stat result for a file that has not arrived: a regular file of its final size."""
    return os.stat_result((0o100444, 0, 0, 1, 0, 0, int(size), 0, 0, 0))


def _data_install_patch():
    """Answer listings and sizes from the manifest, and block a read until its file is here.

    Installed once in the process that runs the body, so anything the body forks inherits it.
    """
    if _DATA_PATCHED:
        return
    _DATA_PATCHED.append(True)
    import builtins
    import io
    real = {
        "open": builtins.open,
        "io_open": io.open,
        "os_open": os.open,
        "listdir": os.listdir,
        "scandir": os.scandir,
        "stat": os.stat,
        "lstat": os.lstat,
    }

    def full(path):
        try:
            return os.path.abspath(os.fspath(path))
        except TypeError:
            return None

    def pending_of(path):
        if not _DATA_CALLS or getattr(_DATA_PLACING, "on", False):
            return None
        text = full(path)
        if text is None:
            return None
        state = _data_state_for(text)
        if state is None:
            return None
        entry = _data_pending_entry(state, text)
        return (state, text, entry) if entry is not None else None

    def wait_for(path):
        found = pending_of(path)
        if found is not None:
            _data_wait(found[1])

    def opened(file, *args, **kwargs):
        wait_for(file)
        return real["open"](file, *args, **kwargs)

    def io_opened(file, *args, **kwargs):
        wait_for(file)
        return real["io_open"](file, *args, **kwargs)

    def os_opened(path, *args, **kwargs):
        wait_for(path)
        return real["os_open"](path, *args, **kwargs)

    def names_of(path):
        """The manifest names of the directory at ``path``, pending ones included."""
        if not _DATA_CALLS or getattr(_DATA_PLACING, "on", False):
            return ()
        text = full(path)
        if text is None:
            return ()
        state = _data_state_for(text)
        if state is None:
            return ()
        prefix = text.rstrip("/") + "/"
        found = []
        for pending in state["pending"]:
            if pending.startswith(prefix) and "/" not in pending[len(prefix):]:
                found.append((pending[len(prefix):], state["pending"][pending][1]))
        return found

    def listed(path="."):
        entries = [name for name in real["listdir"](path) if _DATA_PARTIAL not in name]
        for name, _size in names_of(path):
            if name not in entries:
                entries.append(name)
        return entries

    def scanned(path="."):
        pending = names_of(path)
        found = real["scandir"](path)
        if not pending:
            return found
        entries = [entry for entry in found if _DATA_PARTIAL not in entry.name]
        held = {entry.name for entry in entries}
        prefix = full(path).rstrip("/") + "/"
        for name, size in pending:
            if name not in held:
                entries.append(_PendingEntry(name, prefix + name, size))
        return _Scandir(entries, found)

    def stated(path, **kwargs):
        try:
            return real["stat"](path, **kwargs)
        except FileNotFoundError:
            found = pending_of(path)
            if found is None:
                raise
            return _data_fake_stat(found[2][1])

    def lstated(path, **kwargs):
        try:
            return real["lstat"](path, **kwargs)
        except FileNotFoundError:
            found = pending_of(path)
            if found is None:
                raise
            return _data_fake_stat(found[2][1])

    def disguise(patched, module, name):
        """Give a wrapper the name it replaced, so pickling it finds the name, not the
        closure. A process the body spawns would otherwise be sent this function by value,
        and it holds things no pickle can carry."""
        patched.__module__ = module
        patched.__name__ = name
        patched.__qualname__ = name
        return patched

    builtins.open = disguise(opened, "builtins", "open")
    io.open = disguise(io_opened, "io", "open")
    os.open = disguise(os_opened, "os", "open")
    os.listdir = disguise(listed, "os", "listdir")
    os.scandir = disguise(scanned, "os", "scandir")
    os.stat = disguise(stated, "os", "stat")
    os.lstat = disguise(lstated, "os", "lstat")


def _data_write_manifest(state):
    """Write the call's manifest beside its files, for a process started fresh.

    Spec "What the body sees before a file arrives": a spawned interpreter installs the
    same patch from ``letify_pending.pth`` and reads the layout from here, because it has
    no channel of its own to ask on.
    """
    import json
    # Copies of both maps, so neither the data thread placing a file nor a future caller
    # that forgets the lock can change one while it is being read.
    entries = {}
    for path, (_digest, size) in list(state["pending"].items()):
        entries[path] = size
    for path in list(state["placed"]):
        entries.setdefault(path, os.path.getsize(path) if os.path.isfile(path) else 0)
    try:
        with open(os.path.join(state["dir"], ".letify-pending.json"), "w") as handle:
            json.dump({"dir": state["dir"], "files": entries, "wait": state["wait"]}, handle)
    except OSError:
        return
    directory = os.path.join(os.path.dirname(state["blobs"]), "pending")
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "letify_pending.py"), "w") as handle:
            handle.write(_PENDING_MODULE)
        with open(os.path.join(directory, "letify_pending.pth"), "w") as handle:
            handle.write("import letify_pending\n")
    except OSError:
        return
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if directory not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([directory, *parts])
    os.environ["LETIFY_PENDING"] = state["dir"]


#: Installed in a process the body starts fresh. It answers the same listings from the
#: manifest file and waits for a file to appear, which the worker places as blobs arrive.
_PENDING_MODULE = """
# letify: answer a streaming call's listings in a process started fresh.

import builtins
import io
import json
import os
import time


def _install():
    call_dir = os.environ.get("LETIFY_PENDING")
    if not call_dir:
        return
    try:
        with open(os.path.join(call_dir, ".letify-pending.json")) as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return
    files = record.get("files") or {}
    wait = float(record.get("wait") or 600.0)
    real_open, real_io_open = builtins.open, io.open
    real_listdir, real_scandir = os.listdir, os.scandir
    real_stat = os.stat

    def missing(path):
        try:
            text = os.path.abspath(os.fspath(path))
        except TypeError:
            return None
        if not text.startswith(call_dir) or os.path.exists(text):
            return None
        return text if text in files else None

    def wait_for(path):
        text = missing(path)
        if text is None:
            return
        deadline = time.monotonic() + wait
        while not os.path.exists(text):
            if time.monotonic() >= deadline:
                raise RuntimeError("letify: %s did not arrive within %.0f s" % (text, wait))
            time.sleep(0.02)

    def opened(file, *args, **kwargs):
        wait_for(file)
        return real_open(file, *args, **kwargs)

    def io_opened(file, *args, **kwargs):
        wait_for(file)
        return real_io_open(file, *args, **kwargs)

    def names(path):
        try:
            text = os.path.abspath(os.fspath(path))
        except TypeError:
            return []
        if not text.startswith(call_dir):
            return []
        prefix = text.rstrip("/") + "/"
        return [
            name[len(prefix):]
            for name in files
            if name.startswith(prefix) and "/" not in name[len(prefix):]
        ]

    def listed(path="."):
        found = list(real_listdir(path))
        for name in names(path):
            if name not in found:
                found.append(name)
        return found

    def stated(path, **kwargs):
        try:
            return real_stat(path, **kwargs)
        except FileNotFoundError:
            text = missing(path)
            if text is None:
                raise
            return os.stat_result((0o100444, 0, 0, 1, 0, 0, int(files[text]), 0, 0, 0))

    builtins.open = opened
    io.open = io_opened
    os.listdir = listed
    os.stat = stated
    os.scandir = real_scandir


_install()
"""


def _data_observe(state):
    """Report the read order the body actually takes, as spec "Observing the read order on
    the runtime" describes. Every wrapper calls the original and only reports what it saw."""
    if not state["observe"]:
        return
    try:
        import torch.utils.data as data_module
    except BaseException:
        return
    if getattr(data_module, "_letify_observed", False):
        return
    data_module._letify_observed = True
    prefetch = state["prefetch"]

    def digests_of(paths):
        found = []
        for path in paths:
            try:
                text = os.path.abspath(os.fspath(path))
            except TypeError:
                continue
            for call in list(_DATA_CALLS.values()):
                entry = call["pending"].get(text)
                if entry is not None:
                    found.append(entry[0])
        return found

    def sequence_of(dataset):
        for name in ("samples", "imgs", "files", "paths", "image_paths"):
            found = getattr(dataset, name, None)
            if isinstance(found, (list, tuple)) and found:
                return found
        return None

    loader = getattr(data_module, "DataLoader", None)
    if loader is not None and not getattr(loader, "_letify_wrapped", False):
        original = loader.__iter__

        def __iter__(self):
            try:
                sequence = sequence_of(getattr(self, "dataset", None))
                sampler = getattr(self, "batch_sampler", None) or getattr(self, "sampler", None)
                if sequence is not None and sampler is not None:
                    ahead = max(1, prefetch) * max(1, int(getattr(self, "batch_size", 1) or 1))
                    wanted = []
                    for item in sampler:
                        for index in item if isinstance(item, (list, tuple)) else (item,):
                            if isinstance(index, int) and 0 <= index < len(sequence):
                                wanted.append(sequence[index])
                        if len(wanted) >= ahead:
                            break
                    _data_want(digests_of(wanted))
            except BaseException:
                pass
            return original(self)

        loader.__iter__ = __iter__
        loader._letify_wrapped = True

    dataset = getattr(data_module, "Dataset", None)
    if dataset is not None and not getattr(dataset, "_letify_wrapped", False):
        base = dataset.__getitem__

        def __getitem__(self, index):
            try:
                sequence = sequence_of(self)
                if sequence is not None and isinstance(index, int):
                    ahead = sequence[index : index + max(1, prefetch)]
                    _data_want(digests_of(ahead))
            except BaseException:
                pass
            return base(self, index)

        dataset.__getitem__ = __getitem__
        dataset._letify_wrapped = True


def _call(request):
    fn, args, kwargs = _load_call(request)
    request.clear()
    args = _resolve(args)
    kwargs = _resolve(kwargs)
    try:
        value = fn(*args, **kwargs)
        if hasattr(value, "__await__"):
            # An async declaration is awaitable locally. The body still has to run to
            # completion here, and awaiting it here is what lets it use await inside.
            import asyncio
            value = asyncio.run(value)
    except ModuleNotFoundError as exc:
        # A body that imports letify while running needs it as much as a call that
        # refers to it when loaded, so the same explanation applies.
        if (exc.name or "").split(".")[0] == "letify":
            raise ModuleNotFoundError(_NO_LETIFY, name=exc.name) from exc
        raise
    return {"ok": True, "value": value}


def _disk_path(digest, pickled):
    name = digest + (".pickle" if pickled else "")
    return os.path.join(_DISK["path"], digest[:2], name)


def _disk_find(digest):
    """The file holding a digest on disk, or None. Only complete files have these names."""
    if not _DISK:
        return None
    for pickled in (False, True):
        path = _disk_path(digest, pickled)
        if os.path.isfile(path):
            return path
    return None


def _disk_write(digest, parts, pickled, immutable):
    """Write a blob under the blob directory by rename, then keep the directory in its limit."""
    import struct

    path = _disk_path(digest, pickled)
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = "%s.partial.%d" % (path, os.getpid())
    with open(partial, "wb") as handle:
        if pickled:
            handle.write(_PICKLE_MAGIC + bytes([1 if immutable else 0]))
            handle.write(struct.pack(">I", len(parts)))
            for part in parts:
                handle.write(struct.pack(">Q", memoryview(part).nbytes))
        for part in parts:
            handle.write(part)
    os.replace(partial, path)
    _disk_evict(path)


def _disk_evict(keep):
    """Remove blob files, oldest modification time first, until the total is within limit."""
    files = []
    total = 0
    for shard in os.scandir(_DISK["path"]):
        if not shard.is_dir():
            continue
        for entry in os.scandir(shard.path):
            if ".partial." in entry.name or not entry.is_file():
                continue
            info = entry.stat()
            files.append((info.st_mtime_ns, entry.path, info.st_size))
            total += info.st_size
    files.sort()
    for _mtime, path, size in files:
        if total <= _DISK["limit"]:
            break
        if path == keep:
            continue
        try:
            os.remove(path)
            total -= size
        except OSError:
            pass


def _disk_load(digest):
    """Load a blob held only on disk into the blob table, or answer None."""
    import struct

    path = _disk_find(digest)
    if path is None:
        return None
    with open(path, "rb") as handle:
        payload = handle.read()
    if not path.endswith(".pickle"):
        entry = ("value", payload)
        _SIZES[digest] = len(payload)
    else:
        view = memoryview(payload)
        immutable = view[8] == 1
        count = struct.unpack(">I", view[9:13])[0]
        sizes = struct.unpack(">%dQ" % count, view[13 : 13 + 8 * count])
        offset = 13 + 8 * count
        parts = []
        for size in sizes:
            parts.append(bytes(view[offset : offset + size]))
            offset += size
        head, buffers = parts[0], parts[1:]
        _SIZES[digest] = sum(sizes)
        if immutable:
            entry = ("value", pickle.loads(head, buffers=buffers))
        else:
            entry = ("parts", head, buffers)
    _BLOBS[digest] = entry
    return entry


def _op_blob_dir(request):
    """Write argument blobs under this directory too, kept within limit bytes."""
    os.makedirs(request["path"], exist_ok=True)
    _DISK["path"] = request["path"]
    _DISK["limit"] = int(request["limit"])
    return {"ok": True, "value": None}


def _op_have(request):
    """Report which of these digests the runtime already holds, in memory or on disk."""
    held = []
    for digest in request["digests"]:
        if digest in _BLOBS:
            held.append(digest)
            continue
        path = _disk_find(digest)
        if path is not None:
            try:
                os.utime(path)
            except OSError:
                continue
            held.append(digest)
    return {"ok": True, "value": held}


def _op_put_blob(request):
    """Store a value under its content address.

    An immutable value is kept unpickled, so a repeated argument is not unpickled again.
    Anything else is kept as its pickle and buffers. With a blob directory set, the blob is
    written there as well before the reply.
    """
    digest = request["digest"]
    if request.get("kind") == "bytes":
        value = request.pop("value")
        _BLOBS[digest] = ("value", value)
        _SIZES[digest] = len(value)
        if _DISK:
            _disk_write(digest, [value], False, True)
        return {"ok": True, "value": digest}
    head = bytes(request["head"])
    buffers = list(request.get("buffers") or ())
    _SIZES[digest] = len(head) + sum(memoryview(b).nbytes for b in buffers)
    if _DISK:
        _disk_write(digest, [head, *buffers], True, bool(request.get("immutable")))
    if request.get("immutable"):
        _BLOBS[digest] = ("value", pickle.loads(head, buffers=buffers))
    else:
        _BLOBS[digest] = ("parts", head, buffers)
    return {"ok": True, "value": digest}


def _op_put_file(request):
    """Write a payload to a path inside the runtime."""
    payload = request["payload"]
    path = request["path"]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    if request.get("unpack"):
        _unpack(path, request.get("target") or os.path.dirname(path), request.get("links"))
    return {"ok": True, "value": {"path": path, "size": memoryview(payload).nbytes}}


def _unpack(path, target, links=False):
    """Extract an archive inside target.

    Both filters keep every member path inside target. ``links`` allows symlinks to
    absolute paths, which an environment archive needs because a .venv links its
    interpreter by absolute path.
    """
    os.makedirs(target, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        try:
            archive.extractall(target, filter="tar" if links else "data")
        except TypeError:
            archive.extractall(target)


def _op_pull(request):
    """Download a blob from the backend with a borrowed token, then forget the token.

    The headers are taken out of the request before anything can fail, and the request
    object is the only place they lived, so nothing holds them once this returns.
    """
    import urllib.request

    headers = request.pop("headers", None) or {}
    fetch = urllib.request.Request(request.pop("url"), headers=headers)
    headers = None
    path = request["path"]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    partial = path + ".partial"
    size = 0
    try:
        with urllib.request.urlopen(fetch, timeout=3600) as response, open(partial, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                size += len(chunk)
    finally:
        fetch = None
    os.replace(partial, path)
    if request.get("unpack"):
        _unpack(path, request.get("target") or os.path.dirname(path), request.get("links"))
    return {"ok": True, "value": {"path": path, "size": size}}


def _op_get_file(request):
    """Read a file out of the runtime, for example a checkpoint."""
    with open(request["path"], "rb") as handle:
        payload = handle.read()
    return {
        "ok": True,
        "value": {"payload": payload, "digest": _digest(payload), "size": len(payload)},
    }


def _op_pack_dir(request):
    """Pack a directory inside the runtime and return it as one payload."""
    buffer = io.BytesIO()
    root = request["path"]
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(root, arcname=os.path.basename(root.rstrip("/")), recursive=True)
    payload = buffer.getvalue()
    return {
        "ok": True,
        "value": {"payload": payload, "digest": _digest(payload), "size": len(payload)},
    }


def _op_exec(request):
    """Run plain source inside the worker, sharing its globals."""
    exec(compile(request["source"], "<letify>", "exec"), globals())
    return {"ok": True, "value": None}


def _op_eval(request):
    """Run plain source inside the worker and return what it left in __letify_value__."""
    scope = globals()
    scope.pop("__letify_value__", None)
    exec(compile(request["source"], "<letify>", "exec"), scope)
    return {"ok": True, "value": scope.pop("__letify_value__", None)}


def _reexec(stream, request):
    """Reply, then replace this process with another interpreter on the same pipes.

    The frame reader stopped after it queued this request, and the caller sends nothing
    until it reads the reply, so no byte meant for the new interpreter is read here.
    """
    python = request["python"]
    if not os.access(python, os.X_OK):
        _reply(stream, {"ok": False, "error": "%s is not an executable interpreter" % python,
                        "traceback": ""})
        threading.Thread(target=_read, daemon=True).start()
        return
    _release_output()
    _reply(stream, {"ok": True, "value": None})
    os.execv(python, [python, "-u", "-c", request["bootstrap"]])


#: The PyTorch device executor of this process: its request queue and its thread.
_DEVICE = {}


def _op_device(request):
    """Start the PyTorch device executor in a thread, reading requests on the device stream."""
    thread = _DEVICE.get("thread")
    if thread is None or not thread.is_alive():
        namespace = {"__name__": "letify_device"}
        exec(compile(request["source"], "letify-device", "exec"), namespace)
        inbox = queue.Queue()
        _DEVICE["inbox"] = inbox
        _DEVICE["thread"] = namespace["serve_channel"](request["device"], _SENDER, inbox)
    return {"ok": True, "value": None}


def _op_release(request):
    """Drop blobs the caller no longer needs."""
    for digest in request.get("blobs", ()):
        _BLOBS.pop(digest, None)
        _SIZES.pop(digest, None)
    return {"ok": True, "value": None}


def _op_stat(request):
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        maxrss = None
    return {
        "ok": True,
        "value": {
            "blobs": len(_BLOBS),
            "blob_bytes": sum(_SIZES.values()),
            "pid": os.getpid(),
            "executable": sys.executable,
            "maxrss": maxrss,
        },
    }


def _op_lease(request):
    """Arm or renew the self termination deadline.

    The worker exits on its own if the caller stops renewing, so a crashed or
    killed local process cannot leave a paid session running.
    """
    state = globals().setdefault("_LEASE", {})
    state["deadline"] = time.time() + float(request["grace"])

    def watch():
        while True:
            time.sleep(15)
            if time.time() > state.get("deadline", 0):
                os._exit(0)

    if not state.get("armed"):
        state["armed"] = True
        threading.Thread(target=watch, daemon=True).start()
    return {"ok": True, "value": state["deadline"]}


_OPS = {
    "call": _op_call,
    "have": _op_have,
    "data_have": _op_data_have,
    "data_put": _op_data_put,
    "data_pull": _op_data_pull,
    "data_evict": _op_data_evict,
    "data_cache": _op_data_cache,
    "data_written": _op_data_written,
    "data_stream": _op_data_stream,
    "data_stats": _op_data_stats,
    "blob_dir": _op_blob_dir,
    "put_blob": _op_put_blob,
    "put_file": _op_put_file,
    "pull": _op_pull,
    "get_file": _op_get_file,
    "pack_dir": _op_pack_dir,
    "exec": _op_exec,
    "eval": _op_eval,
    "release": _op_release,
    "device": _op_device,
    "stat": _op_stat,
    "lease": _op_lease,
}

#: Answered by the frame reader at once, so they work while a call runs.
_LIGHT = ("stat", "lease")

#: Answered by several replies, the last one marked. These send their own replies.
_STREAMING = ("data_stream",)

_JOBS = queue.Queue()

#: Data requests, served by a thread of their own so they run while a call runs.
_DATA_JOBS = queue.Queue()
_DATA_THREAD = []


def _data_start():
    """Start the thread that answers data requests, once."""
    if _DATA_THREAD:
        return
    thread = threading.Thread(target=_data_serve, daemon=True)
    _DATA_THREAD.append(thread)
    thread.start()


def _data_serve():
    while True:
        job = _DATA_JOBS.get()
        if job is None:
            return
        stream, request = job
        job = None
        _run(stream, request, False)


def _run(stream, request, settle):
    name = request.get("op")
    op = _OPS.get(name)
    if op is None:
        outcome = {"ok": False, "error": "unknown op %r" % name, "traceback": ""}
    elif name in _STREAMING:
        if settle:
            _settle()
        try:
            op(stream, request)
            return
        except BaseException as exc:
            # The sequence ends with a failed reply, which the caller raises.
            _reply(stream, {
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            })
            return
    else:
        try:
            outcome = op(request)
        except BaseException as exc:
            outcome = {
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
            if isinstance(exc, _DataFailure):
                # Infrastructure rather than the body's own error, so the call may be retried.
                outcome["kind"] = "runtime"
    # Dropped before replying, so a pull token the request carried lives no longer.
    request = None
    if settle:
        _settle()
    _reply(stream, outcome)


_DATA_PREFIX = b"LETIFY-DATA "


def _listen(stream, request):
    """Answer a listen request, then wait for the data connection that carries the token.

    Spec "Modal data channel". Returns the authenticated connection, with every later frame
    already sent over it, or None when standard input became readable, the wait ended, or
    the port could not be bound, so the caller keeps reading standard input.
    """
    import select
    import socket

    token = str(request["token"]).encode("ascii")
    deadline = time.monotonic() + float(request.get("wait") or 60)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", int(request["port"])))
        server.listen(4)
    except OSError as exc:
        server.close()
        _reply(stream, {"ok": False, "error": "OSError: %s" % exc, "traceback": ""})
        return None
    _reply(stream, {"ok": True, "value": None})
    streams = int(request.get("streams") or 1)
    lanes = {}
    stdin = sys.stdin.fileno()
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            ready = select.select([server, stdin], [], [], left)[0]
            if stdin in ready:
                return None
            if server not in ready:
                continue
            connection, _address = server.accept()
            lane = _authenticated(connection, token, streams)
            if lane is None or lane in lanes:
                connection.close()
                continue
            lanes[lane] = connection
            if len(lanes) == streams:
                ordered = [lanes[index] for index in range(streams)]
                lanes = {}
                return _use_connection(ordered)
    finally:
        server.close()
        for connection in lanes.values():
            connection.close()


def _authenticated(connection, token, streams=1):
    """The lane index the connection's first line names with the expected token, or None.

    Spec "Parallel data streams": lane 0 sends ``LETIFY-DATA <token>`` and lane ``i`` sends
    ``LETIFY-DATA <token> <i>``.
    """
    import hmac

    expected = _DATA_PREFIX + token
    line = b""
    try:
        connection.settimeout(10)
        while not line.endswith(b"\n") and len(line) < len(expected) + 4:
            chunk = connection.recv(1)
            if not chunk:
                return None
            line += chunk
        connection.settimeout(None)
    except OSError:
        return None
    if not line.endswith(b"\n"):
        return None
    head, rest = line[: len(expected)], line[len(expected) : -1]
    if not hmac.compare_digest(head, expected):
        return None
    if not rest:
        return 0
    if rest[:1] != b" " or not rest[1:].isdigit():
        return None
    lane = int(rest[1:])
    return lane if 0 < lane < streams else None


def _use_connection(connections):
    """Send hello, then every later frame, over the data connections as binary frames.

    Returns what the frame reader reads from: the one connection, or a ``Striped`` stream
    over every lane.
    """
    global _SENDER
    import socket

    for connection in connections:
        try:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    if len(connections) == 1:
        carrier = connections[0]
    else:
        carrier = Striped(
            [connection.send for connection in connections],
            [connection.recv_into for connection in connections],
        )
    sender = Sender(carrier.send)
    old = _SENDER
    # Under the old lock, so no frame is split between the two transports.
    with old.lock:
        sender.frame(HELLO, 0, ("%d.%d" % sys.version_info[:2]).encode("ascii"))
        _SENDER = sender
    return carrier


def _read():
    """Read frames, answer light requests, and queue the rest for the main thread."""
    receiver = Receiver(sys.stdin.buffer.readinto)
    while True:
        try:
            event = receiver.next_event()
        except EOFError:
            break
        except Exception:
            traceback.print_exc()
            break
        if event is None:
            continue
        kind, stream, value = event
        if kind == SHUTDOWN:
            break
        if kind != REQUEST:
            continue
        if stream == DEVICE_STREAM:
            inbox = _DEVICE.get("inbox")
            if inbox is not None:
                inbox.put(value)
            value = None
            continue
        try:
            request = loads(value[0], value[1])
        except Exception:
            _reply(stream, {
                "ok": False,
                "error": "the request could not be decoded",
                "traceback": traceback.format_exc(),
            })
            continue
        value = None
        op = request.get("op") if isinstance(request, dict) else None
        if op in _LIGHT:
            _run(stream, request, False)
            continue
        if op is not None and op.startswith("data_"):
            # Served beside the call, because the blobs a running call is waiting for
            # cannot queue behind it. Spec "Streaming the rest while the call runs".
            _data_start()
            _DATA_JOBS.put((stream, request))
            continue
        if op == "listen":
            connection = _listen(stream, request)
            request = None
            if connection is not None:
                receiver = Receiver(connection.recv_into)
            continue
        _JOBS.put((stream, request))
        if op == "reexec":
            return
    inbox = _DEVICE.get("inbox")
    if inbox is not None:
        inbox.put(None)
    _JOBS.put(None)


def _serve():
    # The version travels in the hello frame, so the caller can refuse a worker whose
    # interpreter cannot run the bytecode it is about to send.
    _SENDER.frame(HELLO, 0, ("%d.%d" % sys.version_info[:2]).encode("ascii"))
    threading.Thread(target=_read, daemon=True).start()
    while True:
        job = _JOBS.get()
        if job is None:
            break
        stream, request = job
        job = None
        if request.get("op") == "reexec":
            _reexec(stream, request)
            continue
        _run(stream, request, True)
        request = None
    _release_output()


_serve()
'''


def source(*, text_frames: bool = False) -> str:
    """The worker program: the frame code, the transport flag, then the worker body.

    ``text_frames`` makes the worker write each frame as a base64 line, for a transport
    whose output is text, such as a Modal sandbox.
    """
    frames = Path(_wire.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return f"{frames}\n_LETIFY_TEXT_FRAMES = {bool(text_frames)!r}\n{_BODY}"


#: The worker program for a binary pipe.
SOURCE = source()

__all__ = ["BOOTSTRAP", "SOURCE", "source"]
