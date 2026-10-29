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
    "import sys,hashlib;"
    "b=sys.stdin.buffer;"
    "h=b.readline().split();"
    "n=int(h[0]);"
    "s=b.read(n);"
    # Spec "Channels": a source cut or shifted anywhere compiles as whatever construct the
    # damage landed inside, or runs and exits without serving. The count catches a cut and
    # the digest catches a shift, which keeps the count intact.
    "len(s)==n or sys.exit("
    "'the letify worker source arrived short: %d of %d bytes'%(len(s),n));"
    "d=hashlib.blake2b(s,digest_size=16).hexdigest();"
    "d==h[1].decode() or sys.exit("
    "'the letify worker source arrived corrupted: %d bytes, digest %s, expected %s'"
    "%(len(s),d,h[1].decode()));"
    "exec(compile(s.decode(),'letify-worker','exec'))"
)

def count_line(body: bytes) -> bytes:
    """The line the bootstrap stub reads before the source: its length and its digest.

    Spec "Channels": the digest is the first 16 bytes of blake2b in hexadecimal, from
    ``hashlib`` so the stub needs nothing installed.
    """
    import hashlib

    digest = hashlib.blake2b(body, digest_size=16).hexdigest()
    return b"%d %s\n" % (len(body), digest.encode("ascii"))


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

# Spec "Taking over a running call": what the second door and the call it serves share. It is
# defined here, before the output pumps start, because a pump consults it when a write fails.
#   run          id of the call being served, or None. Set from the call's start until its reply
#                has been delivered.
#   path/listener  the door's socket file and listening socket
#   stream       the stream id the call arrived on, on the channel it arrived on
#   live         whether the channel the call's reply goes to is believed open
#   ended_input  whether that channel closed while the call ran
#   moved        whether the call has been adopted away from the channel it arrived on
#   owner        the door connection that adopted the call, and owner_stream its request id
#   finished/outcome  the call's reply once the body has ended and before it is delivered
#   backlog      the tail of output made while no channel could take it
_TAKE = {
    "run": None, "path": None, "listener": None, "stream": None, "live": True,
    "ended_input": False, "moved": False, "owner": None, "owner_stream": None,
    "finished": False, "outcome": None, "primary": None, "new_owner": None,
    "backlog": {STDOUT: bytearray(), STDERR: bytearray()},
}
_TAKE_COND = threading.Condition()

#: The most output kept for a new owner while no channel could take it, per stream.
_BACKLOG_BYTES = 256 << 10

#: Seconds a connection to the door may stay silent before it has adopted a call.
_DOOR_IDLE_S = 30.0


#: Written to a captured descriptor to find out where its reader has got to. A pipe is
#: read in order, so every byte written before this has been framed once it is reached.
#: It is never forwarded. Spec "Worker output".
_DRAIN_TOKEN = b"\x1a\x1aLETIFY-DRAIN-7f3a9c4e\x1a\x1a"


def _held_back(data):
    """How many bytes at the end of ``data`` could be the start of a split token."""
    most = min(len(_DRAIN_TOKEN) - 1, len(data))
    for size in range(most, 0, -1):
        if _DRAIN_TOKEN.startswith(data[len(data) - size:]):
            return size
    return 0


class _Pump(threading.Thread):
    """Reads one captured descriptor and sends what it reads as output frames."""

    def __init__(self, fd, kind):
        threading.Thread.__init__(self, daemon=True)
        self.kind = kind
        self.busy = False
        self.fd = fd
        #: Bytes withheld because they may be the start of a token split across two reads.
        self.tail = b""
        self.reached = threading.Event()
        self.read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)

    def send(self, data):
        if not data:
            return
        self.busy = True
        try:
            _SENDER.frame(self.kind, 0, data)
        except OSError:
            # While a call runs, a channel that closed must not take the pump with it: the body
            # would die of a broken pipe on its next print. The output is kept for a new owner.
            if not _orphan_output(self.kind, data):
                raise
        finally:
            self.busy = False

    def run(self):
        while True:
            try:
                data = os.read(self.read_fd, _READ_SIZE)
            except OSError:
                break
            if not data:
                break
            data = self.tail + data
            self.tail = b""
            try:
                while True:
                    at = data.find(_DRAIN_TOKEN)
                    if at < 0:
                        break
                    self.send(data[:at])
                    data = data[at + len(_DRAIN_TOKEN):]
                    self.reached.set()
                keep = _held_back(data)
                if keep:
                    self.tail = data[len(data) - keep:]
                    data = data[: len(data) - keep]
                self.send(data)
            except OSError:
                break
        os.close(self.read_fd)


def _drain_output(timeout=2.0):
    """Wait until everything written to the captured descriptors has been framed.

    Spec "Worker output": a call's reply waits for the output its body made, so the
    caller's own next print comes after it. Waiting on the reading thread's state is not
    enough, because it has read bytes it has not yet framed; the token goes through the
    pipe behind those bytes instead. It gives up rather than lose a result.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    for pump in _PUMPS:
        pump.reached.clear()
        try:
            os.write(pump.fd, _DRAIN_TOKEN)
        except OSError:
            continue
        pump.reached.wait(timeout)


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
    _forget_run_control()
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
    hasher = _data_new_hasher()
    hasher.update(payload)
    return _data_finish(hasher)


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


def _send_outcome(sender, stream, outcome, last=False):
    """Send a reply on ``sender``. ``last`` marks the final piece of a streamed answer."""
    if last:
        outcome = dict(outcome, last=True)
    try:
        sender.message(REPLY, stream, outcome)
    except OSError:
        raise
    except BaseException:
        failure = {
            "ok": False,
            "error": "the return value could not be serialized",
            "traceback": traceback.format_exc(),
        }
        if last:
            failure["last"] = True
        sender.message(REPLY, stream, failure)


def _reply(stream, outcome):
    try:
        _send_outcome(_SENDER, stream, outcome)
    except OSError:
        _drop_record()
        os._exit(0)


def _op_call(request):
    take = request.pop("takeover", None)
    if take:
        _door_open(take)
    return _op_call_body(request)


def _op_call_body(request):
    data = request.pop("data", None)
    _letify_replica(request.pop("replica", None))
    modules_dir = request.pop("modules_dir", None)
    if modules_dir:
        modules_dir = os.path.expanduser(modules_dir)
        if modules_dir not in sys.path:
            sys.path.insert(0, modules_dir)
        existing = os.environ.get("PYTHONPATH", "")
        parts = [p for p in existing.split(os.pathsep) if p]
        if modules_dir not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([modules_dir, *parts])
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
        _DATA_FAILED.pop(data["dir"], None)
        _DATA_FAILED.pop("", None)
        _DATA_POLL_SEEN.pop(data["dir"], None)
        _DATA_POLL_SENT.pop(data["dir"], None)
        shutil.rmtree(data["dir"], ignore_errors=True)


# Write-back lists of returned calls, by call directory, until data_written takes them.
_DATA_WRITTEN = {}


def _letify_replica(replica):
    """Tell the letify in this runtime which replica this call is, or that it is none.

    Spec "Replicas": the body reads its rank through letify, which is installed here by
    the environment, so the numbers are set on the module rather than passed through the
    user's signature.
    """
    try:
        from letify.declare import replicas as _replicas
    except Exception:
        return
    if replica is None:
        _replicas._WHO.rank = _replicas._WHO.world_size = _replicas._WHO.rate_bps = None
        return
    _replicas._WHO.rank = replica.get("rank")
    _replicas._WHO.world_size = replica.get("world_size")
    _replicas._WHO.rate_bps = replica.get("rate_bps")


def _data_hash_file(path, like=None):
    """The file digest of a file on the runtime: blake3 when present, else blake2b.

    With ``like``, a digest the file may already be named by, the algorithm that digest
    names is used when it can be, so the result is comparable with it.
    """
    hasher = _data_hasher(like) if like is not None else None
    if hasher is None:
        hasher = _data_new_hasher()
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(1 << 22)
            if not piece:
                break
            hasher.update(piece)
    return _data_finish(hasher)


def _data_same_bytes(path, digest):
    """Whether the file still hashes to the digest it is the blob of.

    Spec "Runtime data cache budget": only the ambiguous case is hashed, a size that
    matches with a new timestamp, so the read is paid where a timestamp cannot decide.
    Never raises; a file that cannot be read is not the file the digest promised.
    """
    try:
        return _data_hash_file(path, digest) == digest
    except OSError:
        return False


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
            # The placed record ends with the cache stamp, which is not part of the comparison.
            if known is not None and known[1:4] == (info.st_ino, info.st_size, info.st_mtime_ns):
                continue
            digest = _data_hash_file(full, known[0] if known is not None else None)
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
        if (info.st_size, info.st_mtime_ns) == cached:
            continue
        if info.st_size == cached[0] and _data_same_bytes(source, digest):
            # Spec "Runtime data cache budget": the same bytes with a new timestamp, which
            # is what a file copied back looks like. The digest still holds, so it stays.
            continue
        # Spec "Runtime data cache budget": removing a file a call may still need is worth
        # a line, or it vanishes and only the next failure mentions it.
        print(
            "letify: data cache dropped %s, written through its link (%d bytes at %d, "
            "placed as %d at %d)"
            % (digest, info.st_size, info.st_mtime_ns, cached[0], cached[1]),
            file=sys.stderr,
            flush=True,
        )
        try:
            os.remove(source)
        except OSError:
            pass


def _op_data_written(request):
    """The write-back list of a returned call, removed as it is answered."""
    return {"ok": True, "value": _DATA_WRITTEN.pop(request["dir"], {})}


# What the last output poll of a running call saw, by call directory: the size and the
# modification time in nanoseconds per output file, and the digest already reported per file.
_DATA_POLL_SEEN = {}
_DATA_POLL_SENT = {}


def _op_data_output_poll(request):
    """The output files of a running call that are complete and not yet reported.

    Spec "Writing back", the background write-back. A file is complete when no process has
    it open any more and its size and its modification time in nanoseconds are what the
    poll before saw. The open check is what decides it, because a writer that pauses
    between two writes leaves the size and the time unchanged while the file is still half
    written. A file found complete is hashed and copied into the file blob cache, so the
    local process can fetch it with the same ``data_stream`` request the final write-back
    uses. The copy leaves the output file where it is, because the body may still write
    the next version of it.
    """
    import stat
    call_dir = request["dir"]
    before = _DATA_POLL_SEEN.get(call_dir) or {}
    reported = _DATA_POLL_SENT.setdefault(call_dir, {})
    held = _data_open_files()
    now = {}
    written = {}
    for output in request["outputs"]:
        files = []
        for rel, full in _data_output_files(output):
            try:
                info = os.lstat(full)
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            stamp = (info.st_size, info.st_mtime_ns)
            now[full] = stamp
            if held is not None and (info.st_dev, info.st_ino) in held:
                # Still open somewhere, so the writer is not done with it.
                continue
            if before.get(full) != stamp:
                # Seen once only, so it may still be growing: it waits for the next poll.
                continue
            if reported.get(full) == stamp:
                continue
            try:
                digest = _data_hash_file(full)
                _data_copy_into_cache(request["blobs"], full, digest)
            except OSError:
                continue
            reported[full] = stamp
            files.append([rel, digest, info.st_size])
        if files:
            written[output] = files
    _DATA_POLL_SEEN[call_dir] = now
    return {"ok": True, "value": written}


def _data_open_files():
    """The device and inode of every file any process on this machine holds open.

    Spec "Writing back". Read from ``/proc``, which is how a close is observed without
    hooking it: an output file with no descriptor left pointing at it has been closed by
    the body and by anything the body started. None where ``/proc`` is not there, and the
    poll then rests on the size and the time alone.
    """
    if not os.path.isdir("/proc/self/fd"):
        return None
    held = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        directory = "/proc/%s/fd" % entry
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            try:
                info = os.stat(os.path.join(directory, name))
            except OSError:
                continue
            held.add((info.st_dev, info.st_ino))
    return held


def _data_output_files(output):
    """Each regular file under an output's runtime path, as relative path and full path."""
    skipped = (".git", ".venv", "__pycache__")
    if os.path.isfile(output) and not os.path.islink(output):
        return [("", output)]
    if not (os.path.isdir(output) and not os.path.islink(output)):
        return []
    found = []
    for directory, names, entries in os.walk(output, followlinks=False):
        names[:] = [name for name in names if name not in skipped]
        for name in entries:
            if _DATA_PARTIAL in name:
                continue
            full = os.path.join(directory, name)
            found.append((os.path.relpath(full, output).replace(os.sep, "/"), full))
    found.sort()
    return found


def _data_copy_into_cache(blobs, path, digest):
    """Copy a complete output file into the file blob cache, unless its digest is held."""
    import shutil
    final = _data_file(blobs, digest)
    try:
        if os.stat(final).st_size == os.stat(path).st_size:
            return
    except OSError:
        pass
    os.makedirs(os.path.dirname(final), exist_ok=True)
    partial = "%s.partial.%d" % (final, os.getpid())
    shutil.copyfile(path, partial)
    try:
        os.chmod(partial, 0o444)
    except OSError:
        pass
    os.replace(partial, final)


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


def _data_hasher(digest):
    """A hasher of the algorithm ``digest`` names, or None where it cannot be computed here.

    A digest with the ``b2-`` prefix is blake2b and one without is blake3, so a runtime
    without blake3 cannot check a blake3 digest and trusts the sender.
    """
    if digest.startswith("b2-"):
        return hashlib.blake2b(digest_size=16)
    try:
        import blake3
    except ImportError:
        return None
    return blake3.blake3(max_threads=blake3.blake3.AUTO)


def _data_new_hasher():
    """The hasher this runtime names new content with: blake3 when present, else blake2b."""
    hasher = _data_hasher("")
    return hasher if hasher is not None else hashlib.blake2b(digest_size=16)


def _data_finish(hasher):
    """The digest a hasher has accumulated, with the prefix a blake2b digest carries."""
    if hasher.name == "blake2b":
        return "b2-" + hasher.hexdigest()
    return hasher.hexdigest(length=16)


def _data_commit(partial, final, digest, hasher):
    """Rename a received file into the cache once its digest matches, read-only."""
    if hasher is not None:
        found = _data_finish(hasher)
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


def _digest_of(path):
    """The digest a blob path holds, which is its shard directory and its name joined."""
    return os.path.basename(path)


def _data_digests_in_flight():
    """Every digest a running call lists, so the prune does not take one out from under it.

    Spec "Runtime data cache budget". Read under the lock the data thread writes the maps
    with, so a call registering while this reads cannot change one mid iteration.
    """
    wanted = set()
    with _DATA_READY:
        for state in list(_DATA_CALLS.values()):
            wanted.update(state["by_digest"])
    return wanted


def _op_data_evict(request):
    """Remove least recently used blobs no call links until the cache is within budget.

    Spec "Runtime data cache budget": a digest a running call lists is skipped whatever
    its link count and age, because the call was told the runtime held it and so did not
    send it. ``recent`` is the window a blob no call is using is protected for, carried on
    the request beside the budget.
    """
    started = time.monotonic()
    root = os.path.expanduser(request["dir"])
    files, total = _data_scan(root)
    budget = _data_budget(root, total, request.get("budget"))
    recent = float(request.get("recent", _DATA_RECENT_S))
    wanted = _data_digests_in_flight()
    removed = freed = 0
    if total > budget:
        now = time.time()
        files.sort()
        for used, path, size, links in files:
            if total <= budget:
                break
            if links > 1 or now - used < recent or _digest_of(path) in wanted:
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
        _DATA_OPEN[digest] = (open(partial, "wb"), _data_hasher(digest))
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


def _parallel_partial(root, digest):
    """Where parallel senders write. One name per digest, not per process or thread."""
    return "%s%s.parallel" % (_data_file(root, digest), _DATA_PARTIAL)


def _op_data_drop(request):
    """Remove the partials of a blob nobody is going to finish.

    Spec "Data log line": a sender cancelled when the call ended leaves a partial that no
    later request completes, because the digest it would be sealed under is written only
    once every piece has arrived. A single put's partial is invisible to the budget scan,
    so without this it stays for the life of the runtime.
    """
    digest = request["digest"]
    final = _data_file(request["dir"], digest)
    open_handle = _DATA_OPEN.pop(digest, None)
    if open_handle is not None:
        try:
            open_handle[0].close()
        except OSError:
            pass
    removed = 0
    directory = os.path.dirname(final)
    name = os.path.basename(final)
    try:
        entries = os.listdir(directory)
    except OSError:
        entries = []
    for entry in entries:
        if entry == name or not entry.startswith(name + "."):
            continue
        try:
            os.remove(os.path.join(directory, entry))
            removed += 1
        except OSError:
            pass
    return {"ok": True, "value": removed}


def _op_data_begin(request):
    """Make the file parallel senders will fill, of the size it will end up.

    Spec "Several connections at once". A placement starts here rather than at the write
    of offset zero, because with several connections offset zero is just another range and
    may arrive last: truncating on it would throw away what the others had written.
    """
    partial = _parallel_partial(request["dir"], request["digest"])
    os.makedirs(os.path.dirname(partial), exist_ok=True)
    with open(partial, "wb") as handle:
        handle.truncate(int(request["size"]))
    return {"ok": True, "value": None}


def _op_data_put_at(request):
    """Write one piece of a file blob at its offset, for several senders at once.

    No hashing: a hash over a stream cannot be computed out of order, so ``data_seal``
    does it by reading the finished file.
    """
    partial = _parallel_partial(request["dir"], request["digest"])
    chunk = request.pop("chunk")
    handle = open(partial, "r+b")
    try:
        handle.seek(int(request["offset"]))
        handle.write(chunk)
    finally:
        handle.close()
    chunk = None
    return {"ok": True, "value": None}


def _op_data_seal(request):
    """Hash a file assembled by parallel senders, then commit it as ``data_put`` does."""
    digest = request["digest"]
    root = request["dir"]
    partial = _parallel_partial(root, digest)
    final = _data_file(root, digest)
    size = request.get("size")
    try:
        found = os.path.getsize(partial)
    except OSError as exc:
        raise ValueError("no pieces of file blob %s arrived: %s" % (digest, exc))
    if size is not None and found != int(size):
        raise ValueError(
            "file blob %s is short: %d of %d bytes arrived" % (digest, found, int(size))
        )
    hasher = _data_hasher(digest)
    if hasher is not None:
        with open(partial, "rb") as handle:
            while True:
                piece = handle.read(1 << 22)
                if not piece:
                    break
                hasher.update(piece)
        got = _data_finish(hasher)
        if got != digest:
            os.remove(partial)
            raise ValueError("file blob %s arrived with digest %s" % (digest, got))
    try:
        os.chmod(partial, 0o444)
    except OSError:
        pass
    os.replace(partial, final)
    _data_arrived(root, digest)
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
                hasher = _data_hasher(digest)
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

#: The ``os.open`` flags that mean the body will write. Spec "Copy on write".
_DATA_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_TRUNC


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
        "converting": set(),
        "linked": 0,
        "order": {},
        "arrived": 0,
        # Spec "When a blob does not arrive": set when the local sender stops, so a
        # reader fails at once with its reason rather than waiting out the timeout.
        "failed": _DATA_FAILED.get(data["dir"]) or _DATA_FAILED.get(""),
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
    manifest = data.get("manifest") or {}
    for path, entry in manifest.items():
        digest, size = entry[0], int(entry[1])
        state["pending"][path] = (digest, size)
        state["by_digest"].setdefault(digest, []).append(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not manifest:
        # No manifest travelled, so every link the call carries is already in the cache.
        for path, digest in data.get("links", ()):
            state["pending"][path] = (digest, 0)
            state["by_digest"].setdefault(digest, []).append(path)
            # A single file input names no directory of its own in the request.
            os.makedirs(os.path.dirname(path), exist_ok=True)
    with _DATA_READY:
        _DATA_CALLS[state["dir"]] = state
        for path in list(state["pending"]):
            _data_place(state, path)
        if state["pending"] or state["linked"]:
            # A call still waiting for bytes needs the patch to answer from the manifest,
            # and a call that placed a link needs it for copy on write. Spec "Copy on write".
            _data_install_patch()
        if state["pending"] or state["linked"]:
            # A call still waiting for bytes needs the manifest on disk for the listings,
            # and a call that placed a link needs it so a process the body starts can
            # convert that link before writing through it: the patch a fresh process
            # installs reads the layout from the manifest and installs nothing without it.
            # Spec "Copy on write". Inside the lock: the data thread moves a path out of pending and
            # into placed as each blob completes, and the manifest is written from both maps.
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
    # A link for every entry, a directory's included: the body's first open for writing
    # turns it into a private copy. Spec "Materializing and the rewritten path".
    linked = False
    try:
        if os.path.exists(path) and os.stat(path).st_ino == os.stat(source).st_ino:
            linked = True
    except OSError:
        pass
    if not linked:
        # Linked at a name of its own and renamed over the path, never unlinked first.
        # Unlinking leaves the final name absent for as long as the link takes, and when
        # the link then fails the file that was there is lost: a body that listed the
        # directory is left with a name it cannot stat. Spec "What the body sees before a
        # file arrives". _DATA_PARTIAL names are filtered from every patched listing.
        staged = "%s%slink%d.%d" % (
            path, _DATA_PARTIAL, os.getpid(), threading.get_ident())
        try:
            os.link(source, staged)
            os.replace(staged, path)
            linked = True
        except OSError:
            try:
                os.remove(staged)
            except OSError:
                pass
    if not linked:
        # Copied to a name of its own and renamed into place, so a reader never stats a
        # file that is half written. Spec "What the body sees before a file arrives".
        partial = "%s%s%d.%d" % (
            path, _DATA_PARTIAL, os.getpid(), threading.get_ident())
        try:
            shutil.copyfile(source, partial)
            os.replace(partial, path)
        except OSError:
            try:
                os.remove(partial)
            except OSError:
                pass
            return False
    try:
        info = os.stat(path)
    except OSError:
        # Another thread is placing this same path, so the name is between its link and
        # its rename. Leaving the path pending and reporting nothing placed is right: the
        # other thread finishes the placement. Raising here instead ends the whole call,
        # which is how a race between the opening sweep and the data thread surfaced as
        # `FileNotFoundError` on a file the body had just listed.
        return False
    cached = (info.st_size, info.st_mtime_ns) if linked else None
    state["placed"][path] = (digest, info.st_ino, info.st_size, info.st_mtime_ns, cached)
    state["linked"] += linked
    del state["pending"][path]
    return True


def _data_private(state, path):
    """Replace the link at ``path`` with a private copy before the body writes to it.

    Spec "Copy on write". The copy is made beside the link and renamed over it, so no
    reader sees it half written, and the cache file is never opened for writing.
    """
    with _DATA_READY:
        while path in state["converting"]:
            _DATA_READY.wait(0.5)
        known = state["placed"].get(path)
        if known is None or known[4] is None:
            return
        state["converting"].add(path)
    import shutil
    partial = "%s%s%d" % (path, _DATA_PARTIAL, os.getpid())
    converted = False
    _DATA_PLACING.on = True
    try:
        shutil.copyfile(path, partial)
        os.replace(partial, path)
        converted = True
    except OSError:
        try:
            os.remove(partial)
        except OSError:
            pass
    finally:
        _DATA_PLACING.on = False
        with _DATA_READY:
            state["converting"].discard(path)
            if converted:
                # No inode, size or time to compare against: the write-back hashes it,
                # because the body's first write may land inside the same clock tick.
                state["placed"][path] = (known[0], None, None, None, None)
            _DATA_READY.notify_all()


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


#: Reasons a sender stopped, by call directory, for a call that registers afterwards.
_DATA_FAILED = {}


def _op_data_failed(request):
    """The local sender stopped, so the blobs it had left will never arrive.

    Spec "When a blob does not arrive": every reader waiting on one of them is woken with
    the reason, and a reader that arrives later fails on it at once, because waiting out
    the per blob timeout costs that timeout once for every blob still queued.
    """
    reason = request.get("reason") or "the local process stopped sending"
    where = request.get("dir") or ""
    with _DATA_READY:
        # Remembered by directory as well as set on the live call, because the sender can
        # stop before the call request has been read, and the call then registers after
        # this and would otherwise inherit nothing. Spec "When a blob does not arrive".
        _DATA_FAILED[where] = reason
        for state in _DATA_CALLS.values():
            if where and state["dir"] != where:
                continue
            state["failed"] = reason
        _DATA_READY.notify_all()
    return {"ok": True, "value": True}


def _data_received(blobs, digest):
    """Bytes of this blob that are on the runtime's disk, however they are arriving.

    Spec "When a blob does not arrive": the wait counts time without progress, and this is
    what progress means. Reads the partial a single put writes and the one a parallel put
    fills, so neither shape looks stalled while it grows.
    """
    total = 0
    final = _data_file(blobs, digest)
    directory = os.path.dirname(final)
    name = os.path.basename(final)
    try:
        entries = os.listdir(directory)
    except OSError:
        return 0
    for entry in entries:
        if entry != name and not entry.startswith(name + "."):
            continue
        try:
            total = max(total, os.stat(os.path.join(directory, entry)).st_size)
        except OSError:
            continue
    return total


def _data_want(digests, blocking=False):
    """Ask the local process for these blobs, in this order. Nothing replies to it."""
    if not digests:
        return
    message = {"op": "data_want", "digests": list(digests), "blocking": blocking}
    try:
        _SENDER.message(REQUEST, DATA_STREAM, message)
    except OSError:
        pass


#: Guards ``_REDUCE_TABLES``, and wakes an ``all_reduce`` call waiting on an answer.
_REDUCE_LOCK = threading.Condition()

#: Peer tables and failures the client sent back, keyed by round number, until the
#: waiting ``all_reduce`` call takes them.
_REDUCE_TABLES = {}


def _reduce_join(message):
    """Ask the client to broker an all_reduce round. Nothing replies to this request.

    Mirrors ``_data_want``: it is a one-way request on a reserved stream, and the
    client's answer comes back separately as an ordinary request-reply, op
    "reduce_peers", below.
    """
    try:
        _SENDER.message(REQUEST, REDUCE_STREAM, message)
    except OSError:
        pass


def _op_reduce_peers(request):
    """The client's answer to a join: the peer table for a round, or a failure.

    Spec "Reducing between replicas": the body's ``all_reduce`` is blocked waiting for
    this, so it is answered by the frame reader at once rather than queued behind the
    call, which is why it is in ``_LIGHT`` below.
    """
    with _REDUCE_LOCK:
        _REDUCE_TABLES[request["round"]] = request
        _REDUCE_LOCK.notify_all()
    return {"ok": True, "value": None}


#: Guards ``_DRIVE_ANSWERS``, and wakes a ``stash`` or ``unstash`` waiting on an answer.
_DRIVE_LOCK = threading.Condition()

#: The client's answers to a Drive request, keyed by request id, until the waiting
#: ``stash`` or ``unstash`` takes them.
_DRIVE_ANSWERS = {}

# A process the body starts inherits this and lacks ``_drive_ask``, which is how
# ``stash`` and ``unstash`` tell it is a child of the body and refuse. Spec "Named handoffs
# on the Drive store".
os.environ["LETIFY_WORKER_PID"] = str(os.getpid())


def _drive_ask(message):
    """Ask the client for a Drive grant. Nothing replies to this request.

    Mirrors ``_reduce_join``: a one-way request on a reserved stream, answered separately
    as an ordinary request-reply, op "drive_grant", below.
    """
    try:
        _SENDER.message(REQUEST, DRIVE_STREAM, message)
    except OSError as exc:
        # The stash or unstash then waits out its grant timeout; this line names why.
        print(
            f"letify: drive request {message.get('kind')} {message.get('name')} "
            f"was not sent: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _op_drive_grant(request):
    """The client's answer to a Drive request: a grant, a reply that nothing is there, or a failure.

    The body is blocked waiting for this, so it is answered by the frame reader at once
    rather than queued behind the call, which is why it is in ``_LIGHT`` below.
    """
    with _DRIVE_LOCK:
        _DRIVE_ANSWERS[request["id"]] = request
        _DRIVE_LOCK.notify_all()
    return {"ok": True, "value": None}


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
        if state["failed"]:
            raise _DataFailure(
                "%s was never sent: %s" % (os.path.basename(path), state["failed"])
            )
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
    received = _data_received(state["blobs"], digest)
    try:
        with _DATA_READY:
            while path in state["pending"] and state["dir"] in _DATA_CALLS:
                if state["failed"]:
                    raise _DataFailure(
                        "%s was never sent: %s"
                        % (os.path.basename(path), state["failed"])
                    )
                # Spec "When a blob does not arrive": the wait counts time without
                # progress, because a blob needing longer than the timeout at the link's
                # rate would otherwise fail however healthy its transfer was.
                landed = _data_received(state["blobs"], digest)
                if landed > received:
                    received = landed
                    deadline = time.monotonic() + state["wait"]
                left = deadline - time.monotonic()
                if left <= 0:
                    raise _DataFailure(
                        "the file blob %s of %s did not arrive within %.0f s, "
                        "%.1f MiB of %.1f MiB arrived"
                        % (digest, os.path.basename(path), state["wait"],
                           received / 1048576.0, size / 1048576.0)
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

    def call_of(path):
        """The running call a path belongs to and the absolute path, or None."""
        if not _DATA_CALLS or getattr(_DATA_PLACING, "on", False):
            return None
        text = full(path)
        if text is None:
            return None
        state = _data_state_for(text)
        return (state, text) if state is not None else None

    def pending_of(path):
        found = call_of(path)
        if found is None:
            return None
        entry = _data_pending_entry(found[0], found[1])
        return (found[0], found[1], entry) if entry is not None else None

    def wait_for(path):
        found = pending_of(path)
        if found is not None:
            _data_wait(found[1])

    def ready_for(path, writes):
        """Wait for the file, then give the body its own copy when it means to write."""
        found = call_of(path)
        if found is None:
            return
        state, text = found
        if _data_pending_entry(state, text) is not None:
            _data_wait(text)
        if writes:
            _data_private(state, text)

    def mode_writes(mode):
        try:
            return any(letter in mode for letter in "wax+")
        except TypeError:
            return False

    def opened(file, *args, **kwargs):
        mode = kwargs["mode"] if "mode" in kwargs else (args[0] if args else "r")
        ready_for(file, mode_writes(mode))
        return real["open"](file, *args, **kwargs)

    def io_opened(file, *args, **kwargs):
        mode = kwargs["mode"] if "mode" in kwargs else (args[0] if args else "r")
        ready_for(file, mode_writes(mode))
        return real["io_open"](file, *args, **kwargs)

    def os_opened(path, *args, **kwargs):
        flags = kwargs["flags"] if "flags" in kwargs else (args[0] if args else 0)
        try:
            writes = bool(flags & _DATA_WRITE_FLAGS)
        except TypeError:
            writes = False
        ready_for(path, writes)
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
        # Snapshot the pending map under the lock the data thread writes it with, then
        # iterate the snapshot. The data thread deletes a path from the live map as each
        # blob completes, so iterating the live map would raise "dictionary changed size
        # during iteration" mid listing. Spec "What the body sees before a file arrives".
        with _DATA_READY:
            pending = list(state["pending"].items())
        found = []
        for path, (_digest, size) in pending:
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                found.append((path[len(prefix):], size))
        return found

    def listed(path="."):
        # Snapshot the pending names before reading the real directory, as scanned does.
        # The data thread places a file and then drops it from the pending map, both under
        # the lock, so a file placed between the two steps is in the snapshot or on disk.
        # Read the other way round it is in neither. Spec "What the body sees before a
        # file arrives".
        pending = names_of(path)
        entries = [name for name in real["listdir"](path) if _DATA_PARTIAL not in name]
        for name, _size in pending:
            if name not in entries:
                entries.append(name)
        return entries

    def scanned(path="."):
        pending = names_of(path)
        found = real["scandir"](path)
        if not pending and call_of(path) is None:
            return found
        # Inside a running call the copy of a conversion is never listed either.
        entries = [entry for entry in found if _DATA_PARTIAL not in entry.name]
        held = {entry.name for entry in entries}
        prefix = full(path).rstrip("/") + "/"
        for name, size in pending:
            if name not in held:
                entries.append(_PendingEntry(name, prefix + name, size))
        return _Scandir(entries, found)

    def missing(path, found, real_name, kwargs):
        """Answer a stat that found nothing, from the manifest or by looking again.

        The real stat runs first, so the data thread can place the file and drop it from
        the pending map in between, leaving the path in neither: not on disk when it was
        looked for, not pending when the manifest was asked. Looking again settles it,
        because placement puts the file on disk before it clears pending, so the second
        look finds it. A path that is genuinely absent raises as it should. Spec "What the
        body sees before a file arrives"; the listing states the same rule and takes its
        snapshot first instead, which a stat cannot afford on a data loader's hot path.
        """
        if found is not None:
            return _data_fake_stat(found[2][1])
        return real[real_name](path, **kwargs)

    def stated(path, **kwargs):
        try:
            return real["stat"](path, **kwargs)
        except FileNotFoundError:
            return missing(path, pending_of(path), "stat", kwargs)

    def lstated(path, **kwargs):
        try:
            return real["lstat"](path, **kwargs)
        except FileNotFoundError:
            return missing(path, pending_of(path), "lstat", kwargs)

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
    same patch from ``sitecustomize.py`` and reads the layout from here, because it has
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
        # sitecustomize, not a .pth: site.py runs a .pth only in a directory it adds
        # through addsitedir, and a PYTHONPATH entry is not one, so a .pth there never
        # runs. Spec "What the body sees before a file arrives".
        with open(os.path.join(directory, "sitecustomize.py"), "w") as handle:
            handle.write(_PENDING_MODULE)
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
import shutil
import sys
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
    real_os_open = os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_TRUNC

    # Whether the file exists, asked through the real stat. os.path.exists goes through
    # os.stat, and os.stat is what this patch replaces, so asking it here would call the
    # patch again and recurse without bottom.
    def there(text):
        try:
            real_stat(text)
        except OSError:
            return False
        return True

    def missing(path):
        try:
            text = os.path.abspath(os.fspath(path))
        except TypeError:
            return None
        if not text.startswith(call_dir) or there(text):
            return None
        return text if text in files else None

    def wait_for(path):
        text = missing(path)
        if text is None:
            return
        deadline = time.monotonic() + wait
        while not there(text):
            if time.monotonic() >= deadline:
                raise RuntimeError("letify: %s did not arrive within %.0f s" % (text, wait))
            time.sleep(0.02)

    # A placed file is a hard link to the cache blob, so a write through it writes into
    # the cache. The worker converts the link first and this has to as well; it cannot read
    # the worker's placed map, so a link count above one under the call directory is what
    # it goes on. Spec "What the body sees before a file arrives".
    def convert(path):
        try:
            text = os.path.abspath(os.fspath(path))
        except TypeError:
            return
        if not text.startswith(call_dir):
            return
        try:
            if real_stat(text).st_nlink < 2:
                return
        except OSError:
            return
        partial = '%s.letify-placing.%d' % (text, os.getpid())
        try:
            shutil.copyfile(text, partial)
            os.replace(partial, text)
        except OSError:
            try:
                os.remove(partial)
            except OSError:
                pass

    def writes(mode):
        return any(letter in mode for letter in ('w', 'a', 'x', '+'))

    def ready(path, writing):
        wait_for(path)
        if writing:
            convert(path)

    def opened(file, *args, **kwargs):
        mode = kwargs['mode'] if 'mode' in kwargs else (args[0] if args else 'r')
        ready(file, writes(mode) if isinstance(mode, str) else False)
        return real_open(file, *args, **kwargs)

    def io_opened(file, *args, **kwargs):
        mode = kwargs['mode'] if 'mode' in kwargs else (args[0] if args else 'r')
        ready(file, writes(mode) if isinstance(mode, str) else False)
        return real_io_open(file, *args, **kwargs)

    def os_opened(path, *args, **kwargs):
        flags = kwargs['flags'] if 'flags' in kwargs else (args[0] if args else 0)
        try:
            writing = bool(flags & write_flags)
        except TypeError:
            writing = False
        ready(path, writing)
        return real_os_open(path, *args, **kwargs)

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
    os.open = os_opened
    os.listdir = listed
    os.stat = stated
    os.scandir = real_scandir


# letify's directory goes first on PYTHONPATH, so it shadows a sitecustomize the user
# installed. That one still has to run.
def _chain():
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry:
            continue
        try:
            if os.path.abspath(entry) == here:
                continue
            candidate = os.path.join(entry, 'sitecustomize.py')
            if not os.path.isfile(candidate):
                continue
            spec = importlib.util.spec_from_file_location('letify_other_sitecustomize', candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            pass
        return


_install()
_chain()
"""


#: The prefetch depth the lazy import hook wraps torch.utils.data with, set per call.
_OBSERVE_PREFETCH = None


class _ObserveFinder:
    """A one-shot import hook that wraps torch.utils.data when the body imports it.

    The worker never imports PyTorch to install the wrappers. A call whose body never
    touches PyTorch would otherwise pay that import for nothing, and the import would sit
    on the call's path, ahead of the body and the background sender. So the wrappers are
    installed by hooking the body's own import instead. The finder delegates to the real
    finders for the module, then wraps it once its own loader has run.
    """

    def find_spec(self, name, path=None, target=None):
        if name != "torch.utils.data":
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(name, path, target)
            except ImportError:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        real_exec = getattr(spec.loader, "exec_module", None)
        if real_exec is None:
            return None
        prefetch = _OBSERVE_PREFETCH

        def exec_module(module):
            real_exec(module)
            try:
                _data_wrap(module, prefetch)
            except BaseException:
                pass

        spec.loader.exec_module = exec_module
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        return spec


def _data_observe(state):
    """Install the read-order wrappers of spec "Observing the read order on the runtime".

    The worker does not import PyTorch. It wraps torch.utils.data at once when the body has
    already imported it, and otherwise hooks the body's own import, so the wrappers arrive
    when PyTorch does and never before."""
    global _OBSERVE_PREFETCH
    if not state["observe"]:
        return
    _OBSERVE_PREFETCH = state["prefetch"]
    module = sys.modules.get("torch.utils.data")
    if module is not None:
        _data_wrap(module, state["prefetch"])
        return
    if not any(isinstance(finder, _ObserveFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ObserveFinder())


def _data_wrap(data_module, prefetch):
    """Wrap DataLoader and Dataset once per process. Every wrapper calls the original and
    only reports what it saw to the send order queue."""
    if getattr(data_module, "_letify_observed", False):
        return
    data_module._letify_observed = True

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
    # Spec "Controlling a running call": from here until the body ends, an interrupt may be
    # delivered and exit hooks may be registered.
    _run_enter()
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
    finally:
        _run_leave()
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
    with tarfile.open(path, "r:*") as archive:
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


class _HashedFile:
    """A write-only file that hashes every byte it passes on, so no second read is needed."""

    def __init__(self, handle, hasher):
        self._handle = handle
        self._hasher = hasher
        self.size = 0

    def write(self, data):
        self._handle.write(data)
        self._hasher.update(data)
        self.size += len(data)
        return len(data)

    def flush(self):
        self._handle.flush()

    def tell(self):
        return self.size


def _op_pack_blob(request):
    """Pack a directory into the blob cache as one committed file, hashing as it writes.

    Same archive and member rules as ``pack_dir``, but the archive never exists as one
    bytes object: it is written to a partial file in pieces, and the caller reads it back
    with ``data_stream``. The reply names the directory it used, with ``~`` expanded.
    The archive is gzip unless the request sets ``gzip`` to false, which writes a plain
    tar for a store that moves the archive at a disk's speed.
    """
    root = os.path.expanduser(request["path"])
    directory = os.path.expanduser(request["dir"])
    os.makedirs(directory, exist_ok=True)
    partial = os.path.join(directory, ".pack.%d.partial" % os.getpid())
    hasher = _data_new_hasher()
    try:
        with open(partial, "wb") as handle:
            hashed = _HashedFile(handle, hasher)
            mode = "w:gz" if request.get("gzip", True) else "w"
            with tarfile.open(fileobj=hashed, mode=mode) as archive:
                archive.add(root, arcname=os.path.basename(root.rstrip("/")), recursive=True)
        digest = _data_finish(hasher)
        final = _data_file(directory, digest)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        os.replace(partial, final)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return {"ok": True, "value": {"digest": digest, "size": hashed.size, "dir": directory}}


def _op_push_blob(request):
    """Upload a blob from the blob cache to a resumable upload session, in pieces.

    The session URL authorizes its own upload, so the request carries no access token.
    Every piece but the last is a multiple of 256 KiB, as Drive requires, and is answered
    308 until the last, which is answered with the new file's metadata. The reply names
    the file id.
    """
    import json
    import urllib.error
    import urllib.request

    url = request.pop("url")
    headers = request.pop("headers", None) or {}
    cancel = request.pop("cancel", None)
    path = _data_file(os.path.expanduser(request["dir"]), request["digest"])
    size = os.path.getsize(path)
    piece = int(request.get("piece") or (64 << 20))
    remote_id = None
    with open(path, "rb") as handle:
        offset = 0
        while offset < size:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("the upload was abandoned")
            block = handle.read(piece)
            end = offset + len(block)
            span = "bytes %d-%d/%d" % (offset, end - 1, size)
            send = urllib.request.Request(url, data=block, method="PUT", headers=headers)
            send.add_header("Content-Length", str(len(block)))
            send.add_header("Content-Range", span)
            try:
                with urllib.request.urlopen(send, timeout=3600) as response:
                    remote_id = json.loads(response.read())["id"]
            except urllib.error.HTTPError as exc:
                if exc.code != 308:
                    raise RuntimeError(
                        "the upload returned %d: %s"
                        % (exc.code, exc.read()[:300].decode(errors="replace"))
                    )
            offset = end
    if remote_id is None:
        raise RuntimeError("the upload session never answered with a file")
    return {"ok": True, "value": {"id": remote_id, "size": size}}


#: The one environment pack running beside the session's calls, or None. Spec "Project
#: data / environment archive": the pack is a background job on the worker.
_PACK_JOB = {}


def _pack_push_run(job, request):
    """Pack a directory, wait for the upload target, upload it, and remove the staging dir.

    Runs on its own thread, so the worker's main thread keeps answering calls. The state
    goes packing, packed, uploading, then done or failed. The staging directory is gone
    before done or failed is reported.
    """
    import shutil

    began = time.monotonic()
    directory = os.path.expanduser(request["dir"])
    result = None
    try:
        packed = _op_pack_blob(
            {"path": request["path"], "dir": request["dir"], "gzip": False}
        )["value"]
        job["digest"], job["size"] = packed["digest"], packed["size"]
        job["state"] = "packed"
        job["ready"].wait()
        target = job["target"]
        if target is None:
            raise RuntimeError("the pack was abandoned")
        job["state"] = "uploading"
        sent = _op_push_blob(
            {
                "dir": packed["dir"],
                "digest": packed["digest"],
                "url": target["url"],
                "headers": target.get("headers") or {},
                "piece": target.get("piece"),
                "cancel": job["cancel"],
            }
        )["value"]
        result = {"state": "done", "id": sent["id"], "size": sent["size"]}
    except BaseException as exc:
        result = {"state": "failed", "reason": str(exc) or type(exc).__name__}
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    result["seconds"] = time.monotonic() - began
    job.update(result)


def _op_pack_push_start(request):
    """Start packing a directory on a thread of its own and return at once."""
    import threading

    current = _PACK_JOB.get("job")
    if current is not None and current["state"] in ("packing", "packed", "uploading"):
        return {"ok": False, "error": "a pack is already running", "traceback": ""}
    job = {
        "state": "packing",
        "target": None,
        "ready": threading.Event(),
        "cancel": threading.Event(),
    }
    _PACK_JOB["job"] = job
    threading.Thread(target=_pack_push_run, args=(job, request), daemon=True).start()
    return {"ok": True, "value": None}


def _op_pack_push_target(request):
    """Hand the running pack its upload session, or abandon it when ``url`` is None."""
    job = _PACK_JOB.get("job")
    if job is None:
        return {"ok": False, "error": "no pack is running", "traceback": ""}
    if request.get("url") is None:
        job["cancel"].set()
        job["target"] = None
    else:
        job["target"] = {
            "url": request["url"],
            "headers": request.get("headers") or {},
            "piece": request.get("piece"),
        }
    job["ready"].set()
    return {"ok": True, "value": None}


def _op_pack_push_state(request):
    """Report the pack: its state, and the digest, size, file id, seconds or reason it has."""
    job = _PACK_JOB.get("job")
    if job is None:
        return {"ok": True, "value": {"state": "none"}}
    keys = ("state", "digest", "size", "id", "seconds", "reason")
    return {"ok": True, "value": {key: job[key] for key in keys if key in job}}


def _op_place_blob(request):
    """Move a received file blob to its destination path, and unpack it when asked.

    The sender wrote the blob into a staging directory with ``data_put``. The staging
    directory is removed once the file has moved.
    """
    import shutil

    staging = request["dir"]
    source = _data_file(staging, request["digest"])
    path = request["path"]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    os.replace(source, path)
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass
    shutil.rmtree(staging, ignore_errors=True)
    if request.get("unpack"):
        _unpack(path, request.get("target") or os.path.dirname(path), request.get("links"))
    return {"ok": True, "value": {"path": path, "size": os.path.getsize(path)}}


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
    _record_update(lease_deadline=state["deadline"])

    def watch():
        while True:
            # Wakes at the deadline rather than on a fixed beat, so a short grace is honoured.
            time.sleep(max(0.05, min(15.0, state.get("deadline", 0) - time.time() + 0.05)))
            if time.time() > state.get("deadline", 0):
                _exit_with_hooks(0)

    if not state.get("armed"):
        state["armed"] = True
        threading.Thread(target=watch, daemon=True).start()
    return {"ok": True, "value": state["deadline"]}


# -- the worker's record on its host ---------------------------------------------------
# Spec "Workers on a shell host": a file under the workspace root saying who this worker is,
# written when the client registers it and removed when the worker ends.

_RECORD = {"path": None, "data": None, "lock": threading.Lock()}


def _proc_start():
    """This process's start time in clock ticks since boot, or None where /proc is missing."""
    try:
        with open("/proc/self/stat") as handle:
            return int(handle.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _write_record():
    """Replace the record file whole, so a reader never sees half of it."""
    import json
    path, data = _RECORD["path"], _RECORD["data"]
    if path is None:
        return
    partial = "%s.%d.part" % (path, os.getpid())
    try:
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(data, handle)
        os.replace(partial, path)
    except OSError as exc:
        print("letify: the worker record was not written: %s" % exc, file=sys.stderr)


def _record_update(**fields):
    """Change fields of the record and write it again. Nothing when no record is kept."""
    with _RECORD["lock"]:
        if _RECORD["path"] is None:
            return
        _RECORD["data"].update(fields)
        _write_record()


def _drop_record():
    """Remove the record. Called before every way this process ends."""
    with _RECORD["lock"]:
        path, _RECORD["path"] = _RECORD["path"], None
    if path is not None:
        try:
            os.unlink(path)
        except OSError:
            pass


def _op_register(request):
    """Write ``<workspace>/.letify/workers/<pid>.json`` and keep it current until exit."""
    root = os.path.expanduser(request["workspace"])
    directory = os.path.join(root, ".letify", "workers")
    os.makedirs(directory, exist_ok=True)
    os.chmod(directory, 0o700)
    data = {
        "pid": os.getpid(),
        "proc_start": _proc_start(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": request["runtime"],
        "key": request["key"],
        "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "owner": request["owner"],
        "run": _TAKE["run"],
        "lease_deadline": globals().get("_LEASE", {}).get("deadline"),
    }
    with _RECORD["lock"]:
        _RECORD["path"] = os.path.join(directory, "%d.json" % os.getpid())
        _RECORD["data"] = data
        _write_record()
    return {"ok": True, "value": None}


# -- controlling a running call --------------------------------------------------------
# Spec "Controlling a running call": an interrupt, source run beside the body, and exit hooks.

#: What is running. ``body`` is true from just before the declared function is called until
#: it returns or raises. ``hooks`` are the callables ``letify.on_exit`` registered, dropped
#: when the call ends. ``op_signal`` marks a SIGINT sent by the ``interrupt`` op, and
#: ``raised`` is when a KeyboardInterrupt was last raised by the handler.
_RUN = {"body": False, "hooks": [], "op_signal": False, "raised": 0.0, "ignored": False}
_RUN_LOCK = threading.Lock()

#: Held while hooks run, so a second reason to end waits for the first one's hooks.
_HOOKS_LOCK = threading.Lock()

#: Seconds all exit hooks together may take.
_HOOKS_SECONDS = 120.0


def _run_enter():
    with _RUN_LOCK:
        _RUN["hooks"] = []
        _RUN["body"] = True


def _run_leave():
    with _RUN_LOCK:
        _RUN["body"] = False
        _RUN["hooks"] = []


def _exit_register(fn):
    """Add an exit hook for the running body. False when no body is running."""
    with _RUN_LOCK:
        if not _RUN["body"]:
            return False
        _RUN["hooks"].append(fn)
        return True


def _run_exit_hooks(reason):
    """Run the registered hooks, newest first and each once, within 120 s in total.

    A hook that raises is reported on standard error and the next one runs. The hooks run
    on a thread of their own, so a body blocked in C code cannot hold them up.
    """
    with _HOOKS_LOCK:
        with _RUN_LOCK:
            hooks = list(reversed(_RUN["hooks"]))
            _RUN["hooks"] = []
        if not hooks:
            return

        def go():
            for hook in hooks:
                try:
                    hook()
                except BaseException:
                    print("letify: an exit hook raised (%s):" % reason, file=sys.stderr)
                    traceback.print_exc()

        thread = threading.Thread(target=go, daemon=True)
        thread.start()
        thread.join(_HOOKS_SECONDS)
        if thread.is_alive():
            print("letify: exit hooks did not finish within %d s" % _HOOKS_SECONDS,
                  file=sys.stderr)
        _drain_output()


def _exit_with_hooks(code, reason="the lease ended"):
    _run_exit_hooks(reason)
    _drop_record()
    os._exit(code)


#: Seconds the body's descendants get to end after SIGTERM before they are killed.
CHILD_GRACE_S = 5.0


def _descendants():
    """Pids of every descendant of this process, from /proc, or None where /proc is missing."""
    try:
        names = [n for n in os.listdir("/proc") if n.isdigit()]
    except OSError:
        return None
    children = {}
    for name in names:
        try:
            with open("/proc/%s/stat" % name) as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        children.setdefault(int(fields[1]), []).append(int(name))
    found, queue = [], [os.getpid()]
    while queue:
        for pid in children.get(queue.pop(), ()):
            found.append(pid)
            queue.append(pid)
    return found


def _alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _end_descendants():
    """SIGTERM the processes the body started, wait CHILD_GRACE_S, then SIGKILL survivors."""
    import signal
    pids = _descendants()
    if pids is None:
        # No /proc: the group is signalled only when this worker leads it.
        if hasattr(os, "killpg") and os.getpgrp() == os.getpid():
            try:
                os.killpg(os.getpgrp(), signal.SIGTERM)
                time.sleep(2.0)
            except OSError:
                pass
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    end = time.monotonic() + CHILD_GRACE_S
    while time.monotonic() < end and any(_alive(pid) for pid in pids):
        time.sleep(0.05)
    for pid in pids:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def _end_by_owner():
    """The owner shut down on purpose: run the hooks, then end the body and the worker.

    Spec "A deliberate shutdown". The exit status is the one a received SIGTERM gives. The
    processes the body started are ended first.
    """
    import signal
    _run_exit_hooks("the owner shut down")
    _end_descendants()
    _drop_record()
    os._exit(128 + signal.SIGTERM)


def _on_sigint(signum, frame):
    """Raise KeyboardInterrupt, as the default handler does, except for a stale interrupt.

    A SIGINT the ``interrupt`` op sent can reach the main thread after the body returned,
    and it is dropped then. One that arrives within a second of the last raised interrupt
    is dropped too, because on a Local runtime the terminal's Ctrl-C and the owner's
    interrupt both arrive for one key press.
    """
    now = time.monotonic()
    if _RUN["op_signal"]:
        _RUN["op_signal"] = False
        if not _RUN["body"] or now - _RUN["raised"] < 1.0:
            return
    elif _RUN["ignored"]:
        return
    _RUN["raised"] = now
    raise KeyboardInterrupt


def _on_sigterm(signum, frame):
    # Everything happens on the thread reading the wakeup descriptor, because this runs on
    # the main thread, which may be inside the body.
    return None


def _watch_sigterm(fd):
    import signal
    while True:
        try:
            data = os.read(fd, 64)
        except OSError:
            return
        if not data:
            return
        if signal.SIGTERM in data and signal.getsignal(signal.SIGTERM) is _on_sigterm:
            _run_exit_hooks("SIGTERM")
            _drop_record()
            os._exit(128 + signal.SIGTERM)


def _install_run_control():
    """Install the SIGINT handler and the SIGTERM watcher. Main thread only."""
    try:
        import signal
        if not hasattr(signal, "set_wakeup_fd") or os.name == "nt":
            return
        _RUN["ignored"] = signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        signal.signal(signal.SIGINT, _on_sigint)
        read_fd, write_fd = os.pipe()
        os.set_blocking(write_fd, False)
        signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        return
    threading.Thread(target=_watch_sigterm, args=(read_fd,), daemon=True).start()


def _forget_run_control():
    """In a process the body forked: drop the worker's signal wiring."""
    try:
        import signal
        signal.set_wakeup_fd(-1)
        if signal.getsignal(signal.SIGTERM) is _on_sigterm:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if signal.getsignal(signal.SIGINT) is _on_sigint:
            signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError, AttributeError, NameError):
        pass


def _op_interrupt(request):
    """Raise KeyboardInterrupt in the body's thread, as Ctrl-C does.

    The signal is sent to the main thread, which runs the body, rather than set as an
    asynchronous exception: a body blocked in a system call such as ``time.sleep`` or a
    socket read is woken by the signal and stops at once, where an asynchronous exception
    waits for the call to return. A body inside a long C call that never releases the
    interpreter stops when that call returns, exactly as it does for Ctrl-C.
    """
    import signal
    if not _RUN["body"]:
        return {"ok": True, "value": {"interrupted": False, "reason": "nothing is running"}}
    if not hasattr(signal, "pthread_kill"):
        return {"ok": True, "value": {
            "interrupted": False,
            "reason": "this platform cannot signal the thread that runs the body",
        }}
    _RUN["op_signal"] = True
    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
    return {"ok": True, "value": {"interrupted": True}}


# Source run beside the body, with the body's own frame in reach.

_EXEC = {"count": 0, "wrapped": {}, "sinks": {}, "roots": None}
_EXEC_LOCK = threading.Lock()


class _ThreadStream:
    """Stands in for sys.stdout or sys.stderr and keeps what chosen threads write.

    Any other thread, the body's among them, writes through to the stream it replaced.
    """

    def __init__(self, inner, name):
        self._inner = inner
        self._name = name

    def write(self, text):
        sink = _EXEC["sinks"].get(threading.get_ident())
        if sink is not None:
            return sink[self._name].write(text)
        return self._inner.write(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        if threading.get_ident() not in _EXEC["sinks"]:
            self._inner.flush()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _capture_begin(sink):
    with _EXEC_LOCK:
        for name, attr in (("out", "stdout"), ("err", "stderr")):
            if not isinstance(getattr(sys, attr), _ThreadStream):
                wrapper = _ThreadStream(getattr(sys, attr), name)
                setattr(sys, attr, wrapper)
                _EXEC["wrapped"][attr] = wrapper
        _EXEC["count"] += 1
        _EXEC["sinks"][threading.get_ident()] = sink


def _capture_end():
    with _EXEC_LOCK:
        _EXEC["sinks"].pop(threading.get_ident(), None)
        _EXEC["count"] -= 1
        if _EXEC["count"] == 0:
            for attr, wrapper in _EXEC["wrapped"].items():
                if getattr(sys, attr) is wrapper:
                    setattr(sys, attr, wrapper._inner)
            _EXEC["wrapped"] = {}


def _user_roots():
    """Directories whose frames are not the user's: the standard library, site-packages, letify."""
    if _EXEC["roots"] is None:
        import sysconfig
        roots = []
        for key in ("stdlib", "platstdlib", "purelib", "platlib"):
            path = sysconfig.get_paths().get(key)
            if path:
                roots.append(os.path.normpath(path))
        package = sys.modules.get("letify")
        if package is not None and getattr(package, "__file__", None):
            roots.append(os.path.dirname(os.path.abspath(package.__file__)))
        _EXEC["roots"] = roots
    return _EXEC["roots"]


def _is_user_frame(filename):
    if filename.startswith("<frozen") or filename in ("letify-worker", "<letify>"):
        return False
    parts = filename.replace("\\", "/").split("/")
    if "site-packages" in parts or "dist-packages" in parts:
        return False
    normal = os.path.normpath(filename)
    for root in _user_roots():
        if normal == root or normal.startswith(root + os.sep):
            return False
    return True


class _Locals:
    """A mapping over a frame's locals that records the names the source assigns."""

    def __init__(self, frame):
        self._frame = frame
        self.assigned = []

    def __getitem__(self, key):
        return self._frame.f_locals[key]

    def __setitem__(self, key, value):
        self._frame.f_locals[key] = value
        if key not in self.assigned:
            self.assigned.append(key)

    def __delitem__(self, key):
        del self._frame.f_locals[key]
        if key not in self.assigned:
            self.assigned.append(key)

    def __iter__(self):
        return iter(list(self._frame.f_locals))

    def __len__(self):
        return len(self._frame.f_locals)

    def __contains__(self, key):
        return key in self._frame.f_locals

    def get(self, key, default=None):
        return self._frame.f_locals.get(key, default)

    def keys(self):
        return list(self._frame.f_locals.keys())

    def values(self):
        return list(self._frame.f_locals.values())

    def items(self):
        return list(self._frame.f_locals.items())

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def pop(self, key, *default):
        if key in self._frame.f_locals or not default:
            value = self._frame.f_locals[key]
            del self[key]
            return value
        return default[0]

    def setdefault(self, key, default=None):
        if key not in self._frame.f_locals:
            self[key] = default
        return self._frame.f_locals[key]

    def __repr__(self):
        return repr(dict(self._frame.f_locals))


def _op_inspect_exec(request):
    """Run source on this thread with the body's frame in reach, and answer what it did.

    Standard output and error are captured for this thread only. A single expression
    answers with the repr of its value.
    """
    import ast
    import io
    if not _RUN["body"]:
        return {"ok": True, "value": {"ran": False, "reason": "nothing is running"}}
    chain = []
    top = sys._current_frames().get(threading.main_thread().ident)
    while top is not None:
        # The outermost frame of the worker is the bootstrap stub, which python -c names <string>.
        stub = top.f_back is None and top.f_code.co_filename == "<string>"
        if not stub and _is_user_frame(top.f_code.co_filename):
            chain.append(top)
        top = top.f_back
    namespace = {"__name__": "letify_exec", "frames": list(reversed(chain))}
    mapping = None
    if chain:
        mapping = _Locals(chain[0])
        namespace.update({"frame": chain[0], "L": mapping, "G": chain[0].f_globals})
    else:
        namespace.update({"frame": None, "L": {}, "G": {}})
    answer = {"ran": True, "has_value": False, "value": None, "error": None}
    sink = {"out": io.StringIO(), "err": io.StringIO()}
    _capture_begin(sink)
    try:
        try:
            tree = ast.parse(request["source"], "<letify-exec>", "exec")
            single = len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
            if single:
                code = compile(request["source"].strip(), "<letify-exec>", "eval")
                answer["value"] = repr(eval(code, namespace))
                answer["has_value"] = True
            else:
                exec(compile(tree, "<letify-exec>", "exec"), namespace)
        except BaseException as exc:
            # Past this function's own frame, so the traceback starts at the user's source.
            tb = None if isinstance(exc, SyntaxError) else exc.__traceback__.tb_next
            answer["error"] = "".join(traceback.format_exception(type(exc), exc, tb))
    finally:
        _capture_end()
    answer["stdout"] = sink["out"].getvalue()
    answer["stderr"] = sink["err"].getvalue()
    answer["assigned"] = list(mapping.assigned) if mapping is not None else []
    # Before Python 3.13 f_locals is a snapshot, and writing to it never reaches the frame.
    answer["applied"] = not answer["assigned"] or sys.version_info >= (3, 13)
    namespace.clear()
    return {"ok": True, "value": answer}


_OPS = {
    "call": _op_call,
    "have": _op_have,
    "data_have": _op_data_have,
    "data_put": _op_data_put,
    "data_begin": _op_data_begin,
    "data_drop": _op_data_drop,
    "data_failed": _op_data_failed,
    "data_put_at": _op_data_put_at,
    "data_seal": _op_data_seal,
    "data_pull": _op_data_pull,
    "data_evict": _op_data_evict,
    "data_cache": _op_data_cache,
    "data_written": _op_data_written,
    "data_output_poll": _op_data_output_poll,
    "data_stream": _op_data_stream,
    "data_stats": _op_data_stats,
    "blob_dir": _op_blob_dir,
    "put_blob": _op_put_blob,
    "put_file": _op_put_file,
    "pull": _op_pull,
    "get_file": _op_get_file,
    "pack_dir": _op_pack_dir,
    "pack_blob": _op_pack_blob,
    "push_blob": _op_push_blob,
    "pack_push_start": _op_pack_push_start,
    "pack_push_target": _op_pack_push_target,
    "pack_push_state": _op_pack_push_state,
    "place_blob": _op_place_blob,
    "exec": _op_exec,
    "eval": _op_eval,
    "release": _op_release,
    "device": _op_device,
    "stat": _op_stat,
    "lease": _op_lease,
    "register": _op_register,
    "interrupt": _op_interrupt,
    "inspect_exec": _op_inspect_exec,
    "reduce_peers": _op_reduce_peers,
    "drive_grant": _op_drive_grant,
}

#: Answered by the frame reader at once, so they work while a call runs.
# ``data_failed`` is light because it must not queue behind the call it is about: the call
# is blocked on a blob the sender will never send, so a queued answer would deadlock.
_LIGHT = (
    "stat", "lease", "reduce_peers", "drive_grant", "data_failed",
    "pack_push_state", "pack_push_target", "interrupt",
)

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
    if name == "call":
        _TAKE["stream"] = stream
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
    if name in _DRAINS:
        # Spec "Worker output": what the program wrote is framed before its result goes
        # out. Every op that runs a program drains; a stat or a lease renewal does not,
        # because they write nothing and are on the hot path.
        _drain_output()
    if name == "call" and _TAKE["run"] is not None:
        _deliver_call(stream, outcome)
        return
    _reply(stream, outcome)


#: Ops that run a program, so output of theirs may still be unframed when they answer.
#: Spec "Worker output".
_DRAINS = frozenset(("call", "exec"))

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


def _read(readinto=None, door=None):
    """Read frames, answer light requests, and queue the rest for the main thread.

    With no arguments it reads the channel the worker started on. With ``readinto`` and
    ``door`` it reads a connection to the second door, which spec "Taking over a running call"
    describes: such a connection may only adopt the call until it has, and is then the
    channel the call's output and reply go to.
    """
    receiver = Receiver(readinto or sys.stdin.buffer.readinto)
    shutdown = False
    while True:
        try:
            event = receiver.next_event()
        except EOFError:
            break
        except Exception:
            if door is None:
                traceback.print_exc()
            break
        if event is None:
            continue
        kind, stream, value = event
        if kind == SHUTDOWN:
            shutdown = True
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
        if door is not None:
            if op == "adopt":
                _door_adopt(door, stream, request)
                continue
            if _TAKE["owner"] is not door:
                _answer(door.sender, stream, {
                    "ok": False, "kind": "runtime", "last": True,
                    "error": "adopt the call before sending %r" % (op,),
                })
                continue
        elif _TAKE["moved"]:
            # The call was adopted away from this channel, which gets nothing but this.
            _answer(_TAKE["primary"], stream, _moved_outcome())
            continue
        if op in _LIGHT:
            _run(stream, request, False)
            continue
        if op == "inspect_exec":
            # A thread of its own: the source may run for a while, and the reader must stay
            # free to answer the lease and the stream requests of the body it is looking at.
            threading.Thread(target=_run, args=(stream, request, False), daemon=True).start()
            request = None
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
    _reader_ended(door, shutdown)


def _reader_ended(door, shutdown=False):
    """A channel's reader stopped: it was closed, or sent a shutdown.

    Spec "A deliberate shutdown": a shutdown frame on the channel that owns a running call
    ends the body and the worker. End of input alone leaves the body for a takeover.
    """
    during = False
    with _TAKE_COND:
        if door is None and _TAKE["moved"]:
            # The call was adopted away; this channel closing is the old owner leaving. Read
            # under the lock, because an adoption in progress holds it until it has moved.
            return
        if door is None or (_TAKE["owner"] is door and _TAKE["live"]):
            during = _TAKE["run"] is not None
            if door is not None or during:
                _TAKE["live"] = False
            if during:
                _TAKE["ended_input"] = True
            _TAKE_COND.notify_all()
        else:
            # A connection that never adopted the call: the door goes on listening.
            return
    inbox = _DEVICE.get("inbox")
    if inbox is not None:
        inbox.put(None)
    if shutdown and during:
        _end_by_owner()
        return
    # Spec "Exit hooks": a channel that closes while a body runs is a reason to end, and
    # the hooks run before the main thread is told there is nothing more to serve. Spec
    # "Taking over a running call": the body keeps running, and the worker ends when the
    # call's reply has been delivered or the lease deadline passes, so no end is queued yet.
    _run_exit_hooks("the channel closed")
    if not during:
        _JOBS.put(None)


# -- the second door ----------------------------------------------------------------------
# Spec "Taking over a running call": while a call runs, a Unix socket on this machine lets
# another process become the call's owner.


class _Door:
    """One connection to the second door, with a sender of its own."""

    def __init__(self, connection):
        self.connection = connection
        self.sender = Sender(lambda view: connection.send(view))


def _answer(sender, stream, outcome):
    """Send one reply to a channel that may be gone, ignoring that it is."""
    try:
        sender.message(REPLY, stream, outcome)
    except OSError:
        pass


def _moved_outcome():
    return {
        "ok": False, "moved": True, "run": _TAKE["run"] or "", "owner": _TAKE["new_owner"],
        "error": "run %s now belongs to process %s" % (_TAKE["run"], _TAKE["new_owner"]),
    }



def _orphan_output(kind, data):
    """Keep the tail of output no channel could take. False when no call is being served."""
    with _TAKE_COND:
        if _TAKE["run"] is None:
            return False
        kept = _TAKE["backlog"][kind]
        kept += data
        if len(kept) > _BACKLOG_BYTES:
            del kept[: len(kept) - _BACKLOG_BYTES]
        return True


def _door_open(take):
    """Start listening on the call's door. A door that cannot open leaves the call as it was."""
    _record_update(run=take["run"])
    with _TAKE_COND:
        _TAKE.update(
            run=take["run"], live=True, ended_input=False, finished=False, outcome=None,
            path=take["door"],
        )
        for kept in _TAKE["backlog"].values():
            del kept[:]
    listener = None
    try:
        import socket
        path = os.path.expanduser(take["door"])
        directory = os.path.dirname(path)
        existed = os.path.isdir(directory)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        if not existed:
            os.chmod(directory, 0o700)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        address, descriptor = unix_address(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(address)
        finally:
            if descriptor is not None:
                os.close(descriptor)
        os.chmod(path, 0o600)
        listener.listen(1)
    except OSError as exc:
        if listener is not None:
            listener.close()
        print("letify: run %s has no second door: %s" % (take["run"], exc), file=sys.stderr)
        return
    with _TAKE_COND:
        _TAKE.update(listener=listener, path=path)
    threading.Thread(target=_door_serve, args=(listener,), daemon=True).start()


def _door_serve(listener):
    """Serve one connection at a time until the door closes."""
    version = ("%d.%d" % sys.version_info[:2]).encode("ascii")
    while True:
        try:
            connection, _address = listener.accept()
        except OSError:
            return
        door = _Door(connection)
        try:
            connection.settimeout(_DOOR_IDLE_S)
            door.sender.frame(HELLO, 0, version)
            _read(connection.recv_into, door)
        except OSError:
            pass
        finally:
            connection.close()


def _door_adopt(door, stream, request):
    """Make this connection the owner: switch the call's output and reply to it."""
    global _SENDER
    run = request.get("run")
    with _TAKE_COND:
        if _TAKE["run"] is None or _TAKE["run"] != run:
            _answer(door.sender, stream, {
                "ok": False, "kind": "runtime", "last": True,
                "error": "no call %s is running on this worker" % (run,),
            })
            return
        was_live = _TAKE["live"]
        _TAKE["new_owner"] = request.get("owner")
        former, former_stream = _SENDER, _TAKE["stream"]
        if _TAKE["owner"] is None:
            _TAKE["primary"] = former
        else:
            former_stream = _TAKE["owner_stream"]
        moved = _moved_outcome()
        if _TAKE["owner"] is not None:
            moved["last"] = True
        # Told on a thread, because a former owner that has stopped reading can block the write.
        told = threading.Thread(
            target=_answer, args=(former, former_stream, moved), daemon=True
        )
        told.start()
        told.join(2.0)
        state = {
            "run": run,
            "state": "finished" if _TAKE["finished"] else "running",
            "orphaned": not was_live,
        }
        try:
            door.sender.message(REPLY, stream, {"ok": True, "value": state})
            for kind in (STDOUT, STDERR):
                kept = _TAKE["backlog"][kind]
                if kept:
                    door.sender.frame(kind, 0, bytes(kept))
                    del kept[:]
        except OSError:
            return
        door.connection.settimeout(None)
        _TAKE.update(
            owner=door, owner_stream=stream, moved=True, live=True, ended_input=False
        )
        _SENDER = door.sender
        _TAKE_COND.notify_all()


def _deliver_call(stream, outcome):
    """Send a call's reply to whichever channel owns the call now, then close the door.

    A call that ends while no channel is open waits for an owner until the lease deadline, so
    its result is not lost to a process that died a moment before the body returned. Without a
    lease nothing bounds the wait, so the worker ends as it did before taking over existed.
    """
    failed = False
    with _TAKE_COND:
        _TAKE["finished"] = True
        _TAKE["outcome"] = outcome
        while not _TAKE["live"]:
            deadline = globals().get("_LEASE", {}).get("deadline")
            left = (deadline or 0) - time.time()
            if left <= 0:
                break
            _TAKE_COND.wait(min(left, 1.0))
        owner = _TAKE["owner"]
        try:
            if owner is not None:
                _send_outcome(owner.sender, _TAKE["owner_stream"], outcome, last=True)
            else:
                _send_outcome(_SENDER, stream, outcome)
        except OSError:
            failed = True
        ended = _TAKE["ended_input"]
        listener, path = _TAKE["listener"], _TAKE["path"]
        _TAKE.update(run=None, listener=None, path=None, finished=False, outcome=None)
    _door_close(listener, path)
    _record_update(run=None)
    if failed and owner is None:
        _drop_record()
        os._exit(0)
    if ended:
        _JOBS.put(None)


def _door_close(listener, path):
    """Stop listening and remove the socket file."""
    if listener is not None:
        try:
            # Wakes the accepting thread, which a close alone leaves blocked on Linux.
            import socket
            listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        listener.close()
    if path:
        try:
            os.unlink(os.path.expanduser(path))
        except OSError:
            pass


def _serve():
    _install_run_control()
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
    _drop_record()


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
