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


def _load_call(request):
    # Imported here, not at start: until the worker moves to the project's interpreter it
    # runs on whatever python3 the machine has, and needs the standard library only.
    import cloudpickle

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

_JOBS = queue.Queue()


def _run(stream, request, settle):
    op = _OPS.get(request.get("op"))
    if op is None:
        outcome = {"ok": False, "error": "unknown op %r" % request.get("op"), "traceback": ""}
    else:
        try:
            outcome = op(request)
        except BaseException as exc:
            outcome = {
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
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
            if _authenticated(connection, token):
                _use_connection(connection)
                return connection
            connection.close()
    finally:
        server.close()


def _authenticated(connection, token):
    """Whether the connection's first line is the data prefix and the expected token."""
    import hmac

    expected = _DATA_PREFIX + token + b"\n"
    line = b""
    try:
        connection.settimeout(10)
        while not line.endswith(b"\n") and len(line) < len(expected):
            chunk = connection.recv(len(expected) - len(line))
            if not chunk:
                return False
            line += chunk
        connection.settimeout(None)
    except OSError:
        return False
    return hmac.compare_digest(line, expected)


def _use_connection(connection):
    """Send hello, then every later frame, over the data connection as binary frames."""
    global _SENDER
    import socket

    try:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    sender = Sender(connection.send)
    old = _SENDER
    # Under the old lock, so no frame is split between the two transports.
    with old.lock:
        sender.frame(HELLO, 0, ("%d.%d" % sys.version_info[:2]).encode("ascii"))
        _SENDER = sender


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
