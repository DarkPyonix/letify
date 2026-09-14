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
        try:
            entry = _BLOBS[value.digest]
        except KeyError:
            raise KeyError("blob %s was never sent to this runtime" % value.digest) from None
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


def _op_have(request):
    """Report which of these digests the runtime already holds."""
    held = [d for d in request["digests"] if d in _BLOBS]
    return {"ok": True, "value": held}


def _op_put_blob(request):
    """Store a value under its content address.

    An immutable value is kept unpickled, so a repeated argument is not unpickled again.
    Anything else is kept as its pickle and buffers.
    """
    digest = request["digest"]
    if request.get("kind") == "bytes":
        value = request.pop("value")
        _BLOBS[digest] = ("value", value)
        _SIZES[digest] = len(value)
        return {"ok": True, "value": digest}
    head = bytes(request["head"])
    buffers = list(request.get("buffers") or ())
    _SIZES[digest] = len(head) + sum(memoryview(b).nbytes for b in buffers)
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
