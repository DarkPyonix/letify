"""The worker that runs inside a runtime.

This module holds source code as a string rather than code to import, because it
has to run on the remote side where letify is not installed. The channel sends it
once, the remote Python executes it, and from then on every call is a framed
request over the same pipe.

Keeping one process alive is what makes three features work.

An object table persists, so a value kept with ``keep_remote=True`` can be
referenced by a later call. Without a living process there is nothing for a handle
to point at.

A blob table persists, so a large argument is sent once and reused by name. The
worker answers which digests it already holds before the caller sends anything.

Files written into the runtime survive between calls, so a volume can materialize
an environment archive or a checkpoint and a later call can read it from disk.

The framing is one base64 line per message. That survives an SSH channel, a
WebSocket bridge and a plain pipe without any of them mangling it, which raw
binary framing does not.
"""

from __future__ import annotations

#: Written by the worker on the line before it starts reading requests.
READY = "__LETIFY_WORKER_READY__"

#: Passed to ``python -c`` to get the worker running.
#:
#: ``python -`` cannot be used: it reads all of standard input to EOF before it
#: compiles anything, and the pipe has to stay open for requests. So a stub small
#: enough to survive shell quoting reads a length-prefixed base64 blob, execs it, and
#: leaves standard input where it was.
BOOTSTRAP = (
    "import sys,base64;"
    "n=int(sys.stdin.readline());"
    "exec(compile(base64.b64decode(sys.stdin.read(n)).decode(),'letify-worker','exec'))"
)

#: Prefix of every response line. Anything without it is the user's own output.
REPLY = "__LETIFY_REPLY__"

SOURCE = r'''
import base64, hashlib, io, os, pickle, sys, tarfile, traceback, uuid

_READY = "__LETIFY_WORKER_READY__"
_REPLY = "__LETIFY_REPLY__"

_OBJECTS = {}
_BLOBS = {}

try:
    import cloudpickle
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cloudpickle"], check=True)
    import cloudpickle


def _digest(payload):
    try:
        import blake3
        return blake3.blake3(payload).hexdigest(length=16)
    except ImportError:
        return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _resolve(value):
    """Replace handles and blobs with what they point at, recursively."""
    kind = getattr(value, "__letify_kind__", None)
    if kind == "handle":
        try:
            return _OBJECTS[value.object_id]
        except KeyError:
            raise KeyError(
                "handle %s is not in this runtime's object table. The runtime was "
                "probably restarted after the handle was created." % value.object_id
            ) from None
    if kind == "blob":
        try:
            return pickle.loads(_BLOBS[value.digest])
        except KeyError:
            raise KeyError("blob %s was never sent to this runtime" % value.digest) from None
    if isinstance(value, (list, tuple)):
        return type(value)(_resolve(v) for v in value)
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    return value


def _keep(value):
    object_id = uuid.uuid4().hex
    _OBJECTS[object_id] = value
    return {
        "object_id": object_id,
        "type_name": type(value).__name__,
        "summary": repr(value)[:200],
    }


def _reply(payload):
    sys.stdout.flush()
    blob = pickle.dumps(payload, protocol=5)
    sys.stdout.write(_REPLY + base64.b64encode(blob).decode() + "\n")
    sys.stdout.flush()


def _op_call(request):
    fn, args, kwargs = cloudpickle.loads(base64.b64decode(request["payload"]))
    args = _resolve(args)
    kwargs = _resolve(kwargs)
    value = fn(*args, **kwargs)
    if hasattr(value, "__await__"):
        # An async declaration is awaitable locally. The body still has to run to
        # completion here, and awaiting it here is what lets it use await inside.
        import asyncio
        value = asyncio.run(value)
    if request.get("keep_remote"):
        return {"ok": True, "handle": _keep(value)}
    return {"ok": True, "value": value}


def _op_have(request):
    """Report which of these digests the runtime already holds."""
    held = [d for d in request["digests"] if d in _BLOBS]
    return {"ok": True, "value": held}


def _op_put_blob(request):
    """Store a payload under its content address."""
    payload = base64.b64decode(request["payload"])
    digest = request.get("digest") or _digest(payload)
    _BLOBS[digest] = payload
    return {"ok": True, "value": digest}


def _op_put_file(request):
    """Write a payload to a path inside the runtime."""
    payload = base64.b64decode(request["payload"])
    path = request["path"]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    if request.get("unpack"):
        _unpack(path, request.get("target") or os.path.dirname(path))
    return {"ok": True, "value": {"path": path, "size": len(payload)}}


def _unpack(path, target):
    os.makedirs(target, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        try:
            archive.extractall(target, filter="data")
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
        _unpack(path, request.get("target") or os.path.dirname(path))
    return {"ok": True, "value": {"path": path, "size": size}}


def _op_get_file(request):
    """Read a file out of the runtime, for example a checkpoint."""
    with open(request["path"], "rb") as handle:
        payload = handle.read()
    return {
        "ok": True,
        "value": {
            "payload": base64.b64encode(payload).decode(),
            "digest": _digest(payload),
            "size": len(payload),
        },
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
        "value": {
            "payload": base64.b64encode(payload).decode(),
            "digest": _digest(payload),
            "size": len(payload),
        },
    }


def _op_exec(request):
    """Run plain source inside the worker, sharing its globals."""
    exec(compile(request["source"], "<letify>", "exec"), globals())
    return {"ok": True, "value": None}


def _op_release(request):
    """Drop objects or blobs the caller no longer needs."""
    for object_id in request.get("objects", ()):
        _OBJECTS.pop(object_id, None)
    for digest in request.get("blobs", ()):
        _BLOBS.pop(digest, None)
    return {"ok": True, "value": None}


def _op_stat(request):
    return {
        "ok": True,
        "value": {
            "objects": len(_OBJECTS),
            "blobs": len(_BLOBS),
            "blob_bytes": sum(len(b) for b in _BLOBS.values()),
            "pid": os.getpid(),
            "executable": sys.executable,
        },
    }


def _op_lease(request):
    """Arm or renew the self termination deadline.

    The worker exits on its own if the caller stops renewing, so a crashed or
    killed local process cannot leave a paid session running.
    """
    import threading, time

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
    "release": _op_release,
    "stat": _op_stat,
    "lease": _op_lease,
}


def _serve():
    sys.stdout.write(_READY + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__LETIFY_SHUTDOWN__":
            return
        try:
            # The line is dropped once decoded, so a pull token it carried lives only in
            # the request, and the request is dropped once answered.
            request, line = pickle.loads(base64.b64decode(line)), None
        except Exception:
            _reply({
                "ok": False,
                "error": "the request could not be decoded",
                "traceback": traceback.format_exc(),
            })
            continue
        op = _OPS.get(request.get("op"))
        if op is None:
            _reply({"ok": False, "error": "unknown op %r" % request.get("op"), "traceback": ""})
            continue
        try:
            _reply(op(request))
        except BaseException as exc:
            _reply({
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            })
        request = None


_serve()
'''
