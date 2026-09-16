"""Runtime, one live session.

This is the only object in letify that costs money. The provider, the instance and
the environment are all descriptions; creating a runtime is when Colab or Modal or
a cloud machine powers something on, and shutting it down is when the charge stops.

A runtime ends when the call that needed it finishes. Keeping one longer is opt in,
inside a ``let.keep_alive()`` block, because idle time on a GPU is money for
nothing. The counter-argument is real and the reason that option exists: starting a session
costs provider boot plus environment installation, which on Colab is minutes, so a run of
several separate calls is cheaper with one session than with several.

Nothing ends a session on a timer. The lease is the one thing that ends one without being
asked, and it is for a process killed outright: the worker holds a deadline and exits if this
process stops pushing it forward, which frees the card. Whether that also stops the billing
depends on what the provider charges for.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import protocol
from ..errors import (
    ConfigError,
    EnvironmentFailure,
    InterpreterMismatch,
    RemoteError,
    RuntimeFailure,
)
from .lease import Lease

if TYPE_CHECKING:
    from ..declare.env import Env
    from ..declare.instance import Instance
    from ..providers.base import Provider
    from ..store.volume import Volume
    from .channel import Channel

#: The total size of argument blob files kept under a persistent workspace root, 32 GiB.
BLOB_DISK_LIMIT = 32 * 1024**3


@dataclass
class Runtime:
    """One live session on one provider."""

    name: str
    provider: Provider
    instance: Instance
    env: Env
    volumes: tuple[Volume, ...] = ()

    #: Physical device indices this session reserved, empty where the provider assigns the
    #: device itself. Set for the session's visible devices so the code inside sees its
    #: cards as 0 upward.
    held_devices: tuple[int, ...] = ()
    channel: Channel | None = None

    #: Provider side identifier, such as an Elice allocation id.
    external_id: str | None = None

    #: The worker's process id on its machine, recorded when a persistent channel starts so
    #: the provider does not read this session's own card as busy. None on a one shot channel.
    worker_pid: int | None = None

    started: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    ready: bool = False
    busy: bool = False

    #: How long this session lives, from the declaration that started it.
    lease: Lease | None = field(default=None, repr=False)

    #: How the environment was built: ``"sync"``, ``"archive"``, or None where it was not.
    env_source: str | None = None

    #: The runtime's ``sys.platform`` and machine, such as ``linux-x86_64``, once probed.
    platform: str | None = None

    #: The workspace root with ``~`` expanded on the runtime, once the boot prepared it.
    workspace: str | None = None

    #: Digests this runtime is known to hold, so an argument is sent once.
    _blobs: set[str] = field(default_factory=set, repr=False)

    #: The interpreter the session built or was given, once known. A device worker uses it.
    python: str | None = None

    #: The PyTorch device worker's client, started on the first host='local' call.
    device_client: Any = field(default=None, repr=False)

    #: Digests of immutable arguments already hashed, so a repeated one is not hashed again.
    _digests: protocol.DigestCache = field(default_factory=protocol.DigestCache, repr=False)

    # -- identity ------------------------------------------------------------

    @property
    def key(self) -> str:
        """Pool key. Runtimes with equal keys are interchangeable."""
        return f"{self.instance.key}|{self.env.key}"

    @property
    def idle_for(self) -> float:
        return time.monotonic() - self.last_used

    @property
    def persistent_channel(self) -> bool:
        return bool(self.channel and self.channel.persistent)

    # -- lifecycle -----------------------------------------------------------

    def boot(self) -> None:
        """Open the channel, arm the lease, prepare the workspace root, build the environment,
        check the interpreter, attach volumes."""
        if self.channel is None:
            self.channel = self.provider.open_channel(self)
        self.channel.start()
        if self.channel.persistent:
            # os.execv keeps the process id, so it holds after the interpreter move too.
            self.worker_pid = int(self.stat()["pid"])
            self.provider.add_worker_pid(self.worker_pid)
        self.restrict_devices()
        if self.provider.needs_lease:
            self.lease = Lease(self)
            self.lease.arm()
        self.prepare_workspace()
        self.install_env()
        if self.env_source is not None:
            # The worker moved to the project interpreter, so enter the root again there.
            self.prepare_workspace()
        try:
            self.check_interpreter()
        except InterpreterMismatch:
            # Not retried, so nothing else would end this session.
            self.shutdown()
            raise
        self.keep_blobs_on_disk()
        for volume in self.volumes:
            self.attach(volume)
        self.ready = True

    def keep_blobs_on_disk(self) -> None:
        """Have the worker write argument blobs under the workspace root on a persistent disk.

        Spec "Argument blobs on a persistent disk". A later session on the same machine then
        answers ``have`` from those files, so a repeated argument travels as its digest.
        """
        provider = self.provider
        if not (provider.persistent and provider.prepares_workspace and self.persistent_channel):
            return
        root = self.workspace or provider.workspace_root
        self.request(
            {"op": "blob_dir", "path": f"{root.rstrip('/')}/blobs", "limit": BLOB_DISK_LIMIT},
            timeout=120,
        )

    def shutdown(self) -> None:
        """Stop everything that bills for this runtime."""
        self.ready = False
        if self.device_client is not None:
            client, self.device_client = self.device_client, None
            client.close()
        if self.worker_pid is not None:
            self.provider.remove_worker_pid(self.worker_pid)
            self.worker_pid = None
        if self.lease is not None:
            self.lease.release()
            self.lease = None
        if self.channel is not None:
            try:
                self.channel.close()
            except RuntimeFailure:
                pass
            self.channel = None
        self.provider.stop(self)

    def device(self) -> Any:
        """The PyTorch device worker for host='local', started on first use."""
        if self.device_client is None:
            from ..remoting.device.client import attach

            channel = self.provider.device_channel(self)
            kind = "cuda" if self.instance.gpu else "cpu"
            self.device_client = attach(
                channel,
                device=kind,
                name=f"{self.name}-device",
                auto_fetch=self.provider.config.option("auto_fetch"),
            )
        return self.device_client

    # -- requests ------------------------------------------------------------

    def session(self) -> Runtime:
        """Itself, so a volume can take either a declaration or a live session."""
        return self

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> Any:
        """Send one worker request and return its value."""
        if self.channel is None:
            raise RuntimeFailure(f"{self.name}: the channel is not open")
        self.last_used = time.monotonic()
        value, _logs = self.channel.request(payload, timeout=timeout)
        return value

    def exec(self, source: str, *, timeout: float | None = None) -> None:
        """Run plain source inside the runtime, sharing the worker's globals."""
        self.request({"op": "exec", "source": source}, timeout=timeout)

    def eval(self, source: str, *, timeout: float | None = None) -> Any:
        """Run plain source inside the runtime and return its ``__letify_value__``."""
        if self.channel is None:
            raise RuntimeFailure(f"{self.name}: the channel is not open")
        self.last_used = time.monotonic()
        return self.channel.eval(source, timeout=timeout)

    def stat(self) -> dict[str, Any]:
        """What the worker is holding: blobs, bytes, process id."""
        return self.request({"op": "stat"}, timeout=60)

    def call(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        *,
        timeout: float | None = None,
    ) -> tuple[Any, str]:
        """Run one declared function inside this runtime.

        Large arguments are replaced by content addressed references so the same
        payload is not sent twice.
        """
        if self.channel is None:
            raise RuntimeFailure(f"{self.name}: the channel is not open")
        # Done here rather than at declaration time, because the registration lives in
        # cloudpickle and a process can hold declarations with different environments.
        if self.env.ship_modules:
            protocol.codec.ship_by_value(self.env.ship_modules)
        if not self.persistent_channel:
            self.last_used = time.monotonic()
            return self.channel.call(fn, args, kwargs, timeout=timeout)
        args, kwargs = self._externalize(args, kwargs)
        return self._call_with_data(fn, args, kwargs, timeout)

    def _call_with_data(
        self, fn: Any, args: tuple, kwargs: dict, timeout: float | None
    ) -> tuple[Any, str]:
        """Pickle a call, replacing local data paths, and send their blobs before it.

        Spec "Project data": the call carries the links the worker places before loading it.
        """
        import secrets

        from ..store import pathdata

        assert self.channel is not None
        root = (self.workspace or self.provider.workspace_root).rstrip("/")
        collector = pathdata.Collector(f"{root}/data/calls/{secrets.token_hex(8)}")
        head, buffers = protocol.dumps_call_parts(fn, args, kwargs, data=collector)
        request: dict[str, Any] = {"op": "call", "payload": head, "buffers": buffers}
        blobs = f"{root}/data/blobs"
        added = 0
        if collector.placed:
            if collector.inputs:
                added = pathdata.send(self, collector, blobs)
            request["data"] = collector.request(blobs)
        self.last_used = time.monotonic()
        try:
            outcome = self.channel.request(request, timeout=timeout)
            if collector.outputs:
                added += pathdata.write_back(self, collector, blobs)
            return outcome
        finally:
            if added:
                pathdata.evict(self, blobs)

    # -- content addressed arguments -----------------------------------------

    def _externalize(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Replace large arguments with blob references, uploading what is missing.

        Spec "Argument addressing". A ``bytes`` argument is hashed as it is, anything else
        over its protocol 5 pickle and out-of-band buffers. The digest of an immutable
        argument is cached, so repeating it costs neither hashing nor sending.
        """
        plan: dict[str, Callable[[], dict[str, Any]]] = {}

        def convert(value: Any) -> Any:
            if isinstance(value, protocol.Blob):
                return value
            if type(value) is bytes:
                if len(value) < protocol.INLINE_LIMIT:
                    return value
                known = self._digests.get(value)
                if known is None:
                    known = (protocol.digest_parts((value,)), len(value))
                    self._digests.put(value, *known)
                plan[known[0]] = lambda value=value: {"kind": "bytes", "value": value}
                return protocol.Blob(digest=known[0], size=known[1])
            immutable = protocol.DigestCache.immutable(value)
            known = self._digests.get(value) if immutable else None
            if known is not None:
                plan[known[0]] = lambda value=value: _pickled(value, True)
                return protocol.Blob(digest=known[0], size=known[1])
            try:
                message = _pickled(value, immutable)
            except Exception:
                # Not plainly picklable, so leave it for cloudpickle to carry.
                return value
            parts = [message["head"], *message["buffers"]]
            size = sum(memoryview(part).nbytes for part in parts)
            if size < protocol.INLINE_LIMIT:
                return value
            digest = protocol.digest_parts(parts)
            if immutable:
                self._digests.put(value, digest, size)
            plan[digest] = lambda message=message: message
            return protocol.Blob(digest=digest, size=size)

        new_args = tuple(convert(v) for v in args)
        new_kwargs = {k: convert(v) for k, v in kwargs.items()}
        if plan:
            self._upload_blobs(plan)
        return new_args, new_kwargs

    def _upload_blobs(self, plan: dict[str, Callable[[], dict[str, Any]]]) -> None:
        unknown = [d for d in plan if d not in self._blobs]
        if unknown:
            held = self.request({"op": "have", "digests": unknown}, timeout=120) or []
            self._blobs.update(held)
            unknown = [d for d in unknown if d not in self._blobs]
        for digest in unknown:
            self.request({"op": "put_blob", "digest": digest, **plan[digest]()}, timeout=3600)
            self._blobs.add(digest)

    # -- files ---------------------------------------------------------------

    def put_bytes(
        self,
        payload: bytes,
        path: str,
        *,
        unpack: bool = False,
        target: str | None = None,
        links: bool = False,
    ) -> protocol.RemoteFile:
        """Write bytes to a path inside the runtime, optionally unpacking an archive."""
        value = self.request(
            {
                "op": "put_file",
                "path": path,
                "payload": payload,
                "unpack": unpack,
                "target": target,
                "links": links,
            },
            timeout=3600,
        )
        return protocol.RemoteFile(
            path=value["path"], digest=protocol.digest_of(payload), size=value["size"]
        )

    def pull(
        self,
        source: dict[str, Any],
        path: str,
        *,
        digest: str,
        unpack: bool = False,
        target: str | None = None,
        links: bool = False,
    ) -> protocol.RemoteFile:
        """Have the runtime download a blob from its backend, optionally unpacking it.

        ``source`` is what ``Backend.pull_source`` answered: a URL and headers carrying a
        short-lived token. The worker drops both once the download finishes.
        """
        value = self.request(
            {
                "op": "pull",
                "url": source["url"],
                "headers": dict(source.get("headers") or {}),
                "path": path,
                "unpack": unpack,
                "target": target,
                "links": links,
            },
            timeout=3600,
        )
        return protocol.RemoteFile(path=value["path"], digest=digest, size=value["size"])

    def put_file(self, local: str | Path, path: str, **kwargs: Any) -> protocol.RemoteFile:
        return self.put_bytes(Path(local).read_bytes(), path, **kwargs)

    def get_bytes(self, path: str) -> tuple[bytes, str]:
        """Read a file out of the runtime, returning ``(payload, digest)``."""
        value = self.request({"op": "get_file", "path": path}, timeout=3600)
        return _payload(value["payload"]), value["digest"]

    def pack_dir(self, path: str) -> tuple[bytes, str]:
        """Pack a directory inside the runtime into one archive payload."""
        value = self.request({"op": "pack_dir", "path": path}, timeout=3600)
        return _payload(value["payload"]), value["digest"]

    # -- environment and volumes ---------------------------------------------

    def restrict_devices(self) -> None:
        """Show the worker only the cards this session reserved, numbered from 0.

        Set in the worker's own environment before any user code imports a CUDA library.
        os.execv passes the environment on, so the move to the project interpreter keeps
        it. A provider that assigns the device itself reserves no indices and sets nothing.
        """
        visible = self.provider.visible_devices(self.held_devices)
        if visible is None:
            return
        self.exec(
            "import os\n"
            f"os.environ['CUDA_VISIBLE_DEVICES'] = {visible!r}\n"
            "os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'\n",
            timeout=120,
        )

    def prepare_workspace(self) -> None:
        """Expand, create and enter the workspace root, and point TMPDIR under it.

        A provider whose worker is this machine's subprocess keeps its working directory, and
        its root is expanded here instead.
        """
        root = self.provider.workspace_root
        if not self.provider.prepares_workspace:
            import os

            self.workspace = os.path.expanduser(root)
            return
        from . import bootstrap

        self.workspace = self.eval(bootstrap.workspace_source(root), timeout=120)

    def install_env(self) -> None:
        """Build the project's environment in the session and move the worker onto it.

        An archive cached in a volume is preferred, because unpacking one file is a
        single transfer while syncing from a lock file is thousands of small ones. A
        provider whose account names ``python`` manages its own interpreter, so nothing
        is built there.
        """
        if not self.provider.prepares_env:
            return
        if self.provider.managed_python:
            self.python = self.provider.managed_python
            self.require_cloudpickle()
            return
        from . import bootstrap

        assert self.channel is not None
        files = bootstrap.project_files(self.env)
        env_root = self.provider.env_root
        root = bootstrap.project_dir(
            env_root or self.workspace or self.provider.workspace_root, self.env
        )
        where = self.eval(bootstrap.probe_source(self.env, root), timeout=120)
        self.platform = where["platform"]

        # Spec "Volumes on a persistent runtime": the .venv is already on the runtime's
        # disk, so an archive is neither restored nor packed there.
        archives = () if self.provider.persistent else self.volumes
        for volume in archives:
            digest = volume.cached_env(self.env, self.platform)
            if not digest:
                continue
            archive = f"{where['parent']}/.{digest}.tar.gz"
            volume.materialize(
                self, digest, path=archive, unpack=True, target=where["parent"], links=True
            )
            if self.eval(bootstrap.venv_check_source(where["python"], archive), timeout=600):
                self.env_source = "archive"
                break

        if self.env_source != "archive":
            workspace = self.workspace or self.provider.workspace_root
            # Spec "Environment on the sandbox disk": an env root keeps uv's default cache.
            persistent_cache = self.provider.persistent and env_root is None
            cache = bootstrap.uv_cache_dir(workspace) if persistent_cache else None
            source = bootstrap.sync_source(
                self.env, files, root=root, name=self.name, cache_dir=cache
            )
            try:
                self.eval(source, timeout=3600)
            except RemoteError as exc:
                message = str(exc).split("\n\n--- remote traceback ---")[0]
                raise EnvironmentFailure(
                    message.removeprefix("RuntimeError: "), stderr=exc.remote_traceback
                ) from exc
            self.env_source = "sync"
            if archives:
                archives[0].cache_env_from(self, self.env, where["root"], platform=self.platform)
        self.python = where["python"]
        self.channel.switch_interpreter(where["python"])

    def require_cloudpickle(self) -> None:
        """Refuse a user-managed interpreter that cannot import cloudpickle.

        letify installs nothing into that interpreter, so the fix is the user's. Not
        retried, because a fresh runtime has the same interpreter.
        """
        importable, executable = self.eval(
            "import sys\n"
            "try:\n"
            "    import cloudpickle\n"
            "    __letify_value__ = (True, sys.executable)\n"
            "except ImportError:\n"
            "    __letify_value__ = (False, sys.executable)\n",
            timeout=120,
        )
        if not importable:
            self.shutdown()
            raise ConfigError(
                f"{self.name}: the account's python {executable} cannot import cloudpickle, "
                f"which the worker needs to load a call. letify installs nothing into a "
                f"user-managed interpreter: install cloudpickle there, or remove the python "
                f"option so the runtime builds the project .venv with uv sync"
            )

    def check_interpreter(self) -> None:
        """Refuse a worker whose Python major.minor differs from this process."""
        if not self.provider.prepares_env:
            return
        from . import bootstrap

        assert self.channel is not None
        remote = self.channel.python_version or self.eval(bootstrap.VERSION_SOURCE, timeout=120)
        local = bootstrap.local_python()
        if remote != local:
            raise InterpreterMismatch(
                f"{self.name}: the worker runs Python {remote} and this process runs Python "
                f"{local}. cloudpickle ships a __main__ function as bytecode, which does not "
                f"run across minor versions, so the session is not used"
            )

    def attach(self, volume: Volume) -> None:
        """Make a volume's directory exist under the workspace root."""
        self.exec(
            "import os, pathlib\n"
            f"pathlib.Path(os.path.expanduser({volume.directory(self)!r}))"
            ".mkdir(parents=True, exist_ok=True)\n",
            timeout=120,
        )

    def __repr__(self) -> str:
        state = "ready" if self.ready else "starting"
        return f"<Runtime {self.name} {self.instance.accelerator} {state}>"


def _pickled(value: Any, immutable: bool) -> dict[str, Any]:
    """A ``put_blob`` message for a value: its protocol 5 pickle and out-of-band buffers."""
    head, buffers = protocol.wire.pickle_parts(value)
    return {"kind": "pickle", "immutable": immutable, "head": head, "buffers": buffers}


def _payload(value: Any) -> bytes:
    """File bytes from a reply: raw from a binary channel, base64 text from a one-shot one."""
    if isinstance(value, str):
        import base64

        return base64.b64decode(value)
    return bytes(value)


__all__ = ["Runtime"]
