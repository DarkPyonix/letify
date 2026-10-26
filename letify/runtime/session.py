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

import datetime
import json
import posixpath
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import protocol
from ..errors import (
    EnvironmentFailure,
    InterpreterMismatch,
    ProtocolError,
    RemoteError,
    RuntimeFailure,
)
from . import control
from .lease import Lease

if TYPE_CHECKING:
    from ..declare.env import Env
    from ..declare.instance import Instance
    from ..providers.base import Provider
    from ..store.volume import Volume
    from .channel import Channel

#: The total size of argument blob files kept under a persistent workspace root, 32 GiB.
BLOB_DISK_LIMIT = 32 * 1024**3

#: The largest project directory letify packs into an environment archive when it has no
#: measured rate to judge the transfer by. Spec "Materializing into a runtime": the pack
#: moves the archive as one payload the worker and the client each hold whole.
PACK_WITHOUT_A_RATE = 1024**3

#: Bytes per second a runtime downloads from Google Drive. The slower of two measurements,
#: 176.6 MiB/s from a Colab runtime and 182.9 MiB/s from a Kaggle runtime, for one 1 GiB
#: file. Spec "Materializing into a runtime".
DRIVE_DOWNLOAD_BPS = 176 * 1024**2

#: Bytes per second a runtime extracts an uncompressed tar onto its disk, measured at
#: 170 MB/s on a sandbox disk. Spec "Materializing into a runtime".
EXTRACT_BPS = 170_000_000

#: How many times the estimated restore a build must have taken before the environment is
#: packed to Drive. Spec "Materializing into a runtime": packing has a cost of its own.
DRIVE_PACK_FACTOR = 2

#: Seconds a closing session waits for a pack still running before it abandons it.
PACK_CLOSE_WAIT_S = 60.0

#: Seconds between the checks that hand a finished pack its upload session.
PACK_POLL_S = 2.0


def _against_the_build(size: int, rate: float, build_s: float) -> str:
    """The three numbers both cache decisions weigh, as they are reported.

    Spec "Materializing into a runtime": ``3.0 GB over 13.6 MiB/s is 210 s against a 34 s
    build``. Each decision prefixes its own clause.
    """
    return (
        f"{size / 1_000_000_000:.1f} GB over {rate / (1024 * 1024):.1f} MiB/s is "
        f"{size / rate:.0f} s against a {build_s:.0f} s build"
    )


def _reason_of(error: BaseException) -> str:
    """One line naming an error's type and first message line, for a status message."""
    lines = str(error).strip().splitlines()
    reason = f"{type(error).__name__}: {lines[0]}" if lines else type(error).__name__
    return reason[:197] + "..." if len(reason) > 200 else reason


def _sync_rather_than_restore(archive_bytes: int, rate: float, build_s: float) -> str:
    """The line explaining why a session syncs instead of restoring a cached archive.

    Spec "Materializing into a runtime". This returns the text after the ``<runtime>: ``
    prefix, which ``Runtime._say`` adds.
    """
    return (
        "syncing rather than restoring, because "
        f"{_against_the_build(archive_bytes, rate, build_s)}"
    )


@dataclass
class _DrivePack:
    """A pack the worker is running for a Drive store, as this process tracks it."""

    volume: Any
    began: float = field(default_factory=time.monotonic)
    target_sent: bool = False


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

    #: How the environment was built: ``"sync"``, ``"archive"`` (restored from a store),
    #: ``"workspace"`` (restored from the workspace root), or None where it was not.
    env_source: str | None = None

    #: Seconds the sync took, when ``env_source`` is ``"sync"``. Spec "Materializing into
    #: a runtime": weighed against the project directory's size over the measured link to
    #: decide whether packing an archive is worth it.
    env_build_s: float | None = None

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

    _pool_key: str | None = field(default=None, repr=False)

    #: The pool this runtime was started by, so it can be discarded from the lease thread.
    _pool: Any = field(default=None, repr=False)

    #: Set once ``shutdown`` has run, so a second discard does nothing.
    _shut_down: bool = field(default=False, repr=False)

    #: The environment pack running on the worker, or None. Spec "Materializing into a
    #: runtime": the pack to Drive runs beside the session's calls.
    _pack: _DrivePack | None = field(default=None, repr=False)
    _pack_lock: Any = field(default_factory=threading.RLock, repr=False)

    @property
    def key(self) -> str:
        """Pool key. Runtimes with equal keys are interchangeable."""
        return self._pool_key or f"{self.instance.key}|{self.env.key}"

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
        self.check_devices()
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
            self.shutdown(reason="interpreter_mismatch")
            raise
        # The same rule as _environment_volumes: a project directory that does not
        # survive the session is worth archiving whatever the provider calls itself.
        if self.env_source == "sync" and self._archives_on_workspace():
            self._pack_on_workspace()
        elif self.env_source == "sync" and not (
            self.provider.persistent and self._project_survives()
        ):
            from . import bootstrap

            root = bootstrap.project_dir(
                self.provider.env_root or self.workspace or self.provider.workspace_root, self.env
            )
            if self._worth_packing(root):
                size = self._tree_size(root)
                self._say(
                    "caching the environment"
                    + (f", {size / 1_000_000_000:.1f} GB" if size else "")
                )
                volume = self._environment_volumes()[0]
                if volume.store.backend.moves_on_runtime:
                    self._start_drive_pack(volume, root)
                else:
                    try:
                        volume.cache_env_from(
                            self, self.env, root, platform=self.platform, build_s=self.env_build_s
                        )
                    except Exception as error:
                        # The environment is built and checked, so a failed cache must not
                        # fail the session. Spec "Materializing into a runtime".
                        self._say(f"not caching the environment ({_reason_of(error)})")
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

    def lose(self, message: str) -> None:
        """Declare the runtime gone: wake a waiting call with ``RuntimeLost`` and discard it.

        Spec "Lifetime", the lease. Called from the lease thread when the provider says the
        runtime ended while a request on its channel can no longer be answered.
        """
        if self.lease is not None:
            self.lease.release()
        if self.channel is not None:
            self.channel.declare_lost(message)
        if self._pool is not None:
            self._pool.discard(self, reason="runtime_ended")
        else:
            self.shutdown(reason="runtime_ended")

    def shutdown(self, *, reason: str = "explicit_shutdown") -> None:
        """Stop everything that bills for this runtime and explain the decision."""
        if self._shut_down:
            return
        self._shut_down = True
        self._close_pack()
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        print(
            f"{timestamp} letify: discarding {self.name} provider={self.provider.alias} "
            f"key={self.key} busy={self.busy} reason={reason.split(':', 1)[0]} "
            f"timestamp={timestamp} detail={json.dumps(reason)}",
            file=sys.stderr,
            flush=True,
        )
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

    def stream(self, payload: dict[str, Any], *, timeout: float | None = None) -> Iterator[Any]:
        """Send one worker request answered by several replies, yielding each value."""
        if self.channel is None:
            raise RuntimeFailure(f"{self.name}: the channel is not open")
        self.last_used = time.monotonic()
        return self.channel.stream(payload, timeout=timeout)

    def pipeline(self, *, window: int, timeout: float | None = None) -> Any:
        """Open one stream carrying several worker requests in order, each answered in order."""
        if self.channel is None:
            raise RuntimeFailure(f"{self.name}: the channel is not open")
        self.last_used = time.monotonic()
        return self.channel.pipeline(window=window, timeout=timeout)

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
        data_order: Any = None,
        data_first_wave: int | None = None,
        replica: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        """Run one declared function inside this runtime.

        Large arguments are replaced by content addressed references so the same
        payload is not sent twice.
        """
        if self.channel is None:
            raise RuntimeFailure(f"{self.name}: the channel is not open")
        name = getattr(fn, "__qualname__", None) or repr(fn)
        if not self.persistent_channel:
            self.last_used = time.monotonic()
            # Spec "Controlling a running call": a one-shot call is listed too, so a command
            # against it can say why it is refused.
            run = control.begin(self, name)
            try:
                return self.channel.call(
                    fn, args, kwargs, timeout=timeout, modules=self.env.ship_modules
                )
            finally:
                run.end()
        args, kwargs = self._externalize(args, kwargs)
        self._poll_pack()
        try:
            return self._call_with_data(
                fn, args, kwargs, timeout, data_order, data_first_wave, replica, name
            )
        finally:
            self._poll_pack()

    def _call_with_data(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        timeout: float | None,
        data_order: Any = None,
        data_first_wave: int | None = None,
        replica: dict[str, Any] | None = None,
        name: str = "",
    ) -> tuple[Any, str]:
        """Pickle a call, place the first wave of its data, and send the rest beside it.

        Spec "Project data": the call carries the links and the manifest the worker needs,
        the first wave of the send order is placed before it goes, and a background sender
        pushes the rest while it runs.
        """
        import secrets

        from ..store import handoff, pathdata

        assert self.channel is not None
        root = (self.workspace or self.provider.workspace_root).rstrip("/")
        call_dir = f"{root}/data/calls/{secrets.token_hex(8)}"
        modules_root = f"{root}/modules"
        collector = pathdata.Collector(call_dir)
        ship_digests: set[str] = set()
        if self.env.ship_modules:
            for name in self.env.ship_modules:
                placed = collector.collect_module(name, modules_root)
                for _rel, digest, _size, _loc in placed.entries:
                    ship_digests.add(digest)
        head, buffers = protocol.dumps_call_parts(fn, args, kwargs, data=collector)
        request: dict[str, Any] = {"op": "call", "payload": head, "buffers": buffers}
        if replica is not None:
            # Spec "Replicas": the body reads its rank on the runtime, not here, so the
            # numbers travel with the call rather than being set on a client thread.
            request["replica"] = replica
        blobs = f"{root}/data/blobs"
        stream: pathdata.Stream | None = None
        if self.env.ship_modules:
            request["modules_dir"] = modules_root
        if collector.placed:
            extra_dirs = [modules_root] if self.env.ship_modules else ()
            request["data"] = collector.request(blobs, extra_dirs=extra_dirs)
            if collector.inputs:
                stream = self._stream_data(
                    collector, request["data"], blobs, fn, args, kwargs,
                    data_order, data_first_wave, ship_digests=ship_digests,
                )
        self.last_used = time.monotonic()
        outcome = None
        # Each output file comes back as soon as it is complete, while the call still
        # runs, so a long call's checkpoints do not wait for its return. Spec "Writing back".
        back = pathdata.WriteBack(self, collector, blobs) if collector.outputs else None
        if back is not None:
            back.start()
        # A function body can ask this process for a Drive grant while it runs, as spec
        # "Named handoffs on the Drive store" describes.
        grants = handoff.GrantWatcher(self.channel)
        grants.start()
        # The record exists before the call is sent and goes when it returns or raises.
        run = control.begin(self, name)
        try:
            with run.interruptible():
                outcome = self.channel.request(request, timeout=timeout)
        finally:
            run.end()
            grants.close()
            if back is not None:
                back.cancel()
            added = self._finish_data(stream, call_dir)
            try:
                if outcome is not None and collector.outputs:
                    added += pathdata.write_back(self, collector, blobs, back)
            finally:
                if added:
                    pathdata.evict(self, blobs)
        return outcome

    def _stream_data(
        self,
        collector: Any,
        data: dict[str, Any],
        blobs: str,
        fn: Any,
        args: tuple,
        kwargs: dict,
        data_order: Any,
        data_first_wave: int | None,
        ship_digests: set[str] | None = None,
    ) -> Any:
        """Derive the send order, place the first wave, and start the background sender."""
        from ..store import pathdata, sendorder

        option = self.provider.config.option
        declared = self._declared_order(fn, args, kwargs, data_order)
        order = sendorder.compute(collector, fn, args, kwargs, declared)
        plan = pathdata.prepare(self, collector, blobs, order)
        wave = sendorder.first_wave(
            plan.order,
            plan.sizes,
            set(plan.missing),
            mib=float(option("data_first_wave_mib", pathdata.FIRST_WAVE_MIB)),
            files=int(option("data_first_wave_files", pathdata.FIRST_WAVE_FILES)),
            count=data_first_wave,
        )
        if ship_digests:
            needed = [d for d in plan.missing if d in ship_digests and d not in wave]
            if needed:
                wave = needed + wave
        stream = pathdata.Stream(self, blobs, plan, collector.call_dir)
        if not plan.missing:
            # Nothing to stream: the runtime holds every file, so the call places them all
            # before the body starts. It needs no manifest, no patch and no observation.
            return stream
        data.update(
            {
                "manifest": sendorder.manifest(collector),
                "order": plan.order,
                "observe": bool(option("data_observe", True)),
                "prefetch": int(option("data_prefetch_batches", pathdata.PREFETCH_BATCHES)),
                "wait": float(option("data_wait_timeout", pathdata.WAIT_TIMEOUT_S)),
            }
        )
        stream.place(wave)
        stream.start()
        return stream

    @staticmethod
    def _declared_order(fn: Any, args: tuple, kwargs: dict, data_order: Any) -> Any:
        """``data_order`` from the declaration, with a callable given the bound arguments.

        Spec "The send order and the first wave". A callable that raises is the
        declaration's own error, so it is not caught here.
        """
        if data_order is None or not callable(data_order):
            return data_order
        import inspect

        try:
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            return data_order(dict(kwargs))
        return data_order(dict(bound.arguments))

    def _finish_data(self, stream: Any, call_dir: str) -> int:
        """Cancel the background transfer and print the call's data log line."""
        if stream is None:
            return 0
        stream.cancel()
        if not stream.plan.missing:
            # Nothing was sent and nothing could have waited, so there is nothing to ask for.
            stream.log(None)
            return 0
        try:
            stats = self.request({"op": "data_stats", "dir": call_dir}, timeout=120)
        except (RuntimeFailure, RemoteError, ProtocolError):
            stats = None
        stream.log(stats)
        return stream.sent

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
        if not self.persistent_channel:
            from . import bootstrap

            value = self.eval(
                bootstrap.pull_source(source, path, unpack=unpack, target=target, links=links),
                timeout=3600,
            )
        else:
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

    def pack_dir_to_file(self, path: str, target: Path) -> tuple[str, int]:
        """Pack a directory inside the runtime and stream the archive into a local file.

        Spec "Materializing into a runtime". The runtime writes the archive into its blob
        cache while hashing it, and the client receives it as a stream with the digest
        checked on arrival, so neither side holds the archive whole. Returns
        ``(digest, size)``. Needs a persistent channel.
        """
        from ..store import pathdata

        staging = f"{posixpath.dirname(path.rstrip('/'))}/.letify-pack"
        value = self.request(
            {"op": "pack_blob", "path": path, "dir": staging}, timeout=3600
        )
        digest, size, directory = value["digest"], int(value["size"]), value["dir"]
        meter = pathdata._Downloaded(1, size)
        try:
            pathdata._get(self, directory, digest, target, size, meter)
            meter.finish()
        finally:
            self.exec(
                f"import shutil\nshutil.rmtree({directory!r}, ignore_errors=True)\n",
                timeout=120,
            )
        return digest, size

    def put_path(
        self,
        local: Path,
        path: str,
        *,
        digest: str,
        unpack: bool = False,
        target: str | None = None,
        links: bool = False,
    ) -> protocol.RemoteFile:
        """Send a local file whose digest is known in pieces, then place it at ``path``.

        Spec "Materializing into a runtime". The file is read one piece at a time, so the
        client never holds it whole. The runtime verifies the digest as the pieces arrive.
        """
        from ..store import pathdata

        size = local.stat().st_size
        staging = f"{path}.letify-send"
        meter = pathdata._Uploaded(1, size)
        pipeline = self.pipeline(window=pathdata.WINDOW, timeout=3600)
        try:
            pathdata._put(self, pipeline, staging, digest, str(local), size, meter)
            pipeline.drain()
        finally:
            pipeline.close()
        meter.finish()
        value = self.request(
            {
                "op": "place_blob",
                "dir": staging,
                "digest": digest,
                "path": path,
                "unpack": unpack,
                "target": target,
                "links": links,
            },
            timeout=3600,
        )
        return protocol.RemoteFile(path=value["path"], digest=digest, size=value["size"])

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

    def _archives_on_workspace(self) -> bool:
        """Whether the environment archive lives on the runtime's own workspace root.

        Spec "Volumes on a persistent runtime": true for a provider whose workspace root
        persists and whose project directory lies outside it.
        """
        return bool(
            self.provider.remote_env and self.provider.persistent and not self._project_survives()
        )

    def _workspace_archive(self) -> str:
        """The archive path on the workspace root for this environment and platform."""
        workspace = (self.workspace or self.provider.workspace_root).rstrip("/")
        return f"{workspace}/env-archives/{self.env.key}-{self.platform}.tar"

    def _pack_on_workspace(self) -> None:
        """Tar the project directory onto the workspace root; nothing goes to the client."""
        from . import bootstrap

        root = bootstrap.project_dir(
            self.provider.env_root or self.workspace or self.provider.workspace_root, self.env
        )
        size = self._tree_size(root)
        self._say(
            "caching the environment on the workspace root"
            + (f", {size / 1_000_000_000:.1f} GB" if size else "")
        )
        source = bootstrap.workspace_pack_source(root, self._workspace_archive())
        try:
            self.eval(source, timeout=3600)
        except Exception as error:
            # The environment is built and checked, so a failed cache must not fail the call.
            self._say(f"not caching the environment on the workspace root ({_reason_of(error)})")

    def _environment_volumes(self) -> tuple[Volume, ...]:
        """Explicit archive volumes, or the automatic cache where the project does not survive.

        Spec "Volumes on a persistent runtime": a persistent provider needs no archive
        because its project directory is on a disk that lasts, which holds only while that
        directory is under the workspace root. One that keeps its workspace and builds the
        .venv elsewhere archives like an ephemeral one, because the sync's output dies with
        the session however durable the root is.
        """
        if self.provider.persistent and self._project_survives():
            return ()
        if self._archives_on_workspace():
            # Spec "Volumes on a persistent runtime": the archive is on the workspace root.
            return ()
        if self.volumes:
            return self.volumes
        from ..store.volume import environment_volume

        return (environment_volume(self.provider),)

    def _project_survives(self) -> bool:
        """Whether the directory the sync writes lives on the disk the workspace root is on."""
        root = (self.provider.env_root or "").rstrip("/")
        if not root:
            return True
        workspace = (self.workspace or self.provider.workspace_root or "").rstrip("/")
        return bool(workspace) and (root == workspace or root.startswith(workspace + "/"))

    def _measured_rate(self) -> float | None:
        """This session's measured upload rate in bytes per second, or None where none was.

        Follows how ``pathdata._streams_for`` reaches the link from a runtime: a provider
        with no connection to measure, such as ``Local``, raises reaching for one, and that
        is read the same as a rate of zero. Spec "Materializing into a runtime".
        """
        try:
            link = self.provider.link(self)
        except Exception:
            return None
        rate = getattr(link, "upload_bps", None)
        if not rate or rate <= 0:
            return None
        return float(rate)

    def _tree_size(self, root: str) -> int | None:
        """The byte size of the project directory on the runtime, or None when unknown."""
        from . import bootstrap

        try:
            size = self.eval(bootstrap.tree_size_source(root), timeout=120)
        except Exception:
            return None
        return size if isinstance(size, int) and size > 0 else None

    def _packs_streaming(self) -> bool:
        """Whether the archive of the first environment volume streams in both directions."""
        volumes = self._environment_volumes()
        if not volumes or not self.persistent_channel:
            return False
        return volumes[0].store.streams or volumes[0].store.backend.moves_on_runtime

    def _worth_packing(self, root: str) -> bool:
        """Whether moving the project directory over this session's link beats the build.

        Spec "Materializing into a runtime": the directory is larger than the archive it
        would become, so this overstates the transfer and errs toward building. With no
        measured rate it packs any size when the archive streams, and otherwise only what
        letify is willing to hold in memory, because that pack moves the archive as one
        payload the worker and the client each hold whole.
        """
        if self.env_build_s is None:
            return True
        volumes = self._environment_volumes()
        if volumes and volumes[0].store.backend.moves_on_runtime:
            return self._worth_packing_to_drive(root)
        rate = self._measured_rate()
        size = self._tree_size(root)
        if rate is None:
            if size is None or size <= PACK_WITHOUT_A_RATE or self._packs_streaming():
                return True
            self._say(
                f"not caching the environment, because {size / 1_000_000_000:.1f} GB is over "
                f"the {PACK_WITHOUT_A_RATE / 1024**3:.1f} GiB letify will hold in memory and "
                f"this link was never measured"
            )
            return False
        if size is None:
            return True
        if (size / rate) < self.env_build_s:
            return True
        self._say(
            "not caching the environment, because "
            f"{_against_the_build(size, rate, self.env_build_s)}"
        )
        return False

    def _worth_packing_to_drive(self, root: str) -> bool:
        """Whether the build took at least twice as long as restoring from Drive would.

        Spec "Materializing into a runtime": the archive goes from the runtime to Drive and
        never over this link, so the gate is the estimated restore, the project directory's
        size over the Drive download rate plus the same size over the extraction rate. The
        build must take ``DRIVE_PACK_FACTOR`` times that, because the pack has a cost of its
        own that the comparison leaves out.
        """
        size = self._tree_size(root)
        if size is None or self.env_build_s is None:
            return True
        restore_s = size / DRIVE_DOWNLOAD_BPS + size / EXTRACT_BPS
        if self.env_build_s >= DRIVE_PACK_FACTOR * restore_s:
            return True
        self._say(
            f"not caching the environment, because it built in {self.env_build_s:.0f} s "
            f"and would take an estimated {restore_s:.0f} s to restore from Drive, and a "
            f"cache pays only when the build takes at least {DRIVE_PACK_FACTOR} times as long"
        )
        return False

    # -- the pack that runs beside the calls ---------------------------------

    def _start_drive_pack(self, volume: Volume, root: str) -> None:
        """Have the worker pack the environment and upload it on a thread of its own.

        Spec "Materializing into a runtime": this returns at once, so the first call does
        not wait for the pack. The ref and the build record are written when a poll sees
        the upload finish. A failure to start leaves nothing cached and says so.
        """
        staging = f"{posixpath.dirname(root.rstrip('/'))}/.letify-pack"
        try:
            if not self.persistent_channel:
                raise RuntimeFailure("this channel keeps no worker that could upload the archive")
            self.request(
                {"op": "pack_push_start", "path": root, "dir": staging}, timeout=120
            )
        except Exception as exc:
            self._not_using_drive(exc, after="not caching the environment")
            return
        self._pack = _DrivePack(volume)
        threading.Thread(target=self._feed_pack, name="letify-pack", daemon=True).start()

    def _feed_pack(self) -> None:
        """Hand the pack its upload session soon after the archive is written.

        A call can run for hours, and the upload cannot start until this process has made
        the session, so this does only that. Reporting stays with the calls and the close.
        """
        while True:
            time.sleep(PACK_POLL_S)
            job = self._pack
            if job is None or job.target_sent or self._shut_down:
                return
            self._poll_pack(report=False)

    def _poll_pack(self, *, report: bool = True) -> bool:
        """Ask the worker how the pack is doing and act on the answer.

        A packed archive is given its upload session. A finished upload is recorded, and
        reported when ``report`` is set. False when the worker could not be asked.
        """
        with self._pack_lock:
            job = self._pack
            if job is None or self.channel is None:
                return True
            try:
                state = self.request({"op": "pack_push_state"}, timeout=30)
            except Exception:
                return False
            kind = state.get("state")
            backend = job.volume.store.backend
            if kind == "packed" and not job.target_sent:
                job.target_sent = True
                try:
                    if backend.has(state["digest"]):
                        # Another session stored this archive first, so there is nothing to send.
                        self._send_target(None)
                        self._pack = None
                        self._record_pack(job, state["digest"], int(state["size"]))
                        return True
                    target = backend.begin_upload(state["digest"], int(state["size"]))
                    if target is None:
                        raise RuntimeFailure(f"the {backend.name} backend offered no upload")
                    self._send_target(target)
                except Exception as exc:
                    self._pack = None
                    try:
                        self._send_target(None)
                    except Exception:
                        pass
                    self._not_using_drive(exc, after="not caching the environment")
            elif kind == "done" and report:
                self._pack = None
                backend.record_upload(state["digest"], str(state["id"]))
                self._record_pack(job, state["digest"], int(state["size"]))
                seconds = float(state["seconds"])
                size = int(state["size"])
                self._say(
                    f"cached the environment, {size / 1_000_000_000:.1f} GB in {seconds:.0f} s "
                    f"({size / 1024**2 / max(seconds, 0.001):.1f} MiB/s)"
                )
            elif kind == "failed" and report:
                self._pack = None
                self._not_using_drive(
                    RuntimeFailure(str(state.get("reason"))), after="not caching the environment"
                )
            elif kind == "none":
                self._pack = None
            return True

    def _send_target(self, target: dict[str, Any] | None) -> None:
        """Give the worker the upload session, or None to abandon the pack."""
        payload: dict[str, Any] = {"op": "pack_push_target", "url": None}
        if target is not None:
            payload.update(
                url=target["url"],
                headers=dict(target.get("headers") or {}),  # type: ignore[call-overload]
                piece=target.get("piece"),
            )
        self.request(payload, timeout=60)

    def _record_pack(self, job: _DrivePack, digest: str, size: int) -> None:
        """Write the ref and the build record, which only a finished upload earns."""
        job.volume.record_env(
            self.env, self.platform, digest, size, build_s=self.env_build_s
        )

    def _close_pack(self) -> None:
        """Wait for a running pack, up to ``PACK_CLOSE_WAIT_S``, then abandon it.

        Spec "Materializing into a runtime": an abandoned pack records nothing, so the
        next session syncs and may pack again.
        """
        job = self._pack
        if job is None:
            return
        deadline = time.monotonic() + PACK_CLOSE_WAIT_S
        try:
            while self._pack is not None:
                if not self._poll_pack():
                    self._pack = None
                    return
                if self._pack is None:
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            with self._pack_lock:
                self._pack = None
                try:
                    self._send_target(None)
                except Exception:
                    pass
            self._say(
                "not caching the environment, because the session closed "
                f"{time.monotonic() - job.began:.0f} s into the pack"
            )
        except Exception:
            self._pack = None

    def _restore_from(self, volume: Volume, where: dict[str, Any]) -> bool:
        """Restore the environment from one volume's archive. True when the check passed.

        Spec "Materializing into a runtime": a volume with no archive, or one not worth
        restoring over this session's link, answers False and leaves the sync to build.
        """
        from . import bootstrap

        digest = volume.cached_env(self.env, self.platform)
        if not digest:
            return False
        build = volume.cached_build(self.env, self.platform)
        if build is None:
            # Spec "Materializing into a runtime": written before letify recorded the
            # build, so it is exactly the kind this rule exists to catch.
            self._say(
                "syncing rather than restoring, because the archive carries no "
                "build time to compare against"
            )
            return False
        build_s, archive_bytes = build
        on_drive = volume.store.backend.moves_on_runtime
        restore_s = volume.cached_restore(self.env, self.platform) if on_drive else None
        if restore_s is not None and restore_s >= build_s:
            # Spec "Materializing into a runtime": a restore measured slower than the build
            # is not worth repeating, so this session builds.
            self._say(
                "syncing rather than restoring from Drive, because a restore took "
                f"{restore_s:.0f} s against a {build_s:.0f} s build"
            )
            return False
        rate = None if on_drive else self._measured_rate()
        if rate is not None and (archive_bytes / rate) >= build_s:
            # Spec "Materializing into a runtime": the cache is an optimization, so a
            # link this slow would cost more than the build did.
            self._say(_sync_rather_than_restore(archive_bytes, rate, build_s))
            return False
        self._say(f"restoring the environment, cached as {digest[:12]}")
        archive = f"{where['parent']}/.{digest}" + (".tar" if on_drive else ".tar.gz")
        began = time.monotonic()
        volume.materialize(
            self, digest, path=archive, unpack=True, target=where["parent"], links=True
        )
        if self.eval(bootstrap.venv_check_source(where["python"], archive), timeout=600):
            self.env_source = "archive"
            if on_drive:
                volume.record_restore(self.env, self.platform, time.monotonic() - began)
            return True
        return False

    def _not_using_drive(self, failure: Exception, *, after: str = "syncing") -> None:
        """Say that a Drive request failed and what the session does instead.

        Spec "Materializing into a runtime": a failed Drive request is not retried into a
        slower path, and the archive is never routed through this process. A failed
        restore syncs; a failed pack, which comes after the sync, leaves nothing cached.
        """
        reason = " ".join(str(failure).split()) or type(failure).__name__
        self._say(f"not using the Drive store ({reason[:300]}); {after}")

    def _say(self, message: str) -> None:
        """One decision line about this runtime, as the transport's lines are written.

        Spec "Environment": the build says it is happening. Silent when the launcher was
        built with ``announce=False``.
        """
        from ..transport.announce import printer

        # On the instance dict, not getattr: a provider's __getattr__ answers accelerator
        # names and raises UnknownInstance for anything else, which no default catches.
        announce = self.provider.__dict__.get("announce", True)
        printer(bool(announce))(f"{self.name}: {message}")

    def install_env(self) -> None:
        """Build the project's environment in the session and move the worker onto it.

        An archive cached in a volume is preferred, because unpacking one file is a
        single transfer while syncing from a lock file is thousands of small ones. A
        provider whose account names ``python`` manages its own interpreter, so nothing
        is built there.
        """
        if not self.provider.remote_env:
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

        # Spec "Volumes on a persistent runtime": a provider whose workspace root persists
        # and whose project directory does not keeps the archive on the workspace root, and
        # one whose project directory survives needs no archive at all.
        on_workspace = self._archives_on_workspace()
        found = f"os.path.isfile(os.path.expanduser({self._workspace_archive()!r}))"
        archive_here = on_workspace and self.eval(
            f"import os\n__letify_value__ = {found}\n", timeout=120
        )
        if archive_here:
            self._say("restoring the environment from the workspace root")
            self.eval(
                bootstrap.workspace_restore_source(self._workspace_archive(), where["parent"]),
                timeout=3600,
            )
            if self.eval(bootstrap.venv_check_source(where["python"]), timeout=600):
                self.env_source = "workspace"
            else:
                self._say("the restored environment failed its interpreter check, syncing")
                self.eval(
                    "import os, shutil\n"
                    f"shutil.rmtree(os.path.expanduser({root!r}), ignore_errors=True)\n",
                    timeout=600,
                )

        archives = self._environment_volumes()
        for volume in archives:
            try:
                if self._restore_from(volume, where):
                    break
            except Exception as exc:
                if not volume.store.backend.moves_on_runtime:
                    raise
                self._not_using_drive(exc)

        if self.env_source not in ("archive", "workspace"):
            workspace = self.workspace or self.provider.workspace_root
            # Spec "Remote paths letify writes": uv's cache, its managed Pythons, its tool
            # installs and its own binary stay under the workspace root regardless of an
            # env root or whether the provider is persistent or ephemeral.
            source = bootstrap.sync_source(
                self.env, files, root=root, name=self.name, workspace=workspace,
                archive=bool(archives) or on_workspace,
            )
            # Spec "Environment": the longest step of a first session, and uv's own output
            # is captured rather than streamed, so it says it is happening.
            self._say(
                f"building the environment, {bootstrap.packages_in(files)} packages "
                f"from {self.env.lock}"
            )
            began = time.monotonic()
            try:
                self.eval(source, timeout=3600)
            except RemoteError as exc:
                message = str(exc).split("\n\n--- remote traceback ---")[0]
                raise EnvironmentFailure(
                    message.removeprefix("RuntimeError: "), stderr=exc.remote_traceback
                ) from exc
            self.env_build_s = time.monotonic() - began
            self._say(f"environment built in {self.env_build_s:.1f} s")
            self.env_source = "sync"
        self.python = where["python"]
        self.channel.switch_interpreter(where["python"])

    def check_devices(self) -> None:
        """Refuse a session whose cards are not the ones the instance asked for.

        Spec "Kaggle", the accelerators: a provider may answer a request with a different
        card. Run before the environment is built, so a wrong card costs seconds rather
        than the minutes a sync takes.
        """
        from ..errors import InsufficientDevices
        from . import bootstrap

        instance = getattr(self, "instance", None)
        expected = self.provider.expected_cards(instance)
        if expected is None:
            return
        fragment, count = expected
        try:
            found = self.eval(bootstrap.CARD_SOURCE, timeout=120)
        except Exception:
            # Reading /proc is not worth failing a session over when it cannot be read.
            return
        if not isinstance(found, dict):
            found = {"cards": found, "driver": True, "libcuda": True, "cuinit": 0}
        cards = [str(card) for card in found.get("cards") or ()]
        driver = bool(found.get("driver"))
        libcuda = bool(found.get("libcuda", True))
        cuinit = found.get("cuinit")
        started = libcuda and cuinit == 0
        matched = [card for card in cards if fragment.casefold() in card.casefold()]
        if len(matched) == count and len(cards) == count and driver and started:
            return
        asked = getattr(instance, "accelerator", None) or fragment
        arrived = ", ".join(cards) if cards else "no GPU at all"
        if cards and not driver:
            # Spec "Kaggle": the cards are right and the driver cannot be used, which the
            # first .to("cuda") would have reported minutes into the call instead.
            arrived = (
                f"{arrived}, and no usable driver: /dev/nvidiactl is not there, so CUDA "
                f"cannot start"
            )
        elif cards and not libcuda:
            # Spec "Kaggle": the userspace driver library is not there at all, which is a
            # different fault from CUDA refusing to start, and reads as a GPU session on a
            # CPU image rather than as a CUDA error.
            arrived = (
                f"{arrived}, and no libcuda.so.1 to load: the cards are attached and the "
                f"userspace driver library is missing, so nothing on this session can use "
                f"them"
            )
        elif cards and not started:
            arrived = (
                f"{arrived}, and a driver that does not start: libcuda.so.1 loaded and "
                f"cuInit returned {cuinit}, which is the call CUDA initialization makes, "
                f"so nothing on this session can use those cards"
            )
        message = (
            f"{self.name}: {self.provider.alias} was asked for {count} x {asked} and the "
            f"session has {arrived}. A declaration says where a function runs, so the "
            f"session is not used. Set any_accelerator = true on the account to accept "
            f"whatever card it is given"
        )
        if self.provider.any_accelerator:
            self._say(f"card mismatch accepted: asked {count} x {asked}, got {arrived}")
            return
        raise InsufficientDevices(message)

    def check_interpreter(self) -> None:
        """Refuse a worker whose Python major.minor differs from this process."""
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
    """A ``put_blob`` message for a value: its protocol 5 pickle and out-of-band buffers.

    Pickled with ``codec.pickle_value_parts``, cloudpickle underneath, so a class or
    function the caller defined locally ships by value instead of by name, the same as
    it does when it is small enough to travel inlined in the call payload.
    """
    head, buffers = protocol.codec.pickle_value_parts(value)
    return {"kind": "pickle", "immutable": immutable, "head": head, "buffers": buffers}


def _payload(value: Any) -> bytes:
    """File bytes from a reply: raw from a binary channel, base64 text from a one-shot one."""
    if isinstance(value, str):
        import base64

        return base64.b64decode(value)
    return bytes(value)


__all__ = ["Runtime"]
