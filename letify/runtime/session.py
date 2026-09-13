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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import protocol
from ..errors import RuntimeFailure
from .lease import Lease

if TYPE_CHECKING:
    from ..declare.env import Env
    from ..declare.instance import Instance
    from ..providers.base import Provider
    from ..store.volume import Volume
    from .channel import Channel


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

    started: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    ready: bool = False
    busy: bool = False

    #: How long this session lives, from the declaration that started it.
    lease: Lease | None = field(default=None, repr=False)

    #: Digests this runtime is known to hold, so an argument is sent once.
    _blobs: set[str] = field(default_factory=set, repr=False)

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
        """Open the channel, install the environment, attach volumes, arm the lease."""
        if self.channel is None:
            self.channel = self.provider.open_channel(self)
        self.channel.start()
        if self.provider.needs_lease:
            self.lease = Lease(self)
            self.lease.arm()
        self.install_env()
        for volume in self.volumes:
            self.attach(volume)
        self.ready = True

    def shutdown(self) -> None:
        """Stop everything that bills for this runtime."""
        self.ready = False
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

    def stat(self) -> dict[str, Any]:
        """What the worker is holding: objects, blobs, bytes, process id."""
        return self.request({"op": "stat"}, timeout=60)

    def call(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        *,
        keep_remote: bool = False,
        timeout: float | None = None,
    ) -> tuple[Any, str]:
        """Run one declared function inside this runtime.

        Handles belonging to another runtime are rejected before anything is sent,
        and large arguments are replaced by content addressed references so the
        same payload is not sent twice.
        """
        if self.channel is None:
            raise RuntimeFailure(f"{self.name}: the channel is not open")
        # A handle names the live runtime that holds the object, not the pool key,
        # because two runtimes can share a key and an object lives in only one of them.
        protocol.check_handles(self.name, args, kwargs)
        # Done here rather than at declaration time, because the registration lives in
        # cloudpickle and a process can hold declarations with different environments.
        if self.env.ship_modules:
            protocol.codec.ship_by_value(self.env.ship_modules)
        if self.persistent_channel:
            args, kwargs = self._externalize(args, kwargs)
        self.last_used = time.monotonic()
        return self.channel.call(fn, args, kwargs, keep_remote=keep_remote, timeout=timeout)

    # -- content addressed arguments -----------------------------------------

    def _externalize(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Replace large arguments with blob references, uploading what is missing.

        Hashing is effectively free next to the network: blake3 runs at gigabytes
        per second where an uplink runs at megabytes per second. So every large
        argument is hashed, the runtime is asked which digests it already holds,
        and only the rest is sent.
        """
        import pickle

        plan: dict[str, bytes] = {}

        def convert(value: Any) -> Any:
            if isinstance(value, (protocol.Handle, protocol.Blob)):
                return value
            try:
                payload = pickle.dumps(value, protocol=5)
            except Exception:
                # Not plainly picklable, so leave it for cloudpickle to carry.
                return value
            if len(payload) < protocol.INLINE_LIMIT:
                return value
            digest = protocol.digest_of(payload)
            plan[digest] = payload
            return protocol.Blob(digest=digest, size=len(payload))

        new_args = tuple(convert(v) for v in args)
        new_kwargs = {k: convert(v) for k, v in kwargs.items()}
        if plan:
            self._upload_blobs(plan)
        return new_args, new_kwargs

    def _upload_blobs(self, plan: dict[str, bytes]) -> None:
        import base64

        unknown = [d for d in plan if d not in self._blobs]
        if unknown:
            held = self.request({"op": "have", "digests": unknown}, timeout=120) or []
            self._blobs.update(held)
            unknown = [d for d in unknown if d not in self._blobs]
        for digest in unknown:
            self.request(
                {
                    "op": "put_blob",
                    "digest": digest,
                    "payload": base64.b64encode(plan[digest]).decode(),
                },
                timeout=3600,
            )
            self._blobs.add(digest)

    # -- files ---------------------------------------------------------------

    def put_bytes(
        self, payload: bytes, path: str, *, unpack: bool = False, target: str | None = None
    ) -> protocol.RemoteFile:
        """Write bytes to a path inside the runtime, optionally unpacking an archive."""
        import base64

        value = self.request(
            {
                "op": "put_file",
                "path": path,
                "payload": base64.b64encode(payload).decode(),
                "unpack": unpack,
                "target": target,
            },
            timeout=3600,
        )
        return protocol.RemoteFile(
            path=value["path"], digest=protocol.digest_of(payload), size=value["size"]
        )

    def put_file(self, local: str | Path, path: str, **kwargs: Any) -> protocol.RemoteFile:
        return self.put_bytes(Path(local).read_bytes(), path, **kwargs)

    def get_bytes(self, path: str) -> tuple[bytes, str]:
        """Read a file out of the runtime, returning ``(payload, digest)``."""
        import base64

        value = self.request({"op": "get_file", "path": path}, timeout=3600)
        return base64.b64decode(value["payload"]), value["digest"]

    def pack_dir(self, path: str) -> tuple[bytes, str]:
        """Pack a directory inside the runtime into one archive payload."""
        import base64

        value = self.request({"op": "pack_dir", "path": path}, timeout=3600)
        return base64.b64decode(value["payload"]), value["digest"]

    # -- environment and volumes ---------------------------------------------

    def install_env(self) -> None:
        """Bring the declared environment up inside the session.

        An archive cached in a volume is preferred, because unpacking one file is a
        single transfer while installing from a lock file is thousands of small
        ones.
        """
        if not self.provider.prepares_env:
            return
        from .bootstrap import env_archive_path, install_source

        for volume in self.volumes:
            digest = volume.cached_env(self.env)
            if not digest:
                continue
            payload = volume.store.get_bytes(digest)
            self.put_bytes(
                payload,
                env_archive_path(volume.mount, digest),
                unpack=True,
                target=volume.mount,
            )
            self.exec(f"import sys; sys.path.insert(0, {volume.mount!r})", timeout=120)
            return
        self.exec(install_source(self.env), timeout=3600)

    def attach(self, volume: Volume) -> None:
        """Make a volume's mount point exist and apply its environment variables."""
        self.exec(
            "import os, pathlib\n"
            f"pathlib.Path({volume.mount!r}).mkdir(parents=True, exist_ok=True)\n",
            timeout=120,
        )

    def __repr__(self) -> str:
        state = "ready" if self.ready else "starting"
        return f"<Runtime {self.name} {self.instance.accelerator} {state}>"


__all__ = ["Runtime"]
