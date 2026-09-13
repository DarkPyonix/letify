"""Volume, one named content addressed store on a provider.

A volume is what makes an ephemeral provider behave like a persistent one. With one
attached, the environment archive and the model cache come from storage that outlives
the runtime instead of being rebuilt from their origin.

The numbers behind that. A twenty gigabyte model cache takes about 27 minutes to pull
from a lab server over a 100 Mbit/s link, three to five minutes from the Hugging Face
hub, and 40 to 60 seconds from a bucket in the same infrastructure as the runtime. All
of that time is billed as GPU time, which is why a cache tier is not optional for short
sessions.

Materializing goes through the runtime's channel rather than asking the runtime to
reach the bucket itself. That works with every backend and needs no credentials on the
far side, at the cost of the bytes passing through this process. A provider whose
runtime can read the bucket directly should override that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cas import Store

if TYPE_CHECKING:
    from ..declare.env import Env
    from ..protocol.handle import RemoteFile
    from ..providers.base import Provider
    from ..runtime.session import Runtime

#: Where a runtime keeps the files a volume materializes.
DEFAULT_MOUNT = "/opt/letify"

#: Ref names letify itself uses. The rest of the namespace belongs to the user.
ENV_REF = "env/{key}"
CHECKPOINT_REF = "ckpt/{name}"


@dataclass
class Volume:
    """A named store on one provider."""

    provider: Provider
    name: str
    options: dict[str, Any] = field(default_factory=dict)
    _store: Store | None = field(default=None, repr=False)

    @property
    def mount(self) -> str:
        return str(self.options.get("mount", DEFAULT_MOUNT))

    @property
    def key(self) -> str:
        return f"{self.provider.alias}/{self.name}"

    @property
    def store(self) -> Store:
        """Build the backend on first use, not at declaration time."""
        if self._store is None:
            from . import backends

            backend = str(self.options.get("backend") or self.provider.store_backend())
            options = {
                key: value for key, value in self.options.items() if key not in ("backend", "mount")
            }
            option, value = backends.default_location(backend, self.name)
            options.setdefault(option, value)
            self._store = Store(backends.build(backend, **options))
        return self._store

    # -- environments --------------------------------------------------------

    def env_ref(self, env: Env) -> str:
        return ENV_REF.format(key=env.key)

    def cached_env(self, env: Env) -> str | None:
        """Digest of the prebuilt environment archive, if one was stored."""
        return self.store.resolve(self.env_ref(env))

    def cache_env(self, env: Env, root: str | Path) -> str:
        """Pack an installed environment and remember it under its env key.

        One archive replaces tens of thousands of file transfers. The key is the hash of
        the lock file, so the same declaration reuses the same archive and a changed
        lock file builds a new one.
        """
        return self.store.put_tree(root, key=self.env_ref(env)).digest

    def cache_env_from(self, runtime: Runtime, env: Env, path: str) -> str:
        """Pack an environment that was installed inside a runtime and store it.

        This is how the first session pays the installation cost and every later one
        skips it.
        """
        payload, _digest = runtime.pack_dir(path)
        info = self.store.put_bytes(payload)
        self.store.point(self.env_ref(env), info.digest)
        return info.digest

    # -- checkpoints ---------------------------------------------------------

    def latest_checkpoint(self, name: str) -> str | None:
        """Digest of the newest checkpoint written under this name."""
        return self.store.resolve(CHECKPOINT_REF.format(name=name))

    def put_checkpoint(self, name: str, path: str | Path) -> str:
        """Store a checkpoint from this machine and move the name to point at it.

        The blob is immutable and the ref is a few dozen bytes, so two runtimes writing
        at the same time cannot lose each other's data. The worst case is that one name
        wins, and both blobs remain.
        """
        source = Path(path)
        info = self.store.put_tree(source) if source.is_dir() else self.store.put_file(source)
        self.store.point(CHECKPOINT_REF.format(name=name), info.digest)
        return info.digest

    def absorb(self, runtime: Runtime, path: str, name: str) -> str:
        """Pull a checkpoint out of a runtime and store it under a name."""
        payload, _digest = runtime.get_bytes(path)
        info = self.store.put_bytes(payload)
        self.store.point(CHECKPOINT_REF.format(name=name), info.digest)
        return info.digest

    def fetch_checkpoint(self, name: str, target: str | Path) -> Path | None:
        """Materialize the newest checkpoint for a name onto this machine."""
        digest = self.latest_checkpoint(name)
        if digest is None:
            return None
        destination = Path(target)
        if destination.suffix:
            return self.store.fetch_file(digest, destination)
        return self.store.fetch_tree(digest, destination)

    # -- materializing into a runtime ----------------------------------------

    def materialize(
        self,
        runtime: Runtime,
        digest: str,
        *,
        path: str | None = None,
        unpack: bool = False,
    ) -> RemoteFile:
        """Write a blob into the runtime, optionally unpacking it at the mount."""
        payload = self.store.get_bytes(digest)
        target = path or f"{self.mount.rstrip('/')}/blobs/{digest[:2]}/{digest}"
        return runtime.put_bytes(payload, target, unpack=unpack, target=self.mount)

    def materialize_ref(
        self, runtime: Runtime, ref: str, *, unpack: bool = False
    ) -> RemoteFile | None:
        """Materialize whatever a name currently points at."""
        digest = self.store.resolve(ref)
        if digest is None:
            return None
        return self.materialize(runtime, digest, unpack=unpack)

    def resume(self, runtime: Runtime, name: str, path: str) -> str | None:
        """Put the newest checkpoint for a name inside the runtime at ``path``.

        This is what makes a preempted session cheap to restart: the function asks for
        its own checkpoint and finds it already on disk.
        """
        digest = self.latest_checkpoint(name)
        if digest is None:
            return None
        self.materialize(runtime, digest, path=path)
        return digest

    def __repr__(self) -> str:
        return f"<Volume {self.key} on {self.provider.store_backend()}>"


__all__ = ["CHECKPOINT_REF", "DEFAULT_MOUNT", "ENV_REF", "Volume"]
