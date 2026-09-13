"""Volume, one named content addressed store on a provider.

A volume is what makes an ephemeral provider behave like a persistent one. With one
attached, the environment archive and the model cache come from storage that outlives
the runtime instead of being rebuilt from their origin.

The numbers behind that. A twenty gigabyte model cache takes about 27 minutes to pull
from a lab server over a 100 Mbit/s link, three to five minutes from the Hugging Face
hub, and 40 to 60 seconds from a bucket in the same infrastructure as the runtime. All
of that time is billed as GPU time, which is why a cache tier is not optional for short
sessions.

Materializing has the runtime pull straight from the backend, with a short-lived read
token this process derives from its own login and sends over the channel, so the bytes do
not pass through this process. A backend the runtime has no network path to offers no
pull, and then the bytes go through the channel. Writes from this process, such as
``absorb`` and ``cache_env_from``, go to the backend directly.
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

#: Ref names letify itself uses. The rest of the namespace belongs to the user.
ENV_REF = "env/{key}"
CHECKPOINT_REF = "ckpt/{name}"


def _session(target: Any) -> Runtime:
    """The live session behind a declaration, or a session given directly.

    Both answer ``session()``, so this is one call rather than a type check. Anything else
    is a mistake worth naming, because the alternative is an attribute error from inside a
    transfer.
    """
    asked = getattr(target, "session", None)
    if not callable(asked):
        raise TypeError(
            f"expected a declared function, not {type(target).__name__}. Pass the "
            f"declaration whose calls wrote the file, and letify finds the session."
        )
    return asked()


@dataclass
class Volume:
    """A named store on one provider."""

    provider: Provider
    name: str
    options: dict[str, Any] = field(default_factory=dict)
    _store: Store | None = field(default=None, repr=False)

    @property
    def mount(self) -> str:
        """The volume directory, ``<workspace root>/volumes/<name>``, before ``~`` is expanded.

        The ``mount`` option names another directory for this volume.
        """
        return self.directory(None)

    def directory(self, runtime: Runtime | None) -> str:
        """The volume directory, under the root a booted runtime expanded when there is one."""
        configured = self.options.get("mount")
        if configured:
            return str(configured)
        root = getattr(runtime, "workspace", None) or self.provider.workspace_root
        return f"{root.rstrip('/')}/volumes/{self.name}"

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
            if backend == "modal" and self.provider.kind == "modal":
                # A Modal volume acts as the account that owns it unless told otherwise.
                options.setdefault("account", self.provider.alias)
            self._store = Store(backends.build(backend, **options))
        return self._store

    # -- environments --------------------------------------------------------

    def env_ref(self, env: Env, platform: str | None = None) -> str:
        """``env/<env key>-<platform>``, or ``env/<env key>`` where no platform is named."""
        key = env.key if platform is None else f"{env.key}-{platform}"
        return ENV_REF.format(key=key)

    def cached_env(self, env: Env, platform: str | None = None) -> str | None:
        """Digest of the prebuilt environment archive, if one was stored."""
        return self.store.resolve(self.env_ref(env, platform))

    def cache_env(self, env: Env, root: str | Path, platform: str | None = None) -> str:
        """Pack an installed environment and remember it under its env key.

        One archive replaces tens of thousands of file transfers. The env key covers the
        uv files and the Python version, and the platform names the machine, so the same
        declaration on the same kind of machine reuses the same archive.
        """
        return self.store.put_tree(root, key=self.env_ref(env, platform)).digest

    def cache_env_from(self, target: Any, env: Env, path: str, platform: str | None = None) -> str:
        """Pack an environment that was built inside a runtime and store it.

        This is how the first session pays the sync and every later one skips it.
        """
        payload, _digest = _session(target).pack_dir(path)
        info = self.store.put_bytes(payload)
        self.store.point(self.env_ref(env, platform), info.digest)
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

    def absorb(self, target: Any, path: str, name: str) -> str:
        """Pull a checkpoint out of a session and store it under a name.

        ``target`` is the declaration whose calls wrote the file. Which session that is is
        letify's answer: a caller naming one could name a different session from the one the
        work ran in, and that session holds none of its files.
        """
        payload, _digest = _session(target).get_bytes(path)
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
        target: str | None = None,
        links: bool = False,
    ) -> RemoteFile:
        """Put a blob inside the runtime, optionally unpacking it at ``target``.

        ``target`` defaults to the volume directory. ``links`` allows symlinks to absolute
        paths in the archive, which an environment archive needs. The runtime pulls the blob
        from the backend itself when the backend offers a pull and the channel keeps a worker
        alive to perform it. Otherwise the bytes go through the channel.
        """
        directory = self.directory(runtime)
        destination = path or f"{directory.rstrip('/')}/blobs/{digest[:2]}/{digest}"
        into = target or directory
        if runtime.persistent_channel:
            source = self.store.backend.pull_source(digest)
            if source is not None:
                return runtime.pull(
                    source, destination, digest=digest, unpack=unpack, target=into, links=links
                )
        payload = self.store.get_bytes(digest)
        return runtime.put_bytes(payload, destination, unpack=unpack, target=into, links=links)

    def materialize_ref(
        self, runtime: Runtime, ref: str, *, unpack: bool = False
    ) -> RemoteFile | None:
        """Materialize whatever a name currently points at."""
        digest = self.store.resolve(ref)
        if digest is None:
            return None
        return self.materialize(runtime, digest, unpack=unpack)

    def resume(self, target: Any, name: str, path: str) -> str | None:
        """Put the newest checkpoint for a name inside the session at ``path``.

        This is what makes a preempted session cheap to restart: the function asks for its
        own checkpoint and finds it already on disk. ``target`` is the declaration that will
        look for it, so the file lands in the session its calls are handed.
        """
        digest = self.latest_checkpoint(name)
        if digest is None:
            return None
        self.materialize(_session(target), digest, path=path)
        return digest

    def __repr__(self) -> str:
        return f"<Volume {self.key} on {self.provider.store_backend()}>"


__all__ = ["CHECKPOINT_REF", "ENV_REF", "Volume"]
