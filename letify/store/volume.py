"""Volume, one named content addressed store on a provider.

A volume is what makes an ephemeral provider behave like a persistent one. With a
volume attached, the environment archive and the model cache are fetched from
storage that outlives the runtime instead of being rebuilt from their origin, and
that is the difference between a session that starts working in under a minute and
one that spends twenty minutes downloading.

The numbers behind the design. A twenty gigabyte Hugging Face cache takes about
27 minutes to pull from a lab server over a 100 Mbit/s link, three to five minutes
from the Hugging Face hub, and 40 to 60 seconds from a bucket in the same
infrastructure as the runtime. That difference is charged as GPU time, which is
why a cache tier is not optional for short sessions.

Mutable state is kept in refs rather than by syncing files, so two runtimes that
both write never overwrite each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cas import Store

if TYPE_CHECKING:
    from ..env import Env
    from ..providers.base import Provider

#: Where a runtime keeps the files a volume materializes.
DEFAULT_MOUNT = "/opt/letify"

#: Ref names letify itself uses. Anything else in the namespace is the user's.
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
        value = self.options.get("mount", DEFAULT_MOUNT)
        return str(value)

    @property
    def key(self) -> str:
        return f"{self.provider.alias}/{self.name}"

    @property
    def store(self) -> Store:
        """Build the backend on first use, not at declaration time."""
        if self._store is None:
            from . import backends

            backend_name = self.options.get("backend") or self.provider.store_backend()
            options = dict(self.options)
            for reserved in ("backend", "mount"):
                options.pop(reserved, None)
            options.setdefault(*_default_location(str(backend_name), self.name))
            self._store = Store(backends.build(str(backend_name), **options))
        return self._store

    # -- environments --------------------------------------------------------

    def env_ref(self, env: Env) -> str:
        return ENV_REF.format(key=env.key)

    def cached_env(self, env: Env) -> str | None:
        """Digest of the prebuilt environment archive, if one was stored."""
        return self.store.resolve(self.env_ref(env))

    def cache_env(self, env: Env, root: str | Path) -> str:
        """Pack an installed environment and remember it under its env key.

        One archive replaces tens of thousands of file transfers. The key is the
        hash of the lock file, so the same declaration reuses the same archive and
        a changed lock file builds a new one.
        """
        info = self.store.put_tree(root, key=self.env_ref(env))
        return info.digest

    # -- checkpoints ---------------------------------------------------------

    def latest_checkpoint(self, name: str) -> str | None:
        """Digest of the newest checkpoint written under this name."""
        return self.store.resolve(CHECKPOINT_REF.format(name=name))

    def put_checkpoint(self, name: str, path: str | Path) -> str:
        """Store a checkpoint and move the name to point at it.

        The blob is immutable and the ref is a few dozen bytes, so two runtimes
        writing at the same time cannot lose each other's data. The worst case is
        that one of the two names wins, and both blobs remain.
        """
        source = Path(path)
        info = self.store.put_tree(source) if source.is_dir() else self.store.put_file(source)
        self.store.point(CHECKPOINT_REF.format(name=name), info.digest)
        return info.digest

    def fetch_checkpoint(self, name: str, target: str | Path) -> Path | None:
        """Materialize the newest checkpoint for a name, if there is one."""
        digest = self.latest_checkpoint(name)
        if digest is None:
            return None
        destination = Path(target)
        if destination.suffix:
            return self.store.fetch_file(digest, destination)
        return self.store.fetch_tree(digest, destination)

    def __repr__(self) -> str:
        return f"<Volume {self.key} on {self.provider.store_backend()}>"


def _default_location(backend: str, name: str) -> tuple[str, str]:
    """Pick the option a backend needs when the configuration leaves it out."""
    if backend in ("filesystem", "shell"):
        return "root", str(Path.home() / ".cache" / "letify" / name)
    if backend == "modal":
        return "volume_name", f"letify-{name}"
    return "bucket", name
