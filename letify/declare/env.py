"""Env, the environment a runtime is built from.

An Env is a declaration, not a built artifact. It names a uv lock file and any
extra packages, and its ``key`` is a hash of that declaration. Two runtimes with
the same key are interchangeable, which is what lets the pool reuse them and
what lets the blob store cache one prebuilt archive per environment.

The lock file is resolved for every platform uv supports, so the same lock drives
a Linux runtime from a Windows or macOS client.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_LOCK = "uv.lock"


@dataclass(frozen=True, slots=True)
class Env:
    """A remote environment declared by a uv lock file plus optional extras."""

    lock: str = DEFAULT_LOCK
    packages: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    variables: tuple[tuple[str, str], ...] = ()
    python: str | None = None
    ship_modules: tuple[str, ...] = ()
    _lock_digest: str | None = field(default=None, compare=True)

    @classmethod
    def from_lock(cls, lock: str = DEFAULT_LOCK, **kwargs: object) -> Env:
        """Build an Env from a lock file path.

        ``Env()`` and ``Env.from_lock()`` are the same thing; the named
        constructor exists because it reads better next to a custom path.
        """
        return cls(lock=lock, **kwargs)  # type: ignore[arg-type]

    def pip_install(self, *packages: str) -> Env:
        """Add packages the lock file does not carry."""
        return replace(self, packages=self.packages + tuple(packages))

    def run(self, *commands: str) -> Env:
        """Add shell commands to run after the environment is installed."""
        return replace(self, commands=self.commands + tuple(commands))

    def vars(self, **values: str) -> Env:
        """Set environment variables inside the runtime."""
        return replace(self, variables=self.variables + tuple(sorted(values.items())))

    def ship(self, *modules: str) -> Env:
        """Send these modules by value rather than by name.

        A module in the lock file is installed remotely and referenced by name.
        A module that is not, such as the project's own package or an editable
        install, has to travel with the call because the remote side either does
        not have it or has an older copy. letify infers this from the lock file;
        use this only to override the inference.
        """
        return replace(self, ship_modules=self.ship_modules + tuple(modules))

    @property
    def lock_digest(self) -> str:
        """Hash of the lock file contents, or ``"nolock"`` when there is none."""
        if self._lock_digest is not None:
            return self._lock_digest
        path = Path(self.lock)
        if not path.is_file():
            return "nolock"
        return hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()

    @property
    def key(self) -> str:
        """Identity of the environment. Runtimes are pooled by this value."""
        payload = repr(
            (
                self.lock_digest,
                tuple(sorted(self.packages)),
                self.commands,
                self.variables,
                self.python,
                # Shipped modules travel with every call made against this environment,
                # so two declarations that ship different code are not interchangeable
                # and must not share a pooled session or a cached archive.
                tuple(sorted(self.ship_modules)),
            )
        ).encode()
        return hashlib.blake2b(payload, digest_size=6).hexdigest()

    def __repr__(self) -> str:
        return f"<Env {self.key} lock={self.lock}>"
