"""Backend registry.

A provider names a backend and the volume builds it. Nothing here imports a cloud SDK
until the backend that needs it is actually constructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...errors import ProviderUnavailable
from ..cas import Backend
from .filesystem import FilesystemBackend
from .layout import BLOB_PREFIX, REF_PREFIX, blob_key, ref_key
from .objects import GCSBackend, ModalBackend

#: Backend name to the class that implements it. A provider's ``store_backend`` returns
#: one of these keys.
BACKENDS: dict[str, type[Backend]] = {
    "filesystem": FilesystemBackend,
    "shell": FilesystemBackend,
    "gcs": GCSBackend,
    "modal": ModalBackend,
}


def build(backend: str, **options: Any) -> Backend:
    """Create a backend by name."""
    cls = BACKENDS.get(backend)
    if cls is None:
        known = ", ".join(sorted(BACKENDS))
        raise ProviderUnavailable("store", f"unknown backend {backend!r}. Known: {known}")
    return cls(**options)


def default_location(backend: str, name: str) -> tuple[str, str]:
    """The option a backend needs when the configuration leaves it out."""
    if backend in ("filesystem", "shell"):
        return "root", str(Path.home() / ".cache" / "letify" / name)
    if backend == "modal":
        return "volume_name", f"letify-{name}"
    return "bucket", name


__all__ = [
    "BACKENDS",
    "BLOB_PREFIX",
    "REF_PREFIX",
    "Backend",
    "FilesystemBackend",
    "GCSBackend",
    "ModalBackend",
    "blob_key",
    "build",
    "default_location",
    "ref_key",
]
