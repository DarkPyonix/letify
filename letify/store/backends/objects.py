"""Object store backends: Google Cloud Storage and Modal volumes.

Each client library is imported lazily, so importing letify never pulls in a cloud SDK.

One habit is shared by all three: ``missing`` answers with a single listing rather than
one request per digest. Object level requests are billed and add latency, and avoiding
that is the whole point of diffing against a manifest.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

from ...errors import ProviderUnavailable
from ..cas import Backend
from .layout import BLOB_PREFIX, blob_key, ref_key


class GCSBackend(Backend):
    """A Google Cloud Storage bucket.

    This is the backend for Colab, because a Colab runtime is a Google Compute Engine
    virtual machine and reading the bucket stays inside Google's network.

    Use a multi-region bucket such as ``US``. Colab does not let you choose where the
    runtime lands, and a multi-region bucket avoids a cross-region charge on every read.
    """

    name = "gcs"

    def __init__(self, bucket: str, prefix: str = "letify"):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ProviderUnavailable(
                "gcs", "the google-cloud-storage package is not installed", "gcs"
            ) from exc
        self.prefix = prefix.strip("/")
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def has(self, digest: str) -> bool:
        return self._bucket.blob(self._key(blob_key(digest))).exists()

    def put(self, digest: str, payload: bytes) -> None:
        self._bucket.blob(self._key(blob_key(digest))).upload_from_string(payload)

    def get(self, digest: str) -> bytes:
        return self._bucket.blob(self._key(blob_key(digest))).download_as_bytes()

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        base = self._key(BLOB_PREFIX)
        for blob in self._client.list_blobs(self._bucket, prefix=base):
            name = blob.name.rsplit("/", 1)[-1]
            if name.startswith(prefix):
                yield name

    def missing(self, digests: list[str]) -> list[str]:
        held = set(self.list_digests())
        return [digest for digest in digests if digest not in held]

    def read_ref(self, name: str) -> str | None:
        blob = self._bucket.blob(self._key(ref_key(name)))
        return blob.download_as_text().strip() if blob.exists() else None

    def write_ref(self, name: str, digest: str) -> None:
        self._bucket.blob(self._key(ref_key(name))).upload_from_string(digest)


class ModalBackend(Backend):
    """A Modal volume, mounted from outside the container.

    A Modal volume sits in the same data centre as the GPU and caches read blocks on
    local disk, which is why a persistent provider needs no separate cache tier.
    """

    name = "modal"

    def __init__(self, volume_name: str, prefix: str = "letify"):
        try:
            import modal
        except ImportError as exc:
            raise ProviderUnavailable(
                "modal", "the modal package is not installed", "modal"
            ) from exc
        self.prefix = prefix.strip("/")
        self._volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    def _key(self, key: str) -> str:
        return f"/{self.prefix}/{key}" if self.prefix else f"/{key}"

    def has(self, digest: str) -> bool:
        try:
            next(iter(self._volume.listdir(self._key(blob_key(digest)))))
        except Exception:
            return False
        return True

    def put(self, digest: str, payload: bytes) -> None:
        with self._volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(payload), self._key(blob_key(digest)))

    def get(self, digest: str) -> bytes:
        return b"".join(self._volume.read_file(self._key(blob_key(digest))))

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        for entry in self._volume.listdir(self._key(BLOB_PREFIX), recursive=True):
            name = entry.path.rsplit("/", 1)[-1]
            if name.startswith(prefix):
                yield name

    def missing(self, digests: list[str]) -> list[str]:
        held = set(self.list_digests())
        return [digest for digest in digests if digest not in held]

    def read_ref(self, name: str) -> str | None:
        try:
            payload = b"".join(self._volume.read_file(self._key(ref_key(name))))
        except Exception:
            return None
        return payload.decode().strip()

    def write_ref(self, name: str, digest: str) -> None:
        with self._volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(digest.encode()), self._key(ref_key(name)))


__all__ = ["GCSBackend", "ModalBackend"]
