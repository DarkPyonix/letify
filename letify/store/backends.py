"""Blob store backends.

One backend per kind of storage a provider can reach. All of them keep the same
layout, so a blob written by one can be read by another once both point at the
same bucket:

    blobs/<first two hex characters>/<digest>
    refs/<name>

Each backend's optional dependency is imported lazily, so importing letify never
requires a cloud client library.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..errors import ProviderUnavailable
from .cas import Backend

BLOB_PREFIX = "blobs"
REF_PREFIX = "refs"


def _blob_key(digest: str) -> str:
    return f"{BLOB_PREFIX}/{digest[:2]}/{digest}"


def _ref_key(name: str) -> str:
    return f"{REF_PREFIX}/{name}"


class FilesystemBackend(Backend):
    """A directory on a disk this process can see.

    Used by the local provider, and by any remote machine whose own disk is the
    store. Because the local machine's storage outlives every runtime, this can
    also be the origin that remote runtimes pull from.
    """

    name = "filesystem"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def has(self, digest: str) -> bool:
        return self._path(_blob_key(digest)).is_file()

    def put(self, digest: str, payload: bytes) -> None:
        target = self._path(_blob_key(digest))
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so a reader never sees a partial blob.
        temporary = target.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(target)

    def get(self, digest: str) -> bytes:
        return self._path(_blob_key(digest)).read_bytes()

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        base = self.root / BLOB_PREFIX
        if not base.is_dir():
            return
        for path in base.rglob("*"):
            if path.is_file() and path.name.startswith(prefix):
                yield path.name

    def read_ref(self, name: str) -> str | None:
        path = self._path(_ref_key(name))
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    def write_ref(self, name: str, digest: str) -> None:
        path = self._path(_ref_key(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(digest, encoding="utf-8")


class GCSBackend(Backend):
    """A Google Cloud Storage bucket.

    This is the backend for Colab, because a Colab runtime is a Google Compute
    Engine virtual machine and reading the bucket stays inside Google's network.

    Use a multi-region bucket such as ``US``. Colab does not let you choose where
    the runtime lands, and a multi-region bucket avoids a cross-region charge on
    every read.
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
        return self._bucket.blob(self._key(_blob_key(digest))).exists()

    def put(self, digest: str, payload: bytes) -> None:
        blob = self._bucket.blob(self._key(_blob_key(digest)))
        blob.upload_from_string(payload)

    def get(self, digest: str) -> bytes:
        return self._bucket.blob(self._key(_blob_key(digest))).download_as_bytes()

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        base = self._key(BLOB_PREFIX)
        for blob in self._client.list_blobs(self._bucket, prefix=base):
            name = blob.name.rsplit("/", 1)[-1]
            if name.startswith(prefix):
                yield name

    def missing(self, digests: list[str]) -> list[str]:
        """Answer with one listing instead of one request per digest.

        Object level requests are billed and add latency, so the whole point of
        the manifest style diff is to make this a single call.
        """
        held = set(self.list_digests())
        return [d for d in digests if d not in held]

    def read_ref(self, name: str) -> str | None:
        blob = self._bucket.blob(self._key(_ref_key(name)))
        if not blob.exists():
            return None
        return blob.download_as_text().strip()

    def write_ref(self, name: str, digest: str) -> None:
        self._bucket.blob(self._key(_ref_key(name))).upload_from_string(digest)


class S3Backend(Backend):
    """Any S3 compatible object store.

    This covers Elice Data Hub as well as Amazon S3, and most other providers,
    because the S3 API is what object stores agree on.
    """

    name = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "letify",
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
    ):
        try:
            import boto3
        except ImportError as exc:
            raise ProviderUnavailable("s3", "the boto3 package is not installed", "s3") from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.environ.get("LETIFY_S3_ENDPOINT"),
            region_name=region,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def has(self, digest: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(_blob_key(digest)))
        except ClientError:
            return False
        return True

    def put(self, digest: str, payload: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._key(_blob_key(digest)), Body=payload)

    def get(self, digest: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=self._key(_blob_key(digest)))
        return response["Body"].read()

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(BLOB_PREFIX)):
            for item in page.get("Contents", []):
                name = item["Key"].rsplit("/", 1)[-1]
                if name.startswith(prefix):
                    yield name

    def missing(self, digests: list[str]) -> list[str]:
        held = set(self.list_digests())
        return [d for d in digests if d not in held]

    def read_ref(self, name: str) -> str | None:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=self.bucket, Key=self._key(_ref_key(name)))
        except ClientError:
            return None
        return response["Body"].read().decode("utf-8").strip()

    def write_ref(self, name: str, digest: str) -> None:
        self._client.put_object(
            Bucket=self.bucket, Key=self._key(_ref_key(name)), Body=digest.encode()
        )


class ModalBackend(Backend):
    """A Modal volume, which is mounted from outside the container.

    A Modal volume sits in the same data centre as the GPU and caches read blocks
    on local disk, which is why a persistent provider does not need a separate
    cache tier.
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
            next(iter(self._volume.listdir(self._key(_blob_key(digest)))))
        except Exception:
            return False
        return True

    def put(self, digest: str, payload: bytes) -> None:
        with self._volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(payload), self._key(_blob_key(digest)))

    def get(self, digest: str) -> bytes:
        chunks = list(self._volume.read_file(self._key(_blob_key(digest))))
        return b"".join(chunks)

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        for entry in self._volume.listdir(self._key(BLOB_PREFIX), recursive=True):
            name = entry.path.rsplit("/", 1)[-1]
            if name.startswith(prefix):
                yield name

    def read_ref(self, name: str) -> str | None:
        try:
            return b"".join(self._volume.read_file(self._key(_ref_key(name)))).decode().strip()
        except Exception:
            return None

    def write_ref(self, name: str, digest: str) -> None:
        with self._volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(digest.encode()), self._key(_ref_key(name)))


def build(backend: str, **options: Any) -> Backend:
    """Create a backend by name."""
    table = {
        "filesystem": FilesystemBackend,
        "shell": FilesystemBackend,
        "gcs": GCSBackend,
        "s3": S3Backend,
        "modal": ModalBackend,
    }
    cls = table.get(backend)
    if cls is None:
        known = ", ".join(sorted(table))
        raise ProviderUnavailable("store", f"unknown backend {backend!r}. Known: {known}")
    return cls(**options)  # type: ignore[arg-type]
