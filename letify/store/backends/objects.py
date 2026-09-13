"""Object store backends: Google Cloud Storage and Modal volumes.

Google Cloud Storage is reached with the standard library HTTP client against the JSON
API, with a token borrowed from the user's own Google login. The Modal client is imported
lazily, so importing letify never pulls in a cloud SDK.

One habit is shared by every backend: ``missing`` answers with a single listing rather
than one request per digest. Object level requests are billed and add latency, and
avoiding that is the whole point of diffing against a manifest.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from ...errors import ProviderUnavailable, RuntimeFailure
from ..cas import Backend
from .google_auth import TokenSource
from .layout import BLOB_PREFIX, blob_key, ref_key

#: The Cloud Storage JSON API.
GCS_ENDPOINT = "https://storage.googleapis.com"


class GCSBackend(Backend):
    """A Google Cloud Storage bucket.

    This is the backend for Colab, because a Colab runtime is a Google Compute Engine
    virtual machine and reading the bucket stays inside Google's network.

    Use a multi-region bucket such as ``US``. Colab does not let you choose where the
    runtime lands, and a multi-region bucket avoids a cross-region charge on every read.
    """

    name = "gcs"

    def __init__(
        self,
        bucket: str,
        prefix: str = "letify",
        endpoint: str = GCS_ENDPOINT,
        tokens: TokenSource | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint = endpoint.rstrip("/")
        self.tokens = tokens or TokenSource()

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    # -- HTTP ------------------------------------------------------------------

    def object_url(self, name: str, *, media: bool = False) -> str:
        quoted = urllib.parse.quote(name, safe="")
        url = f"{self.endpoint}/storage/v1/b/{urllib.parse.quote(self.bucket)}/o/{quoted}"
        return url + "?alt=media" if media else url

    def _request(
        self, method: str, url: str, *, data: bytes | None = None, what: str
    ) -> bytes | None:
        """Send one request. A 404 answers None, any other failure raises."""
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.tokens.token()}")
        if data is not None:
            request.add_header("Content-Type", "application/octet-stream")
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            detail = exc.read()[:500].decode(errors="replace")
            raise RuntimeFailure(
                f"gs://{self.bucket}/{what}: {method} returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeFailure(f"gs://{self.bucket}/{what}: {method} failed: {exc}") from exc

    def _upload(self, name: str, payload: bytes) -> None:
        query = urllib.parse.urlencode({"uploadType": "media", "name": name})
        url = f"{self.endpoint}/upload/storage/v1/b/{urllib.parse.quote(self.bucket)}/o?{query}"
        self._request("POST", url, data=payload, what=name)

    def _download(self, name: str) -> bytes | None:
        return self._request("GET", self.object_url(name, media=True), what=name)

    # -- the backend -----------------------------------------------------------

    def has(self, digest: str) -> bool:
        name = self._key(blob_key(digest))
        return self._request("GET", self.object_url(name), what=name) is not None

    def put(self, digest: str, payload: bytes) -> None:
        self._upload(self._key(blob_key(digest)), payload)

    def get(self, digest: str) -> bytes:
        name = self._key(blob_key(digest))
        payload = self._download(name)
        if payload is None:
            raise RuntimeFailure(f"gs://{self.bucket}/{name} does not exist")
        return payload

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        base = self._key(BLOB_PREFIX) + "/"
        listing = f"{self.endpoint}/storage/v1/b/{urllib.parse.quote(self.bucket)}/o"
        page: str | None = None
        while True:
            query = {"prefix": base, "fields": "items(name),nextPageToken"}
            if page:
                query["pageToken"] = page
            url = f"{listing}?{urllib.parse.urlencode(query)}"
            body = json.loads(self._request("GET", url, what=base) or b"{}")
            for item in body.get("items", []):
                name = str(item["name"]).rsplit("/", 1)[-1]
                if name.startswith(prefix):
                    yield name
            page = body.get("nextPageToken")
            if not page:
                return

    def missing(self, digests: list[str]) -> list[str]:
        held = set(self.list_digests())
        return [digest for digest in digests if digest not in held]

    def read_ref(self, name: str) -> str | None:
        payload = self._download(self._key(ref_key(name)))
        return payload.decode().strip() if payload is not None else None

    def write_ref(self, name: str, digest: str) -> None:
        self._upload(self._key(ref_key(name)), digest.encode())


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
