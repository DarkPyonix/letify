"""Object store backends: Google Cloud Storage and Modal volumes.

Google Cloud Storage is reached with the standard library HTTP client against the JSON
API, with a token borrowed from the user's own Google login. A Modal volume is reached
through the Modal adapter, a separate process, so importing letify never pulls in a cloud
SDK.

One habit is shared by every backend: ``missing`` answers with a single listing rather
than one request per digest. Object level requests are billed and add latency, and
avoiding that is the whole point of diffing against a manifest.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from ...errors import ConfigError, ProviderUnavailable, RuntimeFailure
from ..cas import Backend
from .google_auth import TokenSource
from .layout import BLOB_PREFIX, blob_key, ref_key

#: The Cloud Storage JSON API.
GCS_ENDPOINT = "https://storage.googleapis.com"

#: Google's Security Token Service, which downscopes a token with a Credential Access Boundary.
STS_ENDPOINT = "https://sts.googleapis.com/v1/token"


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
        sts_endpoint: str = STS_ENDPOINT,
        tokens: TokenSource | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint = endpoint.rstrip("/")
        self.sts_endpoint = sts_endpoint
        self.tokens = tokens or TokenSource()

    # -- the runtime's pull ------------------------------------------------------

    def access_boundary(self) -> dict[str, object]:
        """Read access to this bucket, narrowed to object names under the prefix."""
        resource = f"projects/_/buckets/{self.bucket}"
        rule: dict[str, object] = {
            "availablePermissions": ["inRole:roles/storage.objectViewer"],
            "availableResource": f"//storage.googleapis.com/{resource}",
        }
        if self.prefix:
            rule["availabilityCondition"] = {
                "expression": f"resource.name.startsWith('{resource}/objects/{self.prefix}/')"
            }
        return {"accessBoundary": {"accessBoundaryRules": [rule]}}

    def read_token(self) -> str:
        """Exchange the local login for a token that can only read this volume.

        A failure raises rather than handing out the unscoped token, because that token can
        do everything the user's login can.
        """
        form = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "subject_token": self.tokens.token(),
                "options": json.dumps(self.access_boundary()),
            }
        ).encode()
        request = urllib.request.Request(self.sts_endpoint, data=form, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return str(json.loads(response.read())["access_token"])
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode(errors="replace")
            raise RuntimeFailure(
                f"downscoping the read token for gs://{self.bucket} returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            raise RuntimeFailure(
                f"downscoping the read token for gs://{self.bucket} failed: {exc}"
            ) from exc

    def pull_source(self, digest: str) -> dict[str, object] | None:
        """A Colab runtime is a Compute Engine VM, so it reads the bucket over Google's network."""
        name = self._key(blob_key(digest))
        return {
            "url": self.object_url(name, media=True),
            "headers": {"Authorization": f"Bearer {self.read_token()}"},
        }

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
    local disk, which is why a persistent provider needs no separate cache tier. Every
    operation goes through the Modal adapter acting as ``account``, so the letify process
    never imports ``modal``.
    """

    name = "modal"

    def __init__(self, volume_name: str, prefix: str = "letify", account: str | None = None):
        from ... import tools
        from ...providers.modal import Adapter

        if not account:
            raise ConfigError(
                f"the modal backend for volume {volume_name!r} needs an account to act as. "
                f"Set account = '<alias of a modal login>' on the volume."
            )
        if tools.find_uv() is None:
            raise ProviderUnavailable("modal", tools.missing_uv_message())
        self.volume_name = volume_name
        self.account = account
        self.prefix = prefix.strip("/")
        self._adapter = Adapter.for_account(account)

    def _key(self, key: str) -> str:
        return f"/{self.prefix}/{key}" if self.prefix else f"/{key}"

    def _put(self, path: str, payload: bytes) -> None:
        data = base64.b64encode(payload).decode()
        self._adapter.request("volume_put", volume=self.volume_name, path=path, data=data)

    def _get(self, path: str) -> bytes | None:
        from ...providers.modal import VolumePathMissing

        try:
            data = self._adapter.request("volume_get", volume=self.volume_name, path=path)
        except VolumePathMissing:
            return None
        return base64.b64decode(str(data))

    def _list(self, path: str) -> list[str]:
        from ...providers.modal import VolumePathMissing

        try:
            found = self._adapter.request("volume_list", volume=self.volume_name, path=path)
        except VolumePathMissing:
            return []
        return [str(entry) for entry in found]

    def has(self, digest: str) -> bool:
        return bool(self._list(self._key(blob_key(digest))))

    def put(self, digest: str, payload: bytes) -> None:
        self._put(self._key(blob_key(digest)), payload)

    def get(self, digest: str) -> bytes:
        path = self._key(blob_key(digest))
        payload = self._get(path)
        if payload is None:
            raise RuntimeFailure(f"modal volume {self.volume_name}:{path} does not exist")
        return payload

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        for path in self._list(self._key(BLOB_PREFIX)):
            name = path.rsplit("/", 1)[-1]
            if name.startswith(prefix):
                yield name

    def missing(self, digests: list[str]) -> list[str]:
        held = set(self.list_digests())
        return [digest for digest in digests if digest not in held]

    def read_ref(self, name: str) -> str | None:
        payload = self._get(self._key(ref_key(name)))
        return payload.decode().strip() if payload is not None else None

    def write_ref(self, name: str, digest: str) -> None:
        self._put(self._key(ref_key(name)), digest.encode())

    def close(self) -> None:
        """Stop the adapter process this backend started."""
        self._adapter.close()


__all__ = ["GCSBackend", "ModalBackend"]
