"""Google Drive as a store for the environment archive.

Owns the Drive v3 calls letify makes with the standard library HTTP client: finding or
creating the ``letify`` folder, listing it, reading and writing small files, creating a
resumable upload session, and naming the URL a runtime downloads a file from. It does not
own moving the archive. The runtime uploads and downloads the archive itself, so the bytes
never cross the client, and this module only hands the runtime a session URL to upload to
and a URL with a short-lived token to download from.

Every request carries ``x-goog-user-project``, because Google answers ``403
SERVICE_DISABLED`` without a project that has the Drive API enabled. Access tokens come from
the refresh token in the named account's ``token.json``, which letify only reads.

One habit is shared with every backend: ``missing`` answers with a single listing rather
than one request per digest.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

from ...errors import ConfigError, ProviderUnavailable, RuntimeFailure
from ..cas import Backend
from .google_auth import TokenSource

#: The Drive API.
DRIVE_ENDPOINT = "https://www.googleapis.com"

#: The folder at the root of My Drive that holds every blob and ref.
FOLDER_NAME = "letify"

#: Bytes of one piece of a resumable upload. Drive wants a multiple of 256 KiB.
UPLOAD_PIECE = 64 << 20

#: The prefix of a ref's file name. A blob's file name is its digest alone.
REF_PREFIX = "ref "

#: Scopes that let letify read and write the files it created.
_SCOPES = (
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
)

_FOLDER_TYPE = "application/vnd.google-apps.folder"


def upload_pieces(
    path: str | os.PathLike[str], url: str, headers: dict[str, str], piece: int
) -> str:
    """Send one file to a resumable session URL in pieces and return the new file's id.

    Every piece but the last is a multiple of 256 KiB, as Drive requires, and is answered
    308 until the last, which is answered with the new file's metadata. This is plain
    standard library code that both this process and a worker can run, so a handoff made
    inside a shipped function uploads from the runtime and the bytes never cross the client.
    """
    size = os.path.getsize(path)
    remote_id: str | None = None
    with open(path, "rb") as handle:
        offset = 0
        while offset < size:
            block = handle.read(piece)
            end = offset + len(block)
            send = urllib.request.Request(url, data=block, method="PUT", headers=headers)
            send.add_header("Content-Length", str(len(block)))
            send.add_header("Content-Range", f"bytes {offset}-{end - 1}/{size}")
            try:
                with urllib.request.urlopen(send, timeout=3600) as response:
                    remote_id = str(json.loads(response.read())["id"])
            except urllib.error.HTTPError as exc:
                if exc.code != 308:
                    detail = exc.read()[:300].decode(errors="replace")
                    raise RuntimeFailure(f"Drive upload returned {exc.code}: {detail}") from exc
            except (urllib.error.URLError, OSError) as exc:
                raise RuntimeFailure(f"Drive upload failed: {exc}") from exc
            offset = end
    if remote_id is None:
        raise RuntimeFailure("Drive upload: the session never answered with a file")
    return remote_id


def download_to(url: str, headers: dict[str, str], destination: str | os.PathLike[str]) -> int:
    """Stream one Drive file into ``destination`` and return its size in bytes."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3600) as response, open(
            destination, "wb"
        ) as out:
            shutil.copyfileobj(response, out, 1 << 20)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode(errors="replace")
        raise RuntimeFailure(f"Drive download returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeFailure(f"Drive download failed: {exc}") from exc
    finally:
        request = None  # type: ignore[assignment]
    return os.path.getsize(destination)


class AccountTokenSource(TokenSource):
    """Access tokens minted from the refresh token in one account's ``token.json``."""

    def __init__(self, alias: str) -> None:
        super().__init__()
        self.alias = alias

    def _fetch(self) -> tuple[str, float]:
        from ...paths import account_directory

        path = account_directory(self.alias) / "token.json"
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProviderUnavailable(
                "drive", f"the login of account {self.alias!r} could not be read from {path}: {exc}"
            ) from exc
        granted = info.get("scopes") or info.get("scope") or []
        if isinstance(granted, str):
            granted = granted.split()
        if not any(scope in _SCOPES for scope in granted):
            raise ConfigError(
                f"the login of account {self.alias!r} does not carry the drive.file scope, "
                f"so letify cannot use it for Google Drive. Log in again with that scope."
            )
        return self._refresh(info, str(path))


def _quoted(text: str) -> str:
    """A string as a Drive query literal."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


class DriveBackend(Backend):
    """The ``letify`` folder in a Google Drive.

    A blob is a file named by its digest and a ref ``<name>`` is a small file named
    ``ref <name>`` holding a digest. Drive is fast for one large file read once, which is
    an environment archive, and slow for many small ones, so this backend is chosen for the
    archive and not for project data.
    """

    name = "drive"
    moves_on_runtime = True

    def __init__(
        self,
        account: str,
        project: str,
        endpoint: str = DRIVE_ENDPOINT,
        tokens: TokenSource | None = None,
    ) -> None:
        self.account = account
        self.project = project
        self.endpoint = endpoint.rstrip("/")
        self.tokens = tokens or AccountTokenSource(account)
        self._folder_id: str | None = None
        self._ids: dict[str, str] = {}
        self._lock = threading.Lock()

    # -- HTTP ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.tokens.token()}",
            "x-goog-user-project": self.project,
        }

    def _call(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        what: str,
    ) -> bytes | None:
        """Send one request. A 404 answers None, any other failure raises."""
        request = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            detail = exc.read()[:500].decode(errors="replace")
            raise RuntimeFailure(
                f"Drive {what}: {method} returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeFailure(f"Drive {what}: {method} failed: {exc}") from exc

    def _json(self, method: str, url: str, *, what: str, body: Any = None) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        reply = self._call(
            method, url, data=data, content_type="application/json" if data else None, what=what
        )
        return dict(json.loads(reply or b"{}"))

    def _list(self, expression: str, fields: str) -> Iterator[dict[str, Any]]:
        """Every file a query matches, one request per page of up to 1000."""
        page: str | None = None
        while True:
            query = {
                "q": expression,
                "pageSize": "1000",
                "fields": f"nextPageToken,files({fields})",
            }
            if page:
                query["pageToken"] = page
            url = f"{self.endpoint}/drive/v3/files?{urllib.parse.urlencode(query)}"
            body = self._json("GET", url, what=f"listing {expression}")
            yield from body.get("files", [])
            page = body.get("nextPageToken")
            if not page:
                return

    # -- layout ------------------------------------------------------------------

    def folder(self) -> str:
        """The id of the ``letify`` folder, found by name or created once."""
        with self._lock:
            if self._folder_id is None:
                found = list(
                    self._list(
                        f"name = {_quoted(FOLDER_NAME)} and 'root' in parents"
                        f" and mimeType = {_quoted(_FOLDER_TYPE)} and trashed = false",
                        "id",
                    )
                )
                if found:
                    self._folder_id = str(found[0]["id"])
                else:
                    made = self._json(
                        "POST",
                        f"{self.endpoint}/drive/v3/files",
                        what="creating the letify folder",
                        body={"name": FOLDER_NAME, "mimeType": _FOLDER_TYPE, "parents": ["root"]},
                    )
                    self._folder_id = str(made["id"])
            return self._folder_id

    def _find(self, name: str) -> str | None:
        found = list(
            self._list(
                f"name = {_quoted(name)} and {_quoted(self.folder())} in parents"
                f" and trashed = false",
                "id",
            )
        )
        return str(found[0]["id"]) if found else None

    def _create(self, name: str, payload: bytes) -> str:
        """Create one small file with a single multipart request."""
        boundary = "letify-drive-boundary"
        meta = json.dumps({"name": name, "parents": [self.folder()]})
        body = (
            f"--{boundary}\r\nContent-Type: application/json\r\n\r\n{meta}\r\n"
            f"--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n"
        ).encode() + payload + f"\r\n--{boundary}--".encode()
        url = f"{self.endpoint}/upload/drive/v3/files?uploadType=multipart&fields=id"
        reply = self._call(
            "POST", url, data=body, content_type=f"multipart/related; boundary={boundary}",
            what=f"creating {name}",
        )
        return str(json.loads(reply or b"{}")["id"])

    def _file_id(self, digest: str) -> str | None:
        if digest not in self._ids:
            found = self._find(digest)
            if found is None:
                return None
            self._ids[digest] = found
        return self._ids[digest]

    def _media_url(self, file_id: str) -> str:
        return f"{self.endpoint}/drive/v3/files/{urllib.parse.quote(file_id)}?alt=media"

    # -- the runtime's pull and push ----------------------------------------------

    def pull_source(self, digest: str) -> dict[str, object] | None:
        """The media URL and the two headers, so the runtime downloads the file itself."""
        file_id = self._file_id(digest)
        if file_id is None:
            raise RuntimeFailure(f"Drive does not hold {digest}")
        return {"url": self._media_url(file_id), "headers": self._headers()}

    def begin_upload(self, digest: str, size: int) -> dict[str, object] | None:
        """Create a resumable upload session and name where the runtime sends the pieces.

        A session URL authorizes its own upload, so the answer carries no access token. The
        project header is not a credential and is sent with every piece.
        """
        url = f"{self.endpoint}/upload/drive/v3/files?uploadType=resumable&fields=id"
        meta = json.dumps({"name": digest, "parents": [self.folder()]}).encode()
        request = urllib.request.Request(url, data=meta, method="POST", headers=self._headers())
        request.add_header("Content-Type", "application/json; charset=UTF-8")
        request.add_header("X-Upload-Content-Length", str(size))
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                location = response.headers["Location"]
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode(errors="replace")
            raise RuntimeFailure(
                f"Drive upload session for {digest}: POST returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeFailure(f"Drive upload session for {digest}: POST failed: {exc}") from exc
        if not location:
            raise RuntimeFailure(f"Drive upload session for {digest} named no upload URL")
        return {
            "url": location,
            "headers": {"x-goog-user-project": self.project},
            "piece": UPLOAD_PIECE,
        }

    def record_upload(self, digest: str, remote_id: str) -> None:
        """Remember the file id of a blob the runtime uploaded."""
        self._ids[digest] = remote_id

    # -- named handoffs ----------------------------------------------------------

    def put_file(self, digest: str, path: str | os.PathLike[str]) -> None:
        """Upload one local file as the blob ``digest``, in pieces, without reading it whole."""
        target = self.begin_upload(digest, os.path.getsize(path))
        assert target is not None
        headers = dict(target.get("headers") or {})  # type: ignore[call-overload]
        remote_id = upload_pieces(path, str(target["url"]), headers, int(target["piece"]))  # type: ignore[call-overload]
        self.record_upload(digest, remote_id)

    def get_file(self, digest: str, destination: str | os.PathLike[str]) -> int:
        """Download the blob ``digest`` into a local file and return its size."""
        source = self.pull_source(digest)
        assert source is not None
        return download_to(
            str(source["url"]), dict(source.get("headers") or {}), destination  # type: ignore[call-overload]
        )

    def delete_blob(self, digest: str) -> None:
        """Delete the blob ``digest``. A blob that is already gone is not an error."""
        file_id = self._file_id(digest)
        if file_id is None:
            return
        url = f"{self.endpoint}/drive/v3/files/{urllib.parse.quote(file_id)}"
        self._call("DELETE", url, what=f"deleting {digest}")
        self._ids.pop(digest, None)

    def refs_with_prefix(self, prefix: str) -> dict[str, str]:
        """Every ref whose name starts with ``prefix``, as name to digest, in one listing."""
        found: dict[str, str] = {}
        wanted = REF_PREFIX + prefix
        for entry in self._list(
            f"{_quoted(self.folder())} in parents and trashed = false", "id,name"
        ):
            name = str(entry["name"])
            if not name.startswith(wanted):
                continue
            payload = self._call(
                "GET", self._media_url(str(entry["id"])), what=f"reading ref {name[4:]}"
            )
            if payload is not None:
                found[name[len(REF_PREFIX) :]] = payload.decode().strip()
        return found

    # -- the backend -----------------------------------------------------------

    def has(self, digest: str) -> bool:
        return self._file_id(digest) is not None

    def put(self, digest: str, payload: bytes) -> None:
        self._ids[digest] = self._create(digest, payload)

    def get(self, digest: str) -> bytes:
        file_id = self._file_id(digest)
        payload = (
            self._call("GET", self._media_url(file_id), what=f"reading {digest}")
            if file_id
            else None
        )
        if payload is None:
            raise RuntimeFailure(f"Drive does not hold {digest}")
        return payload

    def list_digests(self, prefix: str = "") -> Iterator[str]:
        for entry in self._list(
            f"{_quoted(self.folder())} in parents and trashed = false", "name"
        ):
            name = str(entry["name"])
            if not name.startswith(REF_PREFIX) and name.startswith(prefix):
                yield name

    def missing(self, digests: list[str]) -> list[str]:
        held = set(self.list_digests())
        return [digest for digest in digests if digest not in held]

    def read_ref(self, name: str) -> str | None:
        file_id = self._find(REF_PREFIX + name)
        if file_id is None:
            return None
        payload = self._call("GET", self._media_url(file_id), what=f"reading ref {name}")
        return payload.decode().strip() if payload is not None else None

    def write_ref(self, name: str, digest: str) -> None:
        file_id = self._find(REF_PREFIX + name)
        if file_id is None:
            self._create(REF_PREFIX + name, digest.encode())
            return
        url = (
            f"{self.endpoint}/upload/drive/v3/files/{urllib.parse.quote(file_id)}"
            f"?uploadType=media"
        )
        self._call(
            "PATCH", url, data=digest.encode(), content_type="application/octet-stream",
            what=f"updating ref {name}",
        )


__all__ = [
    "DRIVE_ENDPOINT",
    "FOLDER_NAME",
    "AccountTokenSource",
    "DriveBackend",
    "download_to",
    "upload_pieces",
]
