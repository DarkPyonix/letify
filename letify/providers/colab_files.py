"""Bulk transfer to a Colab runtime through the Jupyter file API its proxy exposes.

Owns moving bytes when the Colab link is the provider fallback: uploads split into parts
sent in parallel, each part as chunked contents API ``PUT`` requests, and downloads read
from ``/files/<path>`` in parallel ranges. It does not own running programs on the VM,
which is ``colab exec`` and is handed in as a runner, or choosing the link.

The proxy address and token are the ones the Colab CLI recorded for the session, read
from its state file under the account directory. The standard library HTTP client is the
only client, because letify adds no dependency for a provider.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config.secrets import account_directory
from ..errors import RuntimeFailure

#: Size of one upload part and one download range.
PART_BYTES = 32 * 1024 * 1024

#: Size of one chunk within an upload part, before base64 encoding.
CHUNK_BYTES = 8 * 1024 * 1024

#: Parts or ranges in flight at once, the fastest count measured in docs/NETWORK.md.
PARALLEL = 8

#: Where the Colab CLI keeps its sessions, relative to its home directory.
SESSIONS_FILE = (".config", "colab-cli", "sessions.json")


def session_endpoint(alias: str, name: str) -> tuple[str, str]:
    """The proxy URL and token the Colab CLI recorded for one session of one account."""
    path = account_directory(alias).joinpath(*SESSIONS_FILE)
    try:
        sessions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sessions = {}
    state = sessions.get(name) if isinstance(sessions, dict) else None
    if not isinstance(state, dict) or not state.get("url") or not state.get("token"):
        raise RuntimeFailure(
            f"{path} records no session named {name!r}, so there is no proxy address to "
            f"transfer files through. The session was not created by 'colab new' under this "
            f"account, or it has been stopped"
        )
    return str(state["url"]).rstrip("/"), str(state["token"])


class ContentsTransfer:
    """Uploads and downloads against one Colab runtime proxy."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token

    def _auth_query(self) -> dict[str, str]:
        """Query parameters that authenticate a request. The Colab proxy's by default."""
        return {"authuser": "0", "colab-runtime-proxy-token": self.token}

    def _auth_headers(self) -> dict[str, str]:
        """Headers that authenticate a request. The Colab proxy's by default."""
        return {"X-Colab-Runtime-Proxy-Token": self.token}

    def _address(self, api: str, remote: str, **query: str) -> str:
        quoted = urllib.parse.quote(remote.lstrip("/"), safe="/")
        parameters = urllib.parse.urlencode({**self._auth_query(), **query})
        return f"{self.url}/{api}/{quoted}?{parameters}"

    def _send(
        self,
        method: str,
        address: str,
        remote: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        request = urllib.request.Request(address, data=body, method=method)
        for key, value in {**self._auth_headers(), **(headers or {})}.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode(errors="replace")
            raise RuntimeFailure(
                f"Colab file API {method} {remote} returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeFailure(f"Colab file API {method} {remote} failed: {exc}") from exc

    # -- upload ----------------------------------------------------------------

    def _put_chunk(self, remote: str, piece: bytes, chunk: int) -> None:
        model = {
            "name": remote.rsplit("/", 1)[-1],
            "path": remote,
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(piece).decode("ascii"),
            "chunk": chunk,
        }
        self._send(
            "PUT",
            self._address("api/contents", remote),
            remote,
            body=json.dumps(model).encode(),
            headers={"Content-Type": "application/json"},
        )

    def _put_part(self, remote: str, part: bytes) -> None:
        pieces = [part[i : i + CHUNK_BYTES] for i in range(0, len(part), CHUNK_BYTES)] or [b""]
        for index, piece in enumerate(pieces, start=1):
            # Chunk 1 creates the file, later ones append, and -1 closes a chunked upload.
            last = index == len(pieces) and index > 1
            self._put_chunk(remote, piece, -1 if last else index)

    def upload(self, payload: bytes, remote: str) -> list[str]:
        """Send a payload as parts beside ``remote`` and return their paths, in order."""
        parts = [payload[i : i + PART_BYTES] for i in range(0, len(payload), PART_BYTES)] or [b""]
        names = [f"{remote}.letify-part-{n}" for n in range(len(parts))]
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            for future in [
                pool.submit(self._put_part, n, p) for n, p in zip(names, parts, strict=True)
            ]:
                future.result()
        return names

    # -- download --------------------------------------------------------------

    def size(self, remote: str) -> int:
        model = json.loads(
            self._send("GET", self._address("api/contents", remote, content="0"), remote)
        )
        if model.get("type") != "file":
            raise RuntimeFailure(f"Colab file API: {remote} is a {model.get('type')}, not a file")
        return int(model["size"])

    def download(self, remote: str) -> bytes:
        total = self.size(remote)
        address = self._address("files", remote)

        def read(start: int) -> bytes:
            end = min(start + PART_BYTES, total) - 1
            return self._send("GET", address, remote, headers={"Range": f"bytes={start}-{end}"})

        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            pieces = list(pool.map(read, range(0, total, PART_BYTES)))
        payload = b"".join(pieces)
        if len(payload) != total:
            raise RuntimeFailure(
                f"Colab file API: {remote} is {total} bytes but {len(payload)} arrived"
            )
        return payload


def join_source(path: str, parts: list[str], *, unpack: bool, target: str | None) -> str:
    """A program for the VM that joins uploaded parts in order and unpacks when asked."""
    return (
        "import os, shutil, tarfile\n"
        f"path, parts = {path!r}, {parts!r}\n"
        "with open(path, 'wb') as out:\n"
        "    for part in parts:\n"
        "        with open(part, 'rb') as handle:\n"
        "            shutil.copyfileobj(handle, out, 1 << 20)\n"
        "        os.remove(part)\n"
        f"if {unpack!r}:\n"
        f"    target = {target!r} or os.path.dirname(path)\n"
        "    os.makedirs(target, exist_ok=True)\n"
        "    with tarfile.open(path, 'r:gz') as archive:\n"
        "        try:\n"
        "            archive.extractall(target, filter='data')\n"
        "        except TypeError:\n"
        "            archive.extractall(target)\n"
    )


class ColabFiles:
    """Serves the file requests a one-shot Colab channel cannot run as a program."""

    def __init__(
        self,
        alias: str,
        session: str,
        runner: Callable[[str, float | None], str],
        *,
        workspace: str,
        transfer: Callable[[], ContentsTransfer] | None = None,
    ):
        self.alias = alias
        self.session = session
        self.runner = runner
        #: The workspace root, before ``~`` is expanded on the VM. Temporary files go under it.
        self.workspace = workspace
        #: Builds the transfer for another Jupyter server, such as a Kaggle session.
        self._transfer = transfer

    def transfer(self) -> ContentsTransfer:
        if self._transfer is not None:
            return self._transfer()
        # Read per request, because the CLI writes the state when the session is created.
        return ContentsTransfer(*session_endpoint(self.alias, self.session))

    def serve(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        op = payload.get("op")
        if op == "put_file":
            return self.put_file(payload, timeout)
        if op == "get_file":
            return self.get_file(payload["path"])
        if op == "pack_dir":
            return self.pack_dir(payload["path"], timeout)
        raise RuntimeFailure(f"the Colab file API does not serve {op!r}")

    def put_file(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        path = payload["path"]
        data = base64.b64decode(payload["payload"])
        self.runner(
            f"import os\nos.makedirs(os.path.dirname({path!r}) or '/', exist_ok=True)\n", timeout
        )
        parts = self.transfer().upload(data, path)
        source = join_source(
            path, parts, unpack=bool(payload.get("unpack")), target=payload.get("target")
        )
        self.runner(source, timeout)
        return {"path": path, "size": len(data)}

    def get_file(self, path: str) -> dict[str, Any]:
        from ..protocol import digest_of

        data = self.transfer().download(path)
        return {
            "payload": base64.b64encode(data).decode(),
            "digest": digest_of(data),
            "size": len(data),
        }

    def pack_dir(self, path: str, timeout: float | None) -> dict[str, Any]:
        name = f"letify-pack-{uuid.uuid4().hex}.tar.gz"
        output = self.runner(
            "import os, tarfile\n"
            f"root = {path!r}\n"
            f"folder = os.path.join(os.path.expanduser({self.workspace!r}), 'tmp')\n"
            "os.makedirs(folder, exist_ok=True)\n"
            f"packed = os.path.join(folder, {name!r})\n"
            "with tarfile.open(packed, 'w:gz') as archive:\n"
            "    archive.add(root, arcname=os.path.basename(root.rstrip('/')), recursive=True)\n"
            "print(packed)\n",
            timeout,
        )
        # The VM expands the root, so the path it printed is the one to read back.
        lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
        if not lines:
            raise RuntimeFailure(f"packing {path} on the Colab VM printed no archive path")
        archive = lines[-1]
        try:
            return self.get_file(archive)
        finally:
            self.runner(f"import os\nos.remove({archive!r})\n", timeout)


__all__ = [
    "CHUNK_BYTES",
    "PARALLEL",
    "PART_BYTES",
    "ColabFiles",
    "ContentsTransfer",
    "join_source",
    "session_endpoint",
]
