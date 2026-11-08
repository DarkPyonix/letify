"""Spec "User-Agent": every HTTP request letify sends carries ``User-Agent: letify/<version>``.

Each client is pointed at one local server that records the header of every request.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from typing import ClassVar

import pytest

import letify


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list[str | None]] = []

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        type(self).seen.append(self.headers.get("User-Agent"))
        body = json.dumps(
            {"access_token": "t", "expires_in": 3600, "id": "x", "items": []}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_PUT = _answer

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def echo() -> Iterator[tuple[str, list[str | None]]]:
    handler = type("Handler", (_Recorder,), {"seen": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", handler.seen
    server.shutdown()
    server.server_close()


class _Token:
    def token(self) -> str:
        return "tok"


def _elice(url: str, tmp_path) -> None:
    from letify.providers import elice

    elice.request("GET", "/x", token="t", endpoint=url)


def _colab_refresh(url: str, tmp_path) -> None:
    from letify.providers import colab

    colab._access_token(
        {"client_id": "c", "client_secret": "s", "refresh_token": "r", "token_uri": url}
    )


def _colab_files(url: str, tmp_path) -> None:
    from letify.providers.colab_files import ContentsTransfer

    ContentsTransfer(url, "tok")._send("GET", url + "/x", "x")


def _google_refresh(url: str, tmp_path) -> None:
    from letify.store.backends.google_auth import TokenSource

    TokenSource()._refresh({"token_uri": url}, "test")


def _gcs_request(url: str, tmp_path) -> None:
    from letify.store.backends.objects import GCSBackend

    GCSBackend("b", tokens=_Token())._request("GET", url, what="x")


def _gcs_read_token(url: str, tmp_path) -> None:
    from letify.store.backends.objects import GCSBackend

    GCSBackend("b", sts_endpoint=url, tokens=_Token()).read_token()


def _gcs_put_file(url: str, tmp_path) -> None:
    from letify.store.backends.objects import GCSBackend

    blob = tmp_path / "blob"
    blob.write_bytes(b"data")
    GCSBackend("b", endpoint=url, tokens=_Token()).put_file("d" * 64, str(blob), 4)


def _drive_call(url: str, tmp_path) -> None:
    from letify.store.backends.drive import DriveBackend

    DriveBackend("a", "p", endpoint=url, tokens=_Token())._call("GET", url, what="x")


def _portmap_fetch(url: str, tmp_path) -> None:
    from letify.transport import portmap

    portmap._fetch(url, 5)


def _portmap_soap(url: str, tmp_path) -> None:
    from letify.transport import portmap

    portmap._soap(url, "svc", "act", "<x/>", 5)


def _install_download(url: str, tmp_path) -> None:
    from letify import install

    install._download(url)


@pytest.mark.parametrize(
    "send",
    [
        _elice,
        _colab_refresh,
        _colab_files,
        _google_refresh,
        _gcs_request,
        _gcs_read_token,
        _gcs_put_file,
        _drive_call,
        _portmap_fetch,
        _portmap_soap,
        _install_download,
    ],
    ids=lambda send: send.__name__.lstrip("_"),
)
def test_a_client_request_names_letify_as_its_user_agent(echo, tmp_path, send) -> None:
    url, seen = echo
    try:
        send(url, tmp_path)
    except Exception:
        pass
    assert seen, "the client sent no request"
    assert set(seen) == {f"letify/{letify.__version__}"}


def test_a_runtime_pull_carries_the_user_agent_in_its_headers() -> None:
    """The worker sends the headers it is given, so the producers name letify in them."""
    from letify.store.backends.drive import DriveBackend
    from letify.store.backends.objects import GCSBackend

    expected = f"letify/{letify.__version__}"
    drive = DriveBackend("a", "p", tokens=_Token())
    assert drive._headers()["User-Agent"] == expected
    gcs = GCSBackend("b", tokens=_Token())
    gcs.read_token = lambda: "tok"  # type: ignore[method-assign]
    assert gcs.pull_source("d" * 64)["headers"]["User-Agent"] == expected  # type: ignore[index]
