"""Installing eci after the user confirms.

Spec section pinned here: "Installing external tools" (lookup order, the question, the
verified download into the per-version cache, safe extraction and linking into the
project environment), with its hooks in "Rendezvous", "Logging in" and "Elice machines".

The GitHub releases are replaced by a local HTTP server serving small archives, so no
test reaches the network. Everything else, the download, the digest check, extraction,
hard links and the launcher, runs for real in temporary directories.
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
import subprocess
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from letify import install
from letify.cli import main

ECI_TOP = f"eci-linux-x86_64-{install.ECI_VERSION}"
ECI_ASSET = f"{ECI_TOP}.tar.gz"
ECI_ZIP_TOP = f"eci-windows-x86_64-{install.ECI_VERSION}"
ECI_ZIP = f"{ECI_ZIP_TOP}.zip"
FAKE_ECI = b'#!/bin/sh\necho "eci $*"\n'


# -- helpers ---------------------------------------------------------------------


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tar_gz(members: list[tuple[str, bytes | str, str]]) -> bytes:
    """A tar.gz of (name, data or link target, kind) with kind file, dir or symlink."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                assert isinstance(data, bytes)
                info.size = len(data)
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = str(data)
                archive.addfile(info)
    return buffer.getvalue()


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class ReleaseServer:
    """A local HTTP server answering GET with the bytes registered for a path."""

    def __init__(self) -> None:
        files: dict[str, bytes] = {}
        self.files = files
        self.requests: list[str] = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                requests.append(self.path)
                body = files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def publish(self, tag: str, assets: dict[str, bytes], *, checksums: str | None = None) -> None:
        lines = checksums
        if lines is None:
            lines = "".join(f"{digest(data)}  {name}\n" for name, data in assets.items())
        self.files[f"/{tag}/checksums.txt"] = lines.encode()
        for name, data in assets.items():
            self.files[f"/{tag}/{name}"] = data


@pytest.fixture
def releases(monkeypatch, isolated_home):
    """Serve the release locally, pin its digest, and pretend to be Linux amd64."""
    server = ReleaseServer()
    monkeypatch.setattr(install, "ECI_RELEASE", server.url + "/{version}")
    monkeypatch.setattr(install, "host", lambda: ("Linux", "x86_64"))
    yield server
    server.server.shutdown()


def publish_eci(monkeypatch, server: ReleaseServer, archive: bytes | None = None) -> bytes:
    if archive is None:
        archive = tar_gz(
            [
                (ECI_TOP, b"", "dir"),
                (f"{ECI_TOP}/eci", FAKE_ECI, "file"),
                (f"{ECI_TOP}/_asyncio.so", b"library", "file"),
            ]
        )
    server.publish(install.ECI_VERSION, {ECI_ASSET: archive})
    monkeypatch.setattr(install, "ECI_SHA256", {ECI_ASSET: digest(archive)})
    return archive


@pytest.fixture
def venv(monkeypatch, tmp_path: Path) -> Path:
    """A project virtual environment the lookup links into."""
    root = tmp_path / "project-venv"
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    monkeypatch.setattr(install, "project_environment", lambda: root)
    return root


@pytest.fixture
def nothing_on_path(patch_which):
    patch_which(install, present=False)


def cached(tool: str, version: str) -> Path:
    return Path.home() / ".letify" / "tools" / tool / version / tool


# -- Spec: Installing external tools, installing ---------------------------------------


def test_setup_eci_installs_the_verified_bundle_into_the_version_cache(
    releases, monkeypatch, nothing_on_path, capsys
) -> None:
    publish_eci(monkeypatch, releases)
    assert main(["setup", "eci"]) == 0

    binary = cached("eci", install.ECI_VERSION)
    assert binary.read_bytes() == FAKE_ECI
    assert stat.S_IMODE(binary.stat().st_mode) == 0o555
    assert f"eci {install.ECI_VERSION} at {binary}" in capsys.readouterr().out
    assert f"/{install.ECI_VERSION}/checksums.txt" in releases.requests


def test_a_digest_that_differs_from_the_pinned_one_writes_nothing(
    releases, monkeypatch, nothing_on_path, capsys
) -> None:
    archive = publish_eci(monkeypatch, releases)
    monkeypatch.setattr(install, "ECI_SHA256", {ECI_ASSET: "0" * 64})
    assert main(["setup", "eci"]) == 1

    err = capsys.readouterr().err
    assert "0" * 64 in err and digest(archive) in err
    assert not (Path.home() / ".letify" / "tools" / "eci").exists()


def test_a_checksums_file_that_disagrees_with_the_pin_writes_nothing(
    releases, monkeypatch, nothing_on_path, capsys
) -> None:
    archive = publish_eci(monkeypatch, releases)
    releases.publish(
        install.ECI_VERSION, {ECI_ASSET: archive}, checksums=f"{'1' * 64}  {ECI_ASSET}\n"
    )
    assert main(["setup", "eci"]) == 1
    assert "1" * 64 in capsys.readouterr().err
    assert not cached("eci", install.ECI_VERSION).exists()


@pytest.mark.parametrize(
    "members",
    [
        [(f"{ECI_TOP}/../eci", FAKE_ECI, "file")],
        [("/tmp/eci", FAKE_ECI, "file")],
        [(f"{ECI_TOP}/eci", "../../../../usr/bin/sh", "symlink")],
        [(f"{ECI_TOP}/eci", FAKE_ECI, "file"), ("extra/../../escape", b"x", "file")],
    ],
    ids=["dot-dot", "absolute", "symlink-out", "second-member"],
)
def test_an_archive_with_a_member_escaping_its_directory_is_refused(
    releases, monkeypatch, nothing_on_path, capsys, members
) -> None:
    publish_eci(monkeypatch, releases, tar_gz(members))
    assert main(["setup", "eci"]) == 1
    assert "refused" in capsys.readouterr().err
    assert not cached("eci", install.ECI_VERSION).exists()
    assert not (Path.home() / ".letify" / "escape").exists()


def test_the_windows_release_zip_installs_eci_exe(releases, monkeypatch, nothing_on_path) -> None:
    monkeypatch.setattr(install, "host", lambda: ("Windows", "AMD64"))
    archive = zip_bytes({f"{ECI_ZIP_TOP}/eci.exe": b"MZ fake", f"{ECI_ZIP_TOP}/lib.dll": b"dll"})
    releases.publish(install.ECI_VERSION, {ECI_ZIP: archive})
    monkeypatch.setattr(install, "ECI_SHA256", {ECI_ZIP: digest(archive)})
    path = install.install("eci")
    assert path.name == "eci.exe"
    assert path.read_bytes() == b"MZ fake"
    assert (path.parent / "lib.dll").read_bytes() == b"dll"


def test_a_platform_with_no_release_asset_fails_with_the_releases_page(
    releases, monkeypatch
) -> None:
    monkeypatch.setattr(install, "host", lambda: ("Plan9", "mips"))
    with pytest.raises(install.InstallError, match=r"github\.com/elice-dev/eci-cli/releases"):
        install.install("eci")


def test_eci_installs_its_whole_bundle_with_the_top_directory_stripped(
    releases, monkeypatch, nothing_on_path
) -> None:
    publish_eci(monkeypatch, releases)
    path = install.install("eci")
    assert path == cached("eci", install.ECI_VERSION)
    assert (path.parent / "_asyncio.so").read_bytes() == b"library"
    assert not os.access(path.parent / "_asyncio.so", os.W_OK)


# -- Spec: Installing external tools, linking ------------------------------------------


def test_a_link_left_by_another_version_is_replaced(
    releases, monkeypatch, nothing_on_path, venv
) -> None:
    (venv / "bin" / "eci").write_bytes(b"old version")
    (venv / "bin" / ".letify-eci").write_text("0.1.0\n")
    publish_eci(monkeypatch, releases)
    assert main(["setup", "eci"]) == 0
    assert (venv / "bin" / ".letify-eci").read_text().strip() == install.ECI_VERSION
    assert b"old version" not in (venv / "bin" / "eci").read_bytes()


def test_a_file_letify_did_not_create_in_the_environment_is_left_untouched(
    releases, monkeypatch, nothing_on_path, venv, capsys
) -> None:
    foreign = venv / "bin" / "eci"
    foreign.write_bytes(b"the user's own eci")
    publish_eci(monkeypatch, releases)
    install.install("eci")
    capsys.readouterr()

    assert install.link("eci") == str(cached("eci", install.ECI_VERSION))
    assert foreign.read_bytes() == b"the user's own eci"
    assert "was not created by letify" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="the POSIX launcher")
def test_eci_is_linked_as_a_launcher_that_runs_the_cached_bundle(
    releases, monkeypatch, nothing_on_path, venv
) -> None:
    publish_eci(monkeypatch, releases)
    assert main(["setup", "eci"]) == 0

    launcher = venv / "bin" / "eci"
    ran = subprocess.run([str(launcher), "zone", "list"], capture_output=True, text=True)
    assert ran.stdout.strip() == "eci zone list"
    assert (venv / "bin" / ".letify-eci").read_text().strip() == install.ECI_VERSION


# -- Spec: Installing external tools, lookup -------------------------------------------


def test_the_cache_is_used_before_path_and_is_linked_on_the_way(
    releases, monkeypatch, patch_which, venv
) -> None:
    publish_eci(monkeypatch, releases)
    patch_which(install, present=False)
    install.install("eci")
    patch_which(install, present=True)
    assert install.find("eci") == str(venv / "bin" / "eci")


def test_a_users_own_eci_on_path_is_used_as_the_command_name_and_never_replaced(
    isolated_home, patch_which, venv
) -> None:
    patch_which(install, present=True)
    assert install.find("eci") == "eci"
    assert not (venv / "bin" / "eci").exists()


def test_setup_where_prints_the_cache_the_link_path_and_the_choice(
    isolated_home, patch_which, capsys
) -> None:
    patch_which(install, present=True)
    assert main(["setup", "eci", "--where"]) == 0
    out = capsys.readouterr().out
    assert "missing" in out
    assert "no project environment" in out
    assert "/usr/bin/eci" in out
    assert "uses" in out


# -- Spec: Installing external tools, installing on need -----------------------------


def test_a_confirmed_tool_is_installed_and_both_steps_are_logged(
    releases, monkeypatch, nothing_on_path, venv, capsys
) -> None:
    publish_eci(monkeypatch, releases)
    path = install.ensure("eci", instructions="how to install", confirm=lambda: True)

    assert path == str(venv / "bin" / "eci")
    err = capsys.readouterr().err.splitlines()
    # Download progress lines are spec "Download progress"; the two install lines stay.
    lines = [
        line
        for line in err
        if line.startswith("letify: ") and not line.startswith("letify: downloading")
    ]
    version = install.ECI_VERSION
    assert len(lines) == 2
    assert f"installing eci {version} from {releases.url}/{version}/{ECI_ASSET}" in lines[0]
    assert str(cached("eci", version).parent) in lines[0]
    archive_digest = install.ECI_SHA256[ECI_ASSET]
    assert f"verified sha256 {archive_digest}, linked at {path}" in lines[1]


# -- Spec: hooks in Rendezvous, Logging in and Elice machines --------------------------


def test_the_elice_provider_finds_eci_in_the_cache(releases, monkeypatch, nothing_on_path) -> None:
    from letify.providers import elice

    publish_eci(monkeypatch, releases)
    install.install("eci")
    assert elice.find_eci() == str(cached("eci", install.ECI_VERSION))


def test_pinned_digests_cover_every_published_asset() -> None:
    assert len(install.ECI_SHA256) == 3
    assert all(len(value) == 64 for value in install.ECI_SHA256.values())


# -- Spec: Installing external tools, download progress ------------------------------


class _Terminal(io.StringIO):
    """A standard error stand-in that says it is a terminal."""

    def isatty(self) -> bool:
        return True


def test_a_download_without_a_terminal_logs_progress_at_each_quarter(
    releases, monkeypatch, nothing_on_path, capsys
) -> None:
    archive = tar_gz([(ECI_TOP, b"", "dir"), (f"{ECI_TOP}/eci", os.urandom(4096), "file")])
    publish_eci(monkeypatch, releases, archive)
    monkeypatch.setattr(install, "DOWNLOAD_CHUNK", 64)
    install.install("eci")

    lines = [line for line in capsys.readouterr().err.splitlines() if "downloading" in line]
    assert [
        line.split("] ")[-1].split()[0] if "]" in line else line.split()[4] for line in lines
    ] == [
        "25%",
        "50%",
        "75%",
        "100%",
    ]
    assert all(
        line.startswith(f"letify: downloading eci {install.ECI_VERSION} ") for line in lines
    )
    assert "\r" not in capsys.readouterr().err


def test_a_download_on_a_terminal_redraws_one_line_with_a_gauge(releases, monkeypatch) -> None:
    body = os.urandom(200_000)
    releases.files["/blob"] = body
    stream = _Terminal()
    monkeypatch.setattr(install, "DOWNLOAD_CHUNK", 1024)
    clock = iter(range(10_000))
    monkeypatch.setattr(install.time, "monotonic", lambda: next(clock) * 0.2)

    data = install._download(releases.url + "/blob", label="eci 0.2.1", stream=stream)

    assert data == body
    text = stream.getvalue()
    assert text.count("\r") > 1
    assert text.endswith("\n")
    final = text.rstrip("\n").split("\r")[-1]
    assert final.startswith("letify: downloading eci 0.2.1 [")
    assert "100%" in final and "MiB/s" in final


def test_a_download_with_no_size_shows_bytes_and_rate_without_a_percent(monkeypatch) -> None:
    class NoLength:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.left = [b"x" * 1024] * 3

        def read(self, size: int) -> bytes:
            return self.left.pop() if self.left else b""

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(install.urllib.request, "urlopen", lambda url, timeout: NoLength())
    # The subject here is the progress line, and the host is a deliberately invalid one,
    # so the reachability probe is not what this test is about.
    monkeypatch.setattr(install, "_reachable", lambda url: None)
    stream = io.StringIO()
    assert (
        install._download("http://example.invalid/x", label="eci 0.2.1", stream=stream)
        == b"x" * 3072
    )
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert "%" not in lines[0] and "MiB/s" in lines[0]


# -- Spec: Elice machines, eci installed only after the user says yes -----------------


def test_eci_is_not_installed_without_a_yes(
    releases, monkeypatch, nothing_on_path
) -> None:
    publish_eci(monkeypatch, releases)
    with pytest.raises(install.InstallError, match="letify setup eci"):
        install.ensure("eci", instructions="how to install eci")
    with pytest.raises(install.InstallError, match="letify setup eci"):
        install.ensure("eci", instructions="how to install eci", confirm=lambda: False)
    assert releases.requests == []
    assert not cached("eci", install.ECI_VERSION).exists()


def test_eci_is_installed_when_the_user_says_yes(releases, monkeypatch, nothing_on_path) -> None:
    publish_eci(monkeypatch, releases)
    asked: list[bool] = []
    path = install.ensure(
        "eci", instructions="how to install eci", confirm=lambda: asked.append(True) or True
    )
    assert asked == [True]
    assert path == str(cached("eci", install.ECI_VERSION))


def test_an_elice_login_asks_and_stops_on_no_writing_nothing(
    isolated_home, monkeypatch, nothing_on_path
) -> None:
    from letify.config import login

    monkeypatch.setattr(install, "find", lambda tool, **kw: None)
    installs: list[str] = []
    monkeypatch.setattr(install, "install_and_link", lambda tool, **kw: installs.append(tool))
    prompts: list[str] = []
    monkeypatch.setattr(login, "read_line", lambda prompt: prompts.append(prompt) or "n")
    assert main(["login", "elice", "elice_a"]) != 0
    assert prompts and "Install it now? [y/N]" in prompts[0]
    assert "Elice's software, not part of letify" in prompts[0]
    assert installs == []
    assert not (Path.home() / ".letify" / "accounts" / "elice_a").exists()


def test_an_elice_login_with_no_input_does_not_ask_or_install(
    isolated_home, monkeypatch, nothing_on_path, capsys
) -> None:
    from letify.config import login

    monkeypatch.setattr(install, "find", lambda tool, **kw: None)
    monkeypatch.setattr(login, "read_line", lambda prompt: pytest.fail("asked without a terminal"))
    installs: list[str] = []
    monkeypatch.setattr(install, "install_and_link", lambda tool, **kw: installs.append(tool))
    assert main(["login", "elice", "elice_a", "--no-input", "--token", "t"]) != 0
    assert installs == []
    assert "letify setup eci" in capsys.readouterr().err


def test_a_download_whose_host_cannot_be_reached_fails_fast_and_names_it(monkeypatch) -> None:
    """Spec "Installing external tools": the connection is made first, with a timeout of its own.

    A campus network blocks release-assets.githubusercontent.com while github.com answers,
    so `letify setup eci` sat through the read timeout twice and then said only that the
    download failed.
    """
    import socket

    from letify import install

    tried: list[tuple] = []

    def refuse(address, timeout=None):
        tried.append((address, timeout))
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", refuse)
    with pytest.raises(install.InstallError) as failure:
        install._download("https://release-assets.githubusercontent.com/a/b.tar.gz")
    message = str(failure.value)
    assert "release-assets.githubusercontent.com" in message
    assert "443" in message
    assert "eci_binary" in message
    assert tried and tried[0][0] == ("release-assets.githubusercontent.com", 443)
    assert tried[0][1] == install.CONNECT_TIMEOUT


def test_a_reachable_host_is_downloaded_from_as_before(releases, monkeypatch) -> None:
    """Spec: the probe only decides whether to try, it does not replace the download.

    The release server is a real one on loopback, so the probe connects to it for real.
    """
    archive = publish_eci(monkeypatch, releases)
    asset = install.asset_for("eci")
    assert install._download(asset.url) == archive
