"""Installing tailcat and eci after the user confirms.

Spec section pinned here: "Installing external tools" (lookup order, the question, the
verified download into the per-version cache, safe extraction and linking into the
project environment), with its hooks in "Rendezvous", "Logging in" and "Elice machines".

The GitHub releases are replaced by a local HTTP server serving small archives, so no
test reaches the network. Everything else, the download, the digest check, extraction,
hard links and the launcher, runs for real in temporary directories.
"""

from __future__ import annotations

import errno
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
from letify.transport import setup

TAILCAT_ASSET = f"tailcat_{setup.TAILCAT_VERSION}_linux_amd64.tar.gz"
TAILCAT_ZIP = f"tailcat_{setup.TAILCAT_VERSION}_windows_amd64.zip"
ECI_TOP = f"eci-linux-x86_64-{install.ECI_VERSION}"
ECI_ASSET = f"{ECI_TOP}.tar.gz"
FAKE_TAILCAT = b"#!/bin/sh\necho fake tailcat\n"
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
    """Serve both releases locally, pin their digests, and pretend to be Linux amd64."""
    server = ReleaseServer()
    monkeypatch.setattr(install, "TAILCAT_RELEASE", server.url + "/v{version}")
    monkeypatch.setattr(install, "ECI_RELEASE", server.url + "/{version}")
    monkeypatch.setattr(install, "host", lambda: ("Linux", "x86_64"))
    monkeypatch.delenv("LETIFY_AUTO_INSTALL", raising=False)
    yield server
    server.server.shutdown()


def publish_tailcat(monkeypatch, server: ReleaseServer, archive: bytes, name: str = TAILCAT_ASSET):
    server.publish(f"v{setup.TAILCAT_VERSION}", {name: archive})
    monkeypatch.setattr(install, "TAILCAT_SHA256", {name: digest(archive)})


def publish_eci(monkeypatch, server: ReleaseServer) -> None:
    archive = tar_gz(
        [
            (ECI_TOP, b"", "dir"),
            (f"{ECI_TOP}/eci", FAKE_ECI, "file"),
            (f"{ECI_TOP}/_asyncio.so", b"library", "file"),
        ]
    )
    server.publish(install.ECI_VERSION, {ECI_ASSET: archive})
    monkeypatch.setattr(install, "ECI_SHA256", {ECI_ASSET: digest(archive)})


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


def test_setup_tailcat_with_yes_installs_the_verified_binary_into_the_version_cache(
    releases, monkeypatch, nothing_on_path, capsys
) -> None:
    publish_tailcat(monkeypatch, releases, tar_gz([("tailcat", FAKE_TAILCAT, "file")]))
    assert main(["setup", "tailcat"]) == 0

    binary = cached("tailcat", setup.TAILCAT_VERSION)
    assert binary.read_bytes() == FAKE_TAILCAT
    assert stat.S_IMODE(binary.stat().st_mode) == 0o555
    assert f"tailcat {setup.TAILCAT_VERSION} at {binary}" in capsys.readouterr().out
    assert f"/v{setup.TAILCAT_VERSION}/checksums.txt" in releases.requests


def test_a_digest_that_differs_from_the_pinned_one_writes_nothing(
    releases, monkeypatch, nothing_on_path, capsys
) -> None:
    archive = tar_gz([("tailcat", FAKE_TAILCAT, "file")])
    publish_tailcat(monkeypatch, releases, archive)
    monkeypatch.setattr(install, "TAILCAT_SHA256", {TAILCAT_ASSET: "0" * 64})
    assert main(["setup", "tailcat"]) == 1

    err = capsys.readouterr().err
    assert "0" * 64 in err and digest(archive) in err
    assert not (Path.home() / ".letify" / "tools" / "tailcat").exists()


def test_a_checksums_file_that_disagrees_with_the_pin_writes_nothing(
    releases, monkeypatch, nothing_on_path, capsys
) -> None:
    archive = tar_gz([("tailcat", FAKE_TAILCAT, "file")])
    releases.publish(
        f"v{setup.TAILCAT_VERSION}",
        {TAILCAT_ASSET: archive},
        checksums=f"{'1' * 64}  {TAILCAT_ASSET}\n",
    )
    monkeypatch.setattr(install, "TAILCAT_SHA256", {TAILCAT_ASSET: digest(archive)})
    assert main(["setup", "tailcat"]) == 1
    assert "1" * 64 in capsys.readouterr().err
    assert not cached("tailcat", setup.TAILCAT_VERSION).exists()


@pytest.mark.parametrize(
    "members",
    [
        [("../tailcat", FAKE_TAILCAT, "file")],
        [("/tmp/tailcat", FAKE_TAILCAT, "file")],
        [("tailcat", "../../../../usr/bin/sh", "symlink")],
        [("tailcat", FAKE_TAILCAT, "file"), ("extra/../../escape", b"x", "file")],
    ],
    ids=["dot-dot", "absolute", "symlink-out", "second-member"],
)
def test_an_archive_with_a_member_escaping_its_directory_is_refused(
    releases, monkeypatch, nothing_on_path, capsys, members
) -> None:
    publish_tailcat(monkeypatch, releases, tar_gz(members))
    assert main(["setup", "tailcat"]) == 1
    assert "refused" in capsys.readouterr().err
    assert not cached("tailcat", setup.TAILCAT_VERSION).exists()
    assert not (Path.home() / ".letify" / "escape").exists()


def test_the_windows_release_zip_installs_tailcat_exe(
    releases, monkeypatch, nothing_on_path
) -> None:
    monkeypatch.setattr(install, "host", lambda: ("Windows", "AMD64"))
    archive = zip_bytes({"tailcat.exe": b"MZ fake", "LICENSE": b"BSD"})
    publish_tailcat(monkeypatch, releases, archive, TAILCAT_ZIP)
    path = install.install("tailcat")
    assert path.name == "tailcat.exe"
    assert path.read_bytes() == b"MZ fake"
    assert not (path.parent / "LICENSE").exists()


def test_a_platform_with_no_release_asset_fails_with_the_releases_page(
    releases, monkeypatch
) -> None:
    monkeypatch.setattr(install, "host", lambda: ("Plan9", "mips"))
    with pytest.raises(install.InstallError, match=r"github\.com/tailscale/tailcat/releases"):
        install.install("tailcat")


def test_on_macos_with_homebrew_tailcat_comes_from_brew_and_is_not_cached(
    releases, monkeypatch, patch_which, patch_run
) -> None:
    monkeypatch.setattr(install, "host", lambda: ("Darwin", "arm64"))
    patch_which(install, present=lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    recorder = patch_run(install)
    install.install("tailcat")
    assert recorder.command == ["/opt/homebrew/bin/brew", "install", "tailcat"]
    assert releases.requests == []


def test_eci_installs_its_whole_bundle_with_the_top_directory_stripped(
    releases, monkeypatch, nothing_on_path
) -> None:
    publish_eci(monkeypatch, releases)
    path = install.install("eci")
    assert path == cached("eci", install.ECI_VERSION)
    assert (path.parent / "_asyncio.so").read_bytes() == b"library"
    assert not os.access(path.parent / "_asyncio.so", os.W_OK)


# -- Spec: Installing external tools, linking ------------------------------------------


def test_the_cached_tailcat_is_hard_linked_into_the_project_environment_with_a_marker(
    releases, monkeypatch, nothing_on_path, venv
) -> None:
    publish_tailcat(monkeypatch, releases, tar_gz([("tailcat", FAKE_TAILCAT, "file")]))
    assert main(["setup", "tailcat"]) == 0

    link = venv / "bin" / "tailcat"
    assert link.samefile(cached("tailcat", setup.TAILCAT_VERSION))
    assert (venv / "bin" / ".letify-tailcat").read_text().strip() == setup.TAILCAT_VERSION
    assert install.find("tailcat") == str(link)


def test_a_failed_hard_link_falls_back_to_a_copy_and_says_so(
    releases, monkeypatch, nothing_on_path, venv, capsys
) -> None:
    publish_tailcat(monkeypatch, releases, tar_gz([("tailcat", FAKE_TAILCAT, "file")]))

    def cross_device(source, target):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(install.os, "link", cross_device)
    assert main(["setup", "tailcat"]) == 0

    link = venv / "bin" / "tailcat"
    assert link.read_bytes() == FAKE_TAILCAT
    assert not link.samefile(cached("tailcat", setup.TAILCAT_VERSION))
    assert capsys.readouterr().err.count("linked by copy") == 1


def test_a_link_left_by_another_version_is_replaced(
    releases, monkeypatch, nothing_on_path, venv
) -> None:
    (venv / "bin" / "tailcat").write_bytes(b"old version")
    (venv / "bin" / ".letify-tailcat").write_text("0.5.0\n")
    publish_tailcat(monkeypatch, releases, tar_gz([("tailcat", FAKE_TAILCAT, "file")]))
    assert main(["setup", "tailcat"]) == 0
    assert (venv / "bin" / "tailcat").read_bytes() == FAKE_TAILCAT


def test_a_file_letify_did_not_create_in_the_environment_is_left_untouched(
    releases, monkeypatch, nothing_on_path, venv, capsys
) -> None:
    foreign = venv / "bin" / "tailcat"
    foreign.write_bytes(b"the user's own tailcat")
    publish_tailcat(monkeypatch, releases, tar_gz([("tailcat", FAKE_TAILCAT, "file")]))
    install.install("tailcat")
    capsys.readouterr()

    assert install.link("tailcat") == str(cached("tailcat", setup.TAILCAT_VERSION))
    assert foreign.read_bytes() == b"the user's own tailcat"
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
    publish_tailcat(monkeypatch, releases, tar_gz([("tailcat", FAKE_TAILCAT, "file")]))
    patch_which(install, present=False)
    install.install("tailcat")
    patch_which(install, present=True)
    assert install.find("tailcat") == str(venv / "bin" / "tailcat")


def test_a_users_own_tailcat_on_path_is_used_as_the_command_name_and_never_replaced(
    isolated_home, patch_which, venv
) -> None:
    patch_which(install, present=True)
    assert install.find("tailcat") == "tailcat"
    assert not (venv / "bin" / "tailcat").exists()


def test_setup_where_prints_the_cache_the_link_path_and_the_choice(
    isolated_home, patch_which, capsys
) -> None:
    patch_which(install, present=True)
    assert main(["setup", "tailcat", "--where"]) == 0
    out = capsys.readouterr().out
    assert "missing" in out
    assert "no project environment" in out
    assert "/usr/bin/tailcat" in out
    assert "uses" in out


# -- Spec: Installing external tools, automatic install ------------------------------


def test_a_missing_tool_is_installed_on_first_need_and_both_steps_are_logged(
    releases, monkeypatch, nothing_on_path, venv, capsys
) -> None:
    publish_tailcat(monkeypatch, releases, tar_gz([("tailcat", FAKE_TAILCAT, "file")]))
    path = install.ensure("tailcat", instructions="how to install")

    assert path == str(venv / "bin" / "tailcat")
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("letify: ")]
    version = setup.TAILCAT_VERSION
    assert len(lines) == 2
    assert (
        f"installing tailcat {version} from {releases.url}/v{version}/{TAILCAT_ASSET}" in lines[0]
    )
    assert str(cached("tailcat", version).parent) in lines[0]
    archive_digest = install.TAILCAT_SHA256[TAILCAT_ASSET]
    assert f"verified sha256 {archive_digest}, linked at {path}" in lines[1]


def test_auto_install_false_in_the_home_config_fails_with_the_setup_command(
    releases, nothing_on_path
) -> None:
    (Path.home() / ".letify" / "config.toml").write_text("auto_install = false\n")
    with pytest.raises(install.InstallError) as raised:
        install.ensure("tailcat", instructions="how to install")
    assert "how to install" in str(raised.value)
    assert "letify setup tailcat" in str(raised.value)
    assert releases.requests == []


@pytest.mark.parametrize("value", ["0", "false", "no"])
def test_letify_auto_install_off_in_the_environment_installs_nothing(
    releases, monkeypatch, nothing_on_path, value
) -> None:
    monkeypatch.setenv("LETIFY_AUTO_INSTALL", value)
    with pytest.raises(install.InstallError, match="letify setup eci"):
        install.ensure("eci", instructions="how to install")
    assert releases.requests == []


def test_the_environment_variable_decides_over_the_home_config(
    releases, monkeypatch, nothing_on_path
) -> None:
    (Path.home() / ".letify" / "config.toml").write_text("auto_install = false\n")
    monkeypatch.setenv("LETIFY_AUTO_INSTALL", "1")
    publish_eci(monkeypatch, releases)
    assert install.ensure("eci", instructions="how to install") == str(
        cached("eci", install.ECI_VERSION)
    )


def test_setup_installs_even_when_automatic_install_is_off(
    releases, monkeypatch, nothing_on_path
) -> None:
    monkeypatch.setenv("LETIFY_AUTO_INSTALL", "0")
    publish_eci(monkeypatch, releases)
    assert main(["setup", "eci"]) == 0
    assert cached("eci", install.ECI_VERSION).is_file()


def test_the_auto_install_setting_does_not_declare_a_provider(isolated_home) -> None:
    from letify.config import load

    (Path.home() / ".letify" / "config.toml").write_text("auto_install = false\n")
    assert list(load().providers) == ["local"]


# -- Spec: hooks in Rendezvous, Logging in and Elice machines --------------------------


def test_connect_with_automatic_install_off_names_the_setup_command(
    isolated_home, patch_which, capsys
) -> None:
    patch_which(setup, present=False)
    assert main(["client", "shell", "connect", "--ssh-port", "1"]) == 1
    assert "letify setup tailcat" in capsys.readouterr().err


def test_the_elice_provider_finds_eci_in_the_cache(releases, monkeypatch, nothing_on_path) -> None:
    from letify.providers import elice

    publish_eci(monkeypatch, releases)
    install.install("eci")
    assert elice.find_eci() == str(cached("eci", install.ECI_VERSION))


def test_the_tailcat_strategy_names_the_setup_command_when_tailcat_is_missing(
    nothing_on_path,
) -> None:
    from letify.transport.strategies import TailcatUDP, Target

    reason = TailcatUDP().needs(Target(alias="box"))
    assert reason is not None and "letify setup tailcat" in reason


def test_pinned_digests_cover_every_published_asset() -> None:
    version = setup.TAILCAT_VERSION
    for arch in ("amd64", "arm64", "armv7"):
        assert len(install.TAILCAT_SHA256[f"tailcat_{version}_linux_{arch}.tar.gz"]) == 64
    for arch in ("amd64", "arm64"):
        assert len(install.TAILCAT_SHA256[f"tailcat_{version}_windows_{arch}.zip"]) == 64
    assert len(install.ECI_SHA256) == 3
