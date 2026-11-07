"""Finding and, after the user confirms, installing the external tools letify runs.

Owns the lookup of ``eci`` (project environment, per-version cache,
``PATH``), the confirmed install, the verified download of the tool's pinned GitHub
release from its publisher into ``~/.letify/tools/<tool>/<version>/``, safe extraction,
and linking the cached tool into the project's virtual environment. Spec "Installing
external tools".

It does not own what the tools are asked to do, the install instructions printed when
nothing is installed (``letify.providers.elice`` does), or
any shared ``PATH`` directory, which it never writes to.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import render
from .errors import LetifyError

#: Seconds to wait for a release server.
DOWNLOAD_TIMEOUT = 60.0
#: Seconds to wait for the connection alone. A host a network blocks has to fail in
#: seconds, not after the read timeout, and twice that when a release redirects.
#: Spec "Installing external tools".
CONNECT_TIMEOUT = 10.0

#: Bytes read from a release server at a time, so progress can be reported.
DOWNLOAD_CHUNK = 64 << 10

#: Seconds between redraws of the progress line on a terminal.
PROGRESS_INTERVAL = 0.1

#: Cells in the progress gauge.
PROGRESS_CELLS = 20

#: The eci release letify installs.
ECI_VERSION = "0.2.1"

ECI_RELEASE = "https://github.com/elice-dev/eci-cli/releases/download/{version}"
ECI_RELEASES_PAGE = "https://github.com/elice-dev/eci-cli/releases"

#: SHA-256 of every eci release archive for ``ECI_VERSION``.
ECI_SHA256 = {
    "eci-darwin-arm64-0.2.1.tar.gz": (
        "2818cf974ed0151e4a33a8e1334662e47f3990c8d173b990808779808f321f81"
    ),
    "eci-linux-x86_64-0.2.1.tar.gz": (
        "f9649ca8e8ed63ec5fb1c35b424d26c9ce1ef25bbc789b571b8536728d3c0a99"
    ),
    "eci-windows-x86_64-0.2.1.zip": (
        "d167b7b605d997cbe04e655e95618cf97261e2ea110c48fd4550514f12ccdc49"
    ),
}

TOOLS = ("eci",)

_ECI_ARCH = {
    "Darwin": {"arm64": "arm64", "aarch64": "arm64"},
    "Linux": {"x86_64": "x86_64", "amd64": "x86_64"},
    "Windows": {"amd64": "x86_64", "x86_64": "x86_64"},
}

Say = Callable[[str, str], None]


class InstallError(LetifyError):
    """A tool is missing and was not installed, or its install failed."""


def _say(kind: str, message: str) -> None:
    style = render.Style.for_stream(sys.stderr)
    print(f"{render.mark(kind, style)} {message}", file=sys.stderr)


# -- the host ----------------------------------------------------------------------


def host() -> tuple[str, str]:
    """This operating system and CPU architecture, as ``platform`` names them."""
    return platform.system(), platform.machine()


def version_of(tool: str) -> str:
    return ECI_VERSION


def _executable(tool: str) -> str:
    return f"{tool}.exe" if host()[0] == "Windows" else tool


def tools_home() -> Path:
    from .paths import tools_directory

    return tools_directory()


def cache_path(tool: str) -> Path:
    """Where the pinned version of ``tool`` lives in the cache."""
    return tools_home() / tool / version_of(tool) / _executable(tool)


def project_environment() -> Path | None:
    """The project's virtual environment: the running one, or ``./.venv``."""
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)
    candidate = Path.cwd() / ".venv"
    return candidate if (candidate / "pyvenv.cfg").is_file() else None


def _bin_directory(environment: Path) -> Path:
    return environment / ("Scripts" if host()[0] == "Windows" else "bin")


def log(message: str) -> None:
    """One ``letify:`` line on standard error, as connection decision lines are."""
    print(f"letify: {message}", file=sys.stderr, flush=True)


# -- lookup ------------------------------------------------------------------------


def _link_path(tool: str) -> tuple[Path, Path] | None:
    environment = project_environment()
    if environment is None:
        return None
    directory = _bin_directory(environment)
    name = _executable(tool)
    if tool == "eci" and host()[0] == "Windows":
        name = "eci.cmd"
    return directory / name, directory / f".letify-{tool}"


def _marker_version(marker: Path) -> str | None:
    try:
        return marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def find(tool: str, *, link_cache: bool = True, say: Say = _say) -> str | None:
    """The command to run ``tool`` by, or None. Spec "Installing external tools", Lookup.

    A tool found on ``PATH`` is returned as its bare name, so a command line that names it
    reads the same as one the user would type.
    """
    paths = _link_path(tool)
    if paths is not None and paths[0].is_file():
        recorded = _marker_version(paths[1]) if paths[1].is_file() else None
        if recorded is None or recorded == version_of(tool):
            if not link_cache and recorded == version_of(tool) and cache_path(tool).is_file():
                return str(cache_path(tool))
            return str(paths[0])
    cached = cache_path(tool)
    if cached.is_file():
        return link(tool, say=say) if link_cache else str(cached)
    if shutil.which(tool) is not None:
        return tool
    return None


def link(tool: str, *, say: Say = _say) -> str:
    """Link the cached tool into the project environment and return the path to run."""
    cached = cache_path(tool)
    paths = _link_path(tool)
    if paths is None:
        return str(cached)
    target, marker = paths
    version = version_of(tool)
    if target.exists() or target.is_symlink():
        ours = marker.is_file()
        if not ours:
            try:
                ours = target.samefile(cached)
            except OSError:
                ours = False
        if not ours:
            say("warn", f"{target} exists and was not created by letify; using {cached}")
            return str(cached)
        if _marker_version(marker) == version and target.is_file():
            return str(target)
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_launcher(target, cached)
    marker.write_text(version + "\n", encoding="utf-8")
    return str(target)


def _write_launcher(target: Path, cached: Path) -> None:
    """A script running the cached eci, which must stay next to its libraries."""
    if target.suffix == ".cmd":
        target.write_text(f'@echo off\r\n"{cached}" %*\r\n', encoding="utf-8")
        return
    quoted = str(cached).replace("'", "'\\''")
    target.write_text(f"#!/bin/sh\nexec '{quoted}' \"$@\"\n", encoding="utf-8")
    target.chmod(0o755)


# -- installing --------------------------------------------------------------------


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    checksums_url: str
    pinned: str | None


def asset_for(tool: str) -> Asset:
    """The release archive for this host, or ``InstallError`` naming the releases page."""
    system, machine = host()
    machine = machine.lower()
    arch = _ECI_ARCH.get(system, {}).get(machine)
    extension = "zip" if system == "Windows" else "tar.gz"
    name = f"eci-{system.lower()}-{arch}-{ECI_VERSION}.{extension}"
    base, pins, page = ECI_RELEASE, ECI_SHA256, ECI_RELEASES_PAGE
    if arch is None:
        raise InstallError(
            f"{tool} {version_of(tool)} publishes no release for {system} {machine}; see {page}"
        )
    root = base.format(version=version_of(tool))
    return Asset(name, f"{root}/{name}", f"{root}/checksums.txt", pins.get(name))


class _Progress:
    """The download progress line of spec "Installing external tools", Download progress."""

    def __init__(self, label: str, total: int | None, stream, verb: str = "downloading") -> None:
        self.label = label
        self.verb = verb
        self.total = total if total and total > 0 else None
        self.stream = stream
        self.terminal = render.color_enabled(stream) or _isatty(stream)
        self.style = render.Style.for_stream(stream)
        self.started = time.monotonic()
        self.drawn = 0.0
        self.quarter = 0

    def _line(self, done: int, gauge: bool) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-6)
        rate = done / elapsed / (1 << 20)
        mib = done / (1 << 20)
        head = f"letify: {self.verb} {self.label} "
        if self.total is None:
            return f"{head}{mib:.1f} MiB {rate:.1f} MiB/s"
        fraction = done / self.total
        bar = render.gauge(fraction, PROGRESS_CELLS, self.style) + " " if gauge else ""
        total = self.total / (1 << 20)
        return f"{head}{bar}{fraction * 100:.0f}% {mib:.1f}/{total:.1f} MiB {rate:.1f} MiB/s"

    def update(self, done: int) -> None:
        if self.terminal:
            now = time.monotonic()
            if now - self.drawn >= PROGRESS_INTERVAL:
                self.drawn = now
                self.stream.write("\r" + self._line(done, gauge=True))
                self.stream.flush()
            return
        if self.total is None:
            return
        reached = min(4, done * 4 // self.total)
        while self.quarter < reached:
            self.quarter += 1
            shown = self.total if self.quarter == 4 else self.total * self.quarter // 4
            print(self._line(shown, gauge=False), file=self.stream, flush=True)

    def finish(self, done: int) -> None:
        if self.terminal:
            self.stream.write("\r" + self._line(done, gauge=True) + "\n")
            self.stream.flush()
        elif self.total is None:
            print(self._line(done, gauge=False), file=self.stream, flush=True)
        else:
            self.update(self.total)


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _reachable(url: str) -> None:
    """Refuse a download whose host does not answer, before the request is made.

    Spec "Installing external tools": the error names the host so a blocked network is told what
    to open or to configure around, rather than being told the tool is broken.
    """
    import socket
    import urllib.parse

    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname
    if not host:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.create_connection((host, port), timeout=CONNECT_TIMEOUT).close()
    except OSError as exc:
        raise InstallError(
            f"could not reach {host}:{port} within {CONNECT_TIMEOUT:.0f} s ({exc}), so "
            f"{url} was not downloaded. The host, not letify, is what is unreachable: a "
            f"network that allows github.com can still block the host its releases "
            f"redirect to. Fetch the archive another way and name the binary with the "
            f"account's eci_binary"
        ) from None


def _download(url: str, *, label: str | None = None, stream=None) -> bytes:
    """Fetch ``url``; with ``label``, report progress on ``stream`` (standard error)."""
    _reachable(url)
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            if label is None:
                return response.read()
            length = response.headers.get("Content-Length")
            total = int(length) if length and str(length).isdigit() else None
            progress = _Progress(label, total, stream if stream is not None else sys.stderr)
            parts: list[bytes] = []
            done = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                parts.append(chunk)
                done += len(chunk)
                progress.update(done)
            progress.finish(done)
            return b"".join(parts)
    except OSError as exc:
        raise InstallError(f"could not download {url}: {exc}") from None


def verify(asset: Asset, data: bytes, checksums: str) -> None:
    """Refuse an archive whose digest differs from the pin or from ``checksums.txt``."""
    actual = hashlib.sha256(data).hexdigest()
    if asset.pinned is None:
        raise InstallError(f"{asset.name} has no pinned SHA-256 digest in letify; refused")
    if actual != asset.pinned:
        raise InstallError(
            f"{asset.name} failed verification: expected SHA-256 {asset.pinned}, got {actual}"
        )
    listed = None
    for line in checksums.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].lstrip("*") == asset.name:
            listed = fields[0].lower()
    if listed != asset.pinned:
        raise InstallError(
            f"{asset.name} failed verification: checksums.txt lists {listed}, "
            f"letify pins {asset.pinned}"
        )


@dataclass(frozen=True)
class _Member:
    name: str
    kind: str  # "file", "dir" or "symlink"
    read: Callable[[], bytes]
    target: str = ""


def _safe_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if normalized.startswith("/") or (parts and ":" in parts[0]) or ".." in parts:
        raise InstallError(f"archive member {name!r} escapes its directory; refused")
    return "/".join(parts)


def _members(asset: Asset, data: bytes) -> list[_Member]:
    """Every member of the archive, refusing any that could escape the extraction."""
    found: list[_Member] = []
    if asset.name.endswith(".zip"):
        archive = zipfile.ZipFile(io.BytesIO(data))
        for info in archive.infolist():
            name = _safe_name(info.filename)
            if not name:
                continue
            kind = "dir" if info.is_dir() else "file"
            found.append(_Member(name, kind, lambda info=info: archive.read(info)))
        return found
    # Left open on purpose: members are read lazily while the archive is in memory.
    tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")  # noqa: SIM115
    for info in tar.getmembers():
        name = _safe_name(info.name)
        if not name:
            continue
        if info.isdir():
            found.append(_Member(name, "dir", lambda: b""))
        elif info.isreg():

            def read(info: tarfile.TarInfo = info) -> bytes:
                handle = tar.extractfile(info)
                assert handle is not None
                return handle.read()

            found.append(_Member(name, "file", read))
        elif info.issym():
            joined = posixpath.normpath(posixpath.join(posixpath.dirname(name), info.linkname))
            if info.linkname.startswith("/") or joined.startswith(".."):
                raise InstallError(f"archive member {info.name!r} links outside it; refused")
            found.append(_Member(name, "symlink", lambda: b"", info.linkname))
        else:
            raise InstallError(f"archive member {info.name!r} is not a regular file; refused")
    return found


def _strip_top(members: list[_Member]) -> list[_Member]:
    tops = {member.name.split("/", 1)[0] for member in members}
    if len(tops) != 1 or not any("/" in member.name for member in members):
        return members
    stripped = []
    for member in members:
        rest = member.name.split("/", 1)[1] if "/" in member.name else ""
        if rest:
            stripped.append(_Member(rest, member.kind, member.read, member.target))
    return stripped


def _select(tool: str, members: list[_Member]) -> list[_Member]:
    """The members to keep: the whole eci bundle, because the binary loads libraries beside it."""
    return _strip_top(members)


def install(tool: str, *, say: Say = _say) -> Path:
    """Download, verify and extract the pinned release into the cache. Spec, Installing."""
    asset = asset_for(tool)
    log(f"installing {tool} {version_of(tool)} from {asset.url} into {cache_path(tool).parent}")
    data = _download(asset.url, label=f"{tool} {version_of(tool)}")
    checksums = _download(asset.checksums_url).decode("utf-8", "replace")
    verify(asset, data, checksums)
    members = _select(tool, _members(asset, data))
    binary = _executable(tool)
    if not any(member.name == binary and member.kind == "file" for member in members):
        raise InstallError(f"{asset.name} holds no {binary}; refused")

    final = cache_path(tool).parent
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=final.parent))
    try:
        for member in members:
            path = staging.joinpath(*member.name.split("/"))
            if member.kind == "dir":
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.kind == "symlink":
                os.symlink(member.target, path)
                continue
            path.write_bytes(member.read())
            path.chmod(0o555 if member.name == binary else 0o444)
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
        for entry in list(final.parent.iterdir()):
            if entry.name != final.name and not entry.name.startswith("."):
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final / binary


# -- installing on need -------------------------------------------------------------


def setup_hint(tool: str) -> str:
    return f"Install it with: letify setup {tool}"


def install_and_link(tool: str, *, say: Say = _say) -> str:
    """Install into the cache, link into the project, log the result, return the path."""
    installed = install(tool, say=say)
    if installed.parent.parent.parent != tools_home():
        return str(installed)
    path = link(tool, say=say)
    pinned = asset_for(tool).pinned
    log(f"{tool} {version_of(tool)} verified sha256 {pinned}, linked at {path}")
    return path


def ensure(
    tool: str,
    *,
    instructions: str,
    say: Say = _say,
    confirm: Callable[[], bool] | None = None,
) -> str:
    """The command to run ``tool`` by, installing it first when the user says yes.

    The release carries no license, so ``eci`` installs only when ``confirm`` is given and
    returns true, which a login asks on a terminal.
    """
    found = find(tool, say=say)
    if found is not None:
        return found
    if confirm is None or not confirm():
        raise InstallError(f"{instructions}\n\n{setup_hint(tool)}")
    return install_and_link(tool, say=say)


def where(tool: str) -> list[tuple[str, str]]:
    """The ``--where`` lines: cache, link, PATH and the copy the lookup uses."""
    cached = cache_path(tool)
    lines = [("cache", f"{cached} ({'present' if cached.is_file() else 'missing'})")]
    paths = _link_path(tool)
    if paths is None:
        lines.append(("link", "no project environment"))
    else:
        target, marker = paths
        if not (target.exists() or target.is_symlink()):
            state = "missing"
        elif marker.is_file():
            state = f"letify {_marker_version(marker)}"
        else:
            state = "not letify's"
        lines.append(("link", f"{target} ({state})"))
    lines.append(("PATH", shutil.which(tool) or "none"))
    chosen = find(tool, link_cache=False)
    if chosen == tool:
        chosen = shutil.which(tool)
    lines.append(("uses", chosen or "nothing"))
    return lines


__all__ = [
    "ECI_VERSION",
    "TOOLS",
    "InstallError",
    "cache_path",
    "ensure",
    "find",
    "install",
    "link",
    "where",
]


def remote_link_source(version: str) -> str:
    """Prepare ``letify-link`` on the remote from the letify wheel of this version.

    Spec "Sidecar interface" item 7: the binary changes with every letify release, so a
    hash pinned in this source cannot describe the build that is about to be made. The
    remote therefore fetches the wheel for its own platform at exactly this version, which
    PyPI verifies, and takes the sidecar binary out of it.
    """
    import inspect

    return (
        inspect.getsource(_remote_link)
        + f"\nbinary = _remote_link({version!r})\n"
        + "import json\nprint('LETIFY-LINK ' + json.dumps(binary))\n"
    )


def _remote_link(version):
    """Take letify-link out of the letify wheel for this machine, standard library only."""
    import json
    import os
    import sys
    import tempfile
    import zipfile
    from pathlib import Path

    inside = "letify/remoting/lib/letify-link"
    target = Path(tempfile.gettempdir()) / f"letify-link-{version}"
    if target.is_file() and os.access(target, os.X_OK):
        return str(target)
    directory = tempfile.mkdtemp(prefix="letify-link-")
    done = subprocess.run(
        [
            sys.executable, "-m", "pip", "download", f"letify=={version}",
            "--no-deps", "--only-binary", ":all:", "-d", directory,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if done.returncode != 0:
        raise RuntimeError(
            "could not download the letify wheel for letify-link: "
            + (done.stderr or done.stdout).strip()[-400:]
        )
    wheels = sorted(Path(directory).glob("letify-*.whl"))
    if not wheels:
        raise RuntimeError("pip downloaded no letify wheel")
    with zipfile.ZipFile(wheels[0]) as archive:
        if inside not in archive.namelist():
            raise RuntimeError(f"{wheels[0].name} carries no {inside}")
        target.write_bytes(archive.read(inside))
    target.chmod(0o755)
    json.dumps(str(target))
    return str(target)
