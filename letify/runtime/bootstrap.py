"""Source that prepares a runtime once it is reachable.

These are strings rather than functions because they run on the far side, where
letify is not installed. Keeping them in one module means the set of assumptions
about a remote machine sits in one place: it has Python, it can reach a package
index or already has what it needs, and it can write to a path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..declare.env import Env


def env_archive_path(mount: str, digest: str) -> str:
    """Where a cached environment archive is written before it is unpacked."""
    return f"{mount.rstrip('/')}/blobs/{digest[:2]}/{digest}"


def install_source(env: Env) -> str:
    """Source that installs the declared environment inside a runtime.

    uv is used rather than pip because it resolves a lock file that covers every
    platform, which is what lets one lock file drive a Linux runtime from a Windows
    or macOS client.
    """
    lines = ["import os, shutil, subprocess, sys"]
    for name, value in env.variables:
        lines.append(f"os.environ[{name!r}] = {value!r}")
    lines += [
        "if shutil.which('uv') is None:",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)",
    ]
    if env.packages:
        packages = ", ".join(repr(package) for package in env.packages)
        lines.append(
            f"subprocess.run(['uv', 'pip', 'install', '--system', '-q', {packages}], check=True)"
        )
    for command in env.commands:
        lines.append(f"subprocess.run({command!r}, shell=True, check=True)")
    lines.append("print('letify: environment ready')")
    return "\n".join(lines)


def sync_lock_source(env: Env, lock_path: str) -> str:
    """Source that installs from a lock file already present in the runtime."""
    return (
        "import subprocess, shutil, sys\n"
        "if shutil.which('uv') is None:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)\n"
        f"subprocess.run(['uv', 'sync', '--frozen', '--project', {lock_path!r}], check=True)\n"
        "print('letify: environment synced from the lock file')\n"
    )


__all__ = ["env_archive_path", "install_source", "sync_lock_source"]
