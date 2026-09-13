"""Running provider tools through uv, outside the user's environment.

letify is installed into a research repository, so a provider's own tool, such as the
Colab CLI, is never installed next to it. It runs through ``uv tool run`` in an
environment uv manages, and letify only needs to find uv.

Each account gets its own home directory for the tool, ``~/.letify/accounts/<alias>/``,
because tools such as the Colab CLI keep their login at a fixed path under the home
directory. Pointing ``HOME`` there is what lets two accounts of one provider live on one
machine. uv's own cache and Python installs are pinned to the real home first, so the
changed ``HOME`` does not make uv download everything again per account.

This module does not own what a tool is asked to do. The provider and the login flow do.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config.secrets import account_directory


@dataclass(frozen=True)
class Tool:
    """A command published as a Python package, and the Python it needs."""

    package: str
    executable: str
    python: str
    #: Requirements the tool needs but does not pin itself, added with ``--with``.
    pins: tuple[str, ...] = ()


#: The official Colab CLI. It has no release for Python older than 3.13. Release 0.6.0 calls
#: ``jupyter_kernel_client.KernelClient``, which jupyter-kernel-client 1.0 removed, and does not
#: pin that package, so every command that reaches a kernel fails without the pin.
COLAB = Tool(
    package="google-colab-cli",
    executable="colab",
    python="3.13",
    pins=("jupyter-kernel-client<1",),
)


def find_uv() -> str | None:
    """uv from the ``UV`` variable ``uv run`` sets, then from PATH."""
    given = os.environ.get("UV")
    if given and Path(given).is_file():
        return given
    return shutil.which("uv")


def missing_uv_message() -> str:
    return (
        "uv was not found. letify runs provider tools through uv, so install it from "
        "https://docs.astral.sh/uv/ or set UV to its path"
    )


def command(tool: Tool, uv: str) -> list[str]:
    """The argument list that runs ``tool`` through ``uv``."""
    pinned = [part for pin in tool.pins for part in ("--with", pin)]
    return [
        uv,
        "tool",
        "run",
        "--python",
        tool.python,
        *pinned,
        "--from",
        tool.package,
        tool.executable,
    ]


def environment(alias: str) -> dict[str, str]:
    """The environment a tool runs in for one account, with its home in the account directory."""
    env = dict(os.environ)
    if sys.platform != "win32":
        # On Windows uv keeps its cache under LOCALAPPDATA, which a changed HOME leaves alone.
        real_home = Path.home()
        cache = Path(env.get("XDG_CACHE_HOME") or real_home / ".cache")
        data = Path(env.get("XDG_DATA_HOME") or real_home / ".local" / "share")
        env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(data / "uv" / "python"))
        env.setdefault("UV_TOOL_DIR", str(data / "uv" / "tools"))
    home = account_directory(alias)
    home.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        # The tool writes its token here with its own permissions, so the directory is
        # what keeps other users out.
        home.chmod(0o700)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return env


__all__ = ["COLAB", "Tool", "command", "environment", "find_uv", "missing_uv_message"]
