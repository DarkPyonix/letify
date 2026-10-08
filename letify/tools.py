"""Running provider tools through uv, outside the user's environment.

letify is installed into a research repository, so a provider's own tool, such as the
Colab CLI, is never installed next to it. It runs through ``uv tool run`` in an
environment uv manages, and letify only needs to find uv.

Each account gets its own home directory for the tool, ``~/.letify/accounts/<alias>/``,
because tools such as the Colab CLI keep their login at a fixed path under the home
directory. Pointing ``HOME`` there is what lets two accounts of one provider live on one
machine. uv's own cache and Python installs are pinned to the real home first, so the
changed ``HOME`` does not make uv download everything again per account. Modal takes its
config file path from ``MODAL_CONFIG_PATH`` instead, so its environment keeps ``HOME``.

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


#: Modal's client and CLI. The range is the major version the Modal adapter was written
#: against; ``modal token new`` and the sandbox and volume calls it makes are 1.x API.
MODAL = Tool(
    package="modal",
    executable="modal",
    python="3.12",
    pins=("modal>=1.0,<2",),
)

#: The environment the Kaggle adapter runs in. ``jupyter-kernel-client`` below 1.0 is the
#: release whose ``KernelClient`` the Colab CLI also uses.
KAGGLE_KERNEL = Tool(
    package="jupyter-kernel-client",
    executable="python",
    python="3.13",
    pins=("jupyter-kernel-client<1",),
)

#: The official Kaggle CLI, run with the account's API token, never the cookie. It does
#: everything that does not require the live web session: creating and deleting the
#: ephemeral notebook, and reading the weekly quota.
KAGGLE_CLI = Tool(package="kaggle", executable="kaggle", python="3.13")

#: The Kaggle adapter file, run by path so it needs nothing of letify's own environment.
KAGGLE_ADAPTER = Path(__file__).parent / "providers" / "kaggle_adapter.py"

#: The adapter file, run by path so it needs nothing of letify's own environment.
MODAL_ADAPTER = Path(__file__).parent / "providers" / "modal_adapter.py"

#: Variables that would make Modal act as someone other than the account's file says.
MODAL_OVERRIDES = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE")


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


def _ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.resolve() == target.resolve():
                return
        except OSError:
            pass
        link_path.unlink()
    try:
        link_path.symlink_to(target)
    except OSError:
        shutil.copy2(target, link_path)


def prepare_tool_links(alias: str) -> None:
    """Link credential secrets from the account directory into the redirected tool home."""
    from .paths import tool_cache_home

    acct = account_directory(alias)
    tool_home = tool_cache_home(alias)

    # Colab token
    token_account = acct / "token.json"
    old_colab_token = acct / ".config" / "colab-cli" / "token.json"
    if not token_account.exists() and old_colab_token.is_file():
        token_account.write_bytes(old_colab_token.read_bytes())
        if sys.platform != "win32":
            token_account.chmod(0o600)
    if token_account.is_file():
        colab_config = tool_home / ".config" / "colab-cli"
        colab_config.mkdir(parents=True, exist_ok=True)
        link_target = colab_config / "token.json"
        _ensure_symlink(link_target, token_account)



def sync_tool_credentials(alias: str) -> None:
    """Sync credentials written by provider CLIs back to the account directory."""
    from .paths import ALLOWED_ACCOUNT_CREDENTIAL_FILES, tool_cache_home

    acct = account_directory(alias)
    tool_home = tool_cache_home(alias)

    # Colab token written by colab sessions
    colab_token = tool_home / ".config" / "colab-cli" / "token.json"
    token_account = acct / "token.json"
    if colab_token.is_file():
        is_link_to_target = False
        if colab_token.is_symlink():
            try:
                is_link_to_target = colab_token.resolve() == token_account.resolve()
            except OSError:
                is_link_to_target = False
        if not is_link_to_target:
            token_account.write_bytes(colab_token.read_bytes())
            if sys.platform != "win32":
                token_account.chmod(0o600)
            colab_token.unlink()
            _ensure_symlink(colab_token, token_account)


    # Clean non-credential files and directories from account directory
    if acct.exists():
        for item in list(acct.iterdir()):
            if item.is_dir() or item.name not in ALLOWED_ACCOUNT_CREDENTIAL_FILES:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)


def environment(alias: str) -> dict[str, str]:
    """The environment a tool runs in for one account, with its redirected tool home."""
    from .paths import tool_cache_home

    env = dict(os.environ)
    if sys.platform != "win32":
        # On Windows uv keeps its cache under LOCALAPPDATA, which a changed HOME leaves alone.
        real_home = Path.home()
        cache = Path(env.get("XDG_CACHE_HOME") or real_home / ".cache")
        data = Path(env.get("XDG_DATA_HOME") or real_home / ".local" / "share")
        env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(data / "uv" / "python"))
        env.setdefault("UV_TOOL_DIR", str(data / "uv" / "tools"))
    acct = account_directory(alias)
    acct.mkdir(parents=True, exist_ok=True)
    tool_home = tool_cache_home(alias)
    tool_home.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        acct.chmod(0o700)
        tool_home.chmod(0o700)
    prepare_tool_links(alias)
    env["HOME"] = str(tool_home)
    env["USERPROFILE"] = str(tool_home)
    return env


def script_command(tool: Tool, uv: str, script: Path) -> list[str]:
    """The argument list that runs a Python file in a throwaway uv environment with ``tool``.

    ``-P`` keeps the script's directory off ``sys.path``. Without it a file beside the
    script, such as ``letify/providers/modal.py``, shadows the package the script imports.
    """
    pinned = [part for pin in tool.pins for part in ("--with", pin)]
    return [
        uv,
        "run",
        "--no-project",
        "--python",
        tool.python,
        *pinned,
        "--frozen",
        "--no-sync",
        "python",
        "-P",
        str(script),
    ]


def modal_adapter_command(uv: str) -> list[str]:
    """The argument list that starts the Modal adapter."""
    return script_command(MODAL, uv, MODAL_ADAPTER)


def kaggle_cli_command(uv: str) -> list[str]:
    """The argument list that runs the official Kaggle CLI through uv."""
    return command(KAGGLE_CLI, uv)


def kaggle_cli_environment(alias: str) -> dict[str, str]:
    """Pass this account's token without file whitespace through the child environment."""
    env = dict(os.environ)
    token = (account_directory(alias) / "access_token").read_text(encoding="utf-8").strip()
    env["KAGGLE_API_TOKEN"] = token
    return env


def modal_config_path(alias: str) -> Path:
    """``~/.letify/accounts/<alias>/modal.toml``, where one account's Modal token lives."""
    return account_directory(alias) / "modal.toml"


def modal_environment(alias: str) -> dict[str, str]:
    """The environment Modal runs in for one account.

    ``HOME`` is left alone, because Modal takes the path of its config file from
    ``MODAL_CONFIG_PATH``. Variables that would override that file are removed.
    """
    env = {key: value for key, value in os.environ.items() if key not in MODAL_OVERRIDES}
    path = modal_config_path(alias)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        path.parent.chmod(0o700)
    env["MODAL_CONFIG_PATH"] = str(path)
    return env


__all__ = [
    "COLAB",
    "KAGGLE_CLI",
    "MODAL",
    "MODAL_ADAPTER",
    "Tool",
    "command",
    "environment",
    "find_uv",
    "kaggle_cli_command",
    "kaggle_cli_environment",
    "missing_uv_message",
    "modal_adapter_command",
    "modal_config_path",
    "modal_environment",
    "prepare_tool_links",
    "script_command",
    "sync_tool_credentials",
]
