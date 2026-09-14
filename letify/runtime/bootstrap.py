"""Source that builds a runtime's environment, and the local checks that come before it.

The remote parts are strings rather than functions because they run on the far side, in
the bootstrap interpreter, before the project's environment exists there. Keeping them in
one module means the set of assumptions about a remote machine sits in one place: it has a
Python 3 to start the worker with, it can reach uv's installer and a package index or
already has what it needs, and it can write under the provider's workspace root.

This module does not decide when a step runs or which channel carries it. That is
``Runtime.install_env``.
"""

from __future__ import annotations

import base64
import shlex
import sys
from typing import TYPE_CHECKING

from ..errors import ConfigError, InterpreterMismatch

if TYPE_CHECKING:
    from ..declare.env import Env

#: Where letify writes on a runtime unless the account or its kind says otherwise. Expanded
#: on the runtime.
DEFAULT_WORKSPACE_ROOT = "~/.letify-runtime"

#: The file written and removed to prove a workspace root is writable.
PROBE_FILE = ".letify-probe"


def shell_path(path: str) -> str:
    """Quote a path for a POSIX shell, leaving a leading ``~`` to the remote ``$HOME``."""
    if path == "~" or path.startswith("~/"):
        rest = path[1:]
        return '"$HOME"' + (shlex.quote(rest) if rest else "")
    return shlex.quote(path)


def workspace_check(root: str) -> str:
    """A shell command that creates the root and writes and removes a probe file in it.

    Run as the account's own user, so a root that needs elevated rights fails.
    """
    quoted = shell_path(root)
    probe = shell_path(f"{root.rstrip('/')}/{PROBE_FILE}")
    return f"mkdir -p {quoted} && : > {probe} && rm -f {probe}"


def workspace_source(root: str) -> str:
    """Source that expands and creates the root, enters it, and points TMPDIR under it."""
    return (
        "import os, tempfile\n"
        f"_letify_workspace = os.path.expanduser({root!r})\n"
        "os.makedirs(os.path.join(_letify_workspace, 'tmp'), exist_ok=True)\n"
        "os.chdir(_letify_workspace)\n"
        "os.environ['TMPDIR'] = os.path.join(_letify_workspace, 'tmp')\n"
        "tempfile.tempdir = None\n"
        "__letify_value__ = _letify_workspace\n"
    )


#: The official standalone uv installer. It writes under the home directory and needs no root.
UV_INSTALLER = "https://astral.sh/uv/install.sh"

#: The files a runtime's sync reads, in the order they are looked for.
PROJECT_FILES = ("pyproject.toml", "uv.lock", ".python-version")
REQUIRED_FILES = ("pyproject.toml", "uv.lock")

#: How many trailing lines of a failed command's standard error an error carries.
TAIL_LINES = 40

#: Reports the major.minor of whichever interpreter runs it.
VERSION_SOURCE = "import sys\n__letify_value__ = '%d.%d' % sys.version_info[:2]\n"


def user_agent() -> str:
    """The User-Agent a runtime sends when it downloads the uv installer."""
    from .. import __version__

    return f"letify/{__version__}"


def local_python() -> str:
    """The major.minor of the interpreter running letify, such as ``"3.12"``."""
    return f"{sys.version_info[0]}.{sys.version_info[1]}"


def major_minor(version: str) -> str:
    """Reduce ``3.12.3`` or ``cpython-3.12`` to ``3.12``."""
    text = version.strip().removeprefix("cpython-")
    return ".".join(text.split(".")[:2])


def env_archive_path(mount: str, digest: str) -> str:
    """Where a cached blob is written in the content addressed layout."""
    return f"{mount.rstrip('/')}/blobs/{digest[:2]}/{digest}"


def uv_cache_dir(workspace_root: str) -> str:
    """The uv cache a persistent provider syncs with, before ``~`` is expanded there."""
    return f"{workspace_root.rstrip('/')}/uv-cache"


def project_dir(workspace_root: str, env: Env) -> str:
    """The project directory on the runtime, before ``~`` is expanded there."""
    return f"{workspace_root.rstrip('/')}/project/{env.key}"


def project_files(env: Env) -> dict[str, bytes]:
    """Read the files a runtime syncs from, refusing what cannot give a matching interpreter.

    Called before a session starts, so a missing lock file or a pinned Python that differs
    from this process costs nothing on the provider.
    """
    directory = env.project_dir
    files: dict[str, bytes] = {}
    for name in PROJECT_FILES:
        path = directory / name
        if path.is_file():
            files[name] = path.read_bytes()
        elif name in REQUIRED_FILES:
            raise ConfigError(
                f"a remote runtime builds its environment with uv sync from the project's "
                f"pyproject.toml and uv.lock, and {directory.resolve()} has no {name}. "
                f"Run 'uv lock' in the project first"
            )
    local = local_python()
    pinned = files.get(".python-version")
    if pinned is not None:
        lines = [line.strip() for line in pinned.decode("utf-8", "replace").splitlines()]
        declared = next((major_minor(line) for line in lines if line and line[0] != "#"), "")
        if declared and declared != local:
            raise InterpreterMismatch(
                f".python-version names Python {declared} and this process runs Python "
                f"{local}. A runtime is synced with the local version so cloudpickle's "
                f"bytecode runs there, so the two have to agree: change .python-version to "
                f"{local}, or run letify from a Python {declared} environment"
            )
    if env.python and major_minor(env.python) != local:
        raise InterpreterMismatch(
            f"Env.python is {major_minor(env.python)} and this process runs Python {local}. "
            f"A runtime runs the local version so cloudpickle's bytecode runs there"
        )
    return files


def sync_command(env: Env, uv: str = "uv") -> list[str]:
    """The sync a runtime runs in its project directory."""
    python = major_minor(env.python) if env.python else local_python()
    return [uv, "sync", "--frozen", "--no-install-project", "--python", python]


def _prelude(env: Env, root: str) -> list[str]:
    """Lines shared by every source: variables, the project paths, the active .venv."""
    return [
        "import base64, os, platform, shutil, subprocess, sys, urllib.request",
        f"os.environ.update({dict(env.variables)!r})",
        f"_letify_root = os.path.expanduser({root!r})",
        "_letify_venv = os.path.join(_letify_root, '.venv')",
        "_letify_python = os.path.join(_letify_venv, 'bin', 'python')",
        "os.environ['VIRTUAL_ENV'] = _letify_venv",
        "if not os.environ.get('PATH', '').startswith(os.path.join(_letify_venv, 'bin')):",
        "    os.environ['PATH'] = os.path.join(_letify_venv, 'bin') + os.pathsep"
        " + os.environ.get('PATH', '')",
        "__letify_value__ = {'root': _letify_root, 'parent': os.path.dirname(_letify_root),",
        "    'python': _letify_python, 'platform': sys.platform + '-' + platform.machine()}",
    ]


def probe_source(env: Env, root: str) -> str:
    """Source that sets up the worker's environment and reports where the project lives."""
    return "\n".join(_prelude(env, root)) + "\n"


def venv_check_source(python: str, archive: str | None = None) -> str:
    """Source that answers whether a restored ``.venv`` interpreter starts."""
    lines = [
        "import os, subprocess",
        "try:",
        f"    _done = subprocess.run([{python!r}, '-c', 'pass'], capture_output=True, timeout=300)",
        "    __letify_value__ = _done.returncode == 0",
        "except (OSError, subprocess.SubprocessError):",
        "    __letify_value__ = False",
    ]
    if archive:
        lines += [
            "try:",
            f"    os.remove({archive!r})",
            "except OSError:",
            "    pass",
        ]
    return "\n".join(lines) + "\n"


def sync_source(
    env: Env,
    files: dict[str, bytes],
    *,
    root: str | None = None,
    name: str = "this runtime",
    installer: str = UV_INSTALLER,
    cache_dir: str | None = None,
) -> str:
    """Source that writes the project files, finds or installs uv, and runs the sync.

    ``cache_dir`` becomes ``UV_CACHE_DIR`` for the uv commands only, unless the declared
    variables already name one. Spec "uv cache": a persistent provider passes
    ``<workspace root>/uv-cache`` so uv hard links into the project ``.venv``.

    It raises ``RuntimeError`` with ``uv could not be installed on <name>`` or ``uv sync
    failed on <name>`` so the local side can name the step that failed.
    """
    root = root or project_dir(DEFAULT_WORKSPACE_ROOT, env)
    encoded = {key: base64.b64encode(value).decode() for key, value in files.items()}
    sync_args = sync_command(env)[1:]
    tail = f"'\\n'.join(_letify_text.strip().splitlines()[-{TAIL_LINES}:])"
    lines = [
        *_prelude(env, root),
        "os.makedirs(_letify_root, exist_ok=True)",
        f"for _letify_name, _letify_payload in {encoded!r}.items():",
        "    with open(os.path.join(_letify_root, _letify_name), 'wb') as _letify_file:",
        "        _letify_file.write(base64.b64decode(_letify_payload))",
        "_letify_uv = shutil.which('uv')",
        "_letify_bin = os.path.expanduser('~/.local/bin')",
        "if _letify_uv is None and os.path.isfile(os.path.join(_letify_bin, 'uv')):",
        "    _letify_uv = os.path.join(_letify_bin, 'uv')",
        "if _letify_uv is None:",
        "    _letify_text = ''",
        "    try:",
        # astral.sh answers Python's default urllib agent with 403, so letify names itself.
        f"        _letify_request = urllib.request.Request({installer!r}, "
        f"headers={{'User-Agent': {user_agent()!r}}})",
        "        with urllib.request.urlopen(_letify_request, timeout=300) as _letify_response:",
        "            _letify_script = _letify_response.read()",
        "        _letify_done = subprocess.run(['sh'], input=_letify_script, capture_output=True,",
        "            env=dict(os.environ, UV_INSTALL_DIR=_letify_bin, UV_NO_MODIFY_PATH='1'))",
        "        if _letify_done.returncode != 0:",
        "            _letify_text = 'the installer exited with code %d\\n%s' % (",
        "                _letify_done.returncode, _letify_done.stderr.decode('utf-8', 'replace'))",
        "        elif not os.path.isfile(os.path.join(_letify_bin, 'uv')):",
        "            _letify_text = 'the installer wrote no uv to ' + _letify_bin",
        "    except Exception as _letify_exc:",
        "        _letify_text = '%s: %s' % (type(_letify_exc).__name__, _letify_exc)",
        "    if _letify_text:",
        f"        raise RuntimeError('uv could not be installed on {name} from {installer}: '"
        f" + {tail})",
        "    _letify_uv = os.path.join(_letify_bin, 'uv')",
        "_letify_uv_env = dict(os.environ)",
    ]
    if cache_dir and "UV_CACHE_DIR" not in dict(env.variables):
        lines.append(f"_letify_uv_env['UV_CACHE_DIR'] = os.path.expanduser({cache_dir!r})")
    lines += [
        f"_letify_command = [_letify_uv] + {sync_args!r}",
        "_letify_done = subprocess.run(_letify_command, cwd=_letify_root, capture_output=True,"
        " text=True, env=_letify_uv_env)",
        "if _letify_done.returncode != 0:",
        "    _letify_text = _letify_done.stderr",
        f"    raise RuntimeError('uv sync failed on {name}\\ncommand: ' + ' '.join(_letify_command)"
        f" + '\\n' + {tail})",
    ]
    if env.packages:
        lines += [
            "_letify_command = [_letify_uv, 'pip', 'install', '--python', _letify_python]"
            f" + {list(env.packages)!r}",
            "_letify_done = subprocess.run(_letify_command, cwd=_letify_root,"
            " capture_output=True, text=True, env=_letify_uv_env)",
            "if _letify_done.returncode != 0:",
            "    _letify_text = _letify_done.stderr",
            f"    raise RuntimeError('uv pip install failed on {name}\\ncommand: '"
            f" + ' '.join(_letify_command) + '\\n' + {tail})",
        ]
    for command in env.commands:
        lines.append(f"subprocess.run({command!r}, shell=True, check=True, cwd=_letify_root)")
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_WORKSPACE_ROOT",
    "UV_INSTALLER",
    "VERSION_SOURCE",
    "env_archive_path",
    "local_python",
    "major_minor",
    "probe_source",
    "project_dir",
    "project_files",
    "sync_command",
    "sync_source",
    "uv_cache_dir",
    "venv_check_source",
]
