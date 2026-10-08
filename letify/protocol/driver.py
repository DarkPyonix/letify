"""The one-shot driver script.

Used where a channel can only run a command and collect its output, with no
process left alive between calls. The call travels inside this script, and the
outcome is printed between two markers so it can be found in a stream that also
carries the user's prints.

What this path cannot do is the reason the persistent worker exists. With no
living process nothing a session cache stored survives to the next call, and there
is no blob table, so every large argument travels again on every call.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .codec import BEGIN, END, PROTOCOL_VERSION, dumps_call

_TEMPLATE = """
import base64, os, pickle, sys, traceback
_PAYLOAD = "{payload}"
_BEGIN, _END = "{begin}", "{end}"
_VERSION = {version}
_NO_LETIFY = (
    "letify is not installed in this runtime's environment, so the call, which refers "
    "to letify (for example through letify.session_cache), cannot be loaded. Add it to "
    "the project with 'uv add letify' so uv.lock carries it into the runtime."
)
_MODULES = {modules_code}
_WORKSPACE = {workspace_root!r}
if _MODULES:
    _modules_dir = os.path.join(os.path.expanduser(_WORKSPACE), "modules")
    os.makedirs(_modules_dir, exist_ok=True)
    for _rel, _b64 in _MODULES.items():
        _target = os.path.join(_modules_dir, _rel)
        os.makedirs(os.path.dirname(_target), exist_ok=True)
        with open(_target, "wb") as _f:
            _f.write(base64.b64decode(_b64))
    if _modules_dir not in sys.path:
        sys.path.insert(0, _modules_dir)
    _existing = os.environ.get("PYTHONPATH", "")
    _parts = [p for p in _existing.split(os.pathsep) if p]
    if _modules_dir not in _parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([_modules_dir, *_parts])

def _emit(obj):
    try:
        blob = pickle.dumps(obj, protocol=5)
    except Exception:
        blob = pickle.dumps(
            {{
                "ok": False,
                "error": "the return value could not be serialized",
                "traceback": traceback.format_exc(),
            }},
            protocol=5,
        )
    sys.stdout.flush()
    sys.stdout.write("\\n" + _BEGIN + base64.b64encode(blob).decode() + _END + "\\n")
    sys.stdout.flush()

try:
    import cloudpickle
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cloudpickle"], check=True)
    import cloudpickle

try:
    fn, args, kwargs = cloudpickle.loads(base64.b64decode(_PAYLOAD))
except ModuleNotFoundError as exc:
    _missing = (exc.name or "").split(".")[0] == "letify"
    _emit({{
        "ok": False,
        "error": _NO_LETIFY if _missing else "the call could not be deserialized",
        "traceback": traceback.format_exc(),
    }})
except Exception:
    _emit({{
        "ok": False,
        "error": "the call could not be deserialized",
        "traceback": traceback.format_exc(),
    }})
else:
    try:
        value = fn(*args, **kwargs)
        if hasattr(value, "__await__"):
            import asyncio
            value = asyncio.run(value)
    except BaseException as exc:
        if isinstance(exc, ModuleNotFoundError) and (exc.name or "").split(".")[0] == "letify":
            _error = _NO_LETIFY
        else:
            _error = "{{}}: {{}}".format(type(exc).__name__, exc)
        _emit({{
            "ok": False,
            "error": _error,
            "traceback": traceback.format_exc(),
        }})
    else:
        _emit({{"ok": True, "value": value}})
"""


def build(
    fn: Any,
    args: tuple,
    kwargs: dict,
    *,
    modules: Sequence[str] = (),
    workspace_root: str = "~/.letify-runtime",
) -> str:
    """Return the script that runs one call and prints its outcome."""
    payload = base64.b64encode(dumps_call(fn, args, kwargs)).decode()
    modules_dict: dict[str, str] = {}
    if modules:
        from ..store.pathdata import collect_module

        modules_root = f"{workspace_root.rstrip('/')}/modules"
        for mod_name in modules:
            placed = collect_module(mod_name, modules_root)
            prefix = mod_name.replace(".", "/")
            for rel, _digest, _size, local in placed.entries:
                content = Path(local).read_bytes()
                target_rel = f"{prefix}/{rel}" if placed.directory else prefix + Path(local).suffix
                modules_dict[target_rel] = base64.b64encode(content).decode("ascii")

    return _TEMPLATE.format(
        payload=payload,
        begin=BEGIN,
        end=END,
        version=PROTOCOL_VERSION,
        modules_code=repr(modules_dict),
        workspace_root=workspace_root,
    )


__all__ = ["build"]
