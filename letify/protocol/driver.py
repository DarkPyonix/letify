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
from typing import Any

from .codec import BEGIN, END, PROTOCOL_VERSION, dumps_call

_TEMPLATE = """
import base64, pickle, sys, traceback
_PAYLOAD = "{payload}"
_BEGIN, _END = "{begin}", "{end}"
_VERSION = {version}
_NO_LETIFY = (
    "letify is not installed in this runtime's environment, so the call, which refers "
    "to letify (for example through letify.session_cache), cannot be loaded. Add it to "
    "the project with 'uv add letify' so uv.lock carries it into the runtime."
)

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
        _emit({{
            "ok": False,
            "error": "{{}}: {{}}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }})
    else:
        _emit({{"ok": True, "value": value}})
"""


def build(fn: Any, args: tuple, kwargs: dict) -> str:
    """Return the script that runs one call and prints its outcome."""
    payload = base64.b64encode(dumps_call(fn, args, kwargs)).decode()
    return _TEMPLATE.format(
        payload=payload,
        begin=BEGIN,
        end=END,
        version=PROTOCOL_VERSION,
    )


__all__ = ["build"]
