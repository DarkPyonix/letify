"""The one-shot driver script.

Used where a channel can only run a command and collect its output, with no
process left alive between calls. The call travels inside this script, and the
outcome is printed between two markers so it can be found in a stream that also
carries the user's prints.

What this path cannot do is the reason the persistent worker exists. With no
living process there is no object table, so a handle has nothing to point at, and
no blob table, so every large argument travels again on every call.
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
_KEEP = {keep}

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
        if _KEEP:
            _emit({{
                "ok": False,
                "error": (
                    "keep_remote needs a persistent session, and this provider is "
                    "running one-shot commands. There is no process for the handle "
                    "to point at once the call returns."
                ),
                "traceback": "",
            }})
        else:
            _emit({{"ok": True, "value": value}})
"""


def build(fn: Any, args: tuple, kwargs: dict, *, keep_remote: bool = False) -> str:
    """Return the script that runs one call and prints its outcome."""
    payload = base64.b64encode(dumps_call(fn, args, kwargs)).decode()
    return _TEMPLATE.format(
        payload=payload,
        begin=BEGIN,
        end=END,
        version=PROTOCOL_VERSION,
        keep=bool(keep_remote),
    )


__all__ = ["build"]
