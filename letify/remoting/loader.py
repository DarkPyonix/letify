"""Getting letify-core loaded before the real driver.

This is the one part of forwarding that has to happen in Python, because it has to happen
before the first CUDA library is loaded and therefore before ``import torch``.

Windows has no ``LD_PRELOAD``. What it has is a documented search order, and
``os.add_dll_directory`` puts a directory at the front of it. Since letify-core is named
``nvcuda.dll``, the loader finds ours instead of the real one, and nothing else about the
process changes.

Linux and WSL2 use ``LD_PRELOAD``, which cannot be set from inside a running process for
libraries that are already resolved. So on Linux this reports the command to run rather
than pretending it can inject itself, because a silently ineffective injection would look
like forwarding while the real driver was being used all along.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .probe import core_path


class Injection:
    """What was done, or what the caller has to do instead."""

    def __init__(self, active: bool, instructions: str = ""):
        self.active = active
        self.instructions = instructions

    def __bool__(self) -> bool:
        return self.active

    def __repr__(self) -> str:
        return f"<Injection active={self.active}>"


def inject(agent: str | None = None) -> Injection:
    """Arrange for letify-core to be found before the real CUDA driver.

    Call this before importing torch. Returns an ``Injection`` that is false when the
    caller has to act, carrying the exact command in ``instructions``.
    """
    path = core_path()
    if path is None:
        return Injection(
            False,
            "letify-core is not built. Run `python letify-core/build.py` to build it, or declare "
            "host='remote' to ship the function instead.",
        )

    if agent:
        os.environ["LETIFY_AGENT"] = agent

    if sys.platform.startswith("win"):
        if "torch" in sys.modules:
            return Injection(
                False,
                "torch is already imported, so the real driver may already be resolved. "
                "Call letify.remoting.inject() before importing torch.",
            )
        os.add_dll_directory(str(path.parent))
        # PATH as well, because some loaders consult it for dependent libraries.
        os.environ["PATH"] = f"{path.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        return Injection(True)

    preload = os.environ.get("LD_PRELOAD", "")
    if str(path) in preload:
        return Injection(True)
    return Injection(
        False,
        f"Linux resolves this at process start, so set it before launching Python:\n"
        f"    LD_PRELOAD={path} LETIFY_AGENT={os.environ.get('LETIFY_AGENT', 'host:7654')} "
        f"python your_script.py",
    )


def preload_command(script: str = "your_script.py", agent: str = "host:7654") -> str:
    """The command that runs a script with letify-core in front of the driver."""
    path = core_path()
    if path is None:
        return "python letify-core/build.py   # build letify-core first"
    if sys.platform.startswith("win"):
        return f"python {script}   # letify.remoting.inject() handles this on Windows"
    return f"LD_PRELOAD={path} LETIFY_AGENT={agent} python {script}"


def core_directory() -> Path | None:
    """Where the built library lives, for a caller that wants to place it itself."""
    path = core_path()
    return path.parent if path else None


__all__ = ["Injection", "core_directory", "inject", "preload_command"]
