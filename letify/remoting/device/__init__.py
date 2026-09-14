"""PyTorch forwarding, the mechanism behind ``host="local"``.

This package owns running PyTorch operators from this process on a runtime's device, as
spec "PyTorch forwarding" describes. It does not own starting the session the worker runs
in, which is ``runtime.session``.

PyTorch is the user's own dependency, so nothing here imports it at package import. The
names below that need it load their module on first access.
"""

from __future__ import annotations

from typing import Any

from .guard import check_torch_version, check_worker_version, require_torch

_LAZY = {
    "Client": "client",
    "Stats": "client",
    "connect": "client",
    "worker_command": "client",
    "worker_source": "client",
    "RemoteTensor": "tensor",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(name)
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module}"), name)


__all__ = [
    "Client",
    "RemoteTensor",
    "Stats",
    "check_torch_version",
    "check_worker_version",
    "connect",
    "require_torch",
    "worker_command",
    "worker_source",
]
