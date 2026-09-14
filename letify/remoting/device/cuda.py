"""Mapping ``"cuda"`` onto the runtime's device.

This module owns what lets code written for CUDA run unchanged under ``host="local"``: a
``TorchFunctionMode`` that rewrites CUDA devices, and the ``torch.cuda`` functions letify
provides or refuses, as spec "Mapping cuda" describes. It does not own dispatch, which is
``tensor``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch.overrides import TorchFunctionMode

from ...errors import UnsupportedMode
from .tensor import META, FactoryMode, RemoteTensor

if TYPE_CHECKING:
    from .client import Client

#: torch.cuda attributes that raise, because they name state in the runtime's process.
REFUSED = (
    "Stream",
    "Event",
    "current_stream",
    "stream",
    "CUDAGraph",
    "graph",
    "get_rng_state",
    "set_rng_state",
)


def _cuda_index(value: Any) -> int | None:
    """The CUDA index a device argument names, -1 for an unindexed one, None for no CUDA."""
    if isinstance(value, torch.device):
        device = value
    elif isinstance(value, str):
        try:
            device = torch.device(value)
        except (RuntimeError, ValueError):
            return None
    elif isinstance(value, int) and not isinstance(value, bool):
        device = torch.device("cuda", value)
    else:
        return None
    if device.type != "cuda":
        return None
    return -1 if device.index is None else device.index


def _check_index(index: int) -> None:
    if index not in (-1, 0):
        raise UnsupportedMode(
            f"cuda:{index} was asked for, and one host='local' session forwards to one "
            f"device, which is cuda:0"
        )


class CudaMode(TorchFunctionMode):
    """Rewrites CUDA devices in torch calls to the runtime's device."""

    def __init__(self, client: Client):
        super().__init__()
        self.client = client

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        rewritten = False
        device = kwargs.get("device")
        if device is not None:
            index = _cuda_index(device)
            if index is not None:
                _check_index(index)
                kwargs = {**kwargs, "device": META}
                rewritten = True

        if func is torch.Tensor.cuda:
            target = args[1] if len(args) > 1 else kwargs.get("device")
            if target is not None and not rewritten:
                index = _cuda_index(target)
                _check_index(-1 if index is None else index)
            if isinstance(args[0], RemoteTensor):
                return args[0]
            with FactoryMode(self.client):
                return torch.Tensor.to(
                    args[0], device=META, non_blocking=kwargs.get("non_blocking", False)
                )

        if func is torch.Tensor.to:
            rebuilt = list(args)
            for position, value in enumerate(rebuilt[1:], start=1):
                if isinstance(value, RemoteTensor):
                    rewritten = True
                    continue
                index = _cuda_index(value) if not isinstance(value, int) else None
                if index is not None:
                    _check_index(index)
                    rebuilt[position] = META
                    rewritten = True
            args = tuple(rebuilt)

        if rewritten:
            with FactoryMode(self.client):
                return func(*args, **kwargs)
        return func(*args, **kwargs)


def _refusal(name: str) -> UnsupportedMode:
    return UnsupportedMode(
        f"torch.cuda.{name} is not available under host='local': it names state in the "
        f"runtime's process, which has no counterpart in this one"
    )


def _refuse(name: str) -> Any:
    original = getattr(torch.cuda, name, None)
    if isinstance(original, type):
        # A subclass, so code that checks the class hierarchy, as torch._dynamo does at
        # import, still sees the class it expects. Only constructing one is refused.
        def __new__(cls: type, *args: Any, **kwargs: Any) -> Any:
            raise _refusal(name)

        return type(name, (original,), {"__new__": __new__})

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise _refusal(name)

    return refused


class _Device(contextlib.AbstractContextManager):
    """``torch.cuda.device``, accepting only the forwarded device."""

    def __init__(self, device: Any):
        index = _cuda_index(device) if not isinstance(device, int) else device
        _check_index(-1 if index is None else index)

    def __exit__(self, *exc: Any) -> None:
        return None


def replacements(client: Client) -> dict[str, Any]:
    """The torch.cuda attributes set while forwarding is active."""

    def set_device(device: Any) -> None:
        index = device if isinstance(device, int) else _cuda_index(device)
        _check_index(-1 if index is None else index)

    table: dict[str, Any] = {
        "is_available": lambda: True,
        "is_initialized": lambda: True,
        "init": lambda: None,
        "device_count": lambda: 1,
        "current_device": lambda: 0,
        "set_device": set_device,
        "device": _Device,
        "get_device_name": lambda device=None: client.hello["name"],
        "synchronize": lambda device=None: client.synchronize(),
        "manual_seed": lambda seed: client.call("letify.seed", int(seed)),
        "manual_seed_all": lambda seed: client.call("letify.seed", int(seed)),
        "memory_allocated": lambda device=None: client.call("letify.memory", "memory_allocated"),
        "max_memory_allocated": lambda device=None: client.call(
            "letify.memory", "max_memory_allocated"
        ),
        "memory_reserved": lambda device=None: client.call("letify.memory", "memory_reserved"),
        "empty_cache": lambda: client.call("letify.empty_cache", reply=False),
    }
    for name in REFUSED:
        table[name] = _refuse(name)
    return table


_MISSING = object()

#: The torch.cuda attributes each active mapping replaced, innermost last.
_ACTIVE: list[dict[str, Any]] = []


def _patch(table: dict[str, Any]) -> dict[str, Any]:
    saved = {name: getattr(torch.cuda, name, _MISSING) for name in table}
    for name, value in table.items():
        setattr(torch.cuda, name, value)
    return saved


def _restore(saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        if value is _MISSING:
            delattr(torch.cuda, name)
        else:
            setattr(torch.cuda, name, value)


@contextlib.contextmanager
def mapped(client: Client) -> Iterator[None]:
    """Install the CUDA mapping for the duration of the block."""
    table = replacements(client)
    saved = _patch(table)
    _ACTIVE.append(table)
    foreach_types = _foreach_types()
    registered = foreach_types is not None and RemoteTensor not in foreach_types
    if registered:
        foreach_types.append(RemoteTensor)  # type: ignore[union-attr]
    try:
        with CudaMode(client):
            yield
    finally:
        if registered:
            foreach_types.remove(RemoteTensor)  # type: ignore[union-attr]
        _ACTIVE.pop()
        _restore(saved)


def _foreach_types() -> list | None:
    """The list optimizers read to decide whether a tensor type takes the foreach path."""
    try:
        from torch.utils import _foreach_utils
    except ImportError:  # pragma: no cover - a PyTorch without the module
        return None
    found = getattr(_foreach_utils, "_foreach_supported_types", None)
    return found if isinstance(found, list) else None


@contextlib.contextmanager
def unmapped() -> Iterator[None]:
    """Run the block as if no mapping were active: plain torch.cuda and no device rewrite."""
    from torch.overrides import _pop_mode_temporarily

    if not _ACTIVE:
        yield
        return
    originals = {name: getattr(torch.cuda, name) for name in _ACTIVE[-1]}
    _restore(_ORIGINALS)
    try:
        with _pop_mode_temporarily():
            yield
    finally:
        _patch(originals)


#: torch.cuda as it was before any mapping, read once at import.
_ORIGINALS: dict[str, Any] = {
    name: getattr(torch.cuda, name, _MISSING)
    for name in (
        "is_available",
        "is_initialized",
        "init",
        "device_count",
        "current_device",
        "set_device",
        "device",
        "get_device_name",
        "synchronize",
        "manual_seed",
        "manual_seed_all",
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "empty_cache",
        *REFUSED,
    )
}
