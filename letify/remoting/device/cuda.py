"""Mapping ``"cuda"`` onto the runtime's device.

This module owns what lets code written for CUDA run unchanged under ``host="local"``: a
``TorchFunctionMode`` that rewrites CUDA devices, and the ``torch.cuda`` functions letify
provides or refuses, as spec "Mapping cuda" describes. It does not own dispatch, which is
``tensor``.
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch.overrides import TorchFunctionMode

from ...errors import UnsupportedMode
from .tensor import META, SPECIAL, FactoryMode, RemoteTensor

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


#: Tensor functions that read metadata only, so an unfilled tensor passed to them does not wait.
METADATA = frozenset(
    {
        torch.Tensor.shape.__get__,  # type: ignore[attr-defined]
        torch.Tensor.dtype.__get__,  # type: ignore[attr-defined]
        torch.Tensor.device.__get__,  # type: ignore[attr-defined]
        torch.Tensor.ndim.__get__,  # type: ignore[attr-defined]
        torch.Tensor.is_cuda.__get__,  # type: ignore[attr-defined]
        torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
        torch.Tensor.size,
        torch.Tensor.dim,
        torch.Tensor.numel,
        torch.Tensor.stride,
        torch.Tensor.element_size,
        torch.Tensor.__len__,
    }
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


#: CUDA autocast policy by function name, as spec "Autocast" lists it.
_AUTOCAST: dict[str, str] = {
    **dict.fromkeys(
        [
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "conv_tbc",
            "prelu",
            "addmm",
            "addmv",
            "addr",
            "matmul",
            "__matmul__",
            "__rmatmul__",
            "einsum",
            "mm",
            "mv",
            "linear",
            "bmm",
            "baddbmm",
            "addbmm",
            "chain_matmul",
            "multi_dot",
            "scaled_dot_product_attention",
            "lstm_cell",
            "gru_cell",
            "rnn_tanh_cell",
            "rnn_relu_cell",
        ],
        "lower",
    ),
    **dict.fromkeys(
        [
            "acos",
            "asin",
            "cosh",
            "erfinv",
            "exp",
            "expm1",
            "log",
            "log10",
            "log2",
            "log1p",
            "reciprocal",
            "rsqrt",
            "sinh",
            "tan",
            "pow",
            "__pow__",
            "softplus",
            "layer_norm",
            "group_norm",
            "norm",
            "cosine_similarity",
            "poisson_nll_loss",
            "cosine_embedding_loss",
            "nll_loss",
            "hinge_embedding_loss",
            "kl_div",
            "l1_loss",
            "smooth_l1_loss",
            "huber_loss",
            "mse_loss",
            "margin_ranking_loss",
            "multilabel_margin_loss",
            "soft_margin_loss",
            "triplet_margin_loss",
            "multi_margin_loss",
            "binary_cross_entropy_with_logits",
            "cross_entropy",
            "dist",
            "pdist",
            "cdist",
            "renorm",
            "logsumexp",
            "softmax",
            "log_softmax",
            "sum",
            "prod",
            "cumsum",
            "cumprod",
        ],
        "float32",
    ),
    **dict.fromkeys(
        [
            "addcdiv",
            "addcmul",
            "atan2",
            "bilinear",
            "cross",
            "dot",
            "vdot",
            "grid_sample",
            "index_put",
            "scatter_add",
            "tensordot",
            "cat",
            "stack",
            "index_copy",
        ],
        "widest",
    ),
}


def _autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:  # pragma: no cover - PyTorch before 2.4 takes no device type
        return torch.is_autocast_enabled()


def _autocast_dtype() -> torch.dtype:
    try:
        return torch.get_autocast_dtype("cuda")
    except AttributeError:  # pragma: no cover - PyTorch before 2.4
        return torch.get_autocast_gpu_dtype()


def _eligible(value: Any) -> bool:
    return (
        type(value) is RemoteTensor
        and value.dtype.is_floating_point
        and value.dtype is not torch.float64
    )


def _autocast(policy: str, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """The arguments with autocast's casts for ``policy`` applied."""
    found = [
        leaf
        for value in (*args, *kwargs.values())
        for leaf in (value if isinstance(value, (list, tuple)) else (value,))
        if _eligible(leaf)
    ]
    if not found:
        return args, kwargs
    if policy == "lower":
        dtype = _autocast_dtype()
    elif policy == "float32":
        dtype = torch.float32
    else:
        dtype = max((leaf.dtype for leaf in found), key=lambda d: torch.finfo(d).bits)

    def cast(value: Any) -> Any:
        if _eligible(value) and value.dtype is not dtype:
            return value.to(dtype)
        if isinstance(value, (list, tuple)):
            return type(value)(cast(item) for item in value)
        return value

    return tuple(cast(value) for value in args), {key: cast(v) for key, v in kwargs.items()}


class CudaMode(TorchFunctionMode):
    """Rewrites CUDA devices in torch calls to the runtime's device."""

    def __init__(self, client: Client):
        super().__init__()
        self.client = client

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.client.unfilled and func not in METADATA:
            self.client.wait_used(args)
            if kwargs:
                self.client.wait_used(kwargs.values())
        if args and type(args[0]) is RemoteTensor:
            special = SPECIAL.get(func)
            if special is not None:
                return special(*args, **kwargs)
        policy = _AUTOCAST.get(getattr(func, "__name__", ""))
        if policy is not None and _autocast_enabled():
            args, kwargs = _autocast(policy, args, kwargs)
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

    # Answered from the executor's hello, so no query here initializes CUDA in this process.
    hello = client.hello
    capability = tuple(hello.get("capability", (0, 0)))
    properties = types.SimpleNamespace(
        name=hello["name"],
        major=capability[0],
        minor=capability[1],
        total_memory=hello.get("total_memory", 0),
    )

    table: dict[str, Any] = {
        "is_available": lambda: True,
        "is_initialized": lambda: True,
        "init": lambda: None,
        "device_count": lambda: 1,
        "current_device": lambda: 0,
        "set_device": set_device,
        "device": _Device,
        "get_device_name": lambda device=None: client.hello["name"],
        "get_device_properties": lambda device=None: properties,
        "get_device_capability": lambda device=None: capability,
        "is_current_stream_capturing": lambda: False,
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
    _CLIENTS.append(client)
    registered = [found for found in _foreach_types() if RemoteTensor not in found]
    for found in registered:
        found.append(RemoteTensor)
    try:
        with CudaMode(client):
            yield
    finally:
        for found in registered:
            found.remove(RemoteTensor)
        _CLIENTS.pop()
        _ACTIVE.pop()
        _restore(saved)


#: The client of each active mapping, innermost last. None marks a suspended one.
_CLIENTS: list[Client | None] = []


def current_client() -> Client | None:
    """The client of the innermost active forwarding, or None outside one."""
    return _CLIENTS[-1] if _CLIENTS else None


def _foreach_types() -> list[list]:
    """Every list PyTorch reads to decide whether a tensor type takes the foreach path.

    ``torch.optim.optimizer`` keeps its own list in some versions, 2.5 among them, and
    ``torch.utils._foreach_utils`` keeps another, so both are found when present.
    """
    from importlib import import_module

    found: list[list] = []
    for module_name in ("torch.optim.optimizer", "torch.utils._foreach_utils"):
        try:
            module = import_module(module_name)
        except ImportError:  # pragma: no cover - a PyTorch without the module
            continue
        types = getattr(module, "_foreach_supported_types", None)
        if isinstance(types, list) and all(types is not other for other in found):
            found.append(types)
    return found


@contextlib.contextmanager
def unmapped() -> Iterator[None]:
    """Run the block as if no mapping were active: plain torch.cuda and no device rewrite."""
    from torch.overrides import _pop_mode_temporarily

    if not _ACTIVE:
        yield
        return
    originals = {name: getattr(torch.cuda, name) for name in _ACTIVE[-1]}
    _restore(_ORIGINALS)
    _CLIENTS.append(None)
    try:
        with _pop_mode_temporarily():
            yield
    finally:
        _CLIENTS.pop()
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
        "get_device_properties",
        "get_device_capability",
        "is_current_stream_capturing",
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
