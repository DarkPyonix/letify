"""Mapping ``"cuda"`` onto the runtime's device.

This module owns what lets code written for CUDA run unchanged under ``host="local"``: a
``TorchFunctionMode`` that rewrites CUDA devices, and the ``torch.cuda`` functions letify
provides or refuses, as spec "Mapping cuda" describes. It does not own dispatch, which is
``tensor``.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import types
import warnings
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


#: The signature ``_cross_entropy`` binds a call's arguments against.
_CROSS_ENTROPY = inspect.signature(torch.nn.functional.cross_entropy)

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
        ],
        "widest",
    ),
    "cross_entropy": "cross_entropy",
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


def _cross_entropy(func: Any, args: tuple, kwargs: dict) -> Any:
    """``cross_entropy`` as CUDA autocast runs ``cross_entropy_loss``.

    That operator has no autocast kernel of its own: ``log_softmax`` runs in the input's
    dtype and ``nll_loss`` casts to float32. Probability targets, label smoothing and the
    legacy reduction arguments take the float32 cast of the whole call instead.
    """
    bound = _CROSS_ENTROPY.bind(*args, **kwargs)
    bound.apply_defaults()
    given = bound.arguments
    source, target = given["input"], given["target"]
    if (
        not _eligible(source)
        or target.dtype.is_floating_point
        or given["label_smoothing"]
        or given["size_average"] is not None
        or given["reduce"] is not None
    ):
        args, kwargs = _autocast("float32", args, kwargs)
        return func(*args, **kwargs)
    weight = given["weight"]
    if _eligible(weight) and weight.dtype is not torch.float32:
        weight = weight.to(torch.float32)
    log_probabilities = torch.log_softmax(source, 1 if source.dim() > 1 else 0)
    return torch.nn.functional.nll_loss(
        log_probabilities.to(torch.float32),
        target,
        weight,
        ignore_index=given["ignore_index"],
        reduction=given["reduction"],
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


_BATCH_NORM = inspect.signature(torch.nn.functional.batch_norm)
_ATTENTION = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
_ATTENTION_DEFAULTS = {
    "attn_mask": None,
    "dropout_p": 0.0,
    "is_causal": False,
    "scale": None,
    "enable_gqa": False,
}


def _batch_norm(client: Client, args: tuple, kwargs: dict) -> Any:
    """``batch_norm`` through ``aten.cudnn_batch_norm`` when the runtime picks cuDNN.

    Returns NotImplemented for any other answer, which leaves the call on the ordinary path.
    """
    bound = _BATCH_NORM.bind(*args, **kwargs)
    bound.apply_defaults()
    given = bound.arguments
    x, weight, bias = given["input"], given["weight"], given["bias"]
    mean, var, training = given["running_mean"], given["running_var"], bool(given["training"])
    if any(type(value) is not RemoteTensor for value in (x, weight, bias)):
        return NotImplemented
    if any(value is not None and type(value) is not RemoteTensor for value in (mean, var)):
        return NotImplemented
    if not training and (mean is None or var is None):
        return NotImplemented
    answer = client.kernel(
        "batch_norm",
        (x, weight, bias, mean, var),
        {"training": training, "eps": float(given["eps"])},
    )
    if answer != "Cudnn":
        return NotImplemented
    if training:
        torch.nn.functional._verify_batch_size(x.size())
    momentum = given["momentum"]
    return torch.ops.aten.cudnn_batch_norm.default(
        x,
        weight,
        bias,
        mean,
        var,
        training,
        0.0 if momentum is None else float(momentum),
        float(given["eps"]),
    )[0]


def _attention(client: Client, args: tuple, kwargs: dict) -> Any:
    """Attention through the runtime's fused kernel when it picks one, else NotImplemented."""
    given = dict(_ATTENTION_DEFAULTS)
    given.update(zip(_ATTENTION, args, strict=False))
    given.update(kwargs)
    query, key, value, mask = given["query"], given["key"], given["value"], given["attn_mask"]
    if any(type(tensor) is not RemoteTensor for tensor in (query, key, value)):
        return NotImplemented
    if mask is not None and type(mask) is not RemoteTensor:
        return NotImplemented
    if given["enable_gqa"]:
        return NotImplemented
    dropout, causal, scale = float(given["dropout_p"]), bool(given["is_causal"]), given["scale"]
    flags = {"dropout_p": dropout, "is_causal": causal, "scale": scale, "enable_gqa": False}
    answer = client.kernel("scaled_dot_product_attention", (query, key, value, mask), flags)
    grads = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
    if answer == "FLASH_ATTENTION" and mask is None and query.shape[-1] % 8 == 0:
        return torch.ops.aten._scaled_dot_product_flash_attention.default(
            query, key, value, dropout, causal, False, scale=scale
        )[0]
    if answer == "EFFICIENT_ATTENTION":
        return torch.ops.aten._scaled_dot_product_efficient_attention.default(
            query, key, value, mask, grads, dropout, causal, scale=scale
        )[0]
    if answer == "CUDNN_ATTENTION":
        return torch.ops.aten._scaled_dot_product_cudnn_attention.default(
            query, key, value, mask, grads, dropout, causal, False, scale=scale
        )[0]
    return NotImplemented


#: Functions whose kernel the runtime chooses, as spec "Kernel selection" describes.
_SELECTED = {
    torch.nn.functional.batch_norm: _batch_norm,
    torch.nn.functional.scaled_dot_product_attention: _attention,
}


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
            if policy == "cross_entropy":
                return _cross_entropy(func, args, kwargs)
            args, kwargs = _autocast(policy, args, kwargs)
        selected = _SELECTED.get(func)
        if selected is not None and str(self.client.hello.get("device", "")).startswith("cuda"):
            result = selected(self.client, args, kwargs)
            if result is not NotImplemented:
                return result
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

#: Set on a CPU tensor that ``Tensor.pin_memory`` returned while forwarding was active.
_PINNED_MARK = "_letify_pinned"

#: The ``torch.accelerator`` functions spec "Pinned memory" replaces, where PyTorch has them.
_ACCELERATOR_NAMES = ("is_available", "current_device_index", "set_device_index", "set_device_idx")


def _host_owners() -> list[tuple[Any, str]]:
    """Every ``(owner, name)`` spec "Pinned memory" replaces in this PyTorch."""
    found: list[tuple[Any, str]] = [(torch.Tensor, "pin_memory"), (torch.Tensor, "is_pinned")]
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        names = [name for name in _ACCELERATOR_NAMES if hasattr(accelerator, name)]
        found.extend((accelerator, name) for name in names)
    return found


def host_replacements() -> dict[tuple[Any, str], Any]:
    """Pinning and ``torch.accelerator`` as spec "Pinned memory" describes.

    They are replaced on the class and the module, because the DataLoader pins in a thread
    where the ``TorchFunctionMode`` is not active.
    """
    is_pinned = torch.Tensor.is_pinned

    def pin_memory(self: torch.Tensor, device: Any = None) -> torch.Tensor:
        if getattr(self, _PINNED_MARK, False):
            return self
        with torch._C.DisableTorchFunction():
            copy = self.clone(memory_format=torch.preserve_format)
        setattr(copy, _PINNED_MARK, True)
        return copy

    def pinned(self: torch.Tensor, *args: Any, **kwargs: Any) -> bool:
        if getattr(self, _PINNED_MARK, False):
            return True
        return is_pinned(self, *args, **kwargs)

    def set_device_index(device: Any) -> None:
        index = device if isinstance(device, int) else _cuda_index(device)
        _check_index(-1 if index is None else index)

    table: dict[tuple[Any, str], Any] = {
        (torch.Tensor, "pin_memory"): pin_memory,
        (torch.Tensor, "is_pinned"): pinned,
    }
    values = {
        "is_available": lambda: True,
        "current_device_index": lambda: 0,
        "set_device_index": set_device_index,
        "set_device_idx": set_device_index,
    }
    for owner, name in _host_owners():
        if owner is not torch.Tensor:
            table[(owner, name)] = values[name]
    return table


#: Whether this process has warned that ``torch.compile`` runs eagerly, as a mutable cell.
_COMPILE_WARNED = [False]


def _warn_compile() -> None:
    if _COMPILE_WARNED[0]:
        return
    _COMPILE_WARNED[0] = True
    warnings.warn(
        "torch.compile runs eagerly under host='local': compilation needs a CUDA driver in "
        "this process, and the runtime already receives each repeated step as one entry",
        UserWarning,
        stacklevel=3,
    )


def compile_replacements() -> dict[tuple[Any, str], Any]:
    """``torch.compile`` and ``Module.compile`` as spec "Compilation" describes."""

    def compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        _warn_compile()
        if model is None:
            return lambda function: function
        return model

    def module_compile(self: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        _warn_compile()

    return {(torch, "compile"): compile, (torch.nn.Module, "compile"): module_compile}


def _patch_host(table: dict[tuple[Any, str], Any]) -> dict[tuple[Any, str], Any]:
    saved = {key: getattr(key[0], key[1], _MISSING) for key in table}
    for (owner, name), value in table.items():
        setattr(owner, name, value)
    return saved


def _restore_host(saved: dict[tuple[Any, str], Any]) -> None:
    for (owner, name), value in saved.items():
        if value is _MISSING:
            delattr(owner, name)
        else:
            setattr(owner, name, value)


#: The torch.cuda attributes each active mapping replaced, innermost last.
_ACTIVE: list[dict[str, Any]] = []

#: The pinning and accelerator attributes each active mapping replaced, innermost last.
_ACTIVE_HOST: list[dict[tuple[Any, str], Any]] = []


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
    host = host_replacements()
    host_saved = _patch_host(host)
    compiled_saved = _patch_host(compile_replacements())
    _ACTIVE.append(table)
    _ACTIVE_HOST.append(host)
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
        _ACTIVE_HOST.pop()
        _ACTIVE.pop()
        _restore_host(compiled_saved)
        _restore_host(host_saved)
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
    host = {key: getattr(key[0], key[1]) for key in _ACTIVE_HOST[-1]} if _ACTIVE_HOST else {}
    compiled = {key: getattr(key[0], key[1]) for key in _COMPILE_ORIGINALS}
    _restore(_ORIGINALS)
    _restore_host({key: _HOST_ORIGINALS[key] for key in host if key in _HOST_ORIGINALS})
    _restore_host(_COMPILE_ORIGINALS)
    _CLIENTS.append(None)
    try:
        with _pop_mode_temporarily():
            yield
    finally:
        _CLIENTS.pop()
        _patch_host(compiled)
        _patch_host(host)
        _patch(originals)


def _forget_in_child() -> None:
    """Undo every active mapping in a process just forked, whose channel belongs to the parent."""
    if not _ACTIVE:
        return
    _restore(_ORIGINALS)
    _restore_host(_HOST_ORIGINALS)
    _restore_host(_COMPILE_ORIGINALS)
    for found in _foreach_types():
        while RemoteTensor in found:
            found.remove(RemoteTensor)
    _ACTIVE.clear()
    _ACTIVE_HOST.clear()
    _CLIENTS.clear()
    stack = torch._C._len_torch_function_stack
    while stack() and isinstance(torch._C._get_function_stack_at(stack() - 1), CudaMode):
        torch._C._pop_torch_function_stack()


os.register_at_fork(after_in_child=_forget_in_child)


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

#: Pinning and torch.accelerator as they were before any mapping, read once at import.
_HOST_ORIGINALS: dict[tuple[Any, str], Any] = {
    key: getattr(key[0], key[1], _MISSING) for key in _host_owners()
}

#: torch.compile and Module.compile as they were before any mapping, read once at import.
_COMPILE_ORIGINALS: dict[tuple[Any, str], Any] = {
    key: getattr(key[0], key[1], _MISSING) for key in compile_replacements()
}
