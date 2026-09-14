"""RemoteTensor and operator dispatch.

This module owns the local stand-in for a tensor on the runtime and turning each ATen
operator into a queued request, as spec "Dispatch mechanism" describes: output metadata is
computed locally on meta tensors, and only values travel. It does not own the queue or the
wire, which are ``client``, or the mapping of ``"cuda"``, which is ``cuda``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ...errors import UnsupportedMode

if TYPE_CHECKING:
    from .client import Client

aten = torch.ops.aten

META = torch.device("meta")

#: What a RemoteTensor reports as its device.
REPORTED = torch.device("cuda", 0)

#: CPU tensors at most this large travel inside the head rather than as a buffer, and do
#: not flush the queue.
INLINE_BYTES = 4096


class Ref:
    """One handle on the runtime, shared by every RemoteTensor that names it."""

    __slots__ = ("client", "handle")

    def __init__(self, client: Client, handle: int):
        self.client = client
        self.handle = handle

    def __del__(self) -> None:
        try:
            self.client.released.append(self.handle)
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


class RemoteTensor(torch.Tensor):
    """A tensor whose values live on the runtime and whose metadata lives here."""

    _meta: torch.Tensor
    _ref: Ref

    @staticmethod
    def __new__(cls, meta: torch.Tensor, ref: Ref) -> RemoteTensor:
        made = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            meta.size(),
            strides=meta.stride(),
            storage_offset=meta.storage_offset(),
            dtype=meta.dtype,
            device=META,
            requires_grad=False,
        )
        made._meta = meta
        made._ref = ref
        return made

    def __init__(self, meta: torch.Tensor, ref: Ref):
        pass

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        special = _SPECIAL.get(func)
        if special is not None:
            return special(*args, **kwargs)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dispatch(func, args, kwargs or {}, None)


# -- torch function specials -----------------------------------------------------


def _repr(self: RemoteTensor, *args: Any, **kwargs: Any) -> str:
    text = repr(self.detach().cpu())
    if text.endswith(")"):
        return text[:-1] + f", device='{REPORTED}')"
    return text  # pragma: no cover - torch always closes the parenthesis


def _set_data(self: RemoteTensor, value: Any) -> None:
    with torch._C.DisableTorchFunctionSubclass():
        torch.Tensor.data.__set__(self, value)  # type: ignore[attr-defined]
    if isinstance(value, RemoteTensor):
        self._meta = value._meta
        self._ref = value._ref


_SPECIAL = {
    torch.Tensor.device.__get__: lambda self: REPORTED,  # type: ignore[attr-defined]
    torch.Tensor.is_cuda.__get__: lambda self: True,  # type: ignore[attr-defined]
    torch.Tensor.is_meta.__get__: lambda self: False,  # type: ignore[attr-defined]
    torch.Tensor.get_device: lambda self: 0,
    torch.Tensor.tolist: lambda self: self.detach().cpu().tolist(),
    torch.Tensor.numpy: lambda self, *a, **k: self.detach().cpu().numpy(*a, **k),
    torch.Tensor.__repr__: _repr,
    torch.Tensor.data.__set__: _set_data,  # type: ignore[attr-defined]
}


# -- dispatch ----------------------------------------------------------------------


class FactoryMode(TorchDispatchMode):
    """Sends operators that create a tensor on the runtime's device to the runtime.

    Entered only around a call whose CUDA device ``cuda.CudaMode`` rewrote to meta, so a
    meta device inside that call means the runtime.
    """

    def __init__(self, client: Client):
        super().__init__()
        self.client = client

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if (device is not None and torch.device(device).type == "meta") or _has_remote(args):
            return dispatch(func, args, kwargs, self.client)
        return func(*args, **kwargs)


def _has_remote(values: Any) -> bool:
    for value in values:
        if isinstance(value, RemoteTensor):
            return True
        if type(value) in (list, tuple) and _has_remote(value):
            return True
    return False


class _Prepared:
    """What one pass over an operator's arguments collected."""

    __slots__ = ("big", "buffers", "client", "inputs", "keep")

    def __init__(self, client: Client | None):
        self.client = client
        self.inputs: dict[int, RemoteTensor] = {}
        self.buffers: list[memoryview] = []
        self.keep: list[torch.Tensor] = []
        self.big = False


def _prepare(value: Any, state: _Prepared) -> tuple[Any, Any]:
    """Return ``(meta_value, encoded_value)`` for one argument."""
    kind = type(value)
    if kind in (int, float, bool, str) or value is None:
        return value, value
    if kind is RemoteTensor:
        if state.client is None:
            state.client = value._ref.client
        state.inputs[id(value._meta)] = value
        return value._meta, ("h", value._ref.handle)
    if kind is list or kind is tuple:
        metas = []
        codes = []
        for item in value:
            meta, code = _prepare(item, state)
            metas.append(meta)
            codes.append(code)
        return kind(metas), ("l" if kind is list else "t", codes)
    if isinstance(value, torch.Tensor):
        if isinstance(value, RemoteTensor):  # pragma: no cover - a subclass of RemoteTensor
            return _prepare(torch.Tensor.as_subclass(value, RemoteTensor), state)
        if value.device.type == "meta":
            # The autograd engine's zero gradient, built from the wrapper's metadata.
            dtype = str(value.dtype).split(".")[-1]
            return value, ("z", tuple(value.shape), tuple(value.stride()), dtype)
        if value.device.type != "cpu":
            raise UnsupportedMode(
                f"a tensor on {value.device} cannot be sent to the runtime, only CPU tensors can"
            )
        host = value.detach().contiguous()
        meta = torch.empty_strided(host.shape, host.stride(), dtype=host.dtype, device=META)
        dtype = str(host.dtype).split(".")[-1]
        from .frames import tensor_view

        view = tensor_view(host)
        if view.nbytes <= INLINE_BYTES:
            return meta, ("b", None, dtype, tuple(host.shape), bytes(view))
        state.keep.append(host)
        state.buffers.append(view)
        state.big = True
        return meta, ("b", len(state.buffers) - 1, dtype, tuple(host.shape), None)
    if kind is torch.device:
        if value.type == "meta":
            return value, ("dev", "remote")
        return value, ("dev", str(value))
    if kind in (torch.dtype, torch.memory_format, torch.layout):
        return value, ("a", str(value).split(".")[-1])
    raise UnsupportedMode(f"an argument of type {kind.__name__} cannot be forwarded to the runtime")


def _shared(func: Any) -> bool:
    return func is aten.detach.default or func is aten.alias.default


def dispatch(func: Any, args: tuple, kwargs: dict, client: Client | None) -> Any:
    """Run one ATen operator: metadata here, values on the runtime."""
    state = _Prepared(client)
    meta_args, code_args = _prepare(args, state)
    meta_kwargs = {}
    code_kwargs = {}
    for key, value in kwargs.items():
        meta_kwargs[key], code_kwargs[key] = _prepare(value, state)
    client = state.client
    if client is None:  # pragma: no cover - dispatch is only reached with a client
        return func(*args, **kwargs)
    name = str(func)

    if _shared(func) and isinstance(args[0], RemoteTensor):
        source = args[0]
        return RemoteTensor(func(source._meta), source._ref)

    if func is aten._local_scalar_dense.default:
        results, _buffers = client.request([(name, code_args[1], code_kwargs, None, "value")])
        return results[-1]

    if func is aten._to_copy.default:
        target = kwargs.get("device")
        if target is not None and torch.device(target).type == "cpu":
            dtype = kwargs.get("dtype")
            return client.fetch(args[0], None if dtype is None else str(dtype).split(".")[-1])

    copies_to_host = func is aten.copy_.default and not isinstance(args[0], RemoteTensor)
    if copies_to_host and isinstance(args[1], RemoteTensor):
        args[0].copy_(client.fetch(args[1], None))
        return args[0]

    try:
        meta_out = func(*meta_args, **meta_kwargs)
    except Exception:
        return client.execute_now(name, code_args[1], code_kwargs, state)

    leaves = meta_out if isinstance(meta_out, (list, tuple)) else (meta_out,)
    outs: list[int | None] = []
    results: list[Any] = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            existing = state.inputs.get(id(leaf))
            if existing is not None:
                if leaf.shape != existing.shape or leaf.stride() != existing.stride():
                    raise UnsupportedMode(
                        f"{name} changed the shape of a tensor in place, which host='local' "
                        f"cannot mirror on the local wrapper"
                    )
                outs.append(None)
                results.append(existing)
            else:
                handle = client.new_handle()
                outs.append(handle)
                results.append(RemoteTensor(leaf, Ref(client, handle)))
        else:
            results.append(leaf)
    client.enqueue((name, code_args[1], code_kwargs, outs, None), state)
    if isinstance(meta_out, (list, tuple)):
        return type(meta_out)(results)
    return results[0]


def from_description(client: Client, first: int, described: tuple) -> Any:
    """Build outputs from the runtime's description of an operator it ran at once."""
    shape, leaves = described
    handle = first
    results = []
    for leaf in leaves:
        if leaf[0] == "t":
            _, size, stride, offset, dtype = leaf
            meta = torch.empty_strided(size, stride, dtype=getattr(torch, dtype), device=META)
            if offset:
                meta = meta.as_strided(size, stride, offset)
            results.append(RemoteTensor(meta, Ref(client, handle)))
            handle += 1
        else:
            results.append(leaf[1])
    if shape == "one":
        return results[0]
    return results if shape == "list" else tuple(results)
