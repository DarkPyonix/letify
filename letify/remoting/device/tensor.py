"""RemoteTensor and operator dispatch.

This module owns the local stand-in for a tensor on the runtime and turning each ATen
operator into a structure, scalars, handles and blobs, as spec "Dispatch mechanism"
describes: output metadata is computed locally on meta tensors, and only values travel. It
does not own the queue, templates or step capture, which are ``client`` and ``trace``, or
the mapping of ``"cuda"``, which is ``cuda``.
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

_DETACH = aten.detach.default
_ALIAS = aten.alias.default
_SCALAR = aten._local_scalar_dense.default
_TO_COPY = aten._to_copy.default
_COPY = aten.copy_.default
_make_wrapper = torch.Tensor._make_wrapper_subclass  # type: ignore[attr-defined]


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


class Big:
    """A CPU tensor larger than ``INLINE_BYTES``, sent as an out-of-band buffer."""

    __slots__ = ("keep", "view")

    def __init__(self, view: memoryview, keep: torch.Tensor):
        self.view = view
        self.keep = keep


class RemoteTensor(torch.Tensor):
    """A tensor whose values live on the runtime and whose metadata lives here.

    ``_sig`` is ``(shape, strides, storage offset, dtype)``. The meta tensor is built from it
    only when an inference needs one.
    """

    _sig: tuple
    _ref: Ref
    _m: torch.Tensor | None

    @staticmethod
    def __new__(cls, sig: tuple, ref: Ref) -> RemoteTensor:
        made = _make_wrapper(
            cls,
            sig[0],
            strides=sig[1],
            storage_offset=sig[2],
            dtype=sig[3],
            device=META,
            requires_grad=False,
        )
        made._sig = sig
        #: The signature without its storage offset, which a template and a step compare.
        made._form = (sig[0], sig[1], sig[3])
        made._ref = ref
        made._m = None
        return made

    def __init__(self, sig: tuple, ref: Ref):
        pass

    # Tensor methods enter Python once, in cuda.CudaMode, which answers the specials below.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[assignment]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        for kind in types:
            if kind is not RemoteTensor:
                # Another wrapper subclass, such as a torchao weight, unpacks itself first,
                # as spec "Tensor subclasses" describes.
                return NotImplemented
        return dispatch(func, args, kwargs or {}, None)


def meta_of(tensor: RemoteTensor) -> torch.Tensor:
    """The tensor's meta counterpart, built on first use."""
    meta = tensor._m
    if meta is None:
        meta = tensor._m = _meta_like(*tensor._sig)
    return meta


# -- torch function specials, answered by cuda.CudaMode ------------------------------


def _repr(self: RemoteTensor, *args: Any, **kwargs: Any) -> str:
    text = repr(self.detach().cpu())
    if text.endswith(")"):
        return text[:-1] + f", device='{REPORTED}')"
    return text  # pragma: no cover - torch always closes the parenthesis


def _set_data(self: RemoteTensor, value: Any) -> None:
    with torch._C.DisableTorchFunctionSubclass():
        torch.Tensor.data.__set__(self, value)  # type: ignore[attr-defined]
    if isinstance(value, RemoteTensor):
        self._sig = value._sig
        self._form = value._form
        self._m = value._m
        self._ref = value._ref


SPECIAL = {
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


def signature(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tensor.stride(), tensor.storage_offset(), tensor.dtype)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _read(
    values: Any, parts: list, scalars: list, tensors: list, blobs: list, client_box: list
) -> bool:
    """Append one level of arguments to the structure, scalars, tensors and blobs.

    ``client_box`` collects the storage offset of each tensor argument, which keys metadata
    but not the template. Returns True when a CPU tensor larger than ``INLINE_BYTES`` was read.
    """
    big = False
    for value in values:
        kind = type(value)
        if kind is RemoteTensor:
            tensors.append(value)
            parts.append(value._form)
            client_box.append(value._sig[2])
        elif kind is int or kind is float:
            parts.append(kind)
            scalars.append(value)
        elif kind is list or kind is tuple:
            parts.append((kind, len(value)))
            big = _read(value, parts, scalars, tensors, blobs, client_box) or big
        elif value is None or kind is bool or kind is str:
            parts.append(value)
        elif isinstance(value, torch.Tensor):
            big = _read_tensor(value, parts, scalars, tensors, blobs, client_box) or big
        elif kind is torch.device or kind in (torch.dtype, torch.memory_format, torch.layout):
            parts.append(value)
        else:
            raise UnsupportedMode(
                f"an argument of type {kind.__name__} cannot be forwarded to the runtime"
            )
    return big


def _read_tensor(
    value: torch.Tensor, parts: list, scalars: list, tensors: list, blobs: list, client_box: list
) -> bool:
    if isinstance(value, RemoteTensor):  # pragma: no cover - a subclass of RemoteTensor
        cast = torch.Tensor.as_subclass(value, RemoteTensor)
        cast._sig, cast._ref, cast._m = value._sig, value._ref, value._m
        cast._form = value._form
        return _read((cast,), parts, scalars, tensors, blobs, client_box)
    if value.device.type == "meta":
        # The autograd engine's zero gradient, built from the wrapper's metadata.
        parts.append(("zeros", *signature(value)))
        return False
    if value.device.type != "cpu":
        raise UnsupportedMode(
            f"a tensor on {value.device} cannot be sent to the runtime, only CPU tensors can"
        )
    host = value.detach().contiguous()
    parts.append(("cpu", tuple(host.shape), host.dtype))
    from .frames import tensor_view

    view = tensor_view(host)
    if view.nbytes <= INLINE_BYTES:
        blobs.append(bytes(view))
        return False
    if host.data_ptr() == value.data_ptr():
        # The write happens later on the sender thread, so the bytes are copied now and a
        # change the caller makes to its tensor afterwards does not reach the runtime.
        host = host.clone()
        view = tensor_view(host)
    blobs.append(Big(view, host))
    return True


def layout(args: tuple, kwargs: dict) -> tuple:
    """The argument layout of a template: positions for handles, scalars and blobs."""
    counters = [0, 0, 0]

    def encode(value: Any) -> Any:
        kind = type(value)
        if isinstance(value, RemoteTensor):
            counters[0] += 1
            return ("h", counters[0] - 1)
        if kind is int or kind is float:
            counters[1] += 1
            return ("s", counters[1] - 1)
        if kind is list or kind is tuple:
            return ("l" if kind is list else "t", [encode(item) for item in value])
        if value is None or kind is bool or kind is str:
            return value
        if isinstance(value, torch.Tensor):
            if value.device.type == "meta":
                return ("z", tuple(value.shape), tuple(value.stride()), _dtype_name(value.dtype))
            counters[2] += 1
            return ("b", counters[2] - 1, _dtype_name(value.dtype), tuple(value.shape))
        if kind is torch.device:
            return ("dev", "remote") if value.type == "meta" else ("dev", str(value))
        return ("a", _dtype_name(value))  # a dtype, memory format or layout

    return [encode(value) for value in args], {key: encode(value) for key, value in kwargs.items()}


def _meta_args(value: Any) -> Any:
    kind = type(value)
    if kind is RemoteTensor:
        return meta_of(value)
    if kind is list or kind is tuple:
        return kind(_meta_args(item) for item in value)
    if isinstance(value, torch.Tensor):
        if isinstance(value, RemoteTensor):  # pragma: no cover - a subclass of RemoteTensor
            return _meta_like(*value._sig)
        if value.device.type == "cpu":
            return torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=META)
    return value


class Plan:
    """An operator's outputs, so a later call with the same key can rebuild them."""

    __slots__ = ("container", "leaves", "shape_id")

    def __init__(self, container: type | None, leaves: tuple):
        self.container = container
        self.leaves = leaves
        # Storage offsets are left out, so a slice at a new position each step keeps its id.
        form = tuple(
            (leaf[0], (leaf[1][0], leaf[1][1], leaf[1][3])) if leaf[0] == _NEW else leaf
            for leaf in leaves
        )
        try:
            self.shape_id = _SHAPES.setdefault(form, len(_SHAPES))
        except TypeError:  # pragma: no cover - an output value that cannot be hashed
            self.shape_id = -id(self)


#: Output layouts numbered, so step capture compares one integer per operator.
_SHAPES: dict[tuple, int] = {}

#: Inferred output metadata by operator and argument key. Cleared when it grows past
#: ``_CACHE_LIMIT`` entries, which a training loop with fixed shapes never reaches.
_CACHE: dict[tuple, Plan] = {}
_CACHE_LIMIT = 1 << 16

#: The printable overload name of each operator, whether its float scalars key metadata, and
#: whether its argument storage offsets do.
_NAMES: dict[Any, tuple[str, bool, bool]] = {}


def _aliases(func: Any) -> bool:
    """Whether any return of the operator's schema may alias an argument.

    An operator whose returns carry no alias information returns new tensors, so the
    storage offsets of its arguments do not change its output metadata.
    """
    try:
        return any(ret.alias_info is not None for ret in func._schema.returns)
    except AttributeError:  # pragma: no cover - an operator without a schema, or an old PyTorch
        return True


_NEW = 0
_IN = 1
_VALUE = 2


def _meta_like(shape: tuple, stride: tuple, offset: int, dtype: torch.dtype) -> torch.Tensor:
    if not offset:
        return torch.empty_strided(shape, stride, dtype=dtype, device=META)
    extent = offset + 1 + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))
    if 0 in shape:
        extent = offset
    base = torch.empty(extent, dtype=dtype, device=META)
    return base.as_strided(shape, stride, offset)


#: Returned by ``Client.replay`` when an operator needs the full path.
MISS = object()

#: The client whose repetition of a captured step is running, set and cleared by that client.
REPLAY: list[Client | None] = [None]

_LITERAL_TYPES = (bool, str, torch.device, torch.dtype, torch.memory_format, torch.layout)


def reader(args: tuple, kwargs: dict, floats_keyed: bool, offsets_keyed: bool = True) -> Any:
    """A generated function checking arguments against these ones, per spec "Step capture".

    It returns ``(tensors, scalars, key tail)`` for arguments with the same structure and
    None otherwise. The key tail holds the offsets only when ``offsets_keyed``, as
    ``dispatch`` builds the metadata key. ``reader`` itself returns None when an argument
    has no exact check, such as a plain CPU or meta tensor.
    """
    lines = ["def read(a, k):"]
    constants: dict[str, Any] = {"RT": RemoteTensor}
    tensors: list[str] = []
    scalars: list[str] = []
    keyed: list[str] = []
    counter = [0]

    def name() -> str:
        counter[0] += 1
        return f"v{counter[0]}"

    def constant(value: Any) -> str:
        label = f"c{len(constants)}"
        constants[label] = value
        return label

    def emit(expr: str, value: Any) -> bool:
        kind = type(value)
        if value is None:
            lines.append(f"    if {expr} is not None: return None")
            return True
        var = name()
        lines.append(f"    {var} = {expr}")
        if kind is RemoteTensor:
            form = constant(value._form)
            lines.append(f"    if type({var}) is not RT or {var}._form != {form}: return None")
            tensors.append(var)
            return True
        if kind is int or kind is float:
            lines.append(f"    if type({var}) is not {kind.__name__}: return None")
            scalars.append(var)
            if kind is int or floats_keyed:
                keyed.append(var)
            return True
        if kind is list or kind is tuple:
            lines.append(
                f"    if type({var}) is not {kind.__name__} or len({var}) != {len(value)}: "
                "return None"
            )
            return all(emit(f"{var}[{index}]", item) for index, item in enumerate(value))
        if kind in _LITERAL_TYPES:
            lines.append(
                f"    if type({var}) is not {constant(kind)} or {var} != {constant(value)}: "
                "return None"
            )
            return True
        return False

    lines.append(f"    if len(a) != {len(args)}: return None")
    if kwargs:
        lines.append(f"    if tuple(k) != {constant(tuple(kwargs))}: return None")
    else:
        lines.append("    if k: return None")
    for index, value in enumerate(args):
        if not emit(f"a[{index}]", value):
            return None
    for key, value in kwargs.items():
        if not emit(f"k[{key!r}]", value):
            return None
    tail = ([f"{var}._sig[2]" for var in tensors] if offsets_keyed else []) + keyed

    def pack(names: list[str]) -> str:
        return "(" + "".join(f"{item}, " for item in names) + ")"

    lines.append(f"    return {pack(tensors)}, {pack(scalars)}, {pack(tail)}")
    exec(compile("\n".join(lines), "letify-reader", "exec"), constants)
    return constants["read"]


def _restride(tensor: RemoteTensor, sig: tuple) -> None:
    """Give a wrapper the view metadata an in-place operator gave its meta counterpart."""
    from torch.utils._mode_utils import no_dispatch

    if tensor._sig == sig:
        return
    with no_dispatch(), torch._C.DisableTorchFunction():
        torch.Tensor.as_strided_(tensor, sig[0], sig[1], sig[2])
    tensor._sig = sig
    tensor._form = (sig[0], sig[1], sig[3])
    tensor._m = None


def outputs(plan: Plan, client: Client, tensors: Any) -> tuple[Any, list]:
    """Build an operator's results from its plan, and the new handle of each tensor output."""
    outs: list = []
    results: list = []
    for leaf in plan.leaves:
        tag = leaf[0]
        if tag == _NEW:
            handle = client._next_handle
            client._next_handle = handle + 1
            outs.append(handle)
            results.append(RemoteTensor(leaf[1], Ref(client, handle)))
        elif tag == _IN:
            outs.append(None)
            existing = tensors[leaf[1]]
            if len(leaf) > 2:
                _restride(existing, leaf[2])
            results.append(existing)
        else:
            results.append(leaf[1])
    if plan.container is None:
        return results[0], outs
    return plan.container(results), outs


def dispatch(func: Any, args: tuple, kwargs: dict, client: Client | None) -> Any:
    """Run one ATen operator: metadata here, values on the runtime."""
    if func is _DETACH or func is _ALIAS:
        source = args[0]
        if type(source) is RemoteTensor:
            return RemoteTensor(source._sig, source._ref)

    replaying = REPLAY[0]
    if replaying is not None:
        done = replaying.replay(func, args, kwargs, client)
        if done is not MISS:
            return done

    named = _NAMES.get(func)
    if named is None:
        name = str(func)
        named = _NAMES[func] = (name, "_foreach_" not in name, _aliases(func))
    name, floats_keyed, offsets_keyed = named

    if func is _SCALAR and type(args[0]) is RemoteTensor:
        return args[0]._ref.client.read_value(args[0])

    if func is _TO_COPY and type(args[0]) is RemoteTensor:
        target = kwargs.get("device")
        if target is not None and torch.device(target).type == "cpu":
            dtype = kwargs.get("dtype")
            if kwargs.get("non_blocking"):
                return args[0]._ref.client.read_later(args[0], dtype)
            return args[0]._ref.client.fetch(args[0], None if dtype is None else _dtype_name(dtype))

    if func is _COPY and not isinstance(args[0], RemoteTensor) and type(args[1]) is RemoteTensor:
        if (args[2] if len(args) > 2 else kwargs.get("non_blocking")) and args[0].shape == args[
            1
        ].shape:
            args[1]._ref.client.read_later(args[1], None, args[0])
            return args[0]
        args[0].copy_(args[1]._ref.client.fetch(args[1], None))
        return args[0]

    parts: list = [func]
    scalars: list = []
    tensors: list = []
    blobs: list = []
    box: list = []
    big = _read(args, parts, scalars, tensors, blobs, box)
    if kwargs:
        for key, value in kwargs.items():
            parts.append(key)
            big = _read((value,), parts, scalars, tensors, blobs, box) or big
    if client is None:
        if not tensors:  # pragma: no cover - dispatch is only reached with a client
            return func(*args, **kwargs)
        client = tensors[0]._ref.client
    structure = tuple(parts)
    if not offsets_keyed:
        box = []
    if floats_keyed:
        key = (structure, *box, *scalars)
    else:
        key = (structure, *box, *[value for value in scalars if type(value) is int])
    try:
        plan = _CACHE.get(key)
    except TypeError:  # pragma: no cover - an argument value that cannot be hashed
        key, plan = None, None

    if plan is None:
        try:
            meta_out = func(*_meta_args(args), **{k: _meta_args(v) for k, v in kwargs.items()})
        except Exception:
            return client.execute_now(structure, name, args, kwargs, tensors, scalars, blobs)
        plan = _plan(name, meta_out, tensors)
        if key is not None:
            if len(_CACHE) >= _CACHE_LIMIT:
                _CACHE.clear()
            _CACHE[key] = plan
    else:
        client.stats.cached += 1

    result, outs = outputs(plan, client, tensors)
    client.put(structure, name, args, kwargs, plan, tensors, scalars, blobs, outs, big, key)
    return result


def _plan(name: str, meta_out: Any, tensors: list) -> Plan:
    container = type(meta_out) if isinstance(meta_out, (list, tuple)) else None
    leaves = meta_out if container is not None else (meta_out,)
    positions: dict[int, int] = {}
    for position, tensor in enumerate(tensors):
        if tensor._m is not None:
            positions.setdefault(id(tensor._m), position)
    planned: list[tuple] = []
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            planned.append((_VALUE, leaf))
            continue
        position = positions.get(id(leaf))
        if position is None:
            planned.append((_NEW, signature(leaf)))
            continue
        existing = tensors[position]
        changed = signature(leaf)
        if changed != existing._sig:
            # An in-place restride, such as as_strided_: the wrapper takes the new view.
            planned.append((_IN, position, changed))
            continue
        planned.append((_IN, position))
    return Plan(container, tuple(planned))


def from_description(client: Client, first: int, described: tuple) -> Any:
    """Build outputs from the runtime's description of an operator it ran at once."""
    shape, leaves = described
    handle = first
    results = []
    for leaf in leaves:
        if leaf[0] == "t":
            _, size, stride, offset, dtype = leaf
            sig = (tuple(size), tuple(stride), offset, getattr(torch, dtype))
            results.append(RemoteTensor(sig, Ref(client, handle)))
            handle += 1
        else:
            results.append(leaf[1])
    if shape == "one":
        return results[0]
    return results if shape == "list" else tuple(results)
