"""The device executor: ATen operators and captured steps run on real tensors keyed by handle.

This module owns what runs on the runtime for ``host="local"``: turning template
definitions into argument builders, running eager entries and step repetitions, storing
and releasing tensors, recording the first failure and answering synchronizations, as spec
"The device worker", "Operator templates", "Step capture" and "Failure semantics" describe.
It does not own queueing, dispatch or detection, which are ``client``, ``tensor`` and
``trace``.

It imports the standard library, ``wire``, ``frames`` and PyTorch only. Its source is sent
to the runtime after them and executed there, so the same code runs on a GPU runtime and in
the test suite's CPU worker, parameterized by device.
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
import traceback

try:
    from ...protocol.wire import REPLY, REQUEST
    from .frames import QueueTransport, StreamTransport, TransportClosed, tensor_view
except ImportError:  # pragma: no cover - the concatenated copy inside a worker
    pass

#: Entry kinds in a batch.
E_OP = 0
E_DEFINE = 1
E_REQUEST = 2
E_STEP_DEFINE = 3
E_STEP = 4


class MissingHandle(Exception):
    """An operator read a handle the runtime does not hold."""


def _missing(handle: int) -> MissingHandle:
    return MissingHandle(
        f"handle {handle} is not on the runtime: the operator that produced it "
        f"failed or was skipped after a failure"
    )


class Executor:
    """The runtime's tensor table and entry loop for one device."""

    def __init__(self, device: str):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.tensors: dict = {}
        self.failure: tuple | None = None
        self._ops: dict = {}
        #: Template number to ``(fn, builder, name, scalars, blobs)``.
        self._templates: dict = {}
        #: Step number to ``(operators, handles created before each position)``.
        self._steps: dict = {}
        self._buffers: list = []
        self._current = ""

    # -- identity ----------------------------------------------------------------

    def hello(self) -> dict:
        torch = self.torch
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            name = properties.name
            capability = (properties.major, properties.minor)
            total_memory = properties.total_memory
        else:
            name, capability, total_memory = "cpu", (0, 0), 0
        return {
            "device": str(self.device),
            "name": name,
            "capability": capability,
            "total_memory": total_memory,
            "torch": torch.__version__,
            "pid": os.getpid(),
        }

    # -- definitions -------------------------------------------------------------

    def op(self, name: str):
        found = self._ops.get(name)
        if found is None:
            found = self.torch.ops
            for part in name.split("."):
                found = getattr(found, part)
            self._ops[name] = found
        return found

    def define(self, tid: int, name: str, layout) -> None:
        builder, scalars, blobs = self._compile(layout)
        self._templates[tid] = (self.op(name), builder, name, scalars, blobs)

    def _compile(self, layout):
        """One generated function building ``(args, kwargs)`` from handles, scalars and blobs."""
        torch = self.torch
        constants: list = []
        counts = [0, 0]

        def constant(value) -> str:
            constants.append(value)
            return f"C[{len(constants) - 1}]"

        def expr(leaf) -> str:
            if type(leaf) is not tuple:
                return constant(leaf)
            tag = leaf[0]
            if tag == "h":
                return f"T[{leaf[1]}]"
            if tag == "s":
                counts[0] += 1
                return f"S[{leaf[1]}]"
            if tag == "b":
                counts[1] += 1
                return f"B({leaf[1]}, {constant(leaf[2])}, {constant(tuple(leaf[3]))})"
            if tag == "l":
                return "[" + "".join(expr(item) + ", " for item in leaf[1]) + "]"
            if tag == "t":
                return "(" + "".join(expr(item) + ", " for item in leaf[1]) + ")"
            if tag == "a":
                return constant(getattr(torch, leaf[1]))
            if tag == "dev":
                return constant(self.device if leaf[1] == "remote" else torch.device(leaf[1]))
            if tag == "z":
                return f"Z({constant((tuple(leaf[1]), leaf[3]))})"
            raise ValueError(f"unknown argument tag {tag!r}")

        args, kwargs = layout
        positional = "".join(expr(value) + ", " for value in args)
        keywords = ", ".join(f"{key!r}: {expr(value)}" for key, value in kwargs.items())
        source = f"lambda T, S, B: (({positional}), {{{keywords}}})"
        namespace = {"C": constants, "Z": self._zeros}
        return eval(source, namespace), counts[0], counts[1]

    def _zeros(self, spec):
        shape, dtype = spec
        return self.torch.zeros(shape, dtype=getattr(self.torch, dtype), device=self.device)

    def define_step(self, sid: int, ops) -> None:
        compiled = []
        before = [0]
        for tid, wiring, news in ops:
            fn, builder, name, scalars, blobs = self._templates[tid]
            compiled.append((fn, builder, name, scalars, blobs, tuple(wiring), tuple(news)))
            before.append(before[-1] + sum(news))
        self._steps[sid] = (compiled, before)

    # -- execution ---------------------------------------------------------------

    def _blob_reader(self, blobs, offset: int = 0):
        torch = self.torch
        buffers = self._buffers

        def read(index, dtype, shape):
            data = blobs[offset + index]
            data = buffers[data] if type(data) is int else bytearray(data)
            kind = getattr(torch, dtype)
            if not len(data):
                return torch.empty(shape, dtype=kind)
            return torch.frombuffer(data, dtype=kind).reshape(shape)

        return read

    def run(self, message: dict, buffers: list):
        """Execute one batch. Return ``(head, buffers, keep)`` when it asks for a reply."""
        self._buffers = buffers
        results: list = []
        out_buffers: list = []
        keep: list = []
        for entry in message.get("entries", ()):
            kind = entry[0]
            if kind == E_DEFINE:
                try:
                    self.define(entry[1], entry[2], entry[3])
                except BaseException as exc:
                    if self.failure is None:
                        self.failure = (
                            entry[2],
                            f"{type(exc).__name__}: {exc}",
                            traceback.format_exc(),
                        )
                continue
            if kind == E_STEP_DEFINE:
                try:
                    self.define_step(entry[1], entry[2])
                except BaseException as exc:
                    if self.failure is None:
                        self.failure = (
                            "step",
                            f"{type(exc).__name__}: {exc}",
                            traceback.format_exc(),
                        )
                continue
            want = entry[6] if kind == E_OP else entry[3] if kind == E_REQUEST else None
            if self.failure is not None:
                if want:
                    results.append(None)
                continue
            value = None
            try:
                if kind == E_OP:
                    value = self.execute(entry)
                elif kind == E_REQUEST:
                    self._current = entry[1]
                    value = self.special(entry[1], entry[2], out_buffers, keep)
                else:
                    self.replay(entry)
            except BaseException as exc:
                self.failure = (
                    self._current,
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )
                value = None
            if want:
                results.append(value)
        self._buffers = []
        # Released after the entries, because an operator queued before its input's last
        # RemoteTensor was collected travels in the same batch as that release.
        tensors = self.tensors
        for handle in message.get("release", ()):
            tensors.pop(handle, None)
        if not message.get("reply"):
            return None
        failure, self.failure = self.failure, None
        head = pickle.dumps({"results": results, "failure": failure}, protocol=5)
        views = [tensor_view(tensor) for tensor in out_buffers]
        return head, views, keep

    def _store(self, value, outs) -> None:
        torch = self.torch
        if isinstance(value, torch.Tensor):
            leaves = [value]
        elif isinstance(value, (list, tuple)):
            leaves = [leaf for leaf in value if isinstance(leaf, torch.Tensor)]
        else:
            leaves = []
        tensors = self.tensors
        if type(outs) is int:
            handle = outs
            for leaf in leaves:
                tensors[handle] = leaf
                handle += 1
            return
        for handle, leaf in zip(outs, leaves, strict=True):
            if handle is not None:
                tensors[handle] = leaf

    def execute(self, entry):
        _, tid, handles, scalars, blobs, outs, want = entry
        fn, builder, name, _scalars, _blobs = self._templates[tid]
        self._current = name
        table = self.tensors
        try:
            inputs = [table[handle] for handle in handles]
        except KeyError as exc:
            raise _missing(exc.args[0]) from None
        args, kwargs = builder(inputs, scalars, self._blob_reader(blobs) if blobs else None)
        value = fn(*args, **kwargs)
        if outs:
            self._store(value, outs)
        if want == "describe":
            return self._describe(value)
        return value if want == "value" else None

    def replay(self, entry) -> None:
        _, sid, first, start, stop, externals, scalars, blobs = entry
        ops, before = self._steps[sid]
        table = self.tensors
        torch_tensor = self.torch.Tensor
        handle = first + before[start]
        external = 0
        scalar = 0
        blob = 0
        for index in range(start, stop):
            fn, builder, name, n_scalars, n_blobs, wiring, news = ops[index]
            self._current = name
            inputs = []
            for offset in wiring:
                if offset < 0:
                    key = externals[external]
                    external += 1
                else:
                    key = first + offset
                tensor = table.get(key)
                if tensor is None:
                    raise _missing(key)
                inputs.append(tensor)
            bound = scalars[scalar : scalar + n_scalars] if n_scalars else ()
            scalar += n_scalars
            reader = self._blob_reader(blobs, blob) if n_blobs else None
            blob += n_blobs
            args, kwargs = builder(inputs, bound, reader)
            value = fn(*args, **kwargs)
            if news:
                if isinstance(value, torch_tensor):
                    if news[0]:
                        table[handle] = value
                        handle += 1
                else:
                    leaves = [leaf for leaf in value if isinstance(leaf, torch_tensor)]
                    for flag, leaf in zip(news, leaves, strict=True):
                        if flag:
                            table[handle] = leaf
                            handle += 1

    def _describe(self, value):
        torch = self.torch
        leaves = list(value) if isinstance(value, (list, tuple)) else [value]
        described = []
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                described.append(
                    (
                        "t",
                        tuple(leaf.shape),
                        tuple(leaf.stride()),
                        leaf.storage_offset(),
                        str(leaf.dtype).split(".")[-1],
                    )
                )
            else:
                described.append(("v", leaf))
        if isinstance(value, list):
            return ("list", described)
        return ("tuple" if isinstance(value, tuple) else "one", described)

    def special(self, name, args, out_buffers, keep):
        torch = self.torch
        if name == "letify.fetch":
            handle, dtype = args
            tensor = self.tensors.get(handle)
            if tensor is None:
                raise _missing(handle)
            tensor = tensor.detach()
            if dtype is not None:
                tensor = tensor.to(getattr(torch, dtype))
            host = tensor.to("cpu").contiguous()
            keep.append(host)
            out_buffers.append(host)
            return (len(out_buffers) - 1, tuple(host.shape), str(host.dtype).split(".")[-1])
        if name == "letify.live":
            return len(self.tensors)
        if name == "letify.seed":
            if self.device.type == "cuda":
                torch.cuda.manual_seed(args[0])
            else:
                torch.manual_seed(args[0])
            return None
        if name == "letify.memory":
            if self.device.type != "cuda":
                return 0
            return getattr(torch.cuda, args[0])(self.device)
        if name == "letify.empty_cache":
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None
        if name == "letify.kernel":
            return self.kernel(*args)
        raise ValueError(f"unknown request {name}")

    def kernel(self, function, described, flags):
        """The backend this device's dispatch picks for ``function`` on tensors of ``described``."""
        torch = self.torch

        def empty(entry):
            if entry is None:
                return None
            shape, stride, dtype = entry
            return torch.empty_strided(
                shape, stride, dtype=getattr(torch, dtype), device=self.device
            )

        tensors = [empty(entry) for entry in described]
        if function == "batch_norm":
            select = getattr(torch._C, "_select_batch_norm_backend", None)
            if select is None:
                return "Native"
            x, weight, bias, mean, var = tensors
            backend = select(
                x, weight, bias, mean, var, bool(flags["training"]), float(flags.get("eps", 1e-5))
            )
            return backend.name
        if function == "scaled_dot_product_attention":
            choose = getattr(torch, "_fused_sdp_choice", None)
            if choose is None:
                return "MATH"
            from torch.nn.attention import SDPBackend

            query, key, value, mask = tensors
            choice = int(
                choose(
                    query,
                    key,
                    value,
                    mask,
                    float(flags["dropout_p"]),
                    bool(flags["is_causal"]),
                    flags.get("scale"),
                    bool(flags.get("enable_gqa", False)),
                )
            )
            names = (
                "ERROR",
                "MATH",
                "FLASH_ATTENTION",
                "EFFICIENT_ATTENTION",
                "CUDNN_ATTENTION",
                "OVERRIDEABLE",
            )
            for backend_name in names:
                member = getattr(SDPBackend, backend_name, None)
                if member is not None and int(member) == choice:
                    return backend_name
            return "MATH"
        raise ValueError(f"no kernel selection for {function}")


def serve_transport(device: str, transport) -> None:
    """Announce the executor on ``transport`` and answer its messages until it closes."""
    executor = Executor(device)
    transport.send(pickle.dumps(executor.hello(), protocol=5), [])
    while True:
        try:
            head, buffers = transport.recv()
        except TransportClosed:
            return
        message = pickle.loads(head)
        head = None
        if message.get("close"):
            return
        reply = executor.run(message, buffers)
        buffers = None
        if reply is not None:
            reply_head, views, keep = reply
            try:
                transport.send(reply_head, views)
            except TransportClosed:
                return
            del keep


def serve(device: str) -> None:  # pragma: no cover - runs inside the worker subprocess
    """An executor as its own process: frames on standard input and output until they end."""
    out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    transport = StreamTransport(
        read_fd=0,
        write_fd=out_fd,
        readinto=sys.stdin.buffer.readinto,
        outgoing=REPLY,
        incoming=REQUEST,
    )
    serve_transport(device, transport)


def serve_channel(device: str, sender, inbox) -> threading.Thread:
    """An executor in a thread of the call worker, reading requests the frame reader queues."""
    transport = QueueTransport(sender, inbox)
    thread = threading.Thread(
        target=serve_transport, args=(device, transport), name="letify-device", daemon=True
    )
    thread.start()
    return thread
