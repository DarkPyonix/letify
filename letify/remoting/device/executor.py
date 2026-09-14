"""The device worker: ATen operators executed on real tensors keyed by handle.

This module owns what runs on the runtime for ``host="local"``: decoding operator
arguments, running operators, storing and releasing tensors, recording the first failure
and answering synchronizations, as spec "The device worker" and "Failure semantics"
describe. It does not own queueing or dispatch, which are ``client`` and ``tensor``.

It imports the standard library and PyTorch only. Its source is sent to the runtime after
``frames`` and executed there, so the same code runs on a GPU runtime and in the test
suite's CPU worker, parameterized by device.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback

try:
    from .frames import StreamTransport, TransportClosed, tensor_view
except ImportError:  # pragma: no cover - the concatenated copy inside a worker
    pass


class MissingHandle(Exception):
    """An operator read a handle the runtime does not hold."""


class Executor:
    """The runtime's tensor table and operator loop for one device."""

    def __init__(self, device: str):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.tensors: dict = {}
        self.failure: tuple | None = None
        self._ops: dict = {}
        self._buffers: list = []
        self._base = 0

    # -- identity ----------------------------------------------------------------

    def hello(self) -> dict:
        torch = self.torch
        name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        return {
            "device": str(self.device),
            "name": name,
            "torch": torch.__version__,
            "pid": os.getpid(),
        }

    # -- decoding ----------------------------------------------------------------

    def op(self, name: str):
        found = self._ops.get(name)
        if found is None:
            found = self.torch.ops
            for part in name.split("."):
                found = getattr(found, part)
            self._ops[name] = found
        return found

    def decode(self, value):
        if type(value) is not tuple:
            return value
        tag = value[0]
        if tag == "h":
            try:
                return self.tensors[value[1]]
            except KeyError:
                raise MissingHandle(
                    f"handle {value[1]} is not on the runtime: the operator that produced it "
                    f"failed or was skipped after a failure"
                ) from None
        if tag == "l":
            return [self.decode(item) for item in value[1]]
        if tag == "t":
            return tuple(self.decode(item) for item in value[1])
        if tag == "a":
            return getattr(self.torch, value[1])
        if tag == "dev":
            return self.device if value[1] == "remote" else self.torch.device(value[1])
        if tag == "b":
            _, index, dtype, shape, inline = value
            data = bytearray(inline) if index is None else self._buffers[self._base + index]
            dtype = getattr(self.torch, dtype)
            if not len(data):
                return self.torch.empty(shape, dtype=dtype)
            return self.torch.frombuffer(data, dtype=dtype).reshape(shape)
        if tag == "z":
            _, shape, _stride, dtype = value
            return self.torch.zeros(shape, dtype=getattr(self.torch, dtype), device=self.device)
        raise ValueError(f"unknown argument tag {tag!r}")

    # -- execution ---------------------------------------------------------------

    def run(self, message: dict, buffers: list):
        """Execute one batch. Return ``(head, buffers)`` when it asks for a reply."""
        self._buffers = buffers
        results: list = []
        out_buffers: list = []
        keep: list = []
        for entry in message.get("ops", ()):
            name, args, kwargs, outs, want, self._base = entry
            if self.failure is not None:
                if want:
                    results.append(None)
                continue
            try:
                value = self.execute(name, args, kwargs, outs, want, out_buffers, keep)
            except BaseException as exc:
                self.failure = (name, f"{type(exc).__name__}: {exc}", traceback.format_exc())
                value = None
            if want:
                results.append(value)
        self._buffers = []
        # Released after the operators, because an operator queued before its input's last
        # RemoteTensor was collected travels in the same batch as that release.
        for handle in message.get("release", ()):
            self.tensors.pop(handle, None)
        if not message.get("reply"):
            return None
        failure, self.failure = self.failure, None
        head = pickle.dumps({"results": results, "failure": failure}, protocol=5)
        views = [tensor_view(tensor) for tensor in out_buffers]
        return head, views, keep

    def execute(self, name, args, kwargs, outs, want, out_buffers, keep):
        torch = self.torch
        if name.startswith("letify."):
            return self.special(name, args, out_buffers, keep)
        fn = self.op(name)
        decode = self.decode
        value = fn(*[decode(v) for v in args], **{key: decode(v) for key, v in kwargs.items()})
        leaves = self._leaves(value)
        if outs is not None:
            if isinstance(outs, list):
                tensors = [x for x in leaves if isinstance(x, torch.Tensor)]
                for handle, leaf in zip(outs, tensors, strict=True):
                    if handle is not None:
                        self.tensors[handle] = leaf
            else:
                handle = outs
                for leaf in leaves:
                    if isinstance(leaf, torch.Tensor):
                        self.tensors[handle] = leaf
                        handle += 1
        if want == "value":
            return value
        if want == "describe":
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
        return None

    def _leaves(self, value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def special(self, name, args, out_buffers, keep):
        torch = self.torch
        if name == "letify.fetch":
            handle, dtype = args
            tensor = self.decode(("h", handle)).detach()
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
        raise ValueError(f"unknown request {name}")


def serve(device: str) -> None:  # pragma: no cover - runs inside the worker subprocess
    """Answer frames on standard input and output until the stream ends."""
    out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    transport = StreamTransport(read_fd=0, write_fd=out_fd, readinto=sys.stdin.buffer.readinto)
    executor = Executor(device)
    transport.send(pickle.dumps(executor.hello(), protocol=5), [])
    while True:
        try:
            head, buffers = transport.recv()
        except TransportClosed:
            return
        message = pickle.loads(head)
        if message.get("close"):
            return
        reply = executor.run(message, buffers)
        if reply is not None:
            reply_head, views, _keep = reply
            transport.send(reply_head, views)
