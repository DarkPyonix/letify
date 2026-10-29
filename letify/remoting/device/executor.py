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

import collections
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

#: Compiled variants one step may have in a session, as spec "Compilation" decides.
MAX_VARIANTS = 4

#: Operator name fragments whose segments stay eager, because Inductor draws random numbers
#: from its own generator.
RANDOM_TOKENS = (
    "rand",
    "dropout",
    "bernoulli",
    "normal",
    "multinomial",
    "uniform",
    "exponential",
    "random",
)

#: Bytes of recent uploads the executor keeps by digest and length, least recently used out.
UPLOAD_CACHE_BYTES = 1 << 30


class MissingHandle(Exception):
    """An operator read a handle the runtime does not hold."""


def _missing(handle: int) -> MissingHandle:
    return MissingHandle(
        f"handle {handle} is not on the runtime: the operator that produced it "
        f"failed or was skipped after a failure"
    )


#: Held while a segment is traced and compiled, because tracing patches global state.
_COMPILE_LOCK = threading.Lock()


def _meta(tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, str(tensor.device))


def _varying(previous, current, known) -> tuple:
    """The scalar slots to lift to tensors: ``known`` and those that changed, if numbers."""
    slots = set(known)
    for slot, (before, now) in enumerate(zip(previous, current, strict=True)):
        if type(before) in (int, float) and type(now) is type(before) and before != now:
            slots.add(slot)
    return tuple(sorted(slots))


class _Segment:
    """The compilation state of one ``(step, start, stop)`` segment."""

    def __init__(self, key: tuple, first: tuple, candidates: frozenset):
        self.key = key
        #: Offsets some entry of the segment kept, which a compiled form returns.
        self.candidates = candidates
        #: Scalars of the first sighting.
        self.first = first
        self.varying: tuple = ()
        #: The ready ``_Variant``, if any.
        self.variant = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None
        #: Whether the segment runs eagerly for the rest of the session.
        self.eager = False


class _Variant:
    """One compiled form of a segment and what it was compiled for."""

    def __init__(
        self, pattern: tuple, metas: tuple, scalars: tuple, varying: tuple, candidates: frozenset
    ):
        self.candidates = candidates
        self.pattern = pattern
        self.metas = metas
        self.scalars = scalars
        self.varying = varying
        self.kinds = tuple(type(scalars[slot]) for slot in varying)
        self.compiled = None
        self.survivors: tuple = ()
        self.persistent: frozenset = frozenset()
        self.drops: tuple = ()
        self.blobs: tuple = ()

    def matches(self, scalars: tuple) -> bool:
        """Whether the constants are unchanged and the varying slots keep their type."""
        if len(scalars) != len(self.scalars):
            return False
        varying = self.varying
        for slot, (was, now) in enumerate(zip(self.scalars, scalars, strict=True)):
            if slot in varying:
                if type(now) is not type(was):
                    return False
            elif type(now) is not type(was) or now != was:
                return False
        return True


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
        #: Received upload bytes by ``(digest, length)``, least recently used first.
        self._uploads: collections.OrderedDict = collections.OrderedDict()
        self._upload_bytes = 0
        self._upload_budget = UPLOAD_CACHE_BYTES
        #: ``(mode, options)`` once the client asked for compilation, else None.
        self._compilation: tuple | None = None
        #: Segment key to its ``_Segment``.
        self._segments: dict = {}
        #: Step number to the compilations started for it.
        self._variants: dict = {}
        self._compile_counts = {"compiled": 0, "failed": 0, "runs": 0}
        #: Lines for the client, sent with the next reply.
        self._notes: list = []

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
        last: dict = {}
        made = 0
        for index, (tid, wiring, news) in enumerate(ops):
            fn, builder, name, scalars, blobs = self._templates[tid]
            compiled.append((fn, builder, name, scalars, blobs, tuple(wiring), tuple(news)))
            for offset in wiring:
                if offset >= 0:
                    last[offset] = index
            for flag in news:
                if flag:
                    last[made] = index
                    made += 1
            before.append(made)
        # The offsets last read at each position, released after it as spec "Step capture",
        # **Early release**, describes; the client computes the same schedule in ``trace``.
        frees: list = [[] for _ in ops]
        for offset, index in last.items():
            frees[index].append(offset)
        self._steps[sid] = (compiled, before, tuple(tuple(sorted(group)) for group in frees))

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

    def _evict_uploads(self) -> None:
        uploads = self._uploads
        while self._upload_bytes > self._upload_budget:
            _key, data = uploads.popitem(last=False)
            self._upload_bytes -= len(data)

    def _resolve(self, blobs, buffers: list):
        """Turn upload cache markers into buffer indices, updating the table in order."""
        resolved = []
        uploads = self._uploads
        for blob in blobs:
            if type(blob) is not tuple:
                resolved.append(blob)
            elif blob[0] == "p":
                _, index, key = blob
                if key not in uploads:
                    uploads[key] = buffers[index]
                    self._upload_bytes += key[1]
                uploads.move_to_end(key)
                self._evict_uploads()
                resolved.append(index)
            else:
                key = blob[1]
                uploads.move_to_end(key)
                buffers.append(uploads[key])
                resolved.append(len(buffers) - 1)
        return tuple(resolved)

    def _apply_uploads(self, message: dict, buffers: list) -> list:
        """Apply a batch's upload cache updates before any of its entries runs."""
        budget = message.get("cache_bytes")
        if budget is not None:
            self._upload_budget = budget
            self._evict_uploads()
        entries = message.get("entries", ())
        rebuilt = None
        for position, entry in enumerate(entries):
            kind = entry[0]
            at = 4 if kind == E_OP else 7 if kind == E_STEP else None
            if at is None or not entry[at]:
                continue
            if not any(type(blob) is tuple for blob in entry[at]):
                continue
            if rebuilt is None:
                rebuilt = list(entries)
            rebuilt[position] = (*entry[:at], self._resolve(entry[at], buffers), *entry[at + 1 :])
        return entries if rebuilt is None else rebuilt

    def run(self, message: dict, buffers: list):
        """Execute one batch. Return ``(head, buffers, keep)`` when it asks for a reply."""
        buffers = list(buffers)
        entries = self._apply_uploads(message, buffers)
        self._buffers = buffers
        results: list = []
        out_buffers: list = []
        keep: list = []
        tensors = self.tensors
        # Each release is applied after the last entry of this batch that reads the handle,
        # and at once when none does, as spec "Handles" describes.
        after: dict = {}
        release = message.get("release", ())
        if release:
            wanted = set(release)
            last: dict = {}
            # Entries from which every handle at or above a bound may be read or created: a
            # step entry and an entry whose outputs the runtime describes.
            open_from: list = []
            for index, entry in enumerate(entries):
                kind = entry[0]
                if kind == E_OP:
                    touched = list(entry[2])
                    outs = entry[5]
                    if type(outs) is int:
                        open_from.append((index, outs))
                    elif outs:
                        touched.extend(handle for handle in outs if handle is not None)
                elif kind == E_STEP:
                    touched = entry[5]
                    open_from.append((index, entry[2]))
                elif kind == E_REQUEST and entry[1] == "letify.fetch":
                    touched = (entry[2][0],)
                else:
                    continue
                for handle in touched:
                    if handle in wanted:
                        last[handle] = index
            for index, bound in open_from:
                for handle in release:
                    if handle >= bound and last.get(handle, -1) < index:
                        last[handle] = index
            for handle in release:
                index = last.get(handle)
                if index is None:
                    tensors.pop(handle, None)
                else:
                    after.setdefault(index, []).append(handle)
        for index, entry in enumerate(entries):
            if after and index - 1 in after:
                for handle in after.pop(index - 1):
                    tensors.pop(handle, None)
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
        for handles in after.values():
            for handle in handles:
                tensors.pop(handle, None)
        if not message.get("reply"):
            return None
        failure, self.failure = self.failure, None
        reply = {"results": results, "failure": failure}
        if self._notes:
            reply["notes"], self._notes = self._notes, []
        head = pickle.dumps(reply, protocol=5)
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
        if self._compilation is not None and self._replay_compiled(entry):
            return
        _, sid, first, start, stop, externals, scalars, blobs, keep = entry
        ops, before, frees = self._steps[sid]
        if len(keep) > 8:
            keep = frozenset(keep)
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
            value = None
            inputs = None
            drop = frees[index]
            if drop:
                for offset in drop:
                    if offset not in keep:
                        table.pop(first + offset, None)

    # -- compiled segments -----------------------------------------------------------

    def compile_stats(self, wait: bool) -> dict:
        """Counters of compiled segments, after joining the compilations in progress if asked."""
        if wait:
            for segment in list(self._segments.values()):
                if segment.thread is not None:
                    segment.thread.join()
                    self._adopt(segment)
        eager = sum(1 for segment in self._segments.values() if segment.eager)
        return {**self._compile_counts, "eager": eager}

    def _adopt(self, segment) -> None:
        """Take the result of a finished background compilation."""
        thread = segment.thread
        if thread is None or thread.is_alive():
            return
        segment.thread = None
        if segment.error is not None:
            segment.eager = True
            self._compile_counts["failed"] += 1
            self._notes.append(
                f"letify: step {segment.key[0]} could not be compiled: {segment.error}; "
                f"it runs eagerly"
            )
        else:
            self._compile_counts["compiled"] += 1

    def _replay_compiled(self, entry) -> bool:
        """Run ``entry`` as a compiled segment when one is ready. False means run it eagerly."""
        _, sid, _first, start, stop, _externals, scalars, _blobs, keep = entry
        key = (sid, start, stop)
        segment = self._segments.get(key)
        if segment is None:
            self._segments[key] = _Segment(key, tuple(scalars), frozenset(keep))
            return False
        if segment.thread is not None:
            self._adopt(segment)
        if segment.eager or segment.thread is not None:
            return False
        variant = segment.variant
        if variant is None:
            # The second sighting: the slots that differ from the first are varying.
            segment.varying = _varying(segment.first, scalars, ())
        elif variant.matches(scalars) and self._unchanged(variant, entry):
            if variant.candidates.issuperset(keep):
                return self._run_compiled(segment, variant, entry)
            segment.varying = variant.varying
        else:
            segment.varying = _varying(variant.scalars, scalars, variant.varying)
        # Offsets any entry kept so far stay outputs, so the keeps of a loop whose live
        # tensors come and go with garbage collection converge on one compilation.
        segment.candidates = segment.candidates | frozenset(keep)
        self._start_compile(segment, entry)
        return False

    def _inputs(self, entry):
        """The distinct tensors a segment reads from outside itself and their repeat pattern."""
        _, sid, first, start, stop, externals, _scalars, _blobs, _keep = entry
        ops, before, _frees = self._steps[sid]
        table = self.tensors
        slots: dict = {}
        tensors: list = []
        pattern: list = []
        external = 0
        for index in range(start, stop):
            for offset in ops[index][5]:
                if offset < 0:
                    handle = externals[external]
                    external += 1
                elif offset < before[start]:
                    handle = first + offset
                else:
                    continue
                at = slots.get(handle)
                if at is None:
                    tensor = table.get(handle)
                    if tensor is None:
                        return None, None
                    at = slots[handle] = len(tensors)
                    tensors.append(tensor)
                pattern.append(at)
        return tensors, tuple(pattern)

    def _unchanged(self, variant, entry) -> bool:
        """Whether the inputs of ``entry`` are what ``variant`` was compiled for."""
        tensors, pattern = self._inputs(entry)
        if tensors is None or pattern != variant.pattern:
            return False
        if tuple(_meta(tensor) for tensor in tensors) != variant.metas:
            return False
        pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors}
        return len(pointers) == len(tensors)

    def _start_compile(self, segment, entry) -> None:
        _, sid, _first, start, stop, _externals, scalars, _blobs, _keep = entry
        names = [self._steps[sid][0][index][2] for index in range(start, stop)]
        if any(token in name.split(".")[1] for name in names for token in RANDOM_TOKENS):
            segment.eager = True
            return
        tensors, pattern = self._inputs(entry)
        if tensors is None:
            return
        started = self._variants.get(sid, 0)
        if started >= MAX_VARIANTS:
            segment.eager = True
            self._notes.append(
                f"letify: step {sid} reached {MAX_VARIANTS} compiled variants; it runs eagerly"
            )
            return
        self._variants[sid] = started + 1
        variant = _Variant(
            pattern,
            tuple(_meta(tensor) for tensor in tensors),
            tuple(scalars),
            segment.varying,
            segment.candidates,
        )
        examples = [tensor.detach() for tensor in tensors]
        segment.variant = None
        segment.error = None
        segment.thread = threading.Thread(
            target=self._compile_variant,
            args=(segment, variant, examples),
            name="letify-compile",
            daemon=True,
        )
        segment.thread.start()

    def _compile_variant(self, segment, variant, examples) -> None:
        """Trace and compile one segment variant. Runs in a thread of its own."""
        try:
            sid, start, stop = segment.key
            torch = self.torch
            function, survivors, persistent, drops, blobs = self._segment_function(
                sid, start, stop, variant
            )
            flat = list(examples)
            flat.extend(
                torch.empty(shape, dtype=getattr(torch, dtype)) for _, dtype, shape in blobs
            )
            for slot in variant.varying:
                value = variant.scalars[slot]
                flat.append(
                    torch.tensor(
                        value, dtype=torch.float64 if type(value) is float else torch.int64
                    )
                )
            mode, options = self._compilation
            options = dict(options)
            if mode in ("max-autotune", "max-autotune-no-cudagraphs"):
                options.setdefault("max_autotune", True)
            import torch._dynamo.config as dynamo_config
            import torch._inductor.config as inductor_config
            from torch._inductor.compile_fx import compile_fx
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.proxy_tensor import make_fx
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            # Tracing patches PyTorch's own state, so one compilation runs at a time.
            with (
                _COMPILE_LOCK,
                dynamo_config.patch(capture_scalar_outputs=True),
                inductor_config.patch(options),
            ):
                graph = make_fx(function, tracing_mode="symbolic")(*flat)
                fake = FakeTensorMode(shape_env=ShapeEnv(), static_shapes=True)
                compiled = compile_fx(graph, [fake.from_tensor(tensor) for tensor in flat])
            variant.compiled = compiled
            variant.survivors = survivors
            variant.persistent = persistent
            variant.drops = drops
            variant.blobs = blobs
            segment.variant = variant
        except BaseException as exc:
            segment.error = f"{type(exc).__name__}: {exc}".splitlines()[0]

    def _segment_function(self, sid, start, stop, variant):
        """The function replaying ``start`` to ``stop - 1`` of a step over a table of its own.

        Returns it with the offsets of the survivors it returns, those of them no entry
        releases, the carried offsets a replay would release, and the blobs it takes as
        ``(index, dtype, shape)``.
        """
        ops, before, frees = self._steps[sid]
        released = {offset for index in range(start, stop) for offset in frees[index]}
        persistent = frozenset(
            offset for offset in range(before[start], before[stop]) if offset not in released
        )
        survivors = tuple(
            offset
            for offset in range(before[start], before[stop])
            if offset in persistent or offset in variant.candidates
        )
        drops = tuple(offset for offset in sorted(released) if offset < before[start])
        # The blobs the segment reads: each builder asks its reader once per blob, with
        # placeholders for its tensors, so the dtypes and shapes are known before tracing.
        blobs: list = []
        scalar = 0
        blob = 0
        for index in range(start, stop):
            _fn, builder, _name, n_scalars, n_blobs, wiring, _news = ops[index]

            def record(i, dtype, shape, base=blob):
                blobs.append((base + i, dtype, tuple(shape)))
                return self.torch.empty(shape, dtype=getattr(self.torch, dtype))

            builder([None] * len(wiring), variant.scalars[scalar : scalar + n_scalars], record)
            scalar += n_scalars
            blob += n_blobs
        position = {absolute: at for at, (absolute, _d, _s) in enumerate(blobs)}
        n_inputs = len(set(variant.pattern))
        n_blobs_total = len(blobs)
        dense = self.torch.ops.aten._local_scalar_dense.default
        torch_tensor = self.torch.Tensor
        pattern = variant.pattern
        varying = variant.varying
        scalars = variant.scalars
        first_created = before[start]

        def function(*flat):
            inputs = flat[:n_inputs]
            blob_tensors = flat[n_inputs : n_inputs + n_blobs_total]
            values = list(scalars)
            for at, slot in enumerate(varying):
                values[slot] = dense(flat[n_inputs + n_blobs_total + at])
            local: dict = {}
            handle = first_created
            cursor = 0
            scalar = 0
            blob = 0
            for index in range(start, stop):
                fn, builder, _name, n_scalars, n_blobs, wiring, news = ops[index]
                arguments = []
                for offset in wiring:
                    if offset >= first_created:
                        arguments.append(local[offset])
                    else:
                        arguments.append(inputs[pattern[cursor]])
                        cursor += 1

                def read(i, _dtype, _shape, base=blob):
                    return blob_tensors[position[base + i]]

                args, kwargs = builder(
                    arguments, values[scalar : scalar + n_scalars], read if n_blobs else None
                )
                scalar += n_scalars
                blob += n_blobs
                value = fn(*args, **kwargs)
                if news:
                    if isinstance(value, torch_tensor):
                        if news[0]:
                            local[handle] = value
                            handle += 1
                    else:
                        leaves = [leaf for leaf in value if isinstance(leaf, torch_tensor)]
                        for flag, leaf in zip(news, leaves, strict=True):
                            if flag:
                                local[handle] = leaf
                                handle += 1
            return tuple(local[offset] for offset in survivors)

        return function, survivors, persistent, drops, tuple(blobs)

    def _run_compiled(self, segment, variant, entry) -> bool:
        _, sid, first, _start, _stop, _externals, scalars, blobs, keep = entry
        torch = self.torch
        tensors, _pattern = self._inputs(entry)
        reader = self._blob_reader(blobs)
        flat = list(tensors)
        flat.extend(reader(index, dtype, shape) for index, dtype, shape in variant.blobs)
        for slot in variant.varying:
            value = scalars[slot]
            flat.append(
                torch.tensor(value, dtype=torch.float64 if type(value) is float else torch.int64)
            )
        self._current = f"compiled step {sid}"
        try:
            outputs = variant.compiled(*flat)
        except BaseException:
            segment.eager = True
            raise
        table = self.tensors
        persistent = variant.persistent
        for offset, tensor in zip(variant.survivors, outputs, strict=True):
            if offset in persistent or offset in keep:
                table[first + offset] = tensor
        for offset in variant.drops:
            if offset not in keep:
                table.pop(first + offset, None)
        self._compile_counts["runs"] += 1
        return True

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
            if self.device.type == "cuda":
                return getattr(torch.cuda, args[0])(self.device)
            if args[0] == "memory_stats":
                return {}
            if args[0] == "mem_get_info":
                return (0, 0)
            return None if args[0].startswith("reset_") else 0
        if name == "letify.empty_cache":
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None
        if name == "letify.compile":
            self._compilation = (args[0], dict(args[1]))
            return None
        if name == "letify.compile_stats":
            return self.compile_stats(bool(args[0]))
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
                    scale=flags.get("scale"),
                    enable_gqa=bool(flags.get("enable_gqa", False)),
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
    """Announce the executor on ``transport`` and answer its messages until it closes.

    An exception that ends the executor is sent as a last message, so the client stops
    waiting for a reply that cannot come, as spec "Failure semantics" describes.
    """
    started = False
    try:
        executor = Executor(device)
        transport.send(pickle.dumps(executor.hello(), protocol=5), [])
        started = True
        _answer(executor, transport)
    except TransportClosed:
        return
    except BaseException as exc:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        fatal = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": detail,
            "started": started,
            "missing": exc.name if isinstance(exc, ImportError) else None,
        }
        try:
            transport.send(pickle.dumps({"fatal": fatal}, protocol=5), [])
        except Exception:  # pragma: no cover - the link is gone as well
            pass


def _answer(executor: Executor, transport) -> None:
    while True:
        head, buffers = transport.recv()
        message = pickle.loads(head)
        head = None
        if message.get("close"):
            return
        reply = executor.run(message, buffers)
        buffers = None
        if reply is not None:
            reply_head, views, keep = reply
            transport.send(reply_head, views)
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
