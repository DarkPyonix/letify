"""The order a call's data blobs are sent in, derived from the pickled call.

Owns the analysis of spec "The send order and the first wave": what the pickled arguments
say, what the function's code says, and the manifest order everything else follows. It
answers digests in the order they should travel, and where the first wave ends. It does not
own sending, which is ``pathdata``, nor the order observed on the runtime, which the worker
reports with ``data_want``.

Nothing here imports a provider's library or calls user code, with one exception the spec
names: a sampler is iterated once, because that is where a shuffled epoch's order lives.
"""

from __future__ import annotations

import dis
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pathdata import Collector, Placed

#: Calls that read a path, so a path named by one is ordered at that position.
READS = frozenset(
    {
        "open",
        "read_text",
        "read_bytes",
        "load",
        "memmap",
        "read_csv",
        "read_parquet",
        "load_file",
    }
)

#: Calls that read a whole directory, so the directory's files are ordered there.
WALKS = frozenset({"glob", "rglob", "iterdir", "listdir", "scandir", "walk"})

#: Attribute names a ``Dataset`` keeps its file sequence under, the names PyTorch uses.
SEQUENCES = ("samples", "imgs", "files", "paths", "image_paths")

#: How deep the code scan follows functions named in the globals of the declared function.
DEPTH = 2

#: Containers larger than this are not searched, so a huge argument costs nothing.
WALK_LIMIT = 100_000


def compute(
    collector: Collector,
    fn: Any,
    args: tuple,
    kwargs: dict,
    declared: Any = None,
) -> list[str]:
    """The digests of a call's files, in the order the call is expected to read them.

    ``declared`` is ``data_order`` from the declaration, which outranks the analysis.
    """
    index = _Index(collector)
    if declared is not None:
        index.take(declared)
    else:
        try:
            _from_arguments(index, fn, args, kwargs)
        except Exception:
            # An analysis that fails leaves the manifest order, and the call is unaffected.
            pass
        try:
            _from_code(index, fn)
        except Exception:
            pass
    return index.order()


def first_wave(
    order: list[str],
    sizes: dict[str, int],
    missing: set[str],
    *,
    mib: float,
    files: int,
    count: int | None = None,
) -> list[str]:
    """The leading digests to place before the call, cut at the byte and file limits.

    ``count`` is ``data_first_wave`` from the declaration, which counts entries instead.
    A digest the runtime already holds costs nothing and counts against neither limit.
    """
    wave: list[str] = []
    budget = int(mib * (1 << 20))
    taken = 0
    for digest in order:
        if digest not in missing:
            continue
        if count is not None:
            if len(wave) >= count:
                break
        elif len(wave) >= files or taken + sizes.get(digest, 0) > budget:
            break
        wave.append(digest)
        taken += sizes.get(digest, 0)
    return wave


class _Index:
    """The call's files, and the order they have been placed in so far."""

    def __init__(self, collector: Collector) -> None:
        self.collector = collector
        #: Resolved local file path to digest, for every file of every detected path.
        self.by_file: dict[str, str] = {}
        #: Detected path to the digests under it, in manifest order.
        self.by_placed: dict[str, list[str]] = {}
        self.manifest: list[str] = []
        for key, placed in collector.placed.items():
            digests = []
            for _rel, digest, _size, local in placed.entries:
                self.by_file[local] = digest
                digests.append(digest)
                self.manifest.append(digest)
            self.by_placed[key] = digests
        self.placed_order: list[str] = []
        self.seen: set[str] = set()

    # -- placing ---------------------------------------------------------------

    def add(self, digest: str) -> None:
        if digest in self.seen:
            return
        self.seen.add(digest)
        self.placed_order.append(digest)

    def take(self, paths: Any) -> None:
        """Place every path of an iterable, in the order it gives them."""
        for entry in _limited(paths):
            self.path(entry)

    def path(self, value: Any) -> bool:
        """Place one path, or every file under it when it is a detected directory."""
        resolved = _resolve(value)
        if resolved is None:
            return False
        digest = self.by_file.get(resolved)
        if digest is not None:
            self.add(digest)
            return True
        under = self.by_placed.get(resolved)
        if under is None:
            return False
        for entry in under:
            self.add(entry)
        return True

    def order(self) -> list[str]:
        """What the analysis placed, then everything else in manifest order."""
        rest = [digest for digest in self.manifest if digest not in self.seen]
        seen: set[str] = set()
        ordered = []
        for digest in [*self.placed_order, *rest]:
            if digest not in seen:
                seen.add(digest)
                ordered.append(digest)
        return ordered


def _resolve(value: Any) -> str | None:
    """The resolved path string of an ``os.PathLike``, or None for anything else."""
    if isinstance(value, (str, bytes)) or not hasattr(type(value), "__fspath__"):
        return None
    try:
        return str(Path(os.fspath(value)).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _limited(values: Any) -> list[Any]:
    """At most ``WALK_LIMIT`` items of an iterable, and nothing for anything else."""
    try:
        values = list(values.values()) if isinstance(values, dict) else list(values)
    except TypeError:
        return []
    return values[:WALK_LIMIT]


# -- what the pickled arguments say -----------------------------------------------


def _from_arguments(index: _Index, fn: Any, args: tuple, kwargs: dict) -> None:
    """Place what the call's own objects carry: a file order, a dataset, a loader."""
    for value in _values(fn, args, kwargs):
        _from_object(index, value)


def _values(fn: Any, args: tuple, kwargs: dict) -> list[Any]:
    """The arguments, the defaults, the closure cells and the globals the code names."""
    values: list[Any] = [*args, *kwargs.values()]
    inner = getattr(fn, "__func__", fn)
    values.extend(getattr(inner, "__defaults__", None) or ())
    values.extend((getattr(inner, "__kwdefaults__", None) or {}).values())
    for cell in getattr(inner, "__closure__", None) or ():
        try:
            values.append(cell.cell_contents)
        except ValueError:
            continue
    code = getattr(inner, "__code__", None)
    globals_ = getattr(inner, "__globals__", None)
    if code is not None and isinstance(globals_, dict):
        for name in code.co_names:
            if name in globals_:
                values.append(globals_[name])
    return values


def _from_object(index: _Index, value: Any) -> None:
    """Place one object's file order, if it carries one."""
    reader = getattr(value, "letify_read_order", None)
    if callable(reader):
        index.take(reader())
        return
    if index.path(value):
        return
    if isinstance(value, (list, tuple, dict)):
        entries = _limited(value)
        if entries and all(_resolve(entry) is not None for entry in entries):
            index.take(entries)
        return
    sequence = _loader_order(value)
    if sequence is None:
        sequence = _dataset_order(value)
    if sequence is not None:
        index.take(sequence)


def _dataset_order(value: Any) -> list[Any] | None:
    """A dataset's own file sequence, under one of the names PyTorch's classes use."""
    for name in SEQUENCES:
        sequence = getattr(value, name, None)
        entries = _limited(sequence) if sequence is not None else []
        if entries and all(_resolve(entry) is not None for entry in entries):
            return entries
    return None


def _loader_order(value: Any) -> list[Any] | None:
    """A loader's epoch order: its sampler's indices over its dataset's file sequence."""
    dataset = getattr(value, "dataset", None)
    if dataset is None:
        return None
    sequence = _dataset_order(dataset)
    if sequence is None:
        return None
    sampler = getattr(value, "batch_sampler", None) or getattr(value, "sampler", None)
    if sampler is None:
        return sequence
    ordered: list[Any] = []
    for item in _limited(sampler):
        for position in item if isinstance(item, (list, tuple)) else (item,):
            if isinstance(position, int) and 0 <= position < len(sequence):
                ordered.append(sequence[position])
    return ordered or sequence


# -- what the function's code says --------------------------------------------------


def _from_code(index: _Index, fn: Any) -> None:
    """Place the paths the code reads, in the position of their first read."""
    inner = getattr(fn, "__func__", fn)
    code = getattr(inner, "__code__", None)
    if code is None:
        return
    seen: set[Any] = set()
    stack = [(inner, 0)]
    while stack:
        function, depth = stack.pop(0)
        target = getattr(function, "__code__", None)
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        _scan(index, function, target)
        if depth >= DEPTH:
            continue
        globals_ = getattr(function, "__globals__", None) or {}
        for name in target.co_names:
            found = globals_.get(name)
            if callable(found) and hasattr(found, "__code__"):
                stack.append((found, depth + 1))


def _scan(index: _Index, function: Any, code: Any) -> None:
    """Walk one code object, placing a path where the code reads it."""
    names = _name_values(function, code)
    candidate: Any = None
    for instruction in dis.get_instructions(code):
        name = instruction.opname
        if name in ("LOAD_CONST", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_NAME", "LOAD_FAST"):
            found = _candidate(instruction, names)
            if found is not None:
                candidate = found
            continue
        if name not in ("LOAD_METHOD", "LOAD_ATTR", "CALL", "CALL_FUNCTION_EX"):
            continue
        called = instruction.argval if isinstance(instruction.argval, str) else None
        # A read names one file and a walk names a whole directory, and ``path`` places
        # either, so both kinds of call put the candidate at this position.
        if candidate is not None and (called in WALKS or called in READS):
            index.path(candidate)
            candidate = None


def _name_values(function: Any, code: Any) -> dict[str, Any]:
    """What each name the code loads holds: a global, a closure cell or a constant."""
    values: dict[str, Any] = {}
    globals_ = getattr(function, "__globals__", None) or {}
    for name in code.co_names:
        if name in globals_:
            values[name] = globals_[name]
    cells = getattr(function, "__closure__", None) or ()
    for name, cell in zip(code.co_freevars, cells, strict=False):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            continue
    return values


def _candidate(instruction: Any, names: dict[str, Any]) -> Any:
    """The path an instruction loads, as a value or as a string constant."""
    if instruction.opname == "LOAD_CONST":
        value = instruction.argval
        if isinstance(value, str) and value:
            return Path(value)
        return value if hasattr(type(value), "__fspath__") else None
    return names.get(instruction.argval)


def manifest(collector: Collector) -> dict[str, list[Any]]:
    """Every file of the call, keyed by the runtime path it is placed at.

    Spec "Streaming the rest while the call runs": the worker keeps this for the life of the
    call, and it is what makes a directory listing complete before any byte has arrived. A
    detected file is placed at its own runtime path, a detected directory's files under it.
    """
    entries: dict[str, list[Any]] = {}
    for placed in collector.placed.values():
        for rel, digest, size, _local in placed.entries:
            path = f"{placed.runtime}/{rel}" if placed.directory else placed.runtime
            entries[path] = [digest, size]
    return entries


def sizes_of(collector: Collector) -> dict[str, int]:
    """The size of each digest the call carries."""
    found: dict[str, int] = {}
    for placed in collector.placed.values():
        for _rel, digest, size, _local in placed.entries:
            found.setdefault(digest, size)
    return found


def files_of(collector: Collector) -> dict[str, str]:
    """A local file to read each digest from."""
    found: dict[str, str] = {}
    for placed in collector.placed.values():
        for _rel, digest, _size, local in placed.entries:
            found.setdefault(digest, local)
    return found


def relative_of(collector: Collector, digest: str) -> str:
    """A relative path a digest appears at, for a message a person reads."""
    for placed in collector.placed.values():
        for rel, found, _size, _local in placed.entries:
            if found == digest:
                return rel
    return digest


def placed_of(collector: Collector) -> list[Placed]:
    return list(collector.placed.values())


__all__ = ["READS", "SEQUENCES", "WALKS", "compute", "first_wave", "manifest"]
