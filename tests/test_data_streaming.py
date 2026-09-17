"""Streaming project data: the send order, the first wave and the background sender.

Spec sections pinned here: "The send order and the first wave", "Observing the read order
on the runtime", "Streaming the rest while the call runs", "What the body sees before a
file arrives", "When a blob does not arrive" and "Ordering with write-back and the cache
budget".

Everything runs through the Local provider and the real framed worker, so the worker's
pending patch, its data thread and the background sender are the real ones. The only
stand-in is a ``torch.utils.data`` module written into the test's project, because the
suite does not install PyTorch and the wrapper's contract is the class it wraps.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

import letify
from letify.errors import RuntimeFailure
from letify.store import pathdata, sendorder

#: A module global a declared body reads, set per test.
DATASET: Path | None = None


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """A project root with a pyproject.toml as the working directory, and a private home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = tmp_path / "study"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'study'\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def dataset(root: Path, count: int, size: int = 1 << 16) -> Path:
    """A directory of ``count`` files, each distinct and ``size`` bytes long."""
    directory = root / "set"
    directory.mkdir(exist_ok=True)
    for index in range(count):
        (directory / f"{index:03d}.bin").write_bytes(bytes([index % 251]) * size)
    return directory


def data_line(err: str) -> str:
    """The one data log line of spec "Data log line", without the write-back line."""
    lines = [
        line
        for line in err.splitlines()
        if line.startswith("letify: data ") and not line.startswith("letify: data wrote back ")
    ]
    assert lines, err
    return lines[0]


def before_files(line: str) -> int:
    match = re.search(r"sent (\d+) files [\d.]+ MiB before the call", line)
    assert match, line
    return int(match.group(1))


def during_files(line: str) -> int:
    match = re.search(r"MiB before the call in [\d.]+ s, (\d+) files", line)
    assert match, line
    return int(match.group(1))


def waited_seconds(line: str) -> float:
    match = re.search(r"first access waited ([\d.]+) s", line)
    assert match, line
    return float(match.group(1))


def streaming(launcher_from, **options: object) -> letify.Launcher:
    """A persistent local account with the streaming options a test needs."""
    body = ['[lab]', 'kind = "local"']
    for name, value in options.items():
        rendered = str(value).lower() if isinstance(value, bool) else repr(value)
        body.append(f"{name} = {rendered}")
    return launcher_from("\n".join(body) + "\n")


# -- Spec: The send order and the first wave -----------------------------------


def order_of(collector: pathdata.Collector, fn, args=(), kwargs=None) -> list[str]:
    """The send order the analysis derives, as relative paths in order."""
    entries = sendorder.compute(collector, fn, args, kwargs or {})
    by_digest = {
        digest: f"{placed.local.name}/{rel}" if placed.directory else rel
        for placed in collector.placed.values()
        for rel, digest, _size, _local in placed.entries
    }
    return [by_digest[digest] for digest in entries]


def collect(paths: list[Path]) -> pathdata.Collector:
    collector = pathdata.Collector("/w/data/calls/abc")
    for path in paths:
        collector.reduce(path)
    return collector


def test_a_list_of_paths_in_the_arguments_sets_the_send_order(project) -> None:
    root = dataset(project, 3, size=16)
    files = [root / "002.bin", root / "000.bin", root / "001.bin"]
    collector = collect(files)

    def body(paths):
        return len(paths)

    assert order_of(collector, body, (files,)) == ["002.bin", "000.bin", "001.bin"]


def test_an_object_that_exposes_letify_read_order_is_asked_for_it(project) -> None:
    root = dataset(project, 3, size=16)
    files = [root / "001.bin", root / "002.bin", root / "000.bin"]

    class Corpus:
        def __init__(self, paths):
            self.paths = paths

        def letify_read_order(self):
            return list(self.paths)

    collector = collect(files)

    def body(corpus):
        return len(corpus.paths)

    assert order_of(collector, body, (Corpus(files),)) == ["001.bin", "002.bin", "000.bin"]


def test_a_read_in_the_function_body_orders_the_paths_it_names(project) -> None:
    root = dataset(project, 3, size=16)
    second, first = root / "001.bin", root / "000.bin"
    collector = collect([second, first])

    def body():
        first.read_bytes()
        second.read_bytes()
        return 0

    # The code names ``first`` before ``second``, which outranks the order they were placed.
    assert order_of(collector, body)[:2] == ["000.bin", "001.bin"]


def test_paths_the_analysis_did_not_place_follow_in_manifest_order(project) -> None:
    root = dataset(project, 3, size=16)
    collector = collect([root])

    def body(directory):
        return directory

    assert order_of(collector, body, (root,)) == ["set/000.bin", "set/001.bin", "set/002.bin"]


def test_a_declared_data_order_outranks_the_analysis(launcher_from, project, capsys) -> None:
    let = streaming(launcher_from, data_first_wave_files=1, data_first_wave_mib=1024)
    root = dataset(project, 4, size=1 << 16)
    wanted = [root / "003.bin", root / "002.bin", root / "001.bin", root / "000.bin"]

    @let.function(device=let.providers.lab.CPU, host=letify.remote, data_order=wanted)
    def read(paths: list[Path]) -> bytes:
        return paths[0].read_bytes()[:1]

    assert read(wanted) == bytes([3])
    line = data_line(capsys.readouterr().err)
    # The first wave is one file, and the declaration put 003.bin first, so nothing waited.
    assert before_files(line) == 1
    assert waited_seconds(line) == 0.0


def test_the_first_wave_stops_at_the_file_limit(launcher_from, project, capsys) -> None:
    let = streaming(launcher_from, data_first_wave_files=2)
    root = dataset(project, 6, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def total(directory: Path) -> int:
        return sum(p.stat().st_size for p in sorted(directory.iterdir()))

    assert total(root) == 6 * (1 << 16)
    line = data_line(capsys.readouterr().err)
    assert before_files(line) == 2
    assert during_files(line) == 4


def test_the_first_wave_stops_at_the_byte_limit(launcher_from, project, capsys) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0.125)
    root = dataset(project, 8, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read(directory: Path) -> int:
        return len(sorted(directory.iterdir()))

    assert read(root) == 8
    line = data_line(capsys.readouterr().err)
    # 0.125 MiB of budget over 64 KiB files places two of them before the call.
    assert before_files(line) == 2


def test_a_first_wave_of_zero_sends_the_call_with_nothing_placed(
    launcher_from, project, capsys
) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 3, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read(directory: Path) -> bytes:
        return (directory / "000.bin").read_bytes()[:1]

    assert read(root) == bytes([0])
    assert before_files(data_line(capsys.readouterr().err)) == 0


# -- Spec: Streaming the rest while the call runs --------------------------------


def test_the_rest_of_the_dataset_arrives_while_the_call_runs(
    launcher_from, project, capsys
) -> None:
    let = streaming(launcher_from, data_first_wave_files=1)
    root = dataset(project, 6, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_all(directory: Path) -> list[int]:
        return [p.read_bytes()[0] for p in sorted(directory.iterdir())]

    assert read_all(root) == [index % 251 for index in range(6)]
    line = data_line(capsys.readouterr().err)
    assert before_files(line) == 1
    assert during_files(line) == 5


def test_a_blob_is_never_sent_twice(launcher_from, project, capsys) -> None:
    let = streaming(launcher_from, data_first_wave_files=1)
    root = project / "set"
    root.mkdir()
    for name in ("a.bin", "b.bin", "c.bin"):
        (root / name).write_bytes(b"same contents")

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_all(directory: Path) -> int:
        return sum(len(p.read_bytes()) for p in sorted(directory.iterdir()))

    assert read_all(root) == 39
    line = data_line(capsys.readouterr().err)
    # Three files, one digest: the whole dataset is one blob and it travels once.
    assert before_files(line) + during_files(line) == 3


def test_the_second_call_sends_nothing_and_waits_for_nothing(
    launcher_from, project, capsys
) -> None:
    let = streaming(launcher_from, data_first_wave_files=1)
    root = dataset(project, 4, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_all(directory: Path) -> int:
        return len([p.read_bytes() for p in sorted(directory.iterdir())])

    assert read_all(root) == 4
    capsys.readouterr()
    assert read_all(root) == 4
    line = data_line(capsys.readouterr().err)
    assert "4 files" in line and "already on the runtime" in line
    assert before_files(line) == 0
    assert during_files(line) == 0
    assert waited_seconds(line) == 0.0


# -- Spec: What the body sees before a file arrives -------------------------------


def test_a_directory_listing_is_complete_before_the_files_arrive(
    launcher_from, project
) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 5, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def listing(directory: Path) -> list[str]:
        # Listed before anything is read, so no blob has been asked for yet.
        return sorted(p.name for p in directory.iterdir())

    assert listing(root) == [f"{index:03d}.bin" for index in range(5)]


def test_the_size_of_a_file_that_has_not_arrived_is_its_final_size(
    launcher_from, project
) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 4, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def sizes(directory: Path) -> tuple[list[int], list[bool]]:
        files = sorted(directory.iterdir())
        return [p.stat().st_size for p in files], [p.is_file() for p in files]

    found, are_files = sizes(root)
    assert found == [1 << 16] * 4
    assert are_files == [True] * 4


def test_a_file_is_never_seen_half_written(launcher_from, project) -> None:
    """A file inside a directory is copied rather than linked, because the body may rewrite
    it, and a copy straight to its runtime path can be stat'd with only some of its bytes."""
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 24, size=1 << 18)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def watch(directory: Path) -> tuple[set[int], list[str]]:
        seen = set()
        odd = []
        for _sweep in range(40):
            for entry in sorted(directory.iterdir()):
                if ".letify-placing." in entry.name:
                    odd.append(entry.name)
                seen.add(entry.stat().st_size)
        # Read them all, so the sweeps above overlap files still being placed.
        for entry in sorted(directory.iterdir()):
            entry.read_bytes()
        return seen, odd

    sizes, partials = watch(root)
    assert sizes == {1 << 18}
    assert partials == []


def test_opening_a_file_that_has_not_arrived_waits_for_it(launcher_from, project) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 4, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_last(directory: Path) -> bytes:
        return sorted(directory.iterdir())[-1].read_bytes()[:1]

    assert read_last(root) == bytes([3])


def test_a_call_that_blocked_on_a_read_says_so(
    launcher_from, project, capsys, monkeypatch
) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 4, size=1 << 20)
    sending = pathdata.Stream._pump

    def slowly(self) -> bool:
        # Held back so the body reaches the file first, which is what the backstop is for.
        time.sleep(0.5)
        return sending(self)

    monkeypatch.setattr(pathdata.Stream, "_pump", slowly)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_first(directory: Path) -> bytes:
        return sorted(directory.iterdir())[0].read_bytes()[:1]

    assert read_first(root) == bytes([0])
    err = capsys.readouterr().err
    assert "letify: data waited for " in err
    assert "000.bin" in err


def test_a_process_the_body_spawns_sees_the_pending_files(launcher_from, project) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 3, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def in_a_child(directory: Path) -> tuple[list[str], int]:
        import multiprocessing
        import os

        # Both callables are picklable by reference, so the child needs no test module.
        context = multiprocessing.get_context("spawn")
        with context.Pool(1) as pool:
            names = sorted(pool.apply(os.listdir, (str(directory),)))
            first = pool.apply(Path.read_bytes, (directory / names[0],))
        return names, first[0]

    assert in_a_child(root) == (["000.bin", "001.bin", "002.bin"], 0)


# -- Spec: Observing the read order on the runtime --------------------------------


TORCH_STUB = '''
"""A stand-in for torch.utils.data, carrying the names the wrapper reads."""


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset, batch_size=1, sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler if sampler is not None else range(len(dataset))

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
'''


def write_torch(root: Path) -> None:
    """Write a torch package into the project, so the runtime's body can import it."""
    package = root / "torch"
    (package / "utils").mkdir(parents=True)
    (package / "__init__.py").write_text("from . import utils\n", encoding="utf-8")
    (package / "utils" / "__init__.py").write_text("from . import data\n", encoding="utf-8")
    (package / "utils" / "data.py").write_text(TORCH_STUB, encoding="utf-8")


def test_the_observed_read_order_moves_a_digest_to_the_front(
    launcher_from, project, capsys, monkeypatch
) -> None:
    write_torch(project)
    monkeypatch.syspath_prepend(str(project))
    let = streaming(launcher_from, data_first_wave_files=1, data_prefetch_batches=8)
    root = dataset(project, 8, size=1 << 18)
    # The loader reads the dataset backwards, which the first wave analysis cannot know.
    reversed_files = sorted(root.iterdir(), reverse=True)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def train(paths: list[Path]) -> list[int]:
        # Defined in the body, so the runtime needs nothing of the test module to load it.
        from torch.utils.data import DataLoader, Dataset

        class Files(Dataset):
            def __init__(self, items):
                self.samples = list(items)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                return self.samples[index].read_bytes()[0]

        loader = DataLoader(Files(paths), batch_size=2)
        return [value for batch in loader for value in batch]

    assert train(reversed_files) == [index % 251 for index in range(7, -1, -1)]
    line = data_line(capsys.readouterr().err)
    assert before_files(line) + during_files(line) == 8


def test_data_observe_false_turns_the_wrappers_off(
    launcher_from, project, monkeypatch
) -> None:
    write_torch(project)
    monkeypatch.syspath_prepend(str(project))
    let = streaming(launcher_from, data_first_wave_files=1, data_observe=False)
    root = dataset(project, 4, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def train(directory: Path) -> int:
        from torch.utils.data import DataLoader, Dataset

        class Files(Dataset):
            def __init__(self, items):
                self.samples = list(items)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                return self.samples[index].read_bytes()[0]

        loader = DataLoader(Files(sorted(directory.iterdir())), batch_size=2)
        return sum(len(batch) for batch in loader)

    # The blocking backstop still serves every file, so the call is correct either way.
    assert train(root) == 4


def test_the_worker_does_not_import_torch_to_observe(launcher_from, project) -> None:
    """Spec "Observing the read order on the runtime": the wrapper is lazy.

    The worker installs the DataLoader wrapper by hooking the body's own import of
    ``torch.utils.data``, never by importing PyTorch itself. A call whose body never touches
    PyTorch leaves torch unimported in the worker, so the wrapper's import never sits on the
    call's path and never delays the body or stalls the background sender.
    """
    write_torch(project)
    let = streaming(launcher_from, data_first_wave_mib=0)
    root = dataset(project, 3, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def touched(directory: Path) -> bool:
        import sys

        # The body reads a file, so the call has pending data and the observe setup runs.
        # It still imports nothing of PyTorch, because the body does not.
        (directory / "000.bin").read_bytes()
        return "torch" in sys.modules

    assert touched(root) is False


# -- Spec: When a blob does not arrive ---------------------------------------------


def test_a_blob_that_never_arrives_fails_the_open(
    launcher_from, project, monkeypatch
) -> None:
    let = streaming(launcher_from, data_first_wave_mib=0, data_wait_timeout=2)
    root = dataset(project, 2, size=1 << 16)

    def send_nothing(self) -> None:
        return None

    monkeypatch.setattr(pathdata.Stream, "_pump", send_nothing)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_one(directory: Path) -> bytes:
        return sorted(directory.iterdir())[0].read_bytes()

    with pytest.raises(RuntimeFailure) as failure:
        read_one(root)
    assert "000.bin" in str(failure.value)


def test_a_call_that_returns_early_cancels_the_transfer(
    launcher_from, project, capsys
) -> None:
    let = streaming(launcher_from, data_first_wave_files=1)
    root = dataset(project, 24, size=1 << 20)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def peek(directory: Path) -> str:
        return sorted(directory.iterdir())[0].name

    started = time.monotonic()
    assert peek(root) == "000.bin"
    elapsed = time.monotonic() - started
    line = data_line(capsys.readouterr().err)
    # The body read one file, so the other 23 MiB were cancelled rather than sent.
    assert during_files(line) < 23
    assert elapsed < 20


# -- Spec: Ordering with write-back and the cache budget ----------------------------


def test_a_blob_still_arriving_is_not_evicted(launcher_from, project) -> None:
    let = streaming(
        launcher_from, data_first_wave_files=1, data_cache_gib=1 / (1 << 10)
    )
    root = dataset(project, 8, size=1 << 18)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_all(directory: Path) -> list[int]:
        return [p.read_bytes()[0] for p in sorted(directory.iterdir())]

    # Every file is readable even though the cache budget is smaller than the dataset.
    assert read_all(root) == [index % 251 for index in range(8)]


def test_a_write_back_does_not_wait_for_a_file_that_never_arrived(
    launcher_from, project, capsys
) -> None:
    let = streaming(launcher_from, data_first_wave_files=1)
    root = dataset(project, 12, size=1 << 20)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def write_one(directory: Path) -> str:
        (directory / "result.txt").write_text("done", encoding="utf-8")
        return "done"

    assert write_one(root) == "done"
    assert (root / "result.txt").read_text(encoding="utf-8") == "done"
    err = capsys.readouterr().err
    assert "wrote back 1 files" in err


def test_the_pending_manifest_names_every_file_of_the_call_exactly_once(
    launcher_from, project
) -> None:
    """Spec "What the body sees before a file arrives": the manifest is written from a copy.

    The data thread moves a path out of the pending map and into the placed map as each
    blob completes. A writer that read the live maps could miss a path that was between
    the two, or fail outright on a map that changed size while it was being read, so the
    manifest is written from a copy taken under the lock the data thread takes.
    """
    let = streaming(launcher_from, data_first_wave_files=1)
    root = dataset(project, 16, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_manifest(directory: Path) -> dict:
        import json

        # The manifest sits in the call directory, which is an ancestor of the dataset.
        found = None
        for parent in [directory, *directory.parents]:
            candidate = parent / ".letify-pending.json"
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            return {"manifest": None, "files": sorted(p.name for p in directory.iterdir())}
        written = json.loads(found.read_text())
        return {
            "manifest": sorted(Path(p).name for p in written["files"]),
            "files": sorted(p.name for p in directory.iterdir()),
        }

    answer = read_manifest(root)
    assert answer["manifest"] is not None, "the call had pending files, so a manifest was written"
    # Both listings are taken inside the body, against the same runtime directory, because
    # that is where the two maps live. Comparing against the client's directory would race
    # the write-back instead of the placement this test is about.
    assert answer["manifest"] == answer["files"]
    assert len(answer["manifest"]) == len(set(answer["manifest"])), "a path was named twice"


def test_a_listing_during_streaming_never_sees_a_changing_map(
    launcher_from, project
) -> None:
    """Spec "What the body sees before a file arrives": every listing is complete.

    The data thread places a file and deletes its path from the pending map as each blob
    completes, while the body lists the same directory. Two things can go wrong. Iterating
    the live map raises ``RuntimeError: dictionary changed size during iteration``, so the
    body snapshots the map under the lock the data thread writes it with. And reading the
    real directory before the snapshot misses a file placed between the two steps: it is
    on disk after the listing was read, and out of the map before the snapshot was taken.
    So the body takes the snapshot first. Both are pinned here: the call returning all
    4000 counts proves no listing raised, and every count being the full directory proves
    no listing straddled a placement.
    """
    let = streaming(launcher_from, data_first_wave_files=1)
    count = 400
    root = dataset(project, count, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def list_while_streaming(directory: Path) -> list[int]:
        import sys

        # List repeatedly so the loop overlaps the data thread deleting placed paths from
        # the pending map. Each listing is the union of placed files and pending names. The
        # loop runs long enough to span the streaming window rather than deciding completion,
        # which the body cannot observe.
        #
        # A tiny switch interval makes the interpreter change threads mid listing, so a
        # delete on the data thread lands while the listing iterates the map. Without the
        # snapshot that raises RuntimeError: dictionary changed size during iteration.
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            seen = []
            for _ in range(4000):
                seen.append(len(list(directory.iterdir())))
            return seen
        finally:
            sys.setswitchinterval(previous)

    seen = list_while_streaming(root)
    # The call returning all 4000 counts proves no listing raised: an unsnapshotted iteration
    # would have propagated RuntimeError out of iterdir and failed the call.
    assert len(seen) == 4000, "every listing returned without raising"
    assert set(seen) == {count}, "every listing is the whole directory"


def test_a_call_whose_files_are_all_held_carries_no_streaming_machinery(
    launcher_from, project
) -> None:
    """Spec "Streaming the rest while the call runs": a warm repeat streams nothing.

    The manifest, the send order, the observation settings and the wait statistics request
    all exist to serve bytes that have not arrived. A call whose files the runtime already
    holds has none, so none of them may be sent. This is the repeat run of a dataset being
    iterated on, and it is what the streaming path must not make slower.
    """
    from letify.runtime import channel as channel_module

    let = streaming(launcher_from, data_first_wave_files=1)
    root = dataset(project, 4, size=1 << 16)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_all(directory: Path) -> int:
        return len([p.read_bytes() for p in sorted(directory.iterdir())])

    assert read_all(root) == 4

    seen: list[dict] = []
    original = channel_module.PersistentChannel.request

    def recording(self, request, **kwargs):
        if isinstance(request, dict):
            seen.append({"op": request.get("op", "call"), "data": request.get("data")})
        return original(self, request, **kwargs)

    channel_module.PersistentChannel.request = recording
    try:
        assert read_all(root) == 4
    finally:
        channel_module.PersistentChannel.request = original

    calls = [entry for entry in seen if entry["op"] == "call"]
    assert len(calls) == 1, seen
    data = calls[0]["data"] or {}
    assert "manifest" not in data, data.keys()
    assert "order" not in data, data.keys()
    assert "observe" not in data, data.keys()
    assert [entry for entry in seen if entry["op"] == "data_stats"] == []
