"""Project data: local paths a call reaches travel as content addressed file blobs.

Spec sections pinned here: "Project data" and its subsections "Which paths are data",
"Digests and the digest cache", "Where the bytes come from", "Materializing and the
rewritten path", "Data log line" and "Writing back".

Everything runs through the Local provider and the real framed worker. Only the bucket is a
fake, the Cloud Storage endpoint conftest serves on loopback, because a real one needs an
account.
"""

from __future__ import annotations

import re
import shutil
import threading
from pathlib import Path

import pytest

import letify
from letify.store import pathdata

#: A module global a declared body reads, set per test.
DATASET: Path | None = None


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """A project root with a pyproject.toml as the working directory, and a private home.

    The Local worker inherits HOME, so its workspace root and the digest cache land here.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = tmp_path / "study"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'study'\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def data_lines(err: str) -> list[str]:
    """The upload lines of spec "Data log line", without the write-back lines."""
    return [
        line
        for line in err.splitlines()
        if line.startswith("letify: data ") and not line.startswith("letify: data wrote back ")
    ]


def uploaded_files(line: str) -> int:
    match = re.search(r"uploaded (\d+) files", line)
    assert match, line
    return int(match.group(1))


# -- Spec: Which paths are data ------------------------------------------------


def test_a_path_argument_is_read_back_on_the_runtime(let, cpu, project, capsys) -> None:
    (project / "data").mkdir()
    sample = project / "data" / "sample.txt"
    sample.write_text("hello data", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def read(path: Path) -> tuple[str, str]:
        return str(path), path.read_text(encoding="utf-8")

    where, text = read(sample)
    assert text == "hello data"
    assert where != str(sample)
    assert "/data/calls/" in where
    lines = data_lines(capsys.readouterr().err)
    assert len(lines) == 1
    assert "1 files" in lines[0]


def test_a_path_global_is_read_back_on_the_runtime(let, cpu, project, monkeypatch) -> None:
    (project / "global.bin").write_bytes(b"\x00\x01global")
    monkeypatch.setitem(globals(), "DATASET", project / "global.bin")

    @let.function(device=cpu, host=letify.remote)
    def read() -> bytes:
        assert DATASET is not None
        return DATASET.read_bytes()

    assert read() == b"\x00\x01global"


def test_a_path_that_does_not_exist_outside_the_roots_is_left_as_a_plain_path(
    let, cpu, project, tmp_path, capsys
) -> None:
    absent = tmp_path / "outputs" / "run-1"

    @let.function(device=cpu, host=letify.remote)
    def where(path: Path) -> str:
        return str(path)

    assert where(absent) == str(absent)
    assert data_lines(capsys.readouterr().err) == []
    assert not absent.exists()


def test_a_path_outside_the_allowed_roots_is_left_as_a_plain_path(
    let, cpu, project, tmp_path, capsys
) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not project data", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def where(path: Path) -> str:
        return str(path)

    assert where(outside) == str(outside)
    assert data_lines(capsys.readouterr().err) == []


def test_a_data_root_named_in_pyproject_is_allowed(let, cpu, project, tmp_path) -> None:
    shared = tmp_path / "datasets"
    shared.mkdir()
    (shared / "train.txt").write_text("shared", encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[project]\nname = 'study'\n\n[tool.letify]\ndata_roots = ['../datasets']\n",
        encoding="utf-8",
    )

    @let.function(device=cpu, host=letify.remote)
    def read(path: Path) -> tuple[str, str]:
        return str(path), path.read_text(encoding="utf-8")

    where, text = read(shared / "train.txt")
    assert text == "shared"
    assert where != str(shared / "train.txt")


def test_the_project_root_itself_is_not_data(let, cpu, project, capsys) -> None:
    # Path(__file__).parent names the code and its .venv, not a dataset.
    @let.function(device=cpu, host=letify.remote)
    def where(path: Path) -> str:
        return str(path)

    assert where(project) == str(project)
    assert data_lines(capsys.readouterr().err) == []


# -- Spec: Materializing and the rewritten path ---------------------------------


def test_a_directory_keeps_its_layout_on_the_runtime(let, cpu, project) -> None:
    root = project / "corpus"
    (root / "train" / "deep").mkdir(parents=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "train" / "b.txt").write_text("b", encoding="utf-8")
    (root / "train" / "deep" / "c.txt").write_text("c", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "skip.pyc").write_bytes(b"skip")

    @let.function(device=cpu, host=letify.remote)
    def listing(path: Path) -> tuple[str, dict[str, str]]:
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return path.name, {p.relative_to(path).as_posix(): p.read_text() for p in files}

    name, files = listing(root)
    assert name == "corpus"
    assert files == {"a.txt": "a", "train/b.txt": "b", "train/deep/c.txt": "c"}


def test_the_same_path_twice_in_one_call_is_one_runtime_path(let, cpu, project) -> None:
    (project / "x.txt").write_text("x", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def both(first: Path, second: Path) -> bool:
        return first == second

    assert both(project / "x.txt", Path("x.txt"))


def test_the_call_directory_is_removed_when_the_call_ends(let, cpu, project) -> None:
    (project / "x.txt").write_text("x", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def where(path: Path) -> str:
        return str(path)

    runtime_path = Path(where(project / "x.txt"))
    assert not runtime_path.exists()
    assert not runtime_path.parent.parent.exists()


# -- Spec: Where the bytes come from, a persistent provider ---------------------


def test_a_second_session_on_a_persistent_provider_uploads_nothing(
    let, cpu, project, capsys
) -> None:
    root = project / "set"
    root.mkdir()
    for index in range(3):
        (root / f"{index}.bin").write_bytes(bytes([index]) * 1000)

    @let.function(device=cpu, host=letify.remote)
    def size(path: Path) -> int:
        return sum(p.stat().st_size for p in path.iterdir())

    assert size(root) == 3000
    first = data_lines(capsys.readouterr().err)
    assert size(root) == 3000
    second = data_lines(capsys.readouterr().err)
    assert let.pool.live == []
    assert uploaded_files(first[0]) == 3
    assert uploaded_files(second[0]) == 0
    assert "3 files 0.0 MiB already on the runtime" in second[0]


def test_a_changed_file_uploads_only_that_file(let, cpu, project, capsys) -> None:
    root = project / "set"
    root.mkdir()
    for index in range(3):
        (root / f"{index}.bin").write_bytes(bytes([index]) * 1000)

    @let.function(device=cpu, host=letify.remote)
    def read(path: Path) -> bytes:
        return (path / "1.bin").read_bytes()

    read(root)
    capsys.readouterr()
    (root / "1.bin").write_bytes(b"changed")
    assert read(root) == b"changed"
    assert uploaded_files(data_lines(capsys.readouterr().err)[0]) == 1


# -- Spec: Digests and the digest cache -----------------------------------------


def test_an_unchanged_file_is_not_hashed_again(project, monkeypatch) -> None:
    sample = project / "weights.bin"
    sample.write_bytes(b"w" * 4096)
    cache = pathdata.DigestCache()
    first = cache.digest(sample)
    cache.save()

    def refuse(path: Path, size: int) -> str:
        raise AssertionError(f"{path} was hashed again")

    monkeypatch.setattr(pathdata, "hash_file", refuse)
    assert pathdata.DigestCache().digest(sample) == first


def test_a_changed_file_is_hashed_again(project) -> None:
    sample = project / "weights.bin"
    sample.write_bytes(b"w" * 4096)
    first = pathdata.DigestCache().digest(sample)
    sample.write_bytes(b"v" * 4097)
    assert pathdata.DigestCache().digest(sample) != first


def test_a_file_digest_is_the_argument_digest_of_its_contents(project) -> None:
    sample = project / "weights.bin"
    sample.write_bytes(b"z" * (70 << 20))
    assert pathdata.hash_file(sample, sample.stat().st_size) == letify.protocol.digest_of(
        sample.read_bytes()
    )


# -- Spec: Where the bytes come from, the bucket --------------------------------


def test_an_ephemeral_account_with_a_bucket_pulls_from_it_and_uploads_once(
    launcher_from, project, fake_gcs, capsys
) -> None:
    let = launcher_from(
        "[lab]\n"
        'kind = "local"\n'
        "persistent = false\n"
        f'bucket = "{fake_gcs.bucket}"\n'
        f'bucket_endpoint = "{fake_gcs.endpoint}"\n'
        f'sts_endpoint = "{fake_gcs.endpoint}/v1/token"\n'
    )
    root = project / "set"
    root.mkdir()
    for index in range(3):
        (root / f"{index}.bin").write_bytes(bytes([index]) * 1000)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def size(path: Path) -> int:
        return sum(p.stat().st_size for p in path.iterdir())

    def uploads() -> int:
        return len([r for r in fake_gcs.requests if r["path"].startswith("/upload/")])

    assert size(root) == 3000
    assert uploads() == 3
    # A new ephemeral machine holds nothing, which on Local means removing its cache.
    shutil.rmtree(Path.home() / ".letify-runtime" / "data")
    before = len(fake_gcs.downloads())

    assert size(root) == 3000
    assert uploads() == 3
    pulled = fake_gcs.downloads()[before:]
    assert len(pulled) == 3
    assert {r["authorization"] for r in pulled} == {"Bearer down-token-1"}
    lines = data_lines(capsys.readouterr().err)
    assert "the bucket" in lines[1]
    assert uploaded_files(lines[1]) == 0


# -- Spec: Runtime data cache budget -------------------------------------------------


def blob_files() -> dict[str, int]:
    root = Path.home() / ".letify-runtime" / "data" / "blobs"
    return {p.name: p.stat().st_nlink for p in root.glob("*/*") if ".partial." not in p.name}


def age_blobs(seconds: float) -> None:
    import os
    import time

    past = time.time() - seconds
    for path in (Path.home() / ".letify-runtime" / "data" / "blobs").glob("*/*"):
        os.utime(path, (past, past))


def evictions(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("letify: data cache evicted ")]


def budget_launcher(launcher_from, gib: float):
    return launcher_from(f'[lab]\nkind = "local"\ndata_cache_gib = {gib}\n')


def test_old_blobs_are_evicted_when_a_call_takes_the_cache_over_its_budget(
    launcher_from, project, capsys
) -> None:
    # 20 KiB of budget: two old 16 KiB files and one new one cannot all stay.
    let = budget_launcher(launcher_from, 20 / (1 << 20))

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def size(path: Path) -> int:
        return path.stat().st_size

    for name in ("a", "b"):
        (project / f"{name}.bin").write_bytes(name.encode() * (16 << 10))
        size(project / f"{name}.bin")
    assert len(blob_files()) == 2
    age_blobs(3600)
    capsys.readouterr()
    (project / "c.bin").write_bytes(b"c" * (16 << 10))
    assert size(project / "c.bin") == 16 << 10
    held = blob_files()
    assert held == {pathdata.hash_file(project / "c.bin", 16 << 10): 1}
    lines = evictions(capsys.readouterr().err)
    assert len(lines) == 1
    assert "evicted 2 files 0.0 MiB" in lines[0]
    assert "0.0 GiB" in lines[0]


def test_a_call_that_added_nothing_does_not_check_the_budget(
    launcher_from, project, capsys
) -> None:
    let = budget_launcher(launcher_from, 20 / (1 << 20))

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def size(path: Path) -> int:
        return path.stat().st_size

    for name in ("a", "b"):
        (project / f"{name}.bin").write_bytes(name.encode() * (16 << 10))
        size(project / f"{name}.bin")
    age_blobs(3600)
    capsys.readouterr()
    size(project / "a.bin")
    assert len(blob_files()) == 2
    assert evictions(capsys.readouterr().err) == []


def test_a_linked_or_recent_blob_is_not_evicted(launcher_from, project, tmp_path) -> None:
    let = budget_launcher(launcher_from, 1 / (1 << 20))

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def size(path: Path) -> int:
        return path.stat().st_size

    (project / "linked.bin").write_bytes(b"l" * 4096)
    size(project / "linked.bin")
    age_blobs(3600)
    linked = pathdata.hash_file(project / "linked.bin", 4096)
    # A running call's directory holds a hard link; this one stands in for it.
    cache_file = Path.home() / ".letify-runtime" / "data" / "blobs" / linked[:2] / linked
    (tmp_path / "running-call-link").hardlink_to(cache_file)
    (project / "recent.bin").write_bytes(b"r" * 4096)
    size(project / "recent.bin")
    assert set(blob_files()) == {linked, pathdata.hash_file(project / "recent.bin", 4096)}


def test_the_default_budget_evicts_nothing_from_a_small_cache(let, cpu, project, capsys) -> None:
    @let.function(device=cpu, host=letify.remote)
    def size(path: Path) -> int:
        return path.stat().st_size

    (project / "a.bin").write_bytes(b"a" * 4096)
    size(project / "a.bin")
    age_blobs(3600)
    (project / "b.bin").write_bytes(b"b" * 4096)
    size(project / "b.bin")
    assert len(blob_files()) == 2
    assert evictions(capsys.readouterr().err) == []


# -- Spec: The cache command ---------------------------------------------------------


def test_the_cache_command_shows_the_local_runtime_cache(let, cpu, project, capsys) -> None:
    import json

    from letify.cli import main

    @let.function(device=cpu, host=letify.remote)
    def size(path: Path) -> int:
        return path.stat().st_size

    (project / "a.bin").write_bytes(b"a" * 5000)
    size(project / "a.bin")
    capsys.readouterr()
    assert main(["cache", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    local = next(row for row in record["providers"] if row["alias"] == "local")
    assert local.get("files") == 1, local
    assert local["bytes"] == 5000
    assert local["budget"] > 0


def test_cache_clear_empties_a_providers_runtime_cache(let, cpu, project, capsys) -> None:
    from letify.cli import main

    @let.function(device=cpu, host=letify.remote)
    def size(path: Path) -> int:
        return path.stat().st_size

    for name in ("a", "b"):
        (project / f"{name}.bin").write_bytes(name.encode() * 5000)
        size(project / f"{name}.bin")
    capsys.readouterr()
    assert main(["cache", "clear", "local"]) == 0, capsys.readouterr()
    assert "local: removed 2 files" in capsys.readouterr().out
    assert blob_files() == {}


def test_digest_cache_entries_for_missing_files_are_pruned(project, capsys) -> None:
    import json

    from letify.cli import main

    kept = project / "kept.bin"
    gone = project / "gone.bin"
    kept.write_bytes(b"k")
    gone.write_bytes(b"g")
    cache = pathdata.DigestCache()
    cache.digest(kept)
    cache.digest(gone)
    cache.save()
    gone.unlink()
    assert main(["cache", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["digests"] == {"entries": 1, "pruned": 1}
    assert list(pathdata.DigestCache()._entries) == [str(kept)]


def test_saving_the_digest_cache_drops_missing_files(project) -> None:
    kept = project / "kept.bin"
    gone = project / "gone.bin"
    kept.write_bytes(b"k")
    gone.write_bytes(b"g")
    cache = pathdata.DigestCache()
    cache.digest(kept)
    cache.digest(gone)
    gone.unlink()
    cache.save()
    assert list(pathdata.DigestCache()._entries) == [str(kept)]


# -- Spec: Writing back ----------------------------------------------------------


def wrote_back(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("letify: data wrote back ")]


def test_a_directory_the_body_creates_under_an_absent_path_comes_back(
    let, cpu, project, capsys
) -> None:
    run = Path("runs/exp1")

    @let.function(device=cpu, host=letify.remote)
    def train(out: Path) -> str:
        (out / "logs").mkdir(parents=True)
        (out / "model.bin").write_bytes(b"m" * 5000)
        (out / "logs" / "train.log").write_text("step 1", encoding="utf-8")
        return str(out)

    where = train(run)
    assert where != str(run)
    assert (project / "runs/exp1/model.bin").read_bytes() == b"m" * 5000
    assert (project / "runs/exp1/logs/train.log").read_text(encoding="utf-8") == "step 1"
    lines = wrote_back(capsys.readouterr().err)
    assert len(lines) == 1
    assert "wrote back 2 files" in lines[0]
    assert "0 files 0.0 MiB already on the client" in lines[0]


def test_a_file_the_body_creates_at_an_absent_path_comes_back(let, cpu, project) -> None:
    target = project / "result.json"

    @let.function(device=cpu, host=letify.remote)
    def save(out: Path) -> None:
        out.write_text('{"acc": 0.9}', encoding="utf-8")

    save(target)
    assert target.read_text(encoding="utf-8") == '{"acc": 0.9}'


def test_only_changed_files_come_back_from_an_existing_directory(let, cpu, project, capsys) -> None:
    run = project / "runs" / "exp1"

    @let.function(device=cpu, host=letify.remote)
    def train(out: Path, step: int, checkpoint: bool) -> None:
        out.mkdir(parents=True, exist_ok=True)
        if checkpoint:
            (out / "model.bin").write_bytes(b"w" * 100_000)
        (out / "train.log").write_text(f"step {step}", encoding="utf-8")

    train(run, 1, True)
    capsys.readouterr()
    train(run, 2, False)
    assert (run / "train.log").read_text(encoding="utf-8") == "step 2"
    assert (run / "model.bin").read_bytes() == b"w" * 100_000
    err = capsys.readouterr().err
    assert "uploaded 0 files" in data_lines(err)[0]
    assert "wrote back 1 files" in wrote_back(err)[0]


def test_a_rewritten_file_with_the_same_contents_is_already_on_the_client(
    let, cpu, project, capsys
) -> None:
    run = project / "runs"
    run.mkdir()
    (run / "same.txt").write_text("same", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def rewrite(out: Path) -> None:
        (out / "same.txt").unlink()
        (out / "same.txt").write_text("same", encoding="utf-8")

    rewrite(run)
    line = wrote_back(capsys.readouterr().err)[0]
    assert "wrote back 0 files" in line


def test_a_file_already_identical_on_the_client_is_not_sent(let, cpu, project, capsys) -> None:
    run = project / "runs"

    @let.function(device=cpu, host=letify.remote)
    def save(out: Path, local: str) -> None:
        out.mkdir()
        (out / "same.txt").write_text("same", encoding="utf-8")
        # The Local provider shares this disk, so the client copy can appear meanwhile.
        Path(local).mkdir()
        (Path(local) / "same.txt").write_text("same", encoding="utf-8")

    save(run, str(run))
    line = wrote_back(capsys.readouterr().err)[0]
    assert "wrote back 0 files" in line
    assert "1 files 0.0 MiB already on the client" in line


def test_a_failed_call_writes_nothing_back(let, cpu, project, capsys) -> None:
    run = project / "runs" / "broken"

    @let.function(device=cpu, host=letify.remote)
    def train(out: Path) -> None:
        out.mkdir(parents=True)
        (out / "partial.bin").write_bytes(b"half")
        raise ValueError("diverged")

    with pytest.raises(Exception, match="diverged"):
        train(run)
    assert not run.exists()
    assert wrote_back(capsys.readouterr().err) == []


def test_a_file_deleted_on_the_runtime_stays_on_the_client(let, cpu, project) -> None:
    run = project / "runs"
    run.mkdir()
    (run / "old.log").write_text("old", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def prune(out: Path) -> None:
        (out / "old.log").unlink()
        (out / "new.log").write_text("new", encoding="utf-8")

    prune(run)
    assert (run / "old.log").read_text(encoding="utf-8") == "old"
    assert (run / "new.log").read_text(encoding="utf-8") == "new"


def test_a_file_written_in_place_through_its_link_is_dropped_from_the_runtime_cache(
    let, cpu, project
) -> None:
    state = project / "state.txt"
    state.write_text("before", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def overwrite(target: Path) -> str:
        import os
        import stat
        import time

        # A root process ignores the read-only mode; this stands in for it.
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
        time.sleep(0.01)
        with open(target, "r+", encoding="utf-8") as handle:
            handle.write("AFTER!")
        return target.read_text(encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def read(target: Path) -> str:
        return target.read_text(encoding="utf-8")

    assert overwrite(state) == "AFTER!"
    # An existing file is an input only, so the local copy is untouched.
    assert state.read_text(encoding="utf-8") == "before"
    assert read(state) == "before"


def test_a_large_write_back_shows_the_download_progress_line(let, cpu, project, capsys) -> None:
    run = project / "runs"

    @let.function(device=cpu, host=letify.remote)
    def save(out: Path) -> None:
        out.mkdir()
        (out / "big.bin").write_bytes(b"\x07" * (65 << 20))

    save(run)
    assert (run / "big.bin").stat().st_size == 65 << 20
    err = capsys.readouterr().err
    assert "letify: downloading 1 files" in err
    assert "wrote back 1 files 65.0 MiB" in wrote_back(err)[0]


def test_concurrent_calls_writing_one_path_keep_every_file_and_whole_files(
    let, cpu, project
) -> None:
    run = project / "runs" / "shared"

    @let.function(device=cpu, host=letify.remote)
    def write(out: Path, name: str) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.log").write_text(name, encoding="utf-8")
        (out / "last.bin").write_bytes(name.encode() * 200_000)

    errors: list[BaseException] = []

    def run_one(name: str) -> None:
        try:
            write(run, name)
        except BaseException as exc:  # pragma: no cover - reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=run_one, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert (run / "a.log").read_text(encoding="utf-8") == "a"
    assert (run / "b.log").read_text(encoding="utf-8") == "b"
    assert (run / "last.bin").read_bytes() in (b"a" * 200_000, b"b" * 200_000)
    assert [p.name for p in run.iterdir() if "letify-partial" in p.name] == []


# -- Spec: Runtime data cache budget, with Writing back ------------------------------


def test_blobs_a_write_back_adds_keep_the_cache_within_its_budget(
    launcher_from, project, capsys
) -> None:
    # 20 KiB of budget: an old 16 KiB input and an 8 KiB written file cannot both stay.
    let = budget_launcher(launcher_from, 20 / (1 << 20))

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def size(path: Path) -> int:
        return path.stat().st_size

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def save(out: Path) -> None:
        out.mkdir()
        (out / "model.bin").write_bytes(b"m" * (8 << 10))

    (project / "old.bin").write_bytes(b"o" * (16 << 10))
    size(project / "old.bin")
    age_blobs(3600)
    capsys.readouterr()
    save(project / "runs")
    assert (project / "runs" / "model.bin").read_bytes() == b"m" * (8 << 10)
    held = blob_files()
    assert set(held) == {pathdata.hash_file(project / "runs" / "model.bin", 8 << 10)}
    root = Path.home() / ".letify-runtime" / "data" / "blobs"
    assert sum(p.stat().st_size for p in root.glob("*/*")) <= 20 << 10
    assert len(evictions(capsys.readouterr().err)) == 1
