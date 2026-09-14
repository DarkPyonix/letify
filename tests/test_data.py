"""Project data: local paths a call reaches travel as content addressed file blobs.

Spec sections pinned here: "Project data" and its subsections "Which paths are data",
"Digests and the digest cache", "Where the bytes come from", "Materializing and the
rewritten path" and "Data log line".

Everything runs through the Local provider and the real framed worker. Only the bucket is a
fake, the Cloud Storage endpoint conftest serves on loopback, because a real one needs an
account.
"""

from __future__ import annotations

import re
import shutil
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
    return [line for line in err.splitlines() if line.startswith("letify: data ")]


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


def test_a_path_that_does_not_exist_is_left_as_a_plain_path(let, cpu, project, capsys) -> None:
    absent = project / "outputs" / "run-1"

    @let.function(device=cpu, host=letify.remote)
    def where(path: Path) -> str:
        return str(path)

    assert where(absent) == str(absent)
    assert data_lines(capsys.readouterr().err) == []


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
