"""The content addressed store, its backends, and volumes on top.

Spec sections pinned here: "Storage", "Content addressed layout", "Blob granularity",
"Materializing into a runtime" and "Backends".

The filesystem backend is the real one throughout. The three cloud backends are driven
through in-memory stand-ins from conftest, because a bucket needs an account; what is
still real there is the key layout and the operations letify performs.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

import letify
from letify.store import backends
from letify.store.backends import layout
from letify.store.backends.filesystem import FilesystemBackend
from letify.store.backends.objects import GCSBackend, ModalBackend, S3Backend
from letify.store.cas import Backend, BlobInfo, Store
from letify.store.volume import CHECKPOINT_REF, DEFAULT_MOUNT, ENV_REF, Volume


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(FilesystemBackend(tmp_path / "store"))


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "site"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "nested" / "b.txt").write_text("b", encoding="utf-8")
    return root


@pytest.fixture
def volume(let: letify.Launcher, tmp_path: Path) -> Volume:
    """A volume on the local provider, with the store and the mount in a temp directory."""
    return let.providers.local.volume(
        "cache", root=str(tmp_path / "store"), mount=str(tmp_path / "mount")
    )


@pytest.fixture
def remote_cpu(let: letify.Launcher) -> letify.Instance:
    return let.providers.local.CPU._placed("remote")


# -- Spec: Content addressed layout --------------------------------------------


def test_a_blob_lives_under_the_first_two_characters_of_its_digest() -> None:
    # The prefix keeps any one directory from holding every blob, which matters on a
    # filesystem and costs nothing on an object store.
    assert layout.blob_key("ab12cd34") == "blobs/ab/ab12cd34"
    assert layout.ref_key("ckpt/run-1") == "refs/ckpt/run-1"


def test_the_same_contents_get_the_same_name(store: Store) -> None:
    first = store.put_bytes(b"same contents")
    second = store.put_bytes(b"same contents")
    assert first == second == BlobInfo(first.digest, len(b"same contents"))
    assert store.get_bytes(first.digest) == b"same contents"


def test_different_contents_get_different_names(store: Store) -> None:
    # This is what stops two concurrent writers from overwriting each other's work.
    assert store.put_bytes(b"one").digest != store.put_bytes(b"another").digest


def test_holding_a_digest_is_proof_of_holding_the_contents(store: Store, tmp_path: Path) -> None:
    # So a transfer that already happened is skipped by name alone, with nothing verified
    # a second time.
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"x" * 100)
    store.put_file(payload)
    upload, held = store.plan_upload([payload])
    assert upload == []
    assert len(held) == 1


def test_planning_an_upload_separates_what_is_held_from_what_is_not(
    store: Store, tmp_path: Path
) -> None:
    known = tmp_path / "known.bin"
    known.write_bytes(b"already there")
    fresh = tmp_path / "fresh.bin"
    fresh.write_bytes(b"never seen")
    store.put_file(known)
    upload, held = store.plan_upload([known, fresh])
    assert upload == [fresh]
    assert len(held) == 1


def test_a_ref_carries_the_mutable_part(store: Store) -> None:
    # Refs are a separate, tiny namespace, the way Git keeps branch names apart from
    # objects.
    info = store.put_bytes(b"checkpoint")
    store.point("ckpt/run-1", info.digest)
    assert store.resolve("ckpt/run-1") == info.digest
    assert store.resolve("ckpt/never-written") is None


def test_a_ref_can_be_moved_to_a_newer_blob_and_both_blobs_survive(store: Store) -> None:
    first = store.put_bytes(b"epoch 1")
    second = store.put_bytes(b"epoch 2")
    store.point("ckpt/run-1", first.digest)
    store.point("ckpt/run-1", second.digest)
    assert store.resolve("ckpt/run-1") == second.digest
    assert store.get_bytes(first.digest) == b"epoch 1"


def test_the_names_letify_reserves_are_the_environment_and_the_checkpoint(
    let: letify.Launcher, tmp_path: Path
) -> None:
    # The rest of the ref namespace belongs to the user.
    assert ENV_REF == "env/{key}"
    assert CHECKPOINT_REF == "ckpt/{name}"
    volume = Volume(let.providers.local, "cache", {"root": str(tmp_path)})
    assert volume.env_ref(letify.Env()) == f"env/{letify.Env().key}"


# -- Spec: Blob granularity ----------------------------------------------------


def test_a_tree_of_small_files_travels_as_one_blob(store: Store, tree: Path, tmp_path: Path):
    # An environment is tens of thousands of small files, so packing turns tens of
    # thousands of round trips into one.
    info = store.put_tree(tree, key="env/abc")
    assert store.resolve("env/abc") == info.digest

    restored = store.fetch_tree(info.digest, tmp_path / "out")
    assert (restored / "site" / "nested" / "b.txt").read_text(encoding="utf-8") == "b"


def test_packing_a_tree_without_storing_it_is_a_gzip_archive(store: Store, tree: Path) -> None:
    assert store.pack(tree).startswith(b"\x1f\x8b")


def test_a_tree_can_be_stored_without_naming_it(store: Store, tree: Path) -> None:
    info = store.put_tree(tree)
    assert store.backend.has(info.digest)


def test_a_single_large_file_stands_alone(store: Store, tmp_path: Path) -> None:
    # A model shard is already large, so one file is one blob.
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"y" * 4096)
    info = store.put_file(shard)
    target = store.fetch_file(info.digest, tmp_path / "restored" / "model.safetensors")
    assert target.read_bytes() == b"y" * 4096


def test_an_archive_member_cannot_write_outside_the_destination(
    store: Store, tmp_path: Path
) -> None:
    # Checked before anything is unpacked, so a hostile or careless archive cannot reach
    # the rest of the machine.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    blob = store.put_bytes(buffer.getvalue())

    with pytest.raises(ValueError, match="would escape"):
        store.fetch_tree(blob.digest, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


# -- Spec: Backends ------------------------------------------------------------


def test_a_provider_names_a_backend_and_the_store_builds_it(tmp_path: Path) -> None:
    built = backends.build("filesystem", root=str(tmp_path / "blobs"))
    assert isinstance(built, FilesystemBackend)
    assert built.name == "filesystem"


def test_an_unknown_backend_lists_the_ones_that_exist() -> None:
    with pytest.raises(letify.ProviderUnavailable, match="unknown backend 'tape'"):
        backends.build("tape")


@pytest.mark.parametrize(
    ("backend", "option"),
    [
        ("filesystem", "root"),
        ("shell", "root"),
        ("modal", "volume_name"),
        ("gcs", "bucket"),
        ("s3", "bucket"),
    ],
)
def test_each_backend_has_a_default_location(backend: str, option: str) -> None:
    # So a volume can be declared with a name alone.
    chosen, value = backends.default_location(backend, "study")
    assert chosen == option
    assert "study" in value


def test_a_filesystem_blob_is_never_visible_half_written(tmp_path: Path) -> None:
    # Written beside the target and renamed, so a concurrent reader sees all or nothing.
    backend = FilesystemBackend(tmp_path / "store")
    backend.put("ab12", b"payload")
    assert backend.has("ab12")
    assert backend.get("ab12") == b"payload"
    assert list(backend.list_digests()) == ["ab12"]
    assert list(backend.list_digests(prefix="zz")) == []


def test_listing_a_store_that_holds_nothing_yet_is_empty(tmp_path: Path) -> None:
    assert list(FilesystemBackend(tmp_path / "fresh").list_digests()) == []


def test_a_filesystem_ref_survives_a_rewrite(tmp_path: Path) -> None:
    backend = FilesystemBackend(tmp_path / "store")
    assert backend.read_ref("ckpt/run") is None
    backend.write_ref("ckpt/run", "ab12")
    assert backend.read_ref("ckpt/run") == "ab12"


def test_every_backend_answers_the_missing_question_with_one_listing() -> None:
    for cls in (FilesystemBackend, GCSBackend, S3Backend, ModalBackend):
        assert cls.missing is not Backend.missing, cls.__name__


def test_the_default_missing_answer_asks_once_per_digest(tmp_path: Path) -> None:
    # What the object store backends override, and the cost the design exists to avoid.
    backend = FilesystemBackend(tmp_path / "store")
    backend.put("held", b"x")
    assert backend.missing(["held", "absent"]) == ["absent"]


# -- Spec: Backends, the cloud ones --------------------------------------------


def test_a_bucket_backend_keeps_the_documented_layout_under_its_prefix(fake_gcs) -> None:
    backend = GCSBackend("study-bucket", prefix="letify")
    backend.put("ab12", b"payload")
    assert "letify/blobs/ab/ab12" in fake_gcs
    assert backend.has("ab12") is True
    assert backend.has("nope") is False
    assert backend.get("ab12") == b"payload"
    assert list(backend.list_digests()) == ["ab12"]
    assert backend.missing(["ab12", "absent"]) == ["absent"]
    backend.write_ref("ckpt/run", "ab12")
    assert backend.read_ref("ckpt/run") == "ab12"
    assert backend.read_ref("ckpt/absent") is None


def test_a_bucket_backend_can_be_used_without_a_prefix(fake_gcs) -> None:
    GCSBackend("study-bucket", prefix="").put("ab12", b"payload")
    assert "blobs/ab/ab12" in fake_gcs


def test_an_s3_backend_keeps_the_documented_layout(fake_boto3) -> None:
    # This covers Elice Data Hub as well as Amazon S3, because the S3 API is what object
    # stores agree on.
    backend = S3Backend("study-bucket", endpoint_url="https://datahub.example")
    backend.put("ab12", b"payload")
    assert "letify/blobs/ab/ab12" in fake_boto3.store
    assert backend.has("ab12") is True
    assert backend.has("absent") is False
    assert backend.get("ab12") == b"payload"
    assert list(backend.list_digests()) == ["ab12"]
    assert backend.missing(["ab12", "absent"]) == ["absent"]
    backend.write_ref("ckpt/run", "ab12")
    assert backend.read_ref("ckpt/run") == "ab12"
    assert backend.read_ref("ckpt/absent") is None
    assert fake_boto3.clients[0].options["endpoint_url"] == "https://datahub.example"


def test_an_s3_endpoint_can_come_from_the_environment(fake_boto3, monkeypatch) -> None:
    # An S3 compatible provider is reached by endpoint, and that belongs outside the
    # tracked configuration.
    monkeypatch.setenv("LETIFY_S3_ENDPOINT", "https://from-the-environment")
    S3Backend("study-bucket")
    assert fake_boto3.clients[0].options["endpoint_url"] == "https://from-the-environment"


def test_a_modal_volume_backend_keeps_the_documented_layout(fake_modal) -> None:
    fake_modal()
    backend = ModalBackend("letify-study")
    backend.put("ab12", b"payload")
    assert backend.has("ab12") is True
    assert backend.has("absent") is False
    assert backend.get("ab12") == b"payload"
    assert list(backend.list_digests()) == ["ab12"]
    assert backend.missing(["ab12", "absent"]) == ["absent"]
    backend.write_ref("ckpt/run", "ab12")
    assert backend.read_ref("ckpt/run") == "ab12"
    assert backend.read_ref("ckpt/absent") is None


# -- Spec: Packaging, a backend whose package is absent ------------------------


@pytest.mark.parametrize(
    ("cls", "argument", "module", "extra"),
    [
        (GCSBackend, "bucket", "google.cloud", "gcs"),
        (S3Backend, "bucket", "boto3", "s3"),
        (ModalBackend, "volume", "modal", "modal"),
    ],
)
def test_a_backend_whose_package_is_absent_says_how_to_install_it(
    cls, argument: str, module: str, extra: str, no_module
) -> None:
    no_module(module)
    with pytest.raises(letify.ProviderUnavailable, match=f"letify\\[{extra}\\]"):
        cls(argument)


# -- Spec: Storage, volumes ----------------------------------------------------


def test_a_volume_mounts_where_the_runtime_keeps_materialized_files(let, tmp_path) -> None:
    assert Volume(let.providers.local, "cache", {"root": str(tmp_path)}).mount == DEFAULT_MOUNT
    volume = Volume(let.providers.local, "cache", {"mount": "/mnt/study"})
    assert volume.mount == "/mnt/study"
    assert volume.key == "local/cache"


def test_a_volume_takes_its_backend_from_the_provider_unless_told_otherwise(let, tmp_path) -> None:
    # Spec "Backends": Local uses the filesystem, and a volume may name another.
    volume = let.providers.local.volume("cache", root=str(tmp_path / "a"))
    assert volume.store.backend.name == "filesystem"
    # Built once, on first use rather than at declaration time.
    assert volume.store is volume.store
    assert repr(volume) == "<Volume local/cache on filesystem>"


def test_a_provider_binds_one_volume_per_name(let, tmp_path) -> None:
    provider = let.providers.local
    assert provider.volume("cache", root=str(tmp_path)) is provider.volume("cache")


def test_an_environment_archive_is_keyed_by_the_environment(let, tree, tmp_path) -> None:
    # Spec "Environment": the key is a hash of the lock file and the refinements, so the
    # same declaration reuses the same archive.
    volume = let.providers.local.volume("cache", root=str(tmp_path / "store"))
    env = letify.Env(lock=str(tmp_path / "absent.lock"))
    assert volume.cached_env(env) is None
    digest = volume.cache_env(env, tree)
    assert volume.cached_env(env) == digest
    # A different declaration does not find it.
    assert volume.cached_env(env.pip_install("torch")) is None


def test_a_checkpoint_from_this_machine_can_be_stored_and_fetched_back(let, tmp_path) -> None:
    volume = let.providers.local.volume("cache", root=str(tmp_path / "store"))
    assert volume.latest_checkpoint("run-1") is None

    source = tmp_path / "epoch-1.pt"
    source.write_bytes(b"weights")
    digest = volume.put_checkpoint("run-1", source)
    assert volume.latest_checkpoint("run-1") == digest

    restored = volume.fetch_checkpoint("run-1", tmp_path / "restored" / "epoch-1.pt")
    assert restored.read_bytes() == b"weights"


def test_a_checkpoint_directory_is_stored_as_one_archive(let, tree, tmp_path) -> None:
    volume = let.providers.local.volume("cache", root=str(tmp_path / "store"))
    volume.put_checkpoint("run-2", tree)
    restored = volume.fetch_checkpoint("run-2", tmp_path / "restored")
    assert (restored / "site" / "a.txt").read_text(encoding="utf-8") == "a"


def test_fetching_a_checkpoint_that_was_never_written_returns_nothing(let, tmp_path) -> None:
    volume = let.providers.local.volume("cache", root=str(tmp_path / "store"))
    assert volume.fetch_checkpoint("never-run", tmp_path / "out") is None


# -- Spec: Materializing into a runtime ----------------------------------------


def test_a_volume_writes_into_a_runtime_through_its_channel(
    let, remote_cpu, tmp_path, live
) -> None:
    # Not by asking the runtime to reach the bucket, so this works with every backend and
    # needs no credentials on the far side.
    volume = let.providers.local.volume(
        "cache", root=str(tmp_path / "store"), mount=str(tmp_path / "mount")
    )
    info = volume.store.put_bytes(b"model weights")
    remote = volume.materialize(live(let, remote_cpu), info.digest)
    assert Path(remote.path).read_bytes() == b"model weights"
    assert remote.size == len(b"model weights")
    let.pool.shutdown()


def test_materializing_a_name_follows_it_to_the_current_blob(
    let, remote_cpu, tmp_path, live
) -> None:
    volume = let.providers.local.volume(
        "cache", root=str(tmp_path / "store"), mount=str(tmp_path / "mount")
    )
    runtime = live(let, remote_cpu)
    assert volume.materialize_ref(runtime, "ckpt/absent") is None

    info = volume.store.put_bytes(b"epoch 2")
    volume.store.point("ckpt/run-1", info.digest)
    remote = volume.materialize_ref(runtime, "ckpt/run-1")
    assert Path(remote.path).read_bytes() == b"epoch 2"
    let.pool.shutdown()


def test_resume_puts_the_newest_checkpoint_where_the_function_will_look(
    let, remote_cpu, tmp_path, live
) -> None:
    # This is what makes a preempted session cheap to restart: the body asks for its own
    # checkpoint and finds it already on disk.
    volume = let.providers.local.volume(
        "cache", root=str(tmp_path / "store"), mount=str(tmp_path / "mount")
    )
    runtime = live(let, remote_cpu)
    target = str(tmp_path / "inside" / "epoch.pt")
    assert volume.resume(runtime, "run-1", target) is None

    source = tmp_path / "epoch-2.pt"
    source.write_bytes(b"epoch 2")
    volume.put_checkpoint("run-1", source)
    assert volume.resume(runtime, "run-1", target)
    assert Path(target).read_bytes() == b"epoch 2"
    let.pool.shutdown()


def test_a_checkpoint_written_inside_a_runtime_is_pulled_back_under_a_name(
    let, remote_cpu, tmp_path, live
) -> None:
    volume = let.providers.local.volume("cache", root=str(tmp_path / "store"))
    runtime = live(let, remote_cpu)
    inside = tmp_path / "inside.pt"
    runtime.put_bytes(b"trained", str(inside))

    digest = volume.absorb(runtime, str(inside), "run-1")
    assert volume.latest_checkpoint("run-1") == digest
    assert volume.store.get_bytes(digest) == b"trained"
    let.pool.shutdown()


def test_an_environment_installed_inside_a_runtime_is_cached_for_the_next_session(
    let, remote_cpu, tmp_path, live
) -> None:
    # This is how the first session pays the installation cost and every later one skips
    # it.
    volume = let.providers.local.volume("cache", root=str(tmp_path / "store"))
    runtime = live(let, remote_cpu)
    installed = tmp_path / "site"
    installed.mkdir()
    (installed / "marker.txt").write_text("installed", encoding="utf-8")

    env = letify.Env(lock=str(tmp_path / "absent.lock"))
    digest = volume.cache_env_from(runtime, env, str(installed))
    assert volume.cached_env(env) == digest
    let.pool.shutdown()


# -- Spec: Materializing into a runtime, the declaration names the session ------


def test_a_checkpoint_is_taken_from_the_session_the_declaration_used(
    let, cpu, volume, tmp_path
) -> None:
    # The caller names the declaration. Which session ran the call is letify's answer, and
    # it is the only one holding the file.
    @let.function(device=cpu, host=letify.remote, volumes=[volume])
    def write_a_file(path: str) -> str:
        from pathlib import Path as P

        P(path).parent.mkdir(parents=True, exist_ok=True)
        P(path).write_bytes(b"weights")
        return path

    with let.keep_alive():
        written = write_a_file(path=str(tmp_path / "out" / "adapter.pt"))
        digest = volume.absorb(write_a_file, written, "run-1")
    assert volume.latest_checkpoint("run-1") == digest
    assert volume.store.get_bytes(digest) == b"weights"


def test_a_checkpoint_is_put_back_into_the_session_the_declaration_will_use(
    let, cpu, volume, tmp_path
) -> None:
    # Before the call rather than after, because the training function looks for it as an
    # ordinary path. The session it lands in has to be the one the call is handed.
    source = tmp_path / "seed.pt"
    source.write_bytes(b"seed")
    volume.put_checkpoint("run-2", source)
    target = str(tmp_path / "inside" / "resume.pt")

    @let.function(device=cpu, host=letify.remote, volumes=[volume])
    def read_it_back(path: str) -> bytes:
        from pathlib import Path as P

        return P(path).read_bytes()

    with let.keep_alive():
        assert volume.resume(read_it_back, "run-2", target) is not None
        assert read_it_back(path=target) == b"seed"


def test_resuming_a_name_nothing_was_stored_under_reports_nothing(
    let, cpu, volume, tmp_path
) -> None:
    # Nothing to put back is an answer rather than a failure: a first run has no checkpoint.
    @let.function(device=cpu, host=letify.remote, volumes=[volume])
    def anything() -> int:
        return 1

    with let.keep_alive():
        assert volume.resume(anything, "never-written", str(tmp_path / "x.pt")) is None


def test_a_declaration_asked_twice_is_given_one_session(let, cpu, volume, tmp_path) -> None:
    # Otherwise moving a checkpoint in and then out would cross two sessions, and the second
    # would not hold the file the first wrote.
    @let.function(device=cpu, host=letify.remote, volumes=[volume])
    def note(path: str) -> str:
        from pathlib import Path as P

        P(path).parent.mkdir(parents=True, exist_ok=True)
        P(path).write_bytes(b"once")
        return path

    with let.keep_alive():
        written = note(path=str(tmp_path / "twice" / "a.pt"))
        volume.absorb(note, written, "run-3")
        volume.absorb(note, written, "run-3")
        assert let.status()["live"] == 1


def test_moving_a_checkpoint_outside_keep_alive_is_refused(let, cpu, volume, tmp_path) -> None:
    # The session would end as soon as the call returned, so a resumed checkpoint would vanish
    # before the call that needs it.
    source = tmp_path / "seed.pt"
    source.write_bytes(b"seed")
    volume.put_checkpoint("run-4", source)

    @let.function(device=cpu, host=letify.remote, volumes=[volume])
    def anything() -> int:
        return 1

    with pytest.raises(letify.UnsupportedMode, match="keep_alive"):
        volume.resume(anything, "run-4", str(tmp_path / "x.pt"))
    assert let.pool.live == []
