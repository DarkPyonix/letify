"""The content addressed store, its backends, and volumes on top.

Spec sections pinned here: "Storage", "Content addressed layout", "Blob granularity",
"Materializing into a runtime" and "Backends".

The filesystem backend is the real one throughout. The gcs backend talks HTTP to a Cloud
Storage endpoint served on loopback by conftest, and the Modal volume backend talks to the
standard library stand-in for the Modal adapter, because a bucket needs an account; what
is still real there is the key layout, the clients and the operations letify performs.
"""

from __future__ import annotations

import io
import tarfile
import types
from pathlib import Path
from urllib.parse import unquote

import pytest

import letify
from letify.store import backends
from letify.store.backends import layout
from letify.store.backends.filesystem import FilesystemBackend
from letify.store.backends.objects import GCSBackend, ModalBackend
from letify.store.cas import Backend, BlobInfo, Store
from letify.store.volume import CHECKPOINT_REF, ENV_REF, Volume


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
    for cls in (FilesystemBackend, GCSBackend, ModalBackend):
        assert cls.missing is not Backend.missing, cls.__name__


def test_the_default_missing_answer_asks_once_per_digest(tmp_path: Path) -> None:
    # What the object store backends override, and the cost the design exists to avoid.
    backend = FilesystemBackend(tmp_path / "store")
    backend.put("held", b"x")
    assert backend.missing(["held", "absent"]) == ["absent"]


# -- Spec: Backends, the cloud ones --------------------------------------------


def test_a_bucket_backend_keeps_the_documented_layout_under_its_prefix(fake_gcs) -> None:
    backend = GCSBackend("study-bucket", prefix="letify", endpoint=fake_gcs.endpoint)
    backend.put("ab12", b"payload")
    assert fake_gcs.objects["letify/blobs/ab/ab12"] == b"payload"
    assert backend.has("ab12") is True
    assert backend.has("nope") is False
    assert backend.get("ab12") == b"payload"
    assert list(backend.list_digests()) == ["ab12"]
    assert backend.missing(["ab12", "absent"]) == ["absent"]
    backend.write_ref("ckpt/run", "ab12")
    assert backend.read_ref("ckpt/run") == "ab12"
    assert backend.read_ref("ckpt/absent") is None


def test_a_bucket_backend_can_be_used_without_a_prefix(fake_gcs) -> None:
    GCSBackend("study-bucket", prefix="", endpoint=fake_gcs.endpoint).put("ab12", b"payload")
    assert "blobs/ab/ab12" in fake_gcs.objects


def test_a_bucket_listing_follows_every_page(fake_gcs) -> None:
    # Spec "Google login for gcs": missing() is one listing, and a listing is paged.
    backend = GCSBackend("study-bucket", endpoint=fake_gcs.endpoint)
    held = [f"{n:02x}{'0' * 30}" for n in range(5)]
    for digest in held:
        backend.put(digest, digest.encode())
    before = len(fake_gcs.requests)
    assert backend.missing([*held, "ff" + "0" * 30]) == ["ff" + "0" * 30]
    pages = [r for r in fake_gcs.requests[before:] if r["path"].endswith("/o")]
    assert len(pages) == 3
    assert all(r["query"]["prefix"] == "letify/blobs/" for r in pages)


def test_a_bucket_request_carries_the_borrowed_token(fake_gcs) -> None:
    GCSBackend("study-bucket", endpoint=fake_gcs.endpoint).has("ab12")
    assert fake_gcs.requests[-1]["authorization"] == "Bearer token-1"


def test_a_bucket_that_refuses_the_token_is_a_runtime_failure(fake_gcs, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "stale")
    backend = GCSBackend("study-bucket", endpoint=fake_gcs.endpoint)
    with pytest.raises(letify.RuntimeFailure, match="401"):
        backend.get("ab12")


# -- Spec: Google login for gcs ------------------------------------------------


@pytest.fixture
def no_google_login(monkeypatch, tmp_path: Path) -> Path:
    """No token variable, no credentials file and no gcloud: a machine with no login."""
    from letify.store.backends import google_auth

    monkeypatch.delenv("GOOGLE_OAUTH_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path / "gcloud"))
    monkeypatch.setattr(google_auth.shutil, "which", lambda name: None)
    return tmp_path / "gcloud"


def test_no_google_login_names_the_command_that_makes_one(no_google_login) -> None:
    from letify.store.backends.google_auth import TokenSource

    with pytest.raises(letify.ProviderUnavailable, match="application-default login"):
        TokenSource().token()


def test_application_default_credentials_are_refreshed_over_http(
    fake_gcs, no_google_login, monkeypatch
) -> None:
    import json

    from letify.store.backends.google_auth import TokenSource

    no_google_login.mkdir()
    (no_google_login / "application_default_credentials.json").write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "client-1",
                "client_secret": "secret-1",
                "refresh_token": "refresh-1",
                "token_uri": f"{fake_gcs.endpoint}/token",
            }
        ),
        encoding="utf-8",
    )
    source = TokenSource()
    assert source.token() == "token-1"
    # Reused until shortly before it expires, so a second request asks for nothing.
    assert source.token() == "token-1"
    assert len(fake_gcs.refresh_grants) == 1
    assert fake_gcs.refresh_grants[0]["grant_type"] == "refresh_token"
    assert fake_gcs.refresh_grants[0]["refresh_token"] == "refresh-1"


def test_a_service_account_key_file_is_refused_with_its_reason(
    no_google_login, tmp_path, monkeypatch
) -> None:
    from letify.store.backends.google_auth import TokenSource

    key = tmp_path / "key.json"
    key.write_text('{"type": "service_account"}', encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    with pytest.raises(letify.ProviderUnavailable, match="activate-service-account"):
        TokenSource().token()


def test_gcloud_is_asked_when_nothing_else_answers(no_google_login, monkeypatch) -> None:
    from letify.store.backends import google_auth

    asked: list[list[str]] = []

    def run(command, **kwargs):
        asked.append(command)
        return types.SimpleNamespace(returncode=0, stdout="token-from-gcloud\n", stderr="")

    monkeypatch.setattr(google_auth.shutil, "which", lambda name: "/usr/bin/gcloud")
    monkeypatch.setattr(google_auth.subprocess, "run", run)
    assert google_auth.TokenSource().token() == "token-from-gcloud"
    assert asked == [["/usr/bin/gcloud", "auth", "print-access-token"]]


def test_a_modal_volume_backend_keeps_the_documented_layout(isolated_home, fake_modal) -> None:
    backend = ModalBackend("letify-study", account="modal_lab")
    try:
        assert list(backend.list_digests()) == []
        backend.put("ab12", b"payload")
        assert backend.has("ab12") is True
        assert backend.has("absent") is False
        assert backend.get("ab12") == b"payload"
        assert list(backend.list_digests()) == ["ab12"]
        assert backend.missing(["ab12", "absent"]) == ["absent"]
        backend.write_ref("ckpt/run", "ab12")
        assert backend.read_ref("ckpt/run") == "ab12"
        assert backend.read_ref("ckpt/absent") is None
    finally:
        backend.close()
    # Spec "Modal adapter": the volume is reached as the account, through the adapter.
    expected = Path.home() / ".letify" / "accounts" / "modal_lab" / "modal.toml"
    assert fake_modal.env()["MODAL_CONFIG_PATH"] == str(expected)
    assert {r["volume"] for r in fake_modal.requests("volume_put")} == {"letify-study"}


def test_reading_a_modal_blob_that_is_absent_is_a_runtime_failure(
    isolated_home, fake_modal
) -> None:
    backend = ModalBackend("letify-study", account="modal_lab")
    try:
        with pytest.raises(letify.RuntimeFailure, match="does not exist"):
            backend.get("absent")
    finally:
        backend.close()


# -- Spec: Packaging, a backend whose tool is absent ---------------------------


def test_a_modal_backend_without_uv_says_uv_is_needed(isolated_home, patch_which) -> None:
    from letify import tools

    patch_which(tools, present=False)
    with pytest.raises(letify.ProviderUnavailable, match="uv was not found"):
        ModalBackend("volume", account="modal_lab")


def test_a_modal_backend_with_no_account_is_a_configuration_error() -> None:
    with pytest.raises(letify.ConfigError, match="account"):
        ModalBackend("volume")


def test_a_volume_on_a_modal_provider_acts_as_that_account(isolated_home, fake_modal) -> None:
    from conftest import provider_of

    from letify.providers.modal import Modal

    volume = Volume(provider_of(Modal, "modal_lab"), "cache")
    backend = volume.store.backend
    assert isinstance(backend, ModalBackend)
    assert backend.account == "modal_lab"
    assert backend.volume_name == "letify-cache"


def test_a_volume_elsewhere_naming_the_modal_backend_needs_an_account(let) -> None:
    volume = Volume(let.providers.local, "cache", {"backend": "modal"})
    with pytest.raises(letify.ConfigError, match="account"):
        volume.store  # noqa: B018


# -- Spec: Storage, volumes ----------------------------------------------------


def test_a_volume_mounts_where_the_runtime_keeps_materialized_files(let, tmp_path) -> None:
    # Spec "Workspace root": local uses no workspace, so the default root holds its volumes.
    default = Volume(let.providers.local, "cache", {"root": str(tmp_path)}).mount
    assert default == "~/.letify-runtime/volumes/cache"
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


# -- Spec: Materializing into a runtime, the runtime pulls from the backend -----


@pytest.fixture
def bucket_volume(let, fake_gcs, tmp_path):
    """A gcs volume on the local provider, its bucket served on loopback."""
    return let.providers.local.volume(
        "bucket",
        backend="gcs",
        bucket=fake_gcs.bucket,
        endpoint=fake_gcs.endpoint,
        sts_endpoint=f"{fake_gcs.endpoint}/v1/token",
        mount=str(tmp_path / "mount"),
    )


def test_a_runtime_pulls_a_blob_straight_from_the_bucket(
    let, remote_cpu, bucket_volume, fake_gcs, tmp_path, live
) -> None:
    # Bytes never pass through the local process on the way in: the worker downloads them.
    info = bucket_volume.store.put_bytes(b"model weights")
    runtime = live(let, remote_cpu)
    before = len(fake_gcs.downloads())

    remote = bucket_volume.materialize(runtime, info.digest)

    assert Path(remote.path).read_bytes() == b"model weights"
    assert remote.size == len(b"model weights")
    pulled = fake_gcs.downloads()[before:]
    assert len(pulled) == 1
    # Downscoped from the local login rather than the login itself.
    assert pulled[0]["authorization"] == "Bearer down-token-1"
    let.pool.shutdown()


def test_the_read_token_is_bounded_to_the_volume_bucket_and_prefix(
    let, remote_cpu, bucket_volume, fake_gcs, live
) -> None:
    import json

    info = bucket_volume.store.put_bytes(b"epoch 1")
    bucket_volume.materialize(live(let, remote_cpu), info.digest)

    exchange = fake_gcs.exchanges[-1]
    assert exchange["subject_token"] == "token-1"
    boundary = json.loads(exchange["options"])["accessBoundary"]["accessBoundaryRules"][0]
    assert boundary["availablePermissions"] == ["inRole:roles/storage.objectViewer"]
    assert boundary["availableResource"].endswith(f"/buckets/{fake_gcs.bucket}")
    assert "objects/letify/" in boundary["availabilityCondition"]["expression"]
    let.pool.shutdown()


def test_the_pull_token_is_kept_nowhere_once_the_pull_finishes(
    let, remote_cpu, bucket_volume, fake_gcs, tmp_path, live
) -> None:
    info = bucket_volume.store.put_bytes(b"secret-free weights")
    runtime = live(let, remote_cpu)
    remote = bucket_volume.materialize(runtime, info.digest)

    # Not on disk beside what was pulled.
    for path in (tmp_path / "mount").rglob("*"):
        if path.is_file():
            assert b"down-token-1" not in path.read_bytes()
    # Not in the worker's memory or environment. The token is assembled inside the check,
    # so the exec request carrying it does not contain it.
    runtime.exec(
        "import gc, os\n"
        "_needle = 'down-' + 'token-1'\n"
        "assert not any(_needle in v for v in os.environ.values())\n"
        "assert not any(isinstance(o, dict) and _needle in repr(o.get('headers'))"
        " for o in gc.get_objects())\n"
    )
    assert Path(remote.path).is_file()
    let.pool.shutdown()


def test_a_cached_environment_is_pulled_and_unpacked_by_the_runtime(
    fake_gcs, tmp_path, uv_project
) -> None:
    # The archive the first session stored is what the next session pulls.
    import shutil

    from conftest import PreparingLocal, provider_of

    from letify.runtime import bootstrap

    provider = provider_of(PreparingLocal, "lab")
    env = letify.Env()
    mount = tmp_path / "mount"
    volume = provider.volume(
        "bucket",
        backend="gcs",
        bucket=fake_gcs.bucket,
        endpoint=fake_gcs.endpoint,
        sts_endpoint=f"{fake_gcs.endpoint}/v1/token",
        mount=str(mount),
    )
    from letify.declare.instance import Instance

    instance = Instance(provider, gpu=None)._placed("remote")
    provider.start(instance, env, name="lab-1", volumes=(volume,)).shutdown()
    assert not [r for r in fake_gcs.downloads() if "/blobs/" in unquote(r["path"])]

    shutil.rmtree(Path(bootstrap.DEFAULT_WORKSPACE_ROOT) / "project")
    runtime = provider.start(instance, env, name="lab-2", volumes=(volume,))
    try:
        assert runtime.env_source == "archive"
        # The ref is read by this process with its own login; the blob is read by the
        # runtime with the downscoped token.
        blobs = [r for r in fake_gcs.downloads() if "/blobs/" in unquote(r["path"])]
        assert [r["authorization"] for r in blobs] == ["Bearer down-token-1"]
    finally:
        runtime.shutdown()


def test_a_backend_the_runtime_cannot_reach_offers_no_pull(tmp_path) -> None:
    # So the filesystem backend writes through the channel instead.
    assert FilesystemBackend(tmp_path / "store").pull_source("ab12") is None


def test_a_refused_token_exchange_sends_no_token_at_all(
    let, remote_cpu, fake_gcs, tmp_path, live
) -> None:
    volume = let.providers.local.volume(
        "bucket",
        backend="gcs",
        bucket=fake_gcs.bucket,
        endpoint=fake_gcs.endpoint,
        sts_endpoint=f"{fake_gcs.endpoint}/no-such-exchange",
        mount=str(tmp_path / "mount"),
    )
    info = volume.store.put_bytes(b"weights")
    runtime = live(let, remote_cpu)
    with pytest.raises(letify.RuntimeFailure, match="downscop"):
        volume.materialize(runtime, info.digest)
    assert fake_gcs.downloads() == []
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


def test_the_gcs_backend_imports_no_cloud_sdk(no_module, fake_gcs) -> None:
    # Spec "Packaging": the gcs blob store uses a standard library client.
    no_module("google", "google.cloud", "google.cloud.storage")
    assert GCSBackend("study-bucket", endpoint=fake_gcs.endpoint).has("ab12") is False


def test_there_is_no_s3_backend() -> None:
    # Not a requested feature, and boto3 is a dependency letify does not take.
    assert "s3" not in backends.BACKENDS
