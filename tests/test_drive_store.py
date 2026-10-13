"""The Google Drive store for an environment archive.

Spec section pinned here: "Materializing into a runtime", the paragraphs from "Google Drive
is a store, not a volume" to "Drive's own limits apply". Drive is declared per machine with
``drive_login``, ``drive_project`` and ``drive_endpoint`` in the machine configuration.
Google Drive needs an account, so the endpoint is the stand-in on loopback in
``conftest.py``; everything on this side of it is the real code, including the worker that
does the uploading and the downloading.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from conftest import PreparingLocal, provider_of

from letify.declare.env import Env
from letify.declare.instance import Instance
from letify.errors import ConfigError
from letify.runtime import bootstrap
from letify.runtime.session import Runtime
from letify.store import pathdata
from letify.store.backends.filesystem import FilesystemBackend
from letify.store.cas import Store
from letify.store.volume import environment_volume


def declare_drive(server: Any, **settings: Any) -> None:
    """Write the machine's top-level Drive settings; a None value leaves that setting out."""
    from letify import paths

    values: dict[str, Any] = {
        "drive_login": "drive-login",
        "drive_project": server.project,
        "drive_endpoint": server.endpoint,
    }
    values.update(settings)
    lines = [f'{key} = "{value}"' for key, value in values.items() if value is not None]
    paths.config_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


#: Tests that pin the gate's own rates and arithmetic, so they see the real constants.
REAL_GATE = {
    "test_the_gate_estimates_a_restore_as_size_over_the_download_and_extraction_rates",
    "test_the_gate_packs_only_when_the_build_took_twice_the_estimated_restore",
}


@pytest.fixture(autouse=True)
def pack_whatever_the_build_time(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make the estimated restore negligible, so a session packs however fast it built.

    A test session builds in one or two seconds, and whether that clears twice the
    estimated restore of its small tree depends on the machine. A test that needs the
    pack would otherwise pass or fail by timing. Tests in ``REAL_GATE`` and tests that set
    the rates themselves see their own values.
    """
    if request.node.name in REAL_GATE:
        return
    from letify.runtime import session

    monkeypatch.setattr(session, "DRIVE_DOWNLOAD_BPS", 1e18)
    monkeypatch.setattr(session, "EXTRACT_BPS", 1e18)


def drive_options(server: Any, **extra: Any) -> dict[str, Any]:
    """Account options of an ephemeral account, after the machine has declared Drive."""
    declare_drive(server)
    options: dict[str, Any] = {"persistent": False}
    options.update(extra)
    return options


def remote_instance(provider: Any) -> Instance:
    return Instance(provider, gpu=None)._placed("remote")


def forbid_client_holding_the_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every route that would put the archive in this process raise."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the archive crossed the client")

    monkeypatch.setattr(Store, "put_bytes", refuse)
    monkeypatch.setattr(Runtime, "pack_dir", refuse)
    monkeypatch.setattr(pathdata, "_get", refuse)


# -- configuration -------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "named", "missing"),
    [
        ({"drive_project": None}, "drive_login", "drive_project"),
        ({"drive_login": None}, "drive_project", "drive_login"),
    ],
)
def test_the_machine_with_only_one_of_drive_login_and_drive_project_is_refused(
    fake_drive, settings, named, missing
) -> None:
    declare_drive(fake_drive, **settings)
    provider = provider_of(PreparingLocal, "lab", persistent=False)
    with pytest.raises(ConfigError) as raised:
        environment_volume(provider)
    assert named in str(raised.value)
    assert missing in str(raised.value)


def test_machine_settings_select_drive_for_an_ephemeral_provider(fake_drive) -> None:
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    assert environment_volume(provider).store.backend.name == "drive"


def test_the_environment_volume_is_the_client_filesystem_store_without_drive(
    isolated_home,
) -> None:
    provider = provider_of(PreparingLocal, "lab", persistent=False)
    assert isinstance(environment_volume(provider).store.backend, FilesystemBackend)


def test_a_persistent_provider_ignores_the_machine_drive_as_it_ignores_bucket(
    fake_drive,
) -> None:
    declare_drive(fake_drive)
    provider = provider_of(PreparingLocal, "lab", persistent=True)
    assert isinstance(environment_volume(provider).store.backend, FilesystemBackend)
    assert pathdata.data_bucket(provider) is None


def test_an_account_with_a_bucket_keeps_the_bucket_when_the_machine_declares_drive(
    fake_drive,
) -> None:
    declare_drive(fake_drive)
    provider = provider_of(PreparingLocal, "lab", persistent=False, bucket="some-bucket")
    assert environment_volume(provider).store.backend.name == "gcs"


def test_the_old_account_level_drive_options_no_longer_select_drive(fake_drive) -> None:
    provider = provider_of(
        PreparingLocal, "lab", persistent=False,
        drive="drive-login", drive_project=fake_drive.project,
        drive_endpoint=fake_drive.endpoint,
    )
    assert isinstance(environment_volume(provider).store.backend, FilesystemBackend)


def test_project_data_never_goes_to_drive(fake_drive) -> None:
    # Spec: Drive is chosen for the environment archive and not for project data.
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    assert pathdata.data_bucket(provider) is None


# -- the backend against the stand-in ------------------------------------------


def backend_for(server: Any, **extra: Any):
    provider = provider_of(PreparingLocal, "lab", **drive_options(server, **extra))
    return environment_volume(provider).store.backend


def test_every_request_to_drive_names_the_quota_project(fake_drive) -> None:
    backend = backend_for(fake_drive)
    backend.put("ab" * 16, b"payload")
    backend.write_ref("env/x", "ab" * 16)
    assert backend.has("ab" * 16)
    assert backend.get("ab" * 16) == b"payload"
    assert backend.read_ref("env/x") == "ab" * 16
    assert backend.missing(["ab" * 16, "cd" * 16]) == ["cd" * 16]
    asked = [r for r in fake_drive.requests if r["path"] != "/token"]
    assert asked
    assert all(r["headers"].get("x-goog-user-project") == fake_drive.project for r in asked)


def test_the_archive_lives_in_a_folder_named_letify_with_refs_as_small_files(
    fake_drive,
) -> None:
    backend = backend_for(fake_drive)
    digest = "ab" * 16
    backend.put(digest, b"payload")
    backend.write_ref("env/x", digest)
    backend.write_ref("env/x", "cd" * 16)
    folders = [f for f in fake_drive.files.values() if f["folder"]]
    assert [f["name"] for f in folders] == ["letify"]
    assert folders[0]["parent"] == "root"
    assert fake_drive.blobs() == {digest: b"payload"}
    assert fake_drive.refs() == {"env/x": "cd" * 16}
    # A ref is updated in place, and a second backend finds the folder instead of making one.
    assert len([f for f in fake_drive.files.values() if f["name"] == "ref env/x"]) == 1
    backend_for(fake_drive).has(digest)
    assert len([f for f in fake_drive.files.values() if f["folder"]]) == 1


def test_a_name_with_a_quote_in_it_is_found_again(fake_drive) -> None:
    backend = backend_for(fake_drive)
    backend.write_ref("it's", "ab" * 16)
    assert backend.read_ref("it's") == "ab" * 16


def test_missing_asks_for_one_listing_however_many_digests_it_is_given(fake_drive) -> None:
    backend = backend_for(fake_drive)
    backend.put("ab" * 16, b"x")
    before = len([r for r in fake_drive.requests if r["path"] == "/drive/v3/files"])
    backend.missing([f"{n:02x}" * 16 for n in range(40)])
    listings = [r for r in fake_drive.requests if r["path"] == "/drive/v3/files"][before:]
    assert len(listings) == 1


def test_pull_source_is_the_media_url_with_the_token_and_the_project(fake_drive) -> None:
    backend = backend_for(fake_drive)
    backend.put("ab" * 16, b"x")
    source = backend.pull_source("ab" * 16)
    assert source is not None
    assert source["url"].endswith("?alt=media")
    assert source["headers"] == {
        "Authorization": f"Bearer {fake_drive.token}",
        "x-goog-user-project": fake_drive.project,
    }


def test_a_login_without_the_drive_file_scope_is_refused(fake_drive) -> None:
    fake_drive.login("drive-login", scopes=["https://www.googleapis.com/auth/cloud-platform"])
    backend = backend_for(fake_drive)
    with pytest.raises(ConfigError) as raised:
        backend.has("ab" * 16)
    assert "drive.file" in str(raised.value)


# -- a session with a Drive store ----------------------------------------------


def test_a_pack_puts_the_archive_in_drive_without_the_client_holding_it(
    isolated_home, uv_project: Path, fake_drive, monkeypatch
) -> None:
    forbid_client_holding_the_archive(monkeypatch)
    sent: list[dict[str, Any]] = []
    original = Runtime.request

    def recording(self: Runtime, payload: dict[str, Any], **kwargs: Any) -> Any:
        sent.append(json.loads(json.dumps(payload, default=repr)))
        return original(self, payload, **kwargs)

    monkeypatch.setattr(Runtime, "request", recording)
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    runtime = provider.start(remote_instance(provider), Env(), name="lab-1")
    try:
        assert runtime.env_source == "sync"
    finally:
        runtime.shutdown()
    blobs = fake_drive.blobs()
    assert len(blobs) == 1
    (digest, archive), = blobs.items()
    assert len(archive) > 0
    assert digest in fake_drive.refs().values()
    # The upload used a session the client created, and the worker was handed its URL.
    pushes = [p for p in sent if p.get("op") == "pack_push_target"]
    assert len(pushes) == 1
    assert pushes[0]["url"].startswith(fake_drive.endpoint + "/upload/session/")
    # The access token never reached the worker for the upload.
    assert fake_drive.token not in json.dumps(pushes[0])
    assert all("authorization" not in {k.lower() for k in p["headers"]} for p in pushes)
    pieces = fake_drive.uploads()
    assert pieces and all("authorization" not in r["headers"] for r in pieces)
    assert all(r["headers"].get("x-goog-user-project") == fake_drive.project for r in pieces)
    assert all(r["headers"].get("x-goog-user-project") == fake_drive.project
               for r in fake_drive.requests if r["path"] != "/token")


def test_a_large_archive_goes_up_in_pieces_the_session_acknowledges(
    isolated_home, uv_project: Path, fake_drive, monkeypatch
) -> None:
    from letify.store.backends import drive

    monkeypatch.setattr(drive, "UPLOAD_PIECE", 256 * 1024)
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    runtime = provider.start(remote_instance(provider), Env(), name="lab-1")
    try:
        assert runtime.env_source == "sync"
    finally:
        runtime.shutdown()
    (archive,) = fake_drive.blobs().values()
    assert len(archive) > 256 * 1024
    assert len(fake_drive.uploads()) == -(-len(archive) // (256 * 1024))


def test_a_later_session_restores_from_drive_on_the_runtime(
    isolated_home, uv_project: Path, fake_drive, monkeypatch
) -> None:
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    env = Env()
    first = provider.start(remote_instance(provider), env, name="lab-1")
    try:
        assert first.env_source == "sync"
    finally:
        first.shutdown()
    shutil.rmtree(Path(bootstrap.DEFAULT_WORKSPACE_ROOT))
    forbid_client_holding_the_archive(monkeypatch)
    replacement = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    second = replacement.start(remote_instance(replacement), env, name="lab-2")
    try:
        assert second.env_source == "archive"
    finally:
        second.shutdown()
    (digest,) = fake_drive.blobs()
    blob_id = next(f["id"] for f in fake_drive.files.values() if f["name"] == digest)
    pulled = [r for r in fake_drive.downloads() if r["path"].endswith("/" + blob_id)]
    assert len(pulled) == 1
    assert pulled[0]["headers"]["authorization"] == f"Bearer {fake_drive.token}"
    assert pulled[0]["headers"]["x-goog-user-project"] == fake_drive.project
    assert digest


def test_a_drive_failure_during_pack_leaves_nothing_cached_and_says_so(
    isolated_home, uv_project: Path, fake_drive, capsys, monkeypatch
) -> None:
    forbid_client_holding_the_archive(monkeypatch)
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    fake_drive.fail_with = 500
    runtime = provider.start(remote_instance(provider), Env(), name="lab-1")
    try:
        assert runtime.env_source == "sync"
    finally:
        runtime.shutdown()
    shown = capsys.readouterr()
    text = shown.out + shown.err
    assert "letify: lab-1: not using the Drive store (" in text
    # The pack comes after the sync, so the session says what is left undone, not that it
    # will sync. Spec "Materializing into a runtime".
    assert "); not caching the environment" in text
    assert fake_drive.blobs() == {}


def test_a_drive_failure_during_restore_makes_the_session_sync_and_say_so(
    isolated_home, uv_project: Path, fake_drive, capsys
) -> None:
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    env = Env()
    first = provider.start(remote_instance(provider), env, name="lab-1")
    first.shutdown()
    shutil.rmtree(Path(bootstrap.DEFAULT_WORKSPACE_ROOT))
    capsys.readouterr()
    fake_drive.fail_with = 500
    replacement = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    second = replacement.start(remote_instance(replacement), env, name="lab-2")
    try:
        assert second.env_source == "sync"
    finally:
        second.shutdown()
    shown = capsys.readouterr()
    assert "letify: lab-2: not using the Drive store (" in shown.out + shown.err
    assert "); syncing" in shown.out + shown.err


# -- the archive format, the gate and the restore record -----------------------


def first_session(provider: Any, env: Env, name: str = "lab-1") -> Runtime:
    runtime = provider.start(remote_instance(provider), env, name=name)
    runtime.shutdown()
    return runtime


def test_the_archive_on_drive_is_an_uncompressed_tar_and_the_client_store_still_gzip(
    isolated_home, uv_project: Path, fake_drive
) -> None:
    import io
    import tarfile

    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    first_session(provider, Env())
    (archive,) = fake_drive.blobs().values()
    assert archive[:2] != b"\x1f\x8b"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as packed:
        assert packed.getmembers()
    # Without Drive, the same environment is cached in the client's filesystem store as gzip.
    declare_drive(fake_drive, drive_login=None, drive_project=None)
    plain = provider_of(PreparingLocal, "plain", persistent=False)
    first_session(plain, Env(), name="plain-1")
    store = environment_volume(plain).store
    blobs = [p for p in Path(store.backend.root).rglob("*") if p.is_file() and p.suffix == ""]
    assert any(p.read_bytes()[:2] == b"\x1f\x8b" for p in blobs)


def test_a_build_faster_than_the_estimated_restore_leaves_nothing_on_drive(
    isolated_home, uv_project: Path, fake_drive, capsys, monkeypatch
) -> None:
    from letify.runtime import session

    # A restore estimated at hours for any directory, against a build of seconds.
    monkeypatch.setattr(session, "DRIVE_DOWNLOAD_BPS", 1.0)
    monkeypatch.setattr(session, "EXTRACT_BPS", 1.0)
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    first_session(provider, Env())
    assert fake_drive.blobs() == {}
    assert fake_drive.refs() == {}
    shown = capsys.readouterr()
    assert "not caching the environment" in shown.out + shown.err


def test_the_gate_estimates_a_restore_as_size_over_the_download_and_extraction_rates() -> None:
    from letify.runtime import session

    assert session.DRIVE_DOWNLOAD_BPS == 176 * 1024**2
    assert session.EXTRACT_BPS == 170_000_000


def test_a_restore_records_its_seconds_beside_the_build(
    isolated_home, uv_project: Path, fake_drive
) -> None:
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    env = Env()
    first_session(provider, env)
    (build_name,) = [n for n in fake_drive.refs() if n.endswith(".build")]
    assert len(fake_drive.refs()[build_name].split()) == 2
    shutil.rmtree(Path(bootstrap.DEFAULT_WORKSPACE_ROOT))
    replacement = provider_of(PreparingLocal, "lab", persistent=False)
    second = replacement.start(remote_instance(replacement), env, name="lab-2")
    try:
        assert second.env_source == "archive"
    finally:
        second.shutdown()
    seconds, size, restore = fake_drive.refs()[build_name].split()
    assert float(seconds) > 0 and int(size) > 0
    assert float(restore) >= 0


def test_a_recorded_restore_slower_than_the_build_makes_the_next_session_sync(
    isolated_home, uv_project: Path, fake_drive, capsys
) -> None:
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    env = Env()
    first_session(provider, env)
    (build_name,) = [n for n in fake_drive.refs() if n.endswith(".build")]
    (build_id,) = [f["id"] for f in fake_drive.files.values() if f["name"] == "ref " + build_name]
    fake_drive.files[build_id]["content"] = b"10.0 123456 99.0"
    shutil.rmtree(Path(bootstrap.DEFAULT_WORKSPACE_ROOT))
    capsys.readouterr()
    replacement = provider_of(PreparingLocal, "lab", persistent=False)
    second = replacement.start(remote_instance(replacement), env, name="lab-2")
    try:
        assert second.env_source == "sync"
    finally:
        second.shutdown()
    shown = capsys.readouterr()
    assert (
        "letify: lab-2: syncing rather than restoring from Drive, because a restore took "
        "99 s against a 10 s build"
    ) in shown.out + shown.err


def test_a_record_with_no_restore_seconds_yet_restores(
    isolated_home, uv_project: Path, fake_drive
) -> None:
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    env = Env()
    first_session(provider, env)
    shutil.rmtree(Path(bootstrap.DEFAULT_WORKSPACE_ROOT))
    replacement = provider_of(PreparingLocal, "lab", persistent=False)
    second = replacement.start(remote_instance(replacement), env, name="lab-2")
    try:
        assert second.env_source == "archive"
    finally:
        second.shutdown()


# -- the pack runs beside the session's calls -----------------------------------


def test_the_first_call_runs_while_the_pack_is_still_uploading(
    isolated_home, uv_project: Path, fake_drive, capsys
) -> None:
    import threading

    hold = threading.Event()
    fake_drive.hold_uploads = hold
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    runtime = provider.start(remote_instance(provider), Env(), name="lab-1")
    try:
        value, _logs = runtime.call(lambda: 7, (), {})
        assert value == 7
        # The upload is held, so the pack has not ended and nothing is recorded yet.
        assert fake_drive.refs() == {}
        assert "cached the environment" not in "".join(capsys.readouterr())
        hold.set()
    finally:
        hold.set()
        runtime.shutdown()
    assert fake_drive.refs()


def test_the_pack_says_it_cached_the_environment_when_the_upload_completes(
    isolated_home, uv_project: Path, fake_drive, capsys
) -> None:
    import re

    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    first_session(provider, Env())
    shown = capsys.readouterr()
    text = shown.out + shown.err
    assert re.search(
        r"letify: lab-1: cached the environment, [\d.]+ GB in [\d.]+ s \([\d.]+ MiB/s\)", text
    )
    (build_name,) = [n for n in fake_drive.refs() if n.endswith(".build")]
    assert len(fake_drive.refs()[build_name].split()) == 2
    assert len(fake_drive.blobs()) == 1


def test_the_staging_directory_is_removed_when_the_pack_ends(
    isolated_home, uv_project: Path, fake_drive
) -> None:
    from letify.runtime import bootstrap

    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    first_session(provider, Env())
    assert not list(Path(bootstrap.DEFAULT_WORKSPACE_ROOT).expanduser().rglob(".letify-pack"))


def test_a_session_closed_during_the_pack_abandons_it_and_leaves_no_ref(
    isolated_home, uv_project: Path, fake_drive, capsys, monkeypatch
) -> None:
    import threading

    from letify.runtime import session

    monkeypatch.setattr(session, "PACK_CLOSE_WAIT_S", 0.5)
    hold = threading.Event()
    fake_drive.hold_uploads = hold
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    runtime = provider.start(remote_instance(provider), Env(), name="lab-1")
    try:
        runtime.shutdown()
    finally:
        hold.set()
    shown = capsys.readouterr()
    text = shown.out + shown.err
    assert "letify: lab-1: not caching the environment, because the session closed " in text
    assert " s into the pack" in text
    assert "cached the environment" not in text
    assert fake_drive.refs() == {}


def test_a_failed_upload_is_reported_by_the_session_that_observes_it(
    isolated_home, uv_project: Path, fake_drive, capsys
) -> None:
    fake_drive.fail_with = 500
    provider = provider_of(PreparingLocal, "lab", **drive_options(fake_drive))
    first_session(provider, Env())
    shown = capsys.readouterr()
    assert "); not caching the environment" in shown.out + shown.err
    assert fake_drive.refs() == {}


def test_the_gate_packs_only_when_the_build_took_twice_the_estimated_restore() -> None:
    from types import SimpleNamespace

    from letify.runtime import session

    said: list[str] = []
    restore = 1_000_000_000 / session.DRIVE_DOWNLOAD_BPS + 1_000_000_000 / session.EXTRACT_BPS

    def judged(build_s: float) -> bool:
        stand_in = SimpleNamespace(
            _tree_size=lambda root: 1_000_000_000, env_build_s=build_s, _say=said.append
        )
        return session.Runtime._worth_packing_to_drive(stand_in, "/root")  # type: ignore[arg-type]

    assert judged(restore * 1.5) is False
    assert "not caching the environment" in said[-1]
    assert judged(restore * 2) is True
    assert judged(restore * 3) is True
