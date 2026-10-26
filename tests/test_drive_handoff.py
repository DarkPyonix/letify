"""Named handoffs on the Drive store: ``letify.stash`` and ``letify.unstash``.

Spec section pinned here: "Named handoffs on the Drive store". Google Drive needs an
account, so the endpoint is the stand-in on loopback in ``conftest.py``. Everything on this
side of it is the real code, including the worker that moves the bytes when the call is
made inside a shipped function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_drive_store import declare_drive

import letify
from letify.errors import ConfigError, RuntimeFailure
from letify.store.backends import drive as drive_module

LOCAL = '[a]\nkind = "local"\n'


def make_tree(root: Path) -> Path:
    """A small checkpoint directory with a nested file."""
    (root / "sub").mkdir(parents=True)
    (root / "weights.bin").write_bytes(b"w" * 5000)
    (root / "sub" / "step.txt").write_text("step 7")
    return root


def read_tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def blob_names(server: Any) -> list[str]:
    return sorted(server.blobs())


@pytest.fixture
def worker_sees_this_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the Local worker import the letify under test, not another installed checkout."""
    root = str(Path(letify.__file__).resolve().parent.parent)
    monkeypatch.setenv("PYTHONPATH", root)


# -- the local process ---------------------------------------------------------


def test_a_directory_stashed_in_the_local_process_is_fetched_back_whole(
    fake_drive, tmp_path
) -> None:
    """Spec: stash packs the directory under its base name, fetch lands it at the path."""
    declare_drive(fake_drive)
    source = make_tree(tmp_path / "ckpt")
    digest = letify.stash("run-1", source)
    assert fake_drive.refs() == {"stash/run-1": digest}
    assert blob_names(fake_drive) == [digest]
    target = tmp_path / "elsewhere" / "restored"
    got = letify.unstash("run-1", target)
    assert got == target
    assert isinstance(got, Path)
    assert read_tree(target) == read_tree(source)


def test_a_file_stashed_in_the_local_process_is_fetched_back_over_an_existing_one(
    fake_drive, tmp_path
) -> None:
    """Spec: the tar's one entry replaces what the path held."""
    declare_drive(fake_drive)
    source = tmp_path / "model.pt"
    source.write_bytes(b"abc" * 1000)
    letify.stash("model", source)
    target = tmp_path / "copy.pt"
    target.write_bytes(b"old")
    letify.unstash("model", target)
    assert target.read_bytes() == b"abc" * 1000


def test_fetching_a_directory_replaces_what_the_path_held(fake_drive, tmp_path) -> None:
    declare_drive(fake_drive)
    letify.stash("ckpt", make_tree(tmp_path / "ckpt"))
    target = tmp_path / "restored"
    target.mkdir()
    (target / "stale.txt").write_text("stale")
    letify.unstash("ckpt", target)
    assert "stale.txt" not in read_tree(target)
    assert "weights.bin" in read_tree(target)


def test_the_digest_returned_is_the_blob_name_and_the_upload_is_an_uncompressed_tar(
    fake_drive, tmp_path
) -> None:
    import io
    import tarfile

    declare_drive(fake_drive)
    source = tmp_path / "one.txt"
    source.write_text("hello")
    digest = letify.stash("one", source)
    payload = fake_drive.blobs()[digest]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert archive.getnames() == ["one.txt"]
    assert len(digest) >= 32


def test_fetching_an_unknown_name_returns_none_and_changes_nothing(
    fake_drive, tmp_path
) -> None:
    declare_drive(fake_drive)
    target = tmp_path / "keep"
    target.mkdir()
    (target / "a.txt").write_text("kept")
    assert letify.unstash("never-stashed", target) is None
    assert read_tree(target) == {"a.txt": b"kept"}
    assert fake_drive.downloads() == []


def test_stashing_a_name_again_deletes_the_blob_it_named_before(fake_drive, tmp_path) -> None:
    declare_drive(fake_drive)
    source = tmp_path / "state.txt"
    source.write_text("one")
    first = letify.stash("state", source)
    source.write_text("two")
    second = letify.stash("state", source)
    assert first != second
    assert fake_drive.refs() == {"stash/state": second}
    assert blob_names(fake_drive) == [second]


def test_a_replaced_blob_stays_when_another_stash_ref_names_it(fake_drive, tmp_path) -> None:
    declare_drive(fake_drive)
    source = tmp_path / "state.txt"
    source.write_text("one")
    first = letify.stash("state", source)
    assert letify.stash("copy", source) == first
    source.write_text("two")
    second = letify.stash("state", source)
    assert fake_drive.refs() == {"stash/state": second, "stash/copy": first}
    assert blob_names(fake_drive) == sorted([first, second])


def test_restashing_a_name_downloads_no_other_ref(fake_drive, tmp_path) -> None:
    """Spec: whether another ref names the replaced blob comes from one listing."""
    declare_drive(fake_drive)
    source = tmp_path / "state.txt"
    for index in range(5):
        source.write_text(f"other {index}")
        letify.stash(f"other-{index}", source)
    source.write_text("one")
    first = letify.stash("state", source)
    fake_drive.requests.clear()
    source.write_text("two")
    second = letify.stash("state", source)
    # Only the ref being moved is read; the five others are judged from the listing.
    assert len(fake_drive.downloads()) == 1
    assert first not in fake_drive.blobs()
    assert fake_drive.refs()["stash/state"] == second


def test_stashing_the_same_digest_sends_nothing(fake_drive, tmp_path) -> None:
    declare_drive(fake_drive)
    source = make_tree(tmp_path / "ckpt")
    first = letify.stash("ckpt", source)
    sent = len(fake_drive.uploads())
    assert sent > 0
    posts = [r for r in fake_drive.requests if r["method"] == "POST"]
    assert letify.stash("ckpt", source) == first
    assert len(fake_drive.uploads()) == sent
    assert [r for r in fake_drive.requests if r["method"] == "POST"] == posts
    # A second name for bytes Drive already holds sends nothing either.
    assert letify.stash("other", source) == first
    assert len(fake_drive.uploads()) == sent
    assert fake_drive.refs() == {"stash/ckpt": first, "stash/other": first}


@pytest.mark.parametrize(
    "name",
    ["x" * 129, "", ".hidden", "has space", "a/b", "café", "a:b"],
)
def test_an_invalid_name_is_refused_naming_the_rule(fake_drive, tmp_path, name) -> None:
    declare_drive(fake_drive)
    source = tmp_path / "f.txt"
    source.write_text("x")
    with pytest.raises(ValueError, match="1 to 128"):
        letify.stash(name, source)
    with pytest.raises(ValueError, match="1 to 128"):
        letify.unstash(name, tmp_path / "out")
    assert fake_drive.requests == []


@pytest.mark.parametrize("name", ["a", "x" * 128, "run_1.final-2", "A9"])
def test_a_valid_name_is_accepted(fake_drive, tmp_path, name) -> None:
    declare_drive(fake_drive)
    source = tmp_path / "f.txt"
    source.write_text("x")
    letify.stash(name, source)
    assert list(fake_drive.refs()) == [f"stash/{name}"]


def test_a_machine_with_no_drive_refuses_naming_both_settings(isolated_home, tmp_path) -> None:
    source = tmp_path / "f.txt"
    source.write_text("x")
    for call in (
        lambda: letify.stash("n", source),
        lambda: letify.unstash("n", tmp_path / "out"),
    ):
        with pytest.raises(ConfigError) as raised:
            call()
        assert "drive_login" in str(raised.value)
        assert "drive_project" in str(raised.value)
    assert list(tmp_path.glob("out*")) == []


def test_stashing_a_path_that_does_not_exist_is_refused_before_drive_is_asked(
    fake_drive, tmp_path
) -> None:
    declare_drive(fake_drive)
    with pytest.raises(FileNotFoundError):
        letify.stash("n", tmp_path / "missing")
    assert fake_drive.requests == []


def test_a_blob_whose_digest_does_not_match_is_refused_and_the_path_is_untouched(
    fake_drive, tmp_path
) -> None:
    declare_drive(fake_drive)
    source = tmp_path / "f.txt"
    source.write_text("good")
    digest = letify.stash("n", source)
    for entry in fake_drive.files.values():
        if entry["name"] == digest:
            entry["content"] = entry["content"][:-1] + b"X"
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("kept")
    with pytest.raises(RuntimeFailure, match="digest"):
        letify.unstash("n", target)
    assert read_tree(target) == {"keep.txt": b"kept"}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_a_drive_failure_raises_naming_the_status_and_the_handoff_name(
    fake_drive, tmp_path
) -> None:
    declare_drive(fake_drive)
    source = tmp_path / "f.txt"
    source.write_text("x")
    fake_drive.fail_with = 500
    with pytest.raises(RuntimeFailure) as raised:
        letify.stash("my-ckpt", source)
    assert "500" in str(raised.value)
    assert "my-ckpt" in str(raised.value)
    with pytest.raises(RuntimeFailure) as raised:
        letify.unstash("my-ckpt", tmp_path / "out")
    assert "500" in str(raised.value)
    assert "my-ckpt" in str(raised.value)


def test_fetch_keeps_its_one_argument_tensor_form() -> None:
    """``letify.fetch(tensor)`` predates the handoff and is not the handoff's read."""
    import inspect

    assert list(inspect.signature(letify.fetch).parameters) == ["tensor"]
    assert "stash" in letify.__all__
    assert "unstash" in letify.__all__


# -- inside a shipped function ---------------------------------------------------


def refuse_client_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every route that would move a handoff's bytes through this process raise."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the handoff crossed the client")

    monkeypatch.setattr(drive_module.DriveBackend, "put", refuse)
    monkeypatch.setattr(drive_module.DriveBackend, "get", refuse)
    monkeypatch.setattr(drive_module, "upload_pieces", refuse)
    monkeypatch.setattr(drive_module, "download_to", refuse)


def test_a_function_stashes_and_a_later_call_fetches_with_the_bytes_going_worker_to_drive(
    fake_drive, launcher_from, tmp_path, monkeypatch, worker_sees_this_checkout
) -> None:
    """Spec: inside a shipped function the worker moves the bytes, the client grants."""
    declare_drive(fake_drive)
    let = launcher_from(LOCAL)
    refuse_client_bytes(monkeypatch)
    source = make_tree(tmp_path / "ckpt")
    target = tmp_path / "restored"

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def write_and_stash(path: str) -> str:
        import letify as inner

        return inner.stash("from-worker", path)

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def unstash_back(path: str) -> dict:
        from pathlib import Path

        import letify as inner

        got = inner.unstash("from-worker", path)
        absent = inner.unstash("nothing-here", path + ".none")
        files = {
            str(p.relative_to(path)): p.read_bytes()
            for p in sorted(Path(path).rglob("*"))
            if p.is_file()
        }
        return {"got": str(got), "absent": absent, "files": files}

    with let.keep_alive():
        digest = write_and_stash(str(source))
        assert fake_drive.refs() == {"stash/from-worker": digest}
        assert blob_names(fake_drive) == [digest]
        assert len(fake_drive.blobs()[digest]) > 5000
        puts = fake_drive.uploads()
        assert puts
        # The session URL authorizes its own upload: no access token left the client.
        assert all("authorization" not in r["headers"] for r in puts)
        result = unstash_back(str(target))
    assert result["got"] == str(target)
    assert result["absent"] is None
    assert result["files"] == read_tree(source)
    assert read_tree(target) == read_tree(source)


def test_a_function_restashing_a_name_replaces_the_blob_through_the_client(
    fake_drive, launcher_from, tmp_path, worker_sees_this_checkout
) -> None:
    declare_drive(fake_drive)
    let = launcher_from(LOCAL)
    source = tmp_path / "state.txt"

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def stash_text(path: str, text: str | None) -> str:
        from pathlib import Path

        import letify as inner

        # The tar records the modification time, so unchanged bytes need an untouched file.
        if text is not None:
            Path(path).write_text(text)
        return inner.stash("state", path)

    with let.keep_alive():
        first = stash_text(str(source), "one")
        sent = len(fake_drive.uploads())
        assert stash_text(str(source), None) == first
        assert len(fake_drive.uploads()) == sent
        second = stash_text(str(source), "two")
    assert fake_drive.refs() == {"stash/state": second}
    assert blob_names(fake_drive) == [second]


def test_a_function_restashing_a_name_returns_before_the_replaced_blob_is_deleted(
    fake_drive, launcher_from, tmp_path, worker_sees_this_checkout
) -> None:
    """Spec: the grant answers once the ref is written, and the deletion follows."""
    import threading
    import time

    declare_drive(fake_drive)
    let = launcher_from(LOCAL)
    source = tmp_path / "state.txt"

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def stash_text(path: str, text: str) -> str:
        from pathlib import Path

        import letify as inner

        Path(path).write_text(text)
        return inner.stash("state", path)

    with let.keep_alive():
        first = stash_text(str(source), "one")
        fake_drive.hold_deletes = threading.Event()
        try:
            second = stash_text(str(source), "two")
            assert fake_drive.refs() == {"stash/state": second}
            assert first in fake_drive.blobs()
        finally:
            fake_drive.hold_deletes.set()
        deadline = time.monotonic() + 10
        while first in fake_drive.blobs() and time.monotonic() < deadline:
            time.sleep(0.05)
    assert blob_names(fake_drive) == [second]


def test_a_function_on_a_machine_with_no_drive_fails_naming_both_settings(
    isolated_home, launcher_from, tmp_path, worker_sees_this_checkout
) -> None:
    let = launcher_from(LOCAL)
    source = tmp_path / "f.txt"
    source.write_text("x")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def stash_it(path: str) -> str:
        import letify as inner

        try:
            return inner.stash("n", path)
        except inner.ConfigError as exc:
            return "ConfigError: " + str(exc)

    with let.keep_alive():
        answer = stash_it(str(source))
    assert answer.startswith("ConfigError")
    assert "drive_login" in answer and "drive_project" in answer


def test_a_function_that_fetches_a_corrupt_blob_gets_a_runtime_failure_and_keeps_its_path(
    fake_drive, launcher_from, tmp_path, worker_sees_this_checkout
) -> None:
    declare_drive(fake_drive)
    source = tmp_path / "f.txt"
    source.write_text("good")
    digest = letify.stash("n", source)
    for entry in fake_drive.files.values():
        if entry["name"] == digest:
            entry["content"] = entry["content"][:-1] + b"X"
    let = launcher_from(LOCAL)
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("kept")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def unstash_it(path: str) -> str:
        import letify as inner

        try:
            inner.unstash("n", path)
        except inner.RuntimeFailure as exc:
            return "RuntimeFailure: " + str(exc)
        return "fetched"

    with let.keep_alive():
        answer = unstash_it(str(target))
    assert answer.startswith("RuntimeFailure")
    assert read_tree(target) == {"keep.txt": b"kept"}


# -- a one-shot channel ---------------------------------------------------------------


def test_stash_and_unstash_on_a_one_shot_channel_raise_unsupported_mode_naming_the_provider(
    fake_drive, tmp_path, worker_sees_this_checkout
) -> None:
    """Spec: a one-shot body has no stream back, so it never reads the runtime's own config."""
    from conftest import local_one_shot_runner

    from letify.runtime.channel import OneShotChannel

    # The machine declares Drive, so reaching the local-process path would succeed.
    declare_drive(fake_drive)
    source = tmp_path / "f.txt"
    source.write_text("x")
    channel = OneShotChannel(local_one_shot_runner(), name="rt-1", provider="colab_a")

    def body(path: str) -> list:
        import letify as inner

        seen = []
        for call in (lambda: inner.stash("n", path), lambda: inner.unstash("n", path + ".out")):
            try:
                call()
            except inner.UnsupportedMode as exc:
                seen.append(str(exc))
        return seen

    value, _logs = channel.call(body, (str(source),), {})
    assert len(value) == 2
    assert all("colab_a" in message for message in value)
    assert fake_drive.requests == []


# -- a child process of the body -------------------------------------------------------


def test_stash_and_unstash_in_a_child_process_of_the_body_say_to_call_from_the_main_process(
    fake_drive, launcher_from, tmp_path, worker_sees_this_checkout
) -> None:
    """Spec: a child the body starts has no grant hook, so it raises instead of reading config."""
    declare_drive(fake_drive)
    let = launcher_from(LOCAL)
    source = tmp_path / "f.txt"
    source.write_text("x")

    @let.function(device=let.providers.a.CPU, host=letify.remote)
    def from_children(path: str) -> dict:
        import multiprocessing
        import os
        import subprocess
        import sys
        from pathlib import Path

        import letify as inner

        code = (
            "import sys, letify\n"
            "for call in (lambda: letify.stash('kid', sys.argv[1]),"
            " lambda: letify.unstash('kid', sys.argv[1] + '.out')):\n"
            "    try:\n"
            "        call()\n"
            "    except letify.UnsupportedMode as exc:\n"
            "        print('UM:' + str(exc))\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", code, path], capture_output=True, text=True, check=True
        )
        helper = Path(path).parent / "spawned_helper.py"
        helper.write_text(
            "import letify\n"
            "def unstash_it(queue, path):\n"
            "    try:\n"
            "        letify.unstash('spawned', path)\n"
            "    except letify.UnsupportedMode as exc:\n"
            "        queue.put(str(exc))\n"
            "    else:\n"
            "        queue.put('no error')\n"
        )
        sys.path.insert(0, str(helper.parent))
        old = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(helper.parent) + os.pathsep + old
        import spawned_helper

        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=spawned_helper.unstash_it, args=(queue, path + ".spawn"))
        process.start()
        spawned = queue.get(timeout=120)
        process.join()
        return {
            "subprocess": [x for x in run.stdout.splitlines() if x.startswith("UM:")],
            "spawn": spawned,
            "main_is_not_refused": inner.UnsupportedMode.__name__,
        }

    with let.keep_alive():
        answer = from_children(str(source))
    assert len(answer["subprocess"]) == 2
    for message in answer["subprocess"] + [answer["spawn"]]:
        assert "process letify runs the body in" in message
        assert "child process" in message
    assert "'kid'" in answer["subprocess"][0] and "'kid'" in answer["subprocess"][1]
    assert "'spawned'" in answer["spawn"]
    assert fake_drive.requests == []



def test_every_drive_grant_is_reported_and_a_failed_answer_names_its_error(monkeypatch, capsys) -> None:
    import time

    from letify.store import handoff

    asked = [
        {"id": 1, "kind": "stash", "name": "run-checkpoint"},
        {"id": 2, "kind": "stash", "name": "run-manifest"},
    ]

    class Connection:
        def take_grants(self):
            found, asked[:] = list(asked), []
            return found

    class Channel:
        connection = Connection()

        def request(self, message, timeout):
            if message["id"] == 2:
                raise TimeoutError("no reply within 300 s")

    monkeypatch.setattr(handoff, "grant", lambda ask: {"done": True})
    watcher = handoff.GrantWatcher(Channel(), poll_s=0.01)
    watcher.start()
    deadline = time.time() + 5
    while asked and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    watcher.close()
    err = capsys.readouterr().err
    assert "drive grant stash run-checkpoint: answered in" in err
    assert "drive grant stash run-manifest: the answer was not delivered after" in err
    assert "no reply within 300 s" in err
