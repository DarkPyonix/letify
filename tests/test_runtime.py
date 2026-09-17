"""Channels, live sessions and the pool that decides when they die.

Every channel here is a real one. The local provider starts a Python subprocess running
the same worker source, behind the same framed protocol, that an SSH or WebSocket
channel would, so these tests exercise the protocol and the worker rather than a stub.

Spec sections pinned here: "Channels", "Call protocol", "Failure and retry", "Sessions",
"Pooling", "Lifetime" and "Environment".
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest
from conftest import (
    FakeCompleted,
    LeasingLocal,
    PreparingLocal,
    local_one_shot_runner,
    provider_of,
)

import letify
from letify.declare.env import Env
from letify.declare.instance import Instance
from letify.errors import RuntimeFailure
from letify.protocol.worker import BOOTSTRAP
from letify.providers.local import Local
from letify.runtime import bootstrap, telemetry
from letify.runtime.channel import OneShotChannel, PersistentChannel
from letify.runtime.lease import GRACE, INTERVAL, Lease
from letify.runtime.pool import POLL_INTERVAL, RuntimePool
from letify.store.volume import Volume


@pytest.fixture
def channel():
    """A real worker behind a pipe, started the way every provider starts one."""
    opened = PersistentChannel([sys.executable, "-u", "-c", BOOTSTRAP], name="test-runtime")
    yield opened
    opened.close()


def remote_projects() -> Path:
    """Where PreparingLocal keeps its project directories, under the default workspace root."""
    return Path(bootstrap.DEFAULT_WORKSPACE_ROOT) / "project"


@pytest.fixture
def remote_cpu(let: letify.Launcher) -> Instance:
    # host="remote" throughout, because forwarding needs letify-core built.
    return let.providers.local.CPU._placed("remote")


# -- Spec: Channels, the persistent channel ------------------------------------


def returns_pid():
    """A body that reports which process ran it.

    Built by a factory so it is a nested function, which cloudpickle ships by value the
    way it ships a user's own def. A module level function would travel by reference and
    the worker has no test module to import.
    """

    def read_pid() -> int:
        import os

        return os.getpid()

    return read_pid


def kills_the_worker():
    """A body that takes the whole worker process down, as an out of memory kill does."""

    def die() -> None:
        import os

        os._exit(1)

    return die


def test_a_persistent_channel_keeps_one_process_across_requests(channel) -> None:
    # This is what makes the session cache, the blob table and files on disk survive
    # between calls.
    channel.start()
    first, _logs = channel.call(returns_pid(), (), {})
    second, _logs = channel.call(returns_pid(), (), {})
    assert first == second
    assert channel.alive is True


def test_starting_an_already_running_channel_does_not_start_a_second_worker(channel) -> None:
    channel.start()
    pid, _logs = channel.call(returns_pid(), (), {})
    channel.start()
    assert channel.call(returns_pid(), (), {})[0] == pid


def test_the_user_own_output_comes_back_separately_from_the_outcome(channel) -> None:
    # Replies and prints share one stream, so the caller gets them apart.
    def noisy() -> int:
        print("epoch 1")
        return 5

    value, logs = channel.call(noisy, (), {})
    assert value == 5
    assert "epoch 1" in logs


def test_a_channel_closes_the_worker_it_started(channel) -> None:
    channel.start()
    channel.close()
    assert channel.alive is False
    # Closing what is already closed is part of an ordinary teardown path.
    assert channel.close() is None


def test_a_command_that_cannot_be_run_names_the_command() -> None:
    opened = PersistentChannel(["letify-no-such-program"], name="broken")
    with pytest.raises(RuntimeFailure, match="could not start the worker") as caught:
        opened.start()
    assert caught.value.command == "letify-no-such-program"


def test_a_worker_that_exits_before_it_is_ready_fails_at_start() -> None:
    # A bad interpreter or a failed environment install looks exactly like this, and it
    # has to fail here rather than on the first call.
    # Reads the worker source the way the bootstrap stub does, then exits without ever
    # announcing itself.
    stub = "import sys; n = int(sys.stdin.readline()); sys.stdin.read(n); print('no uv')"
    opened = PersistentChannel([sys.executable, "-u", "-c", stub], name="early-exit")
    with pytest.raises(RuntimeFailure, match="exited before it was ready"):
        opened.start()


def test_a_worker_that_is_already_gone_is_reported_lost(channel) -> None:
    channel.start()
    channel._process.kill()
    channel._process.wait(timeout=30)
    with pytest.raises(letify.RuntimeLost, match="exited with code"):
        channel.call(returns_pid(), (), {})

    # close() kills the process but leaves its pipes to the garbage collector, and
    # finalizing the standard input of a killed process raises on Windows. Closing them
    # here keeps that out of the teardown.
    process, channel._process = channel._process, None
    for pipe in (process.stdin, process.stdout, process.stderr):
        try:
            pipe.close()
        except OSError:
            pass


def test_a_worker_that_dies_mid_call_is_a_protocol_error(channel) -> None:
    # An out of memory kill or a preempted session produces no reply at all, which is
    # not a user code failure and has to be told apart from one.
    with pytest.raises(letify.ProtocolError) as caught:
        channel.call(kills_the_worker(), (), {})
    assert "died before it finished" in str(caught.value)


def test_a_call_that_outlives_its_timeout_is_a_failure(channel) -> None:
    def slow() -> None:
        import time

        time.sleep(3)

    with pytest.raises(RuntimeFailure, match=r"exceeded 0\.3s"):
        channel.call(slow, (), {}, timeout=0.3)


def test_a_body_can_start_forked_processes_while_the_worker_reads_frames(channel) -> None:
    # multiprocessing closes sys.stdin in a forked child, which the frame reader thread is
    # blocked reading at that moment. A DataLoader with num_workers > 0 forks the same way.
    def forks() -> int:
        import multiprocessing

        context = multiprocessing.get_context("fork")
        results = context.Queue()
        children = [context.Process(target=results.put, args=(i,)) for i in range(2)]
        for child in children:
            child.start()
        total = sum(results.get(timeout=20) for _ in children)
        for child in children:
            child.join(20)
        return total

    value, _logs = channel.call(forks, (), {}, timeout=60)
    assert value == 1


@pytest.mark.skipif(sys.platform != "linux", reason="parent death signal is Linux only")
def test_no_forked_child_outlives_a_timed_out_call(channel, tmp_path) -> None:
    marker = tmp_path / "child.pid"

    def forks_and_hangs() -> None:
        import os
        import time

        pid = os.fork()
        if pid == 0:
            time.sleep(120)
            os._exit(0)
        with open(str(marker), "w") as handle:
            handle.write(str(pid))
        time.sleep(120)

    with pytest.raises(RuntimeFailure, match="exceeded"):
        channel.call(forks_and_hangs, (), {}, timeout=3)
    child = int(marker.read_text())
    import time

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        status = Path(f"/proc/{child}/stat")
        if status.exists() and status.read_text().split()[2] == "Z":
            break
        time.sleep(0.1)
    else:
        os.kill(child, 9)
        pytest.fail(f"forked child {child} outlived the timed-out call")


def test_an_operation_the_worker_does_not_know_is_reported_by_name(channel) -> None:
    channel.start()
    with pytest.raises(letify.RemoteError, match="unknown op 'nonsense'"):
        channel.request({"op": "nonsense"})


# -- Spec: Channels, the one-shot channel --------------------------------------


def test_a_one_shot_channel_starts_a_fresh_process_for_every_call() -> None:
    # Nothing persists, which is the whole difference from a persistent channel.
    opened = OneShotChannel(local_one_shot_runner(), name="one-shot")
    assert opened.start() is None
    first, _logs = opened.call(returns_pid(), (), {})
    second, _logs = opened.call(returns_pid(), (), {})
    assert first != second
    assert opened.close() is None
    assert opened.persistent is False


def test_a_one_shot_channel_holds_no_blobs_to_report() -> None:
    # The argument addressing handshake still has to get an answer, and the honest
    # answer is that a fresh process holds nothing.
    opened = OneShotChannel(local_one_shot_runner(), name="one-shot")
    assert opened.request({"op": "have", "digests": ["abc"]}) == ([], "")


def test_a_one_shot_channel_runs_plain_source() -> None:
    seen: list[str] = []
    opened = OneShotChannel(local_one_shot_runner(seen), name="one-shot")
    assert opened.request({"op": "exec", "source": "print('ready')"}) == (None, "")
    assert seen == ["print('ready')"]


def test_a_lease_on_a_one_shot_channel_is_armed_inside_the_command() -> None:
    # Mostly symbolic there, because the process ends with the command and the
    # provider's own session timeout is what bounds the cost.
    seen: list[str] = []
    opened = OneShotChannel(local_one_shot_runner(seen), name="one-shot")
    opened.request({"op": "lease", "grace": 300.0})
    assert "lease armed" in seen[0]
    assert "300.0" in seen[0]


@pytest.mark.parametrize("op", ["stat", "put_blob", "get_file", "pack_dir"])
def test_a_one_shot_channel_refuses_what_needs_a_living_process(op: str) -> None:
    opened = OneShotChannel(local_one_shot_runner(), name="one-shot")
    with pytest.raises(RuntimeFailure, match=f"cannot serve {op!r}"):
        opened.request({"op": op})


def test_a_one_shot_channel_carries_a_call_all_the_way_through() -> None:
    # The driver script, the markers and the decoder, over a runner that only runs a
    # command and collects its output.
    def add(a: int, b: int) -> int:
        return a + b

    opened = OneShotChannel(local_one_shot_runner(), name="one-shot")
    value, logs = opened.call(add, (1,), {"b": 2})
    assert value == 3
    assert logs.strip() == ""


# -- Spec: Sessions ------------------------------------------------------------


def test_a_runtime_opens_its_channel_and_reports_itself_ready(let, remote_cpu, live) -> None:
    runtime = live(let, remote_cpu)
    assert runtime.ready is True
    assert runtime.persistent_channel is True
    assert runtime.idle_for < 60
    assert runtime.stat()["pid"]
    let.pool.shutdown()
    assert runtime.ready is False


def test_the_pool_key_is_the_instance_key_and_the_environment_key(let, remote_cpu, live) -> None:
    env = Env()
    runtime = live(let, remote_cpu, env)
    assert runtime.key == f"{remote_cpu.key}|{env.key}"
    let.pool.shutdown()


def test_a_request_on_a_runtime_whose_channel_is_shut_says_so(let, remote_cpu, live) -> None:
    runtime = live(let, remote_cpu)
    runtime.shutdown()
    with pytest.raises(RuntimeFailure, match="the channel is not open"):
        runtime.stat()
    with pytest.raises(RuntimeFailure, match="the channel is not open"):
        runtime.call(returns_pid(), (), {})


def test_installation_is_skipped_where_the_machine_already_runs_in_the_environment(let) -> None:
    # The local provider is the case the spec names, so booting one must not try to
    # install anything.
    assert let.providers.local.remote_env is False


def test_a_directory_inside_a_runtime_can_be_packed_in_one_payload(let, remote_cpu, tmp_path, live):
    # One archive rather than one transfer per file is where the speedup is.
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"x" * 64)
    runtime = live(let, remote_cpu)
    payload, digest = runtime.pack_dir(str(source))
    assert payload.startswith(b"\x1f\x8b")
    assert digest
    let.pool.shutdown()


def test_a_synced_environment_is_archived_and_the_next_session_restores_it_instead_of_syncing(
    uv_project: Path, tmp_path: Path
) -> None:
    # Spec "Materializing into a runtime": the first session that syncs packs its project
    # directory into its first volume, and a later session with the same key and platform
    # unpacks that archive instead of running uv sync. Spec "Volumes on a persistent
    # runtime" limits this to an ephemeral provider.
    provider = provider_of(PreparingLocal, "lab", persistent=False)
    volume = provider.volume(
        "cache", backend="filesystem", root=str(tmp_path / "store"), mount=str(tmp_path / "mount")
    )
    env = Env()
    instance = Instance(provider, gpu=None)._placed("remote")

    first = provider.start(instance, env, name="lab-1", volumes=(volume,))
    try:
        assert first.env_source == "sync"
        platform = first.platform
    finally:
        first.shutdown()
    assert volume.cached_env(env, platform)

    shutil.rmtree(remote_projects())
    second = provider.start(instance, env, name="lab-2", volumes=(volume,))
    try:
        assert second.env_source == "archive"
        executable, _version, _letify = second.call(reports_interpreter(), (), {})[0]
        assert Path(executable).parent == remote_projects() / env.key / ".venv" / "bin"
    finally:
        second.shutdown()


def test_a_persistent_provider_syncs_every_session_and_never_archives_the_environment(
    uv_project: Path, tmp_path: Path
) -> None:
    # Spec "Volumes on a persistent runtime": the .venv is already on the runtime's disk, so
    # no archive is packed into the volume or restored from it.
    provider = provider_of(PreparingLocal, "lab")
    assert provider.persistent
    volume = provider.volume(
        "cache", backend="filesystem", root=str(tmp_path / "store"), mount=str(tmp_path / "mount")
    )
    env = Env()
    instance = Instance(provider, gpu=None)._placed("remote")
    for name in ("lab-1", "lab-2"):
        runtime = provider.start(instance, env, name=name, volumes=(volume,))
        try:
            assert runtime.env_source == "sync"
            platform = runtime.platform
        finally:
            runtime.shutdown()
    assert volume.cached_env(env, platform) is None


def counting_puts(monkeypatch) -> list[str]:
    """Record the destination of every file written through the channel."""
    from letify.runtime.session import Runtime

    sent: list[str] = []
    original = Runtime.put_bytes

    def put_bytes(self, payload, path, **kwargs):
        sent.append(path)
        return original(self, payload, path, **kwargs)

    monkeypatch.setattr(Runtime, "put_bytes", put_bytes)
    return sent


def test_a_persistent_runtime_is_sent_only_the_files_its_volume_directory_lacks(
    tmp_path: Path, monkeypatch
) -> None:
    # Spec "Volumes on a persistent runtime": a later session holds what an earlier one
    # received, so an unchanged file is not sent and a changed one is.
    sent = counting_puts(monkeypatch)
    provider = provider_of(Local, "lab")
    mount = tmp_path / "mount"
    volume = provider.volume(
        "cache", backend="filesystem", root=str(tmp_path / "store"), mount=str(mount)
    )
    instance = Instance(provider, gpu=None)._placed("remote")
    env = Env(lock=str(tmp_path / "absent.lock"))
    first = volume.store.put_bytes(b"weights-1").digest
    second = volume.store.put_bytes(b"tokens").digest
    changed = volume.store.put_bytes(b"weights-2").digest

    def session(name: str, digests: dict[str, str]) -> None:
        runtime = provider.start(instance, env, name=name, volumes=(volume,))
        try:
            for path, digest in digests.items():
                remote = volume.materialize(runtime, digest, path=str(mount / path))
                assert Path(remote.path).read_bytes() == volume.store.get_bytes(digest)
        finally:
            runtime.shutdown()

    session("lab-1", {"a.bin": first, "b.bin": second})
    assert len(sent) == 2
    session("lab-2", {"a.bin": first, "b.bin": second})
    assert len(sent) == 2
    session("lab-3", {"a.bin": changed, "b.bin": second})
    assert sent[2:] == [str(mount / "a.bin")]


def test_a_file_changed_on_the_runtime_is_sent_again(tmp_path: Path, monkeypatch) -> None:
    # Spec "Volumes on a persistent runtime": the recorded size and time no longer match.
    sent = counting_puts(monkeypatch)
    provider = provider_of(Local, "lab")
    mount = tmp_path / "mount"
    volume = provider.volume(
        "cache", backend="filesystem", root=str(tmp_path / "store"), mount=str(mount)
    )
    instance = Instance(provider, gpu=None)._placed("remote")
    env = Env(lock=str(tmp_path / "absent.lock"))
    digest = volume.store.put_bytes(b"weights").digest
    for name in ("lab-1", "lab-2"):
        runtime = provider.start(instance, env, name=name, volumes=(volume,))
        try:
            volume.materialize(runtime, digest, path=str(mount / "a.bin"))
        finally:
            runtime.shutdown()
        (mount / "a.bin").write_bytes(b"edited on the runtime")
    assert len(sent) == 2
    assert (mount / "a.bin").read_bytes() == b"edited on the runtime"


def test_an_ephemeral_runtime_is_sent_every_file_each_session(tmp_path: Path, monkeypatch) -> None:
    # Spec "Volumes on a persistent runtime": an ephemeral disk is not trusted to keep them.
    sent = counting_puts(monkeypatch)
    provider = provider_of(Local, "lab", persistent=False)
    mount = tmp_path / "mount"
    volume = provider.volume(
        "cache", backend="filesystem", root=str(tmp_path / "store"), mount=str(mount)
    )
    instance = Instance(provider, gpu=None)._placed("remote")
    env = Env(lock=str(tmp_path / "absent.lock"))
    digest = volume.store.put_bytes(b"weights").digest
    for name in ("lab-1", "lab-2"):
        runtime = provider.start(instance, env, name=name, volumes=(volume,))
        try:
            volume.materialize(runtime, digest, path=str(mount / "a.bin"))
        finally:
            runtime.shutdown()
    assert len(sent) == 2


def test_a_runtime_boots_its_channel_then_arms_its_lease(tmp_path: Path) -> None:
    # Spec "Sessions": open the channel, arm the lease, install the environment, attach
    # volumes. A session that could outlive this process gets the lease.
    provider = provider_of(LeasingLocal, "lab")
    instance = Instance(provider, gpu=None)._placed("remote")
    runtime = provider.start(instance, Env(lock=str(tmp_path / "absent.lock")), name="lab-1")
    try:
        assert runtime.ready is True
        assert runtime.lease is not None
        # The deadline is inside the session, so the worker exits on its own if this
        # process stops renewing.
        runtime.exec("assert _LEASE['armed']")
    finally:
        runtime.shutdown()
    assert runtime.lease is None


def test_a_boot_that_fails_stops_the_runtime_it_started(tmp_path: Path) -> None:
    # Spec "Sessions": a session started for a runtime does not outlive a failed boot. The
    # Kaggle run this pins was left alive by a boot that died in uv sync, and the next
    # start then found the notebook's one session taken.
    class FailingLocal(LeasingLocal):
        stopped: ClassVar[list[str]] = []

        def add_worker_pid(self, pid: int) -> None:
            raise letify.errors.RuntimeFailure("the boot fails after the channel is open")

        def stop(self, runtime) -> None:
            self.stopped.append(runtime.name)
            super().stop(runtime)

    provider = provider_of(FailingLocal, "lab")
    instance = Instance(provider, gpu=None)._placed("remote")
    with pytest.raises(letify.errors.RuntimeFailure, match="after the channel is open"):
        provider.start(instance, Env(lock=str(tmp_path / "absent.lock")), name="lab-1")
    assert provider.stopped == ["lab-1"]


def test_a_volume_with_no_cached_archive_is_passed_over(uv_project: Path, tmp_path: Path) -> None:
    # Spec "Blob granularity": the archive is keyed by the environment and the platform, so
    # a volume that does not hold this one is not the place to look.
    provider = provider_of(PreparingLocal, "lab", persistent=False)
    env = Env()
    instance = Instance(provider, gpu=None)._placed("remote")
    stocked = Volume(
        provider,
        "stocked",
        {"backend": "filesystem", "root": str(tmp_path / "store"), "mount": str(tmp_path / "m")},
    )
    provider.start(instance, env, name="lab-1", volumes=(stocked,)).shutdown()

    empty = Volume(
        provider,
        "empty",
        {"backend": "filesystem", "root": str(tmp_path / "empty-store"), "mount": str(tmp_path)},
    )
    shutil.rmtree(remote_projects())
    runtime = provider.start(instance, env, name="lab-2", volumes=(empty, stocked))
    try:
        assert runtime.env_source == "archive"
    finally:
        runtime.shutdown()


def test_a_local_file_can_be_written_into_a_runtime(let, remote_cpu, tmp_path, live) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(b"y" * 32)
    runtime = live(let, remote_cpu)
    remote = runtime.put_file(source, str(tmp_path / "inside" / "weights.bin"))
    assert Path(remote.path).read_bytes() == b"y" * 32
    let.pool.shutdown()


def test_an_argument_plain_pickle_cannot_carry_is_left_to_cloudpickle(let, remote_cpu) -> None:
    # Spec "Argument addressing" only externalizes what pickles. Anything else travels
    # with the call, which is what cloudpickle is there for.
    @let.function(device=remote_cpu, host="remote")
    def apply(fn: object, value: int) -> int:
        return fn(value)

    assert apply(fn=lambda v: v + 1, value=41) == 42


def test_a_worker_whose_pipe_is_gone_is_reported_lost(channel) -> None:
    channel.start()
    channel._process.stdin.close()
    with pytest.raises(letify.RuntimeLost, match="pipe is closed"):
        channel.request({"op": "stat"})


# -- Spec: Building the environment on a runtime -------------------------------

LOCAL_PYTHON = f"{sys.version_info[0]}.{sys.version_info[1]}"
OTHER_PYTHON = "3.11" if LOCAL_PYTHON != "3.11" else "3.12"


def reports_interpreter():
    """A body that reports the interpreter running it and where letify was imported from."""

    def body() -> tuple[str, str, str]:
        import sys

        import letify

        return sys.executable, f"{sys.version_info[0]}.{sys.version_info[1]}", letify.__file__

    return body


def remote_instance(provider) -> Instance:
    return Instance(provider, gpu=None)._placed("remote")


def test_a_default_env_is_synced_with_uv_and_the_worker_runs_from_the_project_venv(
    uv_project: Path,
) -> None:
    # A default Env() takes the same path as any other: uv sync into the project .venv,
    # then the worker moves to that .venv's Python, where letify is importable.
    provider = provider_of(PreparingLocal, "lab")
    env = Env()
    runtime = provider.start(remote_instance(provider), env, name="lab-1")
    try:
        executable, version, imported = runtime.call(reports_interpreter(), (), {})[0]
        venv = remote_projects() / env.key / ".venv"
        assert Path(executable).parent == venv / "bin"
        assert version == LOCAL_PYTHON
        assert Path(imported).is_relative_to(venv)
        assert runtime.env_source == "sync"
        assert runtime.channel.python_version == LOCAL_PYTHON
    finally:
        runtime.shutdown()


def test_a_persistent_provider_syncs_with_the_uv_cache_under_the_workspace_root(
    uv_project: Path,
) -> None:
    # Spec "uv cache": UV_CACHE_DIR is <workspace root>/uv-cache, on the filesystem of the
    # project .venv, so uv hard links the environment instead of copying it.
    provider = provider_of(PreparingLocal, "lab")
    assert provider.persistent
    env = Env()
    runtime = provider.start(remote_instance(provider), env, name="lab-1")
    try:
        cache = Path(bootstrap.DEFAULT_WORKSPACE_ROOT) / "uv-cache"
        assert cache.is_dir()
        venv = remote_projects() / env.key / ".venv"
        site = [p for p in venv.rglob("*.py") if "site-packages" in p.parts]
        # _virtualenv.py is written by uv venv itself, not installed from the cache.
        installed = [p for p in site if p.name != "_virtualenv.py"]
        assert installed and all(p.stat().st_nlink > 1 for p in installed)
    finally:
        runtime.shutdown()


def test_an_ephemeral_provider_keeps_the_default_uv_cache(uv_project: Path) -> None:
    # Spec "uv cache": an ephemeral runtime's disk goes with it, so nothing is moved.
    provider = provider_of(PreparingLocal, "lab", persistent=False)
    runtime = provider.start(remote_instance(provider), Env(), name="lab-1")
    try:
        assert runtime.env_source == "sync"
        assert not (Path(bootstrap.DEFAULT_WORKSPACE_ROOT) / "uv-cache").exists()
    finally:
        runtime.shutdown()


def test_an_env_variable_naming_the_uv_cache_wins_over_the_workspace_cache(
    uv_project: Path, tmp_path: Path
) -> None:
    # Spec "uv cache": an Env.vars entry naming UV_CACHE_DIR wins over the rule.
    chosen = tmp_path / "chosen-cache"
    provider = provider_of(PreparingLocal, "lab")
    runtime = provider.start(
        remote_instance(provider), Env().vars(UV_CACHE_DIR=str(chosen)), name="lab-1"
    )
    try:
        assert chosen.is_dir()
        assert not (Path(bootstrap.DEFAULT_WORKSPACE_ROOT) / "uv-cache").exists()
    finally:
        runtime.shutdown()


def test_the_sync_is_frozen_skips_the_project_and_always_names_the_local_python(
    uv_project: Path,
) -> None:
    # Spec "Interpreter version": --python is passed with or without .python-version.
    expected = ["uv", "sync", "--frozen", "--no-install-project", "--python", LOCAL_PYTHON]
    assert bootstrap.sync_command(Env()) == expected
    assert sorted(bootstrap.project_files(Env())) == ["pyproject.toml", "uv.lock"]

    (uv_project / ".python-version").write_text(LOCAL_PYTHON + "\n", encoding="utf-8")
    assert bootstrap.sync_command(Env()) == expected
    assert sorted(bootstrap.project_files(Env())) == [
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    ]


def test_the_workspace_root_defaults_to_what_each_kind_of_machine_allows() -> None:
    # Spec "Workspace root": per kind, when the account sets no workspace.
    from letify.providers.colab import Colab
    from letify.providers.elice import Elice
    from letify.providers.modal import Modal
    from letify.providers.shell import Shell
    from letify.providers.tunnel import Tunnel

    for kind in (Shell, Tunnel, Elice):
        provider = provider_of(kind, "lab", address="gpu.example")
        assert provider.workspace_root == "~/.letify-runtime"
    assert provider_of(Colab, "lab").workspace_root == "/content/letify"
    assert provider_of(Modal, "lab").workspace_root == "/letify"


def test_an_account_workspace_replaces_the_default_root() -> None:
    from letify.providers.colab import Colab
    from letify.providers.shell import Shell

    shell = provider_of(Shell, "lab", address="gpu.example", workspace="/workspace/me/letify")
    assert shell.workspace_root == "/workspace/me/letify"
    assert provider_of(Colab, "lab", workspace="~/scratch").workspace_root == "~/scratch"


def test_a_workspace_that_is_not_an_absolute_or_home_path_is_refused() -> None:
    from letify.providers.shell import Shell

    provider = provider_of(Shell, "lab", address="gpu.example", workspace="lab-team")
    with pytest.raises(letify.ConfigError, match=r"lab.*workspace"):
        provider.workspace_root  # noqa: B018


def test_a_volume_lives_under_the_workspace_root_unless_it_names_a_mount() -> None:
    # Spec "Materializing into a runtime": <workspace root>/volumes/<volume name>.
    from letify.providers.shell import Shell

    provider = provider_of(Shell, "lab", address="gpu.example", workspace="/workspace/me")
    assert Volume(provider, "cache").mount == "/workspace/me/volumes/cache"
    assert Volume(provider, "cache", {"mount": "/mnt/study"}).mount == "/mnt/study"


def test_no_remote_path_is_hard_coded_outside_the_workspace_defaults() -> None:
    # Spec "Workspace root": every remote path derives from the root. The Colab default is
    # the one place /content may appear, and nothing may write under /opt or /tmp.
    import re

    package = Path(letify.__file__).parent
    offenders = []
    for source in package.rglob("*.py"):
        if "_vendor" in source.parts:
            continue
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for literal in re.findall(r"[\"'](/(?:opt|tmp|content)(?:/[^\"']*)?)[\"']", line):
                if source.name == "colab.py" and literal == "/content/letify":
                    continue
                # SSH control sockets are a client-side path, not a remote one.
                if source.name == "sshopts.py" and literal.startswith("/tmp/letify-"):
                    continue
                offenders.append(f"{source.relative_to(package)}:{number}: {literal}")
    assert offenders == []


class WorkspaceLocal(LeasingLocal):
    """A local provider that prepares a workspace root the way a remote runtime does."""

    prepares_workspace = True

    @property
    def workspace_root(self) -> str:
        return str(self.config.option("workspace"))


def test_a_booted_worker_works_inside_its_workspace_root(tmp_path: Path) -> None:
    # Spec "Sessions", step 3: expand, create, change into it, and TMPDIR under the root.
    root = tmp_path / "ws"
    provider = provider_of(WorkspaceLocal, "lab", python=sys.executable, workspace=str(root))
    instance = Instance(provider, gpu=None)._placed("remote")
    runtime = provider.start(instance, Env(lock=str(tmp_path / "absent.lock")), name="lab-1")
    try:
        assert runtime.workspace == str(root)
        runtime.exec(
            "import os, tempfile\n"
            f"assert os.getcwd() == {str(root)!r}, os.getcwd()\n"
            f"assert tempfile.gettempdir() == {str(root / 'tmp')!r}, tempfile.gettempdir()\n"
        )
    finally:
        runtime.shutdown()


def test_a_volume_materializes_under_the_expanded_workspace_root(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    provider = provider_of(WorkspaceLocal, "lab", python=sys.executable, workspace=str(root))
    volume = provider.volume("cache", backend="filesystem", root=str(tmp_path / "store"))
    info = volume.store.put_bytes(b"weights")
    instance = Instance(provider, gpu=None)._placed("remote")
    runtime = provider.start(
        instance, Env(lock=str(tmp_path / "absent.lock")), name="lab-1", volumes=(volume,)
    )
    try:
        remote = volume.materialize(runtime, info.digest)
        assert Path(remote.path).parent.parent == root / "volumes" / "cache" / "blobs"
        assert Path(remote.path).read_bytes() == b"weights"
    finally:
        runtime.shutdown()


# -- Spec: Argument blobs on a persistent disk ---------------------------------


def _counting_puts(runtime) -> list[str]:
    """Record the digest of every put_blob the runtime sends."""
    sent: list[str] = []
    original = runtime.request

    def request(message, *args, **kwargs):
        if message.get("op") == "put_blob":
            sent.append(message["digest"])
        return original(message, *args, **kwargs)

    runtime.request = request
    return sent


def _start_workspace_runtime(provider, tmp_path: Path, name: str):
    instance = Instance(provider, gpu=None)._placed("remote")
    return provider.start(instance, Env(lock=str(tmp_path / "absent.lock")), name=name)


def test_a_persistent_runtime_receives_a_repeated_argument_from_an_earlier_session_as_a_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    provider = provider_of(
        WorkspaceLocal, "lab", python=sys.executable, workspace=str(root), persistent=True
    )
    payload = b"x" * (256 * 1024)
    first = _start_workspace_runtime(provider, tmp_path, "lab-1")
    try:
        assert first.call(len, (payload,), {})[0] == len(payload)
    finally:
        first.shutdown()
    assert [p for p in (root / "blobs").rglob("*") if p.is_file()], "no blob file was written"

    second = _start_workspace_runtime(provider, tmp_path, "lab-2")
    try:
        sent = _counting_puts(second)
        assert second.call(len, (payload,), {})[0] == len(payload)
        assert sent == []
    finally:
        second.shutdown()


def test_a_mutable_argument_blob_on_disk_still_arrives_as_a_fresh_copy(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    provider = provider_of(
        WorkspaceLocal, "lab", python=sys.executable, workspace=str(root), persistent=True
    )
    value = bytearray(b"y" * (256 * 1024))

    def mutate(buffer: bytearray) -> int:
        first = buffer[0]
        buffer[0] = 0
        return first

    first = _start_workspace_runtime(provider, tmp_path, "lab-1")
    try:
        first.call(mutate, (value,), {})
    finally:
        first.shutdown()
    second = _start_workspace_runtime(provider, tmp_path, "lab-2")
    try:
        sent = _counting_puts(second)
        assert second.call(mutate, (value,), {})[0] == ord("y")
        assert second.call(mutate, (value,), {})[0] == ord("y")
        assert sent == []
    finally:
        second.shutdown()


def test_an_ephemeral_runtime_writes_no_argument_blob_to_disk(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    provider = provider_of(
        WorkspaceLocal, "lab", python=sys.executable, workspace=str(root), persistent=False
    )
    runtime = _start_workspace_runtime(provider, tmp_path, "lab-1")
    try:
        runtime.call(len, (b"z" * (256 * 1024),), {})
    finally:
        runtime.shutdown()
    assert not (root / "blobs").exists()


def test_argument_blobs_on_disk_are_evicted_oldest_first_beyond_the_limit(
    tmp_path: Path, monkeypatch
) -> None:
    from letify.runtime import session as session_module

    monkeypatch.setattr(session_module, "BLOB_DISK_LIMIT", 600 * 1024, raising=False)
    root = tmp_path / "ws"
    provider = provider_of(
        WorkspaceLocal, "lab", python=sys.executable, workspace=str(root), persistent=True
    )
    runtime = _start_workspace_runtime(provider, tmp_path, "lab-1")
    try:
        for fill in (b"a", b"b", b"c"):
            runtime.call(len, (fill * (256 * 1024),), {})
            time.sleep(0.05)
    finally:
        runtime.shutdown()
    files = [p for p in (root / "blobs").rglob("*") if p.is_file()]
    assert len(files) == 2
    assert sorted(p.read_bytes()[:1] for p in files) == [b"b", b"c"]


def test_modal_builds_the_project_environment_like_every_remote_runtime() -> None:
    # Spec "Building the environment on a runtime": a Modal sandbox syncs the project .venv
    # too, because nothing in its image carries letify.
    from letify.providers.modal import Modal

    modal = provider_of(Modal, "lab")
    assert modal.remote_env is True


def test_every_remote_path_derives_from_the_provider_workspace_root(uv_project: Path) -> None:
    provider = provider_of(PreparingLocal, "lab")
    assert provider.workspace_root == bootstrap.DEFAULT_WORKSPACE_ROOT
    env = Env()
    assert bootstrap.project_dir(provider.workspace_root, env) == f"{remote_projects()}/{env.key}"


def test_an_env_records_the_local_interpreter_version() -> None:
    assert Env().python == LOCAL_PYTHON


def test_the_sync_source_carries_the_declared_refinements(uv_project: Path) -> None:
    env = Env().pip_install("torch").run("nvidia-smi").vars(HF_HOME="/opt/cache")
    source = bootstrap.sync_source(env, bootstrap.project_files(env))
    compile(source, "<sync>", "exec")
    assert "'torch'" in source
    assert "'nvidia-smi'" in source
    assert "'HF_HOME': '/opt/cache'" in source
    assert "'--system'" not in source


@pytest.mark.parametrize("declared", ["python-version", "env"])
def test_a_python_other_than_the_local_one_is_refused_before_a_session_starts(
    uv_project: Path, monkeypatch, declared: str
) -> None:
    provider = provider_of(PreparingLocal, "lab")
    created: list[object] = []
    monkeypatch.setattr(provider, "create_session", lambda *args: created.append(args))
    if declared == "python-version":
        (uv_project / ".python-version").write_text(OTHER_PYTHON + ".4\n", encoding="utf-8")
        env = Env()
    else:
        env = Env(python=OTHER_PYTHON)

    with pytest.raises(letify.InterpreterMismatch) as caught:
        provider.start(remote_instance(provider), env, name="lab-1")
    assert OTHER_PYTHON in str(caught.value)
    assert LOCAL_PYTHON in str(caught.value)
    assert created == []


def test_a_project_without_a_lock_file_is_refused_before_a_session_starts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    provider = provider_of(PreparingLocal, "lab")
    with pytest.raises(letify.ConfigError, match="uv lock"):
        provider.start(remote_instance(provider), Env(), name="lab-1")


# -- Spec: Channels and Sessions, the bootstrap interpreter --------------------


def without_cloudpickle_or_pip(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Stand-ins that make importing cloudpickle fail and any pip use fail loudly.

    Returns the directory to put first on PYTHONPATH, the directory to put first on PATH,
    and the file a pip stand-in writes when anything runs it.
    """
    modules = tmp_path / "standins"
    marker = tmp_path / "pip-was-run"
    for name in ("cloudpickle", "pip"):
        (modules / name).mkdir(parents=True)
    (modules / "cloudpickle" / "__init__.py").write_text(
        "raise ImportError(\"No module named 'cloudpickle'\", name='cloudpickle')\n",
        encoding="utf-8",
    )
    (modules / "pip" / "__init__.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nraise SystemExit('pip must not run')\n",
        encoding="utf-8",
    )
    (modules / "pip" / "__main__.py").write_text("import pip\n", encoding="utf-8")
    binaries = tmp_path / "standin-bin"
    binaries.mkdir()
    for name in ("pip", "pip3"):
        script = binaries / name
        script.write_text(f"#!/bin/sh\n: > {marker}\nexit 9\n", encoding="utf-8")
        script.chmod(0o755)
    return modules, binaries, marker


def test_the_worker_reaches_the_project_interpreter_without_cloudpickle_or_pip(
    uv_project: Path, tmp_path: Path
) -> None:
    # The lab_docker case: a system Python with no cloudpickle that refuses pip install.
    # Hello, workspace, environment build and the move all run on the standard library.
    # Driven through the real PersistentChannel, so the source hand-off and the frames are
    # the ones a runtime receives, and a failed request raises instead of returning.
    import os

    modules, binaries, marker = without_cloudpickle_or_pip(tmp_path)
    uv = binaries / "uv"
    uv.write_text(
        f"#!/bin/sh\nmkdir -p .venv/bin && ln -sf {sys.executable} .venv/bin/python\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = {
        **os.environ,
        "PYTHONPATH": str(modules),
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
    }
    channel = PersistentChannel(
        [sys.executable, "-u", "-c", BOOTSTRAP], name="no-cloudpickle", env=env
    )
    version = "{}.{}".format(*sys.version_info[:2])
    try:
        channel.start()
        assert channel.python_version == version
        workspace = tmp_path / "workspace"
        channel.eval(bootstrap.workspace_source(str(workspace)), timeout=60)
        env_ = Env()
        root = str(workspace / "project" / env_.key)
        files = bootstrap.project_files(env_)
        synced = channel.eval(bootstrap.sync_source(env_, files, root=root), timeout=120)
        python = synced["python"]
        assert Path(python).parent == Path(root) / ".venv" / "bin"
        channel.switch_interpreter(python, timeout=60)
        assert channel.eval("import sys\n__letify_value__ = sys.executable") == python
        channel.request({"op": "stat"}, timeout=60)
    finally:
        channel.close()
    stderr = channel._stderr.text()
    assert not marker.exists()
    assert "pip" not in stderr


# -- Spec: uv on the runtime ---------------------------------------------------


def run_without_uv(source: str, home: Path) -> subprocess.CompletedProcess[str]:
    """Run bootstrap source in a fresh interpreter whose PATH and home hold no uv."""
    return subprocess.run(
        [sys.executable, "-c", source],
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_default_installer_is_the_official_standalone_script_over_https() -> None:
    assert bootstrap.UV_INSTALLER == "https://astral.sh/uv/install.sh"


def test_a_runtime_with_neither_curl_nor_wget_fails_early_naming_them(tmp_path: Path) -> None:
    """Spec "uv on the runtime": the installer needs curl or wget, so the worker checks first.

    With uv, curl and wget all off PATH, the worker must refuse before running the installer
    and say curl or wget is required, rather than let the installer fail with an obscure error.
    """
    home = tmp_path / "home"
    home.mkdir()
    empty = tmp_path / "empty-path"
    empty.mkdir()
    source = bootstrap.sync_source(
        Env(), bootstrap.project_files(Env()), root=str(tmp_path / "proj")
    )
    script = tmp_path / "boot.py"
    script.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        env={"HOME": str(home), "PATH": str(empty)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "curl or wget" in result.stderr


def test_a_runtime_without_uv_installs_it_under_home_and_then_syncs(
    uv_project: Path, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    record = tmp_path / "uv-arguments"
    installer = tmp_path / "install.sh"
    # Stands in for the standalone installer: it writes a uv into UV_INSTALL_DIR, and that
    # uv records its arguments and makes the .venv the worker expects.
    installer.write_text(
        'test "$UV_NO_MODIFY_PATH" = 1 || exit 9\n'
        'mkdir -p "$UV_INSTALL_DIR"\n'
        "cat > \"$UV_INSTALL_DIR/uv\" <<'EOF'\n"
        "#!/bin/sh\n"
        f'echo "$@" > {record}\n'
        f"{sys.executable} -m venv --without-pip .venv\n"
        "EOF\n"
        'chmod +x "$UV_INSTALL_DIR/uv"\n',
        encoding="utf-8",
    )
    source = bootstrap.sync_source(
        Env(), bootstrap.project_files(Env()), installer=installer.as_uri()
    )
    result = run_without_uv(source, home)
    assert result.returncode == 0, result.stderr
    assert (home / ".local" / "bin" / "uv").is_file()
    assert record.read_text(encoding="utf-8").split() == [
        "sync",
        "--frozen",
        "--no-install-project",
        "--python",
        LOCAL_PYTHON,
    ]


def test_a_failed_uv_install_is_named(uv_project: Path, tmp_path: Path) -> None:
    installer = tmp_path / "install.sh"
    installer.write_text("exit 3\n", encoding="utf-8")
    source = bootstrap.sync_source(
        Env(), bootstrap.project_files(Env()), installer=installer.as_uri()
    )
    result = run_without_uv(source, tmp_path / "home")
    assert result.returncode != 0
    assert "uv could not be installed" in result.stderr


def test_a_failed_sync_raises_environment_failure_naming_uv_sync(
    uv_project: Path, tmp_path: Path
) -> None:
    # A lock file that names a package no index has makes the real uv fail.
    (uv_project / "uv.lock").write_text("version = 1\nnot a lock\n", encoding="utf-8")
    provider = provider_of(PreparingLocal, "lab")
    with pytest.raises(letify.EnvironmentFailure, match="uv sync failed"):
        provider.start(remote_instance(provider), Env(), name="lab-1")


# -- Spec: Materializing into a runtime, the archive key -----------------------


def test_the_environment_archive_is_keyed_by_env_key_python_version_and_platform(
    tmp_path: Path,
) -> None:
    volume = provider_of(PreparingLocal, "lab").volume(
        "cache", backend="filesystem", root=str(tmp_path / "store")
    )
    env = Env()
    assert volume.env_ref(env, "linux-x86_64") == f"env/{env.key}-linux-x86_64"
    assert Env(python="3.11").key != Env(python="3.12").key


# -- Spec: Interpreter check ---------------------------------------------------


def test_the_ready_line_carries_the_worker_python_version(channel) -> None:
    channel.start()
    assert channel.python_version == LOCAL_PYTHON


def test_a_worker_on_another_python_version_fails_the_start_naming_both(
    uv_project: Path, monkeypatch
) -> None:
    # A real locked project, because every runtime builds its environment now. The account
    # setting that skipped the build, and with it this check, is gone.
    monkeypatch.setattr(bootstrap, "local_python", lambda: "3.99")
    provider = provider_of(PreparingLocal, "lab")
    with pytest.raises(letify.InterpreterMismatch) as caught:
        provider.start(remote_instance(provider), Env(), name="lab-1")
    assert "3.99" in str(caught.value)
    assert LOCAL_PYTHON in str(caught.value)


def test_no_account_can_name_the_bootstrap_interpreter() -> None:
    # An account that named its own interpreter also turned the environment build off, which
    # is the hole Kaggle's batch mode used to skip the interpreter check. There is no such
    # setting now: every runtime bootstraps on python3 and then moves onto the project .venv.
    from letify.providers.base import Provider
    from letify.providers.shell import Shell

    plain = provider_of(Shell, "lab", address="gpu.example")
    assert plain.remote_python == "python3"
    assert not hasattr(plain, "managed_python")
    assert not hasattr(Provider, "managed_python")


def test_a_one_shot_channel_moved_to_the_venv_runs_each_program_with_that_python(
    tmp_path: Path,
) -> None:
    # What colab exec offers: one program per request. After the move every program runs as
    # a child of the given interpreter, which a symlinked name makes visible.
    link = tmp_path / "bin" / "python"
    link.parent.mkdir()
    link.symlink_to(sys.executable)
    record: list[str] = []
    channel = OneShotChannel(local_one_shot_runner(record), name="colab-1")
    channel.switch_interpreter(str(link))
    assert channel.python_version == LOCAL_PYTHON
    value, _logs = channel.request(
        {"op": "eval", "source": "import sys\n__letify_value__ = sys.executable\n"}
    )
    assert value == str(link)
    assert repr(str(link)) in record[-1]


def test_the_local_provider_still_runs_in_the_local_environment(let) -> None:
    assert let.providers.local.remote_env is False


def test_a_cached_environment_archive_lands_in_the_content_addressed_layout() -> None:
    assert bootstrap.env_archive_path("/opt/letify/", "ab12cd") == "/opt/letify/blobs/ab/ab12cd"


# -- Spec: Lifetime, the lease -------------------------------------------------


def test_the_lease_renews_well_inside_the_grace_period() -> None:
    # The gap between the two is the tolerance for a dropped connection, so a flaky link
    # does not kill a training run.
    assert INTERVAL == 30.0
    assert GRACE == 300.0
    assert GRACE > INTERVAL * 2


def test_arming_a_lease_sets_a_deadline_inside_the_session(let, remote_cpu, live) -> None:
    runtime = live(let, remote_cpu)
    # Nothing is armed until the lease asks for it.
    with pytest.raises(letify.RemoteError):
        runtime.exec("assert _LEASE['armed']")
    lease = Lease(runtime, interval=60.0, grace=120.0)
    lease.arm()
    try:
        runtime.exec("assert _LEASE['armed'] and _LEASE['deadline'] > 0")
    finally:
        lease.release()
        let.pool.shutdown()


def test_a_lease_keeps_pushing_the_deadline_forward(renewal_recorder) -> None:
    lease = Lease(renewal_recorder, interval=0.02, grace=90.0)
    lease.arm()
    try:
        deadline = time.monotonic() + 5
        while len(renewal_recorder.renewals) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        lease.release()
    assert len(renewal_recorder.renewals) >= 3
    assert renewal_recorder.renewals[0] == 90.0


def test_arming_a_lease_twice_does_not_start_a_second_renewer(renewal_recorder) -> None:
    lease = Lease(renewal_recorder, interval=60.0, grace=90.0)
    lease.arm()
    lease.arm()
    try:
        assert len(renewal_recorder.renewals) == 1
    finally:
        lease.release()


def test_a_lease_stops_renewing_once_the_session_is_gone(renewal_recorder) -> None:
    # The pool notices on its next call, so the renewer has nothing left to do.
    renewal_recorder.fail_after = 2
    lease = Lease(renewal_recorder, interval=0.02, grace=90.0)
    lease.arm()
    time.sleep(0.2)
    lease.release()
    # Two renewals happened and then the loop stopped, rather than retrying forever.
    assert len(renewal_recorder.renewals) == 2


# -- Spec: Pooling -------------------------------------------------------------


def test_the_pool_has_no_ceiling_and_no_timer() -> None:
    # A ceiling would be a guess about hardware the provider entry already describes. A timer
    # would overrule a declaration that said to keep the session.
    pool = RuntimePool()
    assert not hasattr(pool, "max_runtimes")
    assert not hasattr(pool, "idle_timeout")
    # The one interval left is how long a call waiting for a card sleeps between looks, which
    # bounds nothing except how long a lost notification goes unnoticed.
    assert POLL_INTERVAL == 30.0


def test_a_call_that_finds_every_card_taken_waits_for_one_to_come_free(
    one_card_cpu, tmp_path
) -> None:
    # Asking the provider for a machine it would refuse is worse than waiting.
    remote_cpu = one_card_cpu
    pool = RuntimePool()
    first_lock = tmp_path / "a.lock"
    first_lock.write_text("a", encoding="utf-8")
    second_lock = tmp_path / "b.lock"
    second_lock.write_text("b", encoding="utf-8")

    held = pool.acquire(remote_cpu, Env(lock=str(first_lock)))
    waiting: list[object] = []

    def second() -> None:
        waiting.append(pool.acquire(remote_cpu, Env(lock=str(second_lock))))

    thread = threading.Thread(target=second, daemon=True)
    thread.start()
    time.sleep(0.5)
    try:
        assert waiting == []
        assert len(pool.live) == 1
        pool.discard(held)
        thread.join(timeout=30)
        assert len(waiting) == 1
    finally:
        pool.shutdown()


def test_a_session_that_fails_to_start_gives_its_slot_back(
    launcher_from, tmp_path, live, monkeypatch
) -> None:
    # Otherwise one failed start would permanently shrink the ceiling. The failure has to
    # land after the slot is reserved, which is why the channel is what refuses: a provider
    # that is refused earlier never takes a slot to give back.
    let = launcher_from('[broken]\nkind = "local"\n')
    provider = let.providers.broken

    def refuse(runtime):
        raise RuntimeFailure("the worker did not start")

    # On the instance, so the working session below still opens its own channel.
    monkeypatch.setattr(provider, "open_channel", refuse)
    broken = provider.CPU._placed("remote")
    with pytest.raises(RuntimeFailure):
        live(let, broken)
    assert let.pool.live == []
    # The slot is free again, so a working session still starts.
    working = let.providers.local.CPU._placed("remote")
    assert live(let, working).ready is True
    let.pool.shutdown()


def test_a_runtime_is_reused_while_it_is_free_and_not_while_it_is_busy(let, remote_cpu) -> None:
    pool = let.pool
    pool.hold()
    first = pool.acquire(remote_cpu, Env())
    pool.release(first)
    assert pool.acquire(remote_cpu, Env()) is first
    pool.shutdown()


# -- Spec: Lifetime ------------------------------------------------------------


def test_a_released_runtime_ends_unless_something_says_otherwise(let, remote_cpu) -> None:
    # The default is that a runtime dies when the work that needed it is done.
    pool = let.pool
    runtime = pool.acquire(remote_cpu, Env())
    pool.release(runtime)
    assert pool.live == []


def test_a_held_invocation_keeps_released_runtimes_until_the_last_hold_goes(
    let, remote_cpu
) -> None:
    # This is what lets overlapping calls reuse a session until the last of them finishes.
    pool = let.pool
    pool.hold()
    assert pool.holding is True
    runtime = pool.acquire(remote_cpu, Env())
    pool.release(runtime)
    assert pool.live == [runtime]
    pool.unhold()
    assert pool.holding is False
    assert pool.live == []


def test_a_declared_session_is_not_ended_on_a_timer(let, remote_cpu) -> None:
    # A keep_alive block says the session lives until the block ends. A thread ending it after
    # some idle period overrules that, which is the same mistake as keeping one alive on the
    # chance a call arrives.
    let.pool.hold()
    runtime = let.pool.acquire(remote_cpu, Env())
    let.pool.release(runtime)
    assert let.pool.live == [runtime]

    assert not hasattr(let, "idle_timeout")
    assert not hasattr(let, "reap_idle")
    assert not hasattr(let.pool, "idle_timeout")
    assert not hasattr(let.pool, "shutdown_idle")
    assert [t.name for t in threading.enumerate() if "reap" in t.name] == []
    with pytest.raises(TypeError):
        letify.Launcher(idle_timeout=1.0)

    # It still goes at process exit, which is what the registered shutdown does.
    assert let.pool.shutdown() == [runtime.name]


def test_a_session_started_early_is_the_one_the_first_call_uses(let, remote_cpu, live) -> None:
    started = live(let, remote_cpu)

    @let.function(device=let.providers.local.CPU, host=letify.remote)
    def noop() -> None:
        return None

    noop()
    assert let.pool.live == [started]


def test_leaving_the_keep_alive_block_ends_the_idle_sessions(let, remote_cpu) -> None:
    # The pool side of let.keep_alive(): released runtimes stay while held and go when the
    # last hold does.
    pool = let.pool
    pool.hold()
    runtime = pool.acquire(remote_cpu, Env())
    pool.release(runtime)
    assert pool.live == [runtime]
    pool.unhold()
    assert pool.live == []


def test_shutdown_takes_everything_including_a_runtime_still_marked_busy(let, remote_cpu) -> None:
    # Registered at process exit, so nothing survives this process.
    pool = let.pool
    runtime = pool.acquire(remote_cpu, Env())
    assert runtime.busy is True
    assert pool.shutdown() == [runtime.name]
    assert pool.live == []


# -- Spec: Failure and retry ---------------------------------------------------


def test_an_infrastructure_failure_is_retried_on_a_fresh_runtime(let, remote_cpu, tmp_path) -> None:
    ledger = tmp_path / "attempts.txt"

    @let.function(device=remote_cpu, host="remote", retries=2)
    def dies(path: str) -> None:
        import os

        with open(path, "a", encoding="utf-8") as handle:
            handle.write("attempt\n")
        os._exit(1)

    with pytest.raises(letify.RuntimeLost) as caught:
        dies(path=str(ledger))

    message = str(caught.value)
    assert "dies failed after 3 attempt(s)" in message
    assert "local:cpu" in message
    assert ledger.read_text(encoding="utf-8").count("attempt") == 3
    # Each failed runtime is discarded rather than handed out again.
    assert let.pool.live == []


class DiagnosingLocal(Local):
    """A local provider that reads every failure as a spot preemption, counting them."""

    diagnosed: ClassVar[list[str]] = []

    def diagnose(self, runtime, failure):
        from letify.errors import SpotPreempted

        type(self).diagnosed.append(str(failure))
        return SpotPreempted("local-cpu was preempted", machine="local-cpu", state="idle", at=1.0)


def test_a_diagnosed_failure_is_retried_and_raised_as_it_is_after_the_last_retry(
    tmp_path, monkeypatch
) -> None:
    # Spec "Failure and retry": the provider may name the failure more precisely before the
    # retry, and a SpotPreempted left after the last retry is not wrapped.
    from letify import providers
    from letify.errors import SpotPreempted

    monkeypatch.setitem(providers.KINDS, "local", DiagnosingLocal)
    DiagnosingLocal.diagnosed = []
    project = tmp_path / "project" / ".letify"
    project.mkdir(parents=True)
    let = letify.Launcher(project, home=False, announce=False)
    device = let.providers.local.CPU._placed("remote")
    ledger = tmp_path / "attempts.txt"

    @let.function(device=device, host="remote", retries=1)
    def dies(path: str) -> None:
        import os

        with open(path, "a", encoding="utf-8") as handle:
            handle.write("attempt\n")
        os._exit(1)

    with pytest.raises(SpotPreempted) as caught:
        dies(path=str(ledger))
    assert caught.value.machine == "local-cpu"
    assert ledger.read_text(encoding="utf-8").count("attempt") == 2
    assert len(DiagnosingLocal.diagnosed) == 2
    assert let.pool.live == []


def test_the_last_failures_command_and_stderr_survive_the_retry(
    let, remote_cpu, monkeypatch
) -> None:
    from letify.runtime.session import Runtime

    def fail(self, *args, **kwargs):
        raise RuntimeFailure(
            "`colab exec` exited 1", command="colab exec -s s", stderr="SyntaxError: bad"
        )

    monkeypatch.setattr(Runtime, "call", fail)

    @let.function(device=remote_cpu, host="remote", retries=1)
    def noop() -> None:
        return None

    with pytest.raises(letify.RuntimeLost) as caught:
        noop()
    assert "SyntaxError: bad" in str(caught.value)
    assert caught.value.stderr == "SyntaxError: bad"
    assert caught.value.command == "colab exec -s s"


def test_user_code_failure_is_never_retried(let, remote_cpu, tmp_path) -> None:
    # Retrying a body that raises only reproduces it, so the traceback comes straight
    # back instead.
    ledger = tmp_path / "attempts.txt"

    @let.function(device=remote_cpu, host="remote", retries=3)
    def boom(path: str) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("attempt\n")
        raise ValueError("intentional")

    with pytest.raises(letify.RemoteError, match="intentional"):
        boom(path=str(ledger))

    assert ledger.read_text(encoding="utf-8").count("attempt") == 1
    assert let.pool.live == []


# -- Spec: GPU utilization -----------------------------------------------------


def test_a_reading_is_one_record_per_device_with_memory_in_gibibytes() -> None:
    # nvidia-smi writes mebibytes under --nounits, and the instance carries gibibytes, so
    # the two have to be comparable.
    devices = telemetry.parse_smi(
        "0, NVIDIA RTX PRO 6000, 87, 40960, 98304, 71, 412.5\n"
        "1, NVIDIA RTX 4050, 0, 512, 6144, 44, 8.1\n"
    )
    assert [d.index for d in devices] == [0, 1]
    assert devices[0].name == "NVIDIA RTX PRO 6000"
    assert devices[0].utilization_percent == 87.0
    assert devices[0].memory_used_gb == 40.0
    assert devices[0].memory_total_gb == 96.0
    assert devices[0].temperature_c == 71.0
    assert devices[0].power_w == 412.5
    assert devices[1].memory_percent is not None


def test_a_reading_a_card_does_not_support_is_a_gap_rather_than_a_zero() -> None:
    # A card reporting no power draw is not a card drawing no power.
    device = telemetry.parse_smi("0, NVIDIA T4, [N/A], 1024, 16384, [Not Supported], [N/A]\n")[0]
    assert device.utilization_percent is None
    assert device.power_w is None
    assert device.temperature_c is None
    assert device.memory_total_gb == 16.0
    assert "load unknown" in device.describe()


def test_output_that_is_not_a_reading_is_skipped_rather_than_guessed_at() -> None:
    # A driver error message on stdout must not become a device with a made up load.
    assert telemetry.parse_smi("") == []
    assert telemetry.parse_smi("NVIDIA-SMI has failed because it could not\n") == []
    assert telemetry.parse_smi("index, name, 1, 2, 3, 4, 5\n") == []


def test_a_machine_without_the_tool_reports_nothing_rather_than_failing(
    patch_which,
) -> None:
    # There is nothing to ask, which is an answer and not an error.
    patch_which(telemetry, present=False)
    assert telemetry.read_smi() == ""
    assert telemetry.local_load() == []


def test_a_tool_that_fails_reports_nothing(patch_run, patch_which) -> None:
    patch_which(telemetry, present=True)
    patch_run(telemetry, result=FakeCompleted(returncode=9, stderr="driver mismatch"))
    assert telemetry.read_smi() == ""


def test_the_reading_is_asked_for_in_one_pass_over_the_devices(patch_run, patch_which) -> None:
    # One nvidia-smi call for every device, because a per-device call would measure them
    # at different moments and the point is a snapshot.
    patch_which(telemetry, present=True)
    recorder = patch_run(
        telemetry, result=FakeCompleted(stdout="0, NVIDIA T4, 5, 1024, 16384, 40, 30\n")
    )
    assert len(telemetry.local_load()) == 1
    assert len(recorder.commands) == 1
    assert recorder.commands[0][0] == "nvidia-smi"


# -- Spec: Pooling, capacity is the inventory ----------------------------------


def test_the_launcher_has_no_session_ceiling(let) -> None:
    # A number there would be a guess about hardware the provider entry already describes,
    # and when the two disagreed the smaller would win silently.
    assert not hasattr(let, "max_runtimes")
    assert "max_runtimes" not in let.status()
    with pytest.raises(TypeError):
        letify.Launcher(max_runtimes=3)


def test_a_session_reserves_the_devices_its_instance_asks_for(reserving) -> None:
    provider = reserving(A100={"indices": "0-3"})
    first = provider.reserve(provider.A100)
    second = provider.reserve(provider.A100 * 2)
    assert first == (0,)
    assert second == (1, 2)
    assert provider.free("A100") == (3,)


def test_a_reservation_that_cannot_be_met_is_refused_rather_than_halved(reserving) -> None:
    # Half the cards a run asked for is not a smaller version of the run.
    provider = reserving(A100={"indices": "0-1"})
    assert provider.reserve(provider.A100 * 2) == (0, 1)
    assert provider.reserve(provider.A100) is None


def test_releasing_a_reservation_gives_the_cards_back(reserving) -> None:
    provider = reserving(A100={"indices": "0-1"})
    held = provider.reserve(provider.A100 * 2)
    provider.unreserve("A100", held)
    assert provider.free("A100") == (0, 1)


def test_a_card_another_process_is_using_is_skipped(reserving, patch_smi) -> None:
    # Registered is permission, not availability. A colleague computing on card one is not
    # something to fight over.
    provider = reserving(A100={"indices": "0-2"})
    patch_smi(busy=[1])
    assert provider.reserve(provider.A100 * 2) == (0, 2)


def test_a_provider_that_assigns_its_own_devices_reserves_by_count(reserving) -> None:
    # Colab hands out the accelerator itself, so there is nothing to index and the only
    # question is whether the account has a slot left.
    provider = reserving(G4={"count": 2})
    assert provider.reserve(provider.G4) == ()
    assert provider.reserve(provider.G4) == ()
    assert provider.reserve(provider.G4) is None


# -- Spec: Inventory, the visible devices of a reserved session ---------------

VISIBLE_SOURCE = (
    "import os\n"
    "__letify_value__ = (os.environ.get('CUDA_VISIBLE_DEVICES'),"
    " os.environ.get('CUDA_DEVICE_ORDER'))\n"
)


def started_with_reservation(provider, instance) -> letify.Runtime:
    held = provider.reserve(instance)
    return provider.start(instance, Env(), name="box-1", held=held)


def test_a_reserved_session_sees_only_its_one_card(reserving, patch_smi) -> None:
    provider = reserving(A100={"indices": "0-3"})
    patch_smi(busy=[0, 1])
    runtime = started_with_reservation(provider, provider.A100._placed("remote"))
    try:
        assert list(runtime.eval(VISIBLE_SOURCE)) == ["2", "PCI_BUS_ID"]
    finally:
        runtime.shutdown()


def test_a_session_of_two_cards_sees_exactly_those_two(reserving, patch_smi) -> None:
    provider = reserving(A100={"indices": "0-3"})
    patch_smi(busy=[1])
    runtime = started_with_reservation(provider, (provider.A100 * 2)._placed("remote"))
    try:
        assert list(runtime.eval(VISIBLE_SOURCE)) == ["0,2", "PCI_BUS_ID"]
    finally:
        runtime.shutdown()


def test_the_visible_devices_survive_the_move_to_the_project_interpreter(
    uv_project: Path, patch_smi, monkeypatch
) -> None:
    provider = provider_of(PreparingLocal, "lab", devices={"A100": {"indices": "0-3"}})
    monkeypatch.setattr(
        provider,
        "discover",
        lambda: {name: Instance(provider, gpu=name) for name in provider.inventory},
    )
    patch_smi(busy=[0])
    runtime = started_with_reservation(provider, provider.A100._placed("remote"))
    try:
        assert runtime.env_source == "sync"
        assert list(runtime.eval(VISIBLE_SOURCE)) == ["1", "PCI_BUS_ID"]
    finally:
        runtime.shutdown()


def test_a_call_waits_for_a_card_rather_than_asking_for_a_refusal(launcher_from) -> None:
    # One card, two declarations. The second waits for the first to finish instead of
    # starting a session the machine cannot serve.
    import threading

    let = launcher_from('[one]\nkind = "local"\n[one.devices]\nCPU = { count = 1 }\n')

    started = threading.Event()

    @let.function(device=let.provider("one").CPU, host=letify.remote)
    def wait_a_moment() -> int:
        return 1

    with let.keep_alive():
        assert wait_a_moment() == 1
        assert let.status()["live"] == 1
        started.set()


# -- Spec: Pooling, devices that cannot be allocated ------------------------------


def test_a_card_held_by_an_idle_kept_session_cannot_be_allocated(launcher_from) -> None:
    # The only card is kept by an idle session the block is holding, and a call with a different
    # environment needs a card of its own. Nothing running would free it, so waiting would
    # never end.
    let = launcher_from('[one]\nkind = "local"\n[one.devices]\nCPU = { count = 1 }\n')
    card = let.provider("one").CPU

    @let.function(device=card, host=letify.remote)
    def prepare() -> int:
        return 1

    @let.function(device=card, host=letify.remote, env=Env().vars(STAGE="evaluate"), retries=2)
    def evaluate() -> int:
        return 2

    with let.keep_alive():
        assert prepare() == 1
        started = time.monotonic()
        with pytest.raises(letify.InsufficientDevices, match="keep_alive"):
            evaluate()
        # Raised at once rather than after a wait, and not retried.
        assert time.monotonic() - started < 5
        assert let.status()["live"] == 1


def test_asking_for_more_cards_than_the_account_has_is_refused_at_once(launcher_from) -> None:
    let = launcher_from('[one]\nkind = "local"\n[one.devices]\nCPU = { count = 1 }\n')

    @let.function(device=let.provider("one").CPU * 2, host=letify.remote)
    def wide() -> int:
        return 1

    with pytest.raises(letify.InsufficientDevices, match="2"):
        wide()


def test_cards_another_process_is_using_cannot_be_allocated(reserving, patch_smi) -> None:
    # letify cannot know when someone else's job ends, so there is nothing to wait for.
    provider = reserving(A100={"indices": "0"})
    patch_smi(busy=[0])
    pool = RuntimePool()
    with pytest.raises(letify.InsufficientDevices, match="another user"):
        pool.acquire(provider.A100._placed("remote"), Env())
    assert pool.live == []


def test_the_refusal_names_the_busy_cards_and_the_users_computing_on_them(
    reserving, patch_smi
) -> None:
    provider = reserving(A100={"indices": "0-2"})
    patch_smi(busy=[0, 2], owners={0: ("alice",), 2: ("bob", "unknown")})
    provider.reserve(provider.A100)
    pool = RuntimePool()
    with pytest.raises(letify.InsufficientDevices, match=r"0 \(alice\), 2 \(bob, unknown\)"):
        pool.acquire((provider.A100 * 2)._placed("remote"), Env())


def test_a_reservation_takes_the_first_free_index_and_the_session_is_given_it(
    reserving, patch_smi
) -> None:
    provider = reserving(A100={"indices": "0-3"})
    patch_smi(busy=[0, 1, 3])
    runtime = started_with_reservation(provider, provider.A100._placed("remote"))
    try:
        assert runtime.held_devices == (2,)
        assert list(runtime.eval(VISIBLE_SOURCE)) == ["2", "PCI_BUS_ID"]
    finally:
        runtime.shutdown()


def test_a_session_records_its_worker_process_so_its_card_is_not_read_as_busy(
    reserving, monkeypatch
) -> None:
    provider = reserving(A100={"indices": "0-1"})
    asked: list[set[int]] = []

    def busy_indices(exclude_pids=None, **kwargs):
        asked.append(set(exclude_pids or ()))
        return ()

    monkeypatch.setattr(telemetry, "busy_indices", busy_indices)
    runtime = started_with_reservation(provider, provider.A100._placed("remote"))
    try:
        pid = runtime.stat()["pid"]
        assert runtime.worker_pid == pid
        provider.free("A100")
        assert pid in asked[-1]
    finally:
        runtime.shutdown()
    provider.free("A100")
    assert pid not in asked[-1]


def test_devices_that_cannot_be_allocated_are_not_an_infrastructure_failure() -> None:
    # A retry asks for the same devices from the same inventory, so it is never retried.
    assert issubclass(letify.InsufficientDevices, letify.LetifyError)
    assert not issubclass(letify.InsufficientDevices, letify.RuntimeFailure)


def test_holders_separate_other_users_from_the_login_users_own_processes() -> None:
    from letify.runtime import telemetry

    uuids = {"GPU-a": 0, "GPU-b": 1, "GPU-c": 2}
    output = "GPU-a, 10\nGPU-b, 20\n#owners\n10 brew\n20 alice\n#login\nbrew\n"
    holders = telemetry.parse_holders(output, uuids)
    assert holders == {0: ("mine", ()), 1: ("others", ("alice",)), 2: ("free", ())}


def test_a_worker_of_this_client_marks_its_card_as_the_login_users() -> None:
    from letify.runtime import telemetry

    output = "GPU-a, 10\n#owners\n10 root\n#login\nroot\n"
    assert telemetry.parse_holders(output, {"GPU-a": 0}, {10}) == {0: ("mine", ())}


# -- environment on the sandbox disk: spec "Environment on the sandbox disk" --------------


def test_a_provider_with_an_env_root_builds_the_venv_there_with_the_default_uv_cache(
    uv_project: Path, tmp_path: Path
) -> None:
    disk = tmp_path / "sandbox-disk"
    kind = type("DiskEnvLocal", (PreparingLocal,), {"env_root": str(disk)})
    provider = provider_of(kind, "lab")
    assert provider.persistent
    env = Env()
    runtime = provider.start(remote_instance(provider), env, name="lab-1")
    try:
        executable, _version, imported = runtime.call(reports_interpreter(), (), {})[0]
        venv = disk / "project" / env.key / ".venv"
        assert Path(executable).parent == venv / "bin"
        assert Path(imported).is_relative_to(venv)
        assert runtime.env_source == "sync"
        assert not (remote_projects() / env.key).exists()
        assert not (Path(bootstrap.DEFAULT_WORKSPACE_ROOT) / "uv-cache").exists()
    finally:
        runtime.shutdown()


def test_only_modal_sets_an_env_root() -> None:
    from letify.providers.base import Provider
    from letify.providers.modal import Modal

    assert Provider.env_root is None
    assert Modal.env_root == "/root/.letify-env"


def test_the_uv_installer_is_fetched_with_a_user_agent_a_cdn_accepts(
    uv_project: Path, tmp_path: Path
) -> None:
    # Spec "Environment on the runtime": astral.sh answers Python's default urllib agent
    # with 403, as a live Elice machine showed, so the download names letify instead.
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    home = tmp_path / "home"
    script = (
        'mkdir -p "$UV_INSTALL_DIR"\n'
        "cat > \"$UV_INSTALL_DIR/uv\" <<'EOF'\n"
        "#!/bin/sh\n"
        f"{sys.executable} -m venv --without-pip .venv\n"
        "EOF\n"
        'chmod +x "$UV_INSTALL_DIR/uv"\n'
    ).encode()
    agents: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            agent = self.headers.get("User-Agent", "")
            agents.append(agent)
            if agent.startswith("Python-urllib"):
                self.send_response(403)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(script)))
            self.end_headers()
            self.wfile.write(script)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/uv/install.sh"
        source = bootstrap.sync_source(Env(), bootstrap.project_files(Env()), installer=url)
        result = run_without_uv(source, home)
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert agents and agents[0].startswith("letify/")


# -- Spec: Child processes of a call -----------------------------------------------

_MAIN_SCRIPT = """
import multiprocessing


class Offset:
    def __init__(self, by):
        self.by = by


def square(rank, offset, queue):
    queue.put((rank, rank * rank + offset.by))


@let.function(device=let.providers.local.CPU, host=letify.remote)
def spawn_two():
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    children = [
        context.Process(target=square, args=(rank, Offset(10), queue)) for rank in range(2)
    ]
    for child in children:
        child.start()
    results = sorted(queue.get(timeout=60) for _ in children)
    for child in children:
        child.join(60)
    return results, [child.exitcode for child in children]
"""


def test_a_spawned_child_inside_a_call_runs_a_target_defined_in_the_callers_main(let) -> None:
    # The script's functions and classes belong to __main__, which cloudpickle ships by value
    # and which the worker does not have as a module, as in a user's script.
    namespace = {"__name__": "__main__", "let": let, "letify": letify}
    exec(compile(_MAIN_SCRIPT, "user_script.py", "exec"), namespace)
    assert namespace["spawn_two"]() == ([(0, 10), (1, 11)], [0, 0])
