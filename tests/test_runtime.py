"""Channels, live sessions and the pool that decides when they die.

Every channel here is a real one. The local provider starts a Python subprocess running
the same worker source, behind the same framed protocol, that an SSH or WebSocket
channel would, so these tests exercise the protocol and the worker rather than a stub.

Spec sections pinned here: "Channels", "Call protocol", "Failure and retry", "Sessions",
"Pooling", "Lifetime" and "Environment".
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

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
    # This is what makes the object table, the blob table and files on disk survive
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
    assert let.providers.local.prepares_env is False


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


def test_a_prebuilt_environment_archive_is_unpacked_instead_of_being_installed(
    tmp_path: Path,
) -> None:
    # Spec "Blob granularity" and "Materializing into a runtime": the first session pays
    # the installation and every later one unpacks one blob at the mount.
    provider = provider_of(PreparingLocal, "lab")
    env = Env(lock=str(tmp_path / "absent.lock"))
    installed = tmp_path / "site"
    installed.mkdir()
    (installed / "marker.txt").write_text("cached", encoding="utf-8")

    mount = tmp_path / "mount"
    volume = provider.volume(
        "cache", backend="filesystem", root=str(tmp_path / "store"), mount=str(mount)
    )
    volume.cache_env(env, installed)
    assert volume.cached_env(env)

    instance = Instance(provider, gpu=None)._placed("remote")
    runtime = provider.start(instance, env, name="lab-1", volumes=(volume,))
    try:
        assert (mount / "site" / "marker.txt").read_text(encoding="utf-8") == "cached"
    finally:
        runtime.shutdown()


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


def test_a_volume_with_no_cached_archive_is_passed_over(tmp_path: Path) -> None:
    # Spec "Blob granularity": the archive is keyed by the environment, so a volume that
    # does not hold this one is not the place to look.
    provider = provider_of(PreparingLocal, "lab")
    env = Env(lock=str(tmp_path / "absent.lock"))
    installed = tmp_path / "site"
    installed.mkdir()
    (installed / "marker.txt").write_text("cached", encoding="utf-8")

    empty = Volume(
        provider, "empty", {"backend": "filesystem", "root": str(tmp_path / "empty-store")}
    )
    mount = tmp_path / "mount"
    stocked = Volume(
        provider,
        "stocked",
        {"backend": "filesystem", "root": str(tmp_path / "store"), "mount": str(mount)},
    )
    stocked.cache_env(env, installed)

    instance = Instance(provider, gpu=None)._placed("remote")
    runtime = provider.start(instance, env, name="lab-1", volumes=(empty, stocked))
    try:
        assert (mount / "site" / "marker.txt").read_text(encoding="utf-8") == "cached"
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


# -- Spec: Environment, the source that prepares a runtime ---------------------


def test_the_install_source_uses_uv_and_carries_the_declared_refinements() -> None:
    # uv rather than pip because a uv lock file resolves for every platform, which is
    # what lets one lock file drive a Linux runtime from a Windows client.
    source = bootstrap.install_source(
        Env().pip_install("torch").run("nvidia-smi").vars(HF_HOME="/opt/cache")
    )
    compile(source, "<install>", "exec")
    assert "'uv', 'pip', 'install'" in source
    assert "'torch'" in source
    assert "'nvidia-smi'" in source
    assert "os.environ['HF_HOME'] = '/opt/cache'" in source


def test_syncing_from_a_lock_file_already_in_the_runtime_is_frozen() -> None:
    source = bootstrap.sync_lock_source(Env(), "/opt/project")
    compile(source, "<sync>", "exec")
    assert "'uv', 'sync', '--frozen', '--project', '/opt/project'" in source


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


def test_a_session_that_fails_to_start_gives_its_slot_back(launcher_from, tmp_path, live) -> None:
    # Otherwise one failed start would permanently shrink the ceiling.
    let = launcher_from('[broken]\nkind = "local"\npython = "letify-no-such-python"\n')
    broken = let.providers.broken.CPU._placed("remote")
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
    # This is what makes a sweep start its runtimes once and release them once.
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


def test_a_handle_scope_error_is_not_retried_either(let, remote_cpu) -> None:
    # It is a mistake in the call, not a misbehaving session, so the runtime it was
    # aimed at stays usable.
    @let.function(device=let.providers.local.CPU, host=letify.remote, retries=2)
    def consume(value: object) -> object:
        return value

    stranger = letify.Handle(runtime="elsewhere", object_id="00", type_name="dict")
    with let.keep_alive():
        with pytest.raises(letify.HandleScopeError):
            consume(value=stranger)
        assert len(let.pool.live) == 1
        assert consume(value=1) == 1


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
    with pytest.raises(letify.InsufficientDevices, match="another process"):
        pool.acquire(provider.A100._placed("remote"), Env())
    assert pool.live == []


def test_devices_that_cannot_be_allocated_are_not_an_infrastructure_failure() -> None:
    # A retry asks for the same devices from the same inventory, so it is never retried.
    assert issubclass(letify.InsufficientDevices, letify.LetifyError)
    assert not issubclass(letify.InsufficientDevices, letify.RuntimeFailure)
