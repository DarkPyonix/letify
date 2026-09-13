"""End to end tests through the local provider.

Nothing here is mocked. The local provider starts the same worker behind the same framed
protocol that a remote runtime would, so a passing test means the protocol, the object
table, the pool and the release rule all work rather than that a stub returned what it
was told to.

Narrower tests live beside this file: declaration values in test_declare.py, the wire in
test_protocol.py, channels and the pool in test_runtime.py, providers in
test_providers.py, storage in test_store.py.

Spec sections pinned here: "Invocation", "Concurrency", "Call protocol", "Handles",
"Argument addressing", "Failure and retry", "Pooling" and "Lifetime", which covers keep_alive.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

import letify

# -- Spec: Invocation ----------------------------------------------------------


def test_a_sync_call_returns_its_value(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host="remote")
    def double(x: int) -> int:
        return x * 2

    assert double(x=21) == 42


def test_a_call_needs_no_scope(let: letify.Launcher, cpu: letify.Instance) -> None:
    # No context manager anywhere. The runtime starts on the call and ends with it.
    @let.function(device=cpu, host="remote")
    def double(x: int) -> int:
        return x * 2

    assert double(x=1) == 2
    assert let.pool.live == []


def test_an_async_declaration_returns_a_coroutine_the_standard_library_accepts(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote")
    async def double(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    call = double(x=4)
    assert inspect.iscoroutine(call)
    assert asyncio.run(call) == 8


def test_an_async_body_can_await_inside_the_runtime(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    # The body runs to completion on the far side, which is what lets it use await.
    @let.function(device=cpu, host="remote")
    async def work() -> str:
        import asyncio as remote_asyncio

        await remote_asyncio.sleep(0.01)
        return "finished"

    assert asyncio.run(work()) == "finished"


# -- Spec: Concurrency ---------------------------------------------------------


def test_gathered_calls_return_their_values_in_the_order_they_were_made(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote")
    async def slow(n: int) -> int:
        import asyncio as remote_asyncio

        # The later calls finish first, so completion order is not call order.
        await remote_asyncio.sleep(0.05 / n)
        return n

    async def run() -> list[int]:
        with let.keep_alive():
            return list(await asyncio.gather(*(slow(n) for n in (1, 2, 3))))

    assert asyncio.run(run()) == [1, 2, 3]


# -- Spec: Handles -------------------------------------------------------------


def test_a_kept_value_stays_in_the_runtime(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host=letify.remote, keep_remote=True)
    def build() -> dict[str, list[int]]:
        return {"weights": [1, 2, 3]}

    @let.function(device=cpu, host=letify.remote)
    def total(model: dict[str, list[int]]) -> int:
        return sum(model["weights"])

    with let.keep_alive():
        handle = build()
        assert isinstance(handle, letify.Handle)
        assert handle.type_name == "dict"
        # Resolving it in a later call is what the persistent worker exists for.
        assert total(model=handle) == 6


def test_a_handle_from_another_runtime_is_refused(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host=letify.remote)
    def consume(value: object) -> object:
        return value

    stranger = letify.Handle(runtime="elsewhere", object_id="00", type_name="dict")
    with pytest.raises(letify.HandleScopeError, match="belongs to runtime"):
        consume(value=stranger)


# -- Spec: Argument addressing -------------------------------------------------


def test_a_large_argument_is_sent_once(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host=letify.remote)
    def size(payload: bytes) -> int:
        return len(payload)

    big = b"x" * 200_000
    with let.keep_alive():
        assert size(payload=big) == 200_000
        assert size(payload=big) == 200_000
        stat = let.pool.live[0].stat()
    # One blob, not two, even though the argument was passed twice.
    assert stat["blobs"] == 1
    assert stat["blob_bytes"] > 200_000 - 1


def test_a_small_argument_travels_with_the_call(let: letify.Launcher, cpu: letify.Instance) -> None:
    # Below the inline limit there is nothing to gain from a second round trip.
    @let.function(device=cpu, host=letify.remote)
    def size(payload: bytes) -> int:
        return len(payload)

    with let.keep_alive():
        assert size(payload=b"x" * 1024) == 1024
        assert let.pool.live[0].stat()["blobs"] == 0


# -- Spec: Channels, what one living process buys ------------------------------


def test_the_worker_is_one_process_across_calls(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host=letify.remote)
    def noop() -> None:
        return None

    with let.keep_alive():
        noop()
        first = let.pool.live[0].stat()["pid"]
        noop()
        assert let.pool.live[0].stat()["pid"] == first


def test_files_written_into_a_runtime_survive_between_calls(
    let: letify.Launcher, cpu: letify.Instance, tmp_path
) -> None:
    @let.function(device=cpu, host=letify.remote)
    def read(path: str) -> str:
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    @let.function(device=cpu, host=letify.remote)
    def touch() -> None:
        return None

    with let.keep_alive():
        touch()
        runtime = let.pool.live[0]
        target = str(tmp_path / "materialized.txt")
        runtime.put_bytes(b"from the store", target)
        assert read(path=target) == "from the store"
        payload, digest = runtime.get_bytes(target)
    assert payload == b"from the store"
    assert digest


def test_the_prints_a_body_made_reach_the_caller(let, cpu, capsys) -> None:
    # The user's own stdout comes back separately from the outcome, and letify passes it
    # through rather than swallowing it.
    @let.function(device=cpu, host=letify.remote)
    def train() -> int:
        print("epoch 1 loss 0.5")
        return 1

    assert train() == 1
    assert "epoch 1 loss 0.5" in capsys.readouterr().err


# -- Spec: Failure and retry ---------------------------------------------------


def test_a_remote_exception_arrives_with_its_traceback(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host=letify.remote, retries=0)
    def boom() -> None:
        raise ValueError("intentional")

    with pytest.raises(letify.RemoteError) as caught:
        boom()

    assert "intentional" in str(caught.value)
    assert "ValueError" in caught.value.remote_traceback


# -- Spec: Lifetime and Pooling ------------------------------------------------


def test_a_runtime_dies_when_its_call_finishes(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host=letify.remote)
    def noop() -> None:
        return None

    noop()
    assert let.pool.live == []


def test_keep_alive_keeps_sessions_until_the_block_ends(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host=letify.remote)
    def noop() -> None:
        return None

    with let.keep_alive():
        noop()
        assert len(let.pool.live) == 1
        noop()
        # The second call reused the session instead of starting another.
        assert len(let.pool.live) == 1
    assert let.pool.live == []


def test_only_the_outermost_keep_alive_ends_sessions(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host=letify.remote)
    def noop() -> None:
        return None

    with let.keep_alive():
        with let.keep_alive():
            noop()
        assert len(let.pool.live) == 1
    assert let.pool.live == []


def test_keep_alive_ends_its_sessions_when_the_block_raises(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    # A block has a visible end whichever way it is left, so nothing outlives the code that
    # asked for it.
    @let.function(device=cpu, host=letify.remote)
    def noop() -> None:
        return None

    with pytest.raises(RuntimeError), let.keep_alive():
        noop()
        raise RuntimeError("user code failed")
    assert let.pool.live == []


def test_two_declarations_on_one_device_share_a_session(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host=letify.remote)
    def first() -> int:
        return 1

    @let.function(device=cpu, host=letify.remote)
    def second() -> int:
        return 2

    with let.keep_alive():
        first()
        second()
        # Pooling is by instance and environment, so two declarations share a session with
        # nothing said about it.
        assert len(let.pool.live) == 1


def test_concurrent_calls_run_no_wider_than_the_inventory(launcher_from) -> None:
    # Two cards declared means at most two sessions, however many calls are gathered, and the
    # calls beyond the inventory wait for a card rather than failing.
    limited = launcher_from('[box]\nkind = "local"\n[box.devices]\nCPU = { count = 2 }\n')
    here = limited.provider("box").CPU

    @limited.function(device=here, host=letify.remote)
    async def slow(n: int) -> int:
        import asyncio as remote_asyncio

        await remote_asyncio.sleep(0.05)
        return n

    widest = 0

    async def watch() -> None:
        nonlocal widest
        for _ in range(40):
            widest = max(widest, len(limited.pool.live))
            await asyncio.sleep(0.01)

    async def run() -> list[int]:
        with limited.keep_alive():
            calls = asyncio.gather(*(slow(n) for n in (1, 2, 3, 4)))
            _, results = await asyncio.gather(watch(), calls)
            return list(results)

    assert asyncio.run(run()) == [1, 2, 3, 4]
    assert 1 <= widest <= 2
    assert limited.pool.live == []


def test_overlapping_calls_release_their_sessions_when_the_last_one_finishes(
    launcher_from,
) -> None:
    # Outside keep_alive, calls that overlap in time share one span, so nothing outlives them.
    limited = launcher_from('[box]\nkind = "local"\n[box.devices]\nCPU = { count = 2 }\n')
    here = limited.provider("box").CPU

    @limited.function(device=here, host=letify.remote)
    async def slow(n: int) -> int:
        import asyncio as remote_asyncio

        await remote_asyncio.sleep(0.05)
        return n

    async def run() -> list[int]:
        return list(await asyncio.gather(*(slow(n) for n in (1, 2, 3))))

    assert asyncio.run(run()) == [1, 2, 3]
    assert limited.pool.live == []


def test_the_providers_holding_a_session_can_be_named(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    # The quickest answer to what is costing money.
    @let.function(device=cpu, host=letify.remote)
    def noop() -> None:
        return None

    with let.keep_alive():
        noop()
        assert let.providers.active["local"] == [let.pool.live[0].name]
    assert let.providers.active == {}


# -- Spec: Status reporting ----------------------------------------------------


def test_status_counts_the_sessions_against_the_ceiling(let, cpu) -> None:
    # Counts rather than a description, so a reader can see whether the ceiling is the
    # reason a call is waiting.
    empty = let.status()
    assert empty["live"] == 0
    assert empty["busy"] == 0
    assert "max_runtimes" not in empty

    @let.function(device=cpu, host=letify.remote)
    def answer() -> int:
        return 7

    with let.keep_alive():
        assert answer() == 7
        running = let.status()
    assert running["live"] == 1
    assert running["busy"] == 0
    assert running["live"] == len(running["runtimes"])


def test_status_does_not_report_the_pools_own_bookkeeping(let, cpu) -> None:
    # The pool holds a guard so a session released by one call is not ended while an
    # overlapping call is still running. Whether that guard is open is a fact about the pool rather than
    # about what is running, and a boolean sitting among counts gets read as a count.
    report = let.status()
    assert "holding" not in report
    assert all(isinstance(report[field], int) for field in ("live", "busy"))


def test_a_reported_session_says_what_it_is(let, cpu) -> None:
    @let.function(device=cpu, host=letify.remote)
    def answer() -> int:
        return 7

    with let.keep_alive():
        answer()
        row = let.status()["runtimes"][0]
    assert row["provider"] == "local"
    assert row["accelerator"] == cpu.accelerator
    assert row["placement"] == "remote"
    assert "lifetime" not in row
    assert row["busy"] is False
    assert row["idle_seconds"] >= 0


def test_status_describes_this_process_only(let, cpu) -> None:
    # The pool lives in the process that owns it, so a session started elsewhere is not
    # here. What a machine itself is doing is what utilization answers.
    assert let.status()["name"] == let.name
    assert let.status()["live"] == 0


# -- Spec: Pooling, a session is not a value the caller holds -------------------


def test_nothing_public_hands_out_a_session(let) -> None:
    # A caller holding a session has to have asked for it with the same instance the
    # declaration uses, and the declaration folds the host placement into that instance, so
    # asking with the bare one starts a second session holding none of the first one's
    # files. On one machine that still passes, because both see the same disk; on a rented
    # one it fails. So there is nothing to hold.
    public = [name for name in dir(let) if not name.startswith("_")]
    assert "runtime" not in public
    assert "release" not in public
    assert "shutdown" not in public


def test_status_reports_the_inventory_against_what_is_reserved(launcher_from) -> None:
    # So a reader can see whether a call is waiting for a card rather than for a slot in a
    # number somebody guessed.
    let = launcher_from('[lab]\nkind = "local"\n[lab.devices]\nCPU = { count = 2 }\n')
    report = let.status()
    assert report["devices"]["lab"]["CPU"] == {"count": 2, "reserved": 0, "indices": []}
