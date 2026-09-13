"""Tests that exercise the real code path through the local provider.

Nothing here is mocked. The local provider starts the same worker behind the same
framed protocol that a remote runtime would, so a passing test means the protocol, the
object table, the pool and the release rule all work rather than that a stub returned
what it was told to.
"""

from __future__ import annotations

import asyncio

import pytest

import letify


@pytest.fixture
def let() -> letify.Launcher:
    # home=False keeps the developer's own accounts out of the test run.
    return letify.Launcher(home=False, announce=False)


@pytest.fixture
def cpu(let: letify.Launcher) -> letify.Instance:
    return let.providers.local.CPU


# -- declarations --------------------------------------------------------------


def test_env_key_is_stable_for_the_same_declaration() -> None:
    assert letify.Env().key == letify.Env().key
    assert letify.Env().key != letify.Env().pip_install("torch").key


def test_grid_takes_the_product_and_zip_pairs() -> None:
    assert len(letify.grid(lr=[1e-4, 3e-4], bs=[16, 32])) == 4
    assert len(letify.zip(lr=[1e-4, 3e-4], bs=[16, 32])) == 2


def test_zip_rejects_axes_of_different_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        letify.zip(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32])


def test_grid_union_drops_duplicates() -> None:
    assert len(letify.grid(lr=[1e-4, 3e-4]) | letify.grid(lr=[3e-4, 1e-3])) == 3


def test_the_host_defaults_to_this_process(cpu: letify.Instance) -> None:
    assert cpu.placement is letify.Host.local
    assert cpu.on_host("remote").placement is letify.Host.remote
    # An override produces a new value rather than mutating the registered shape.
    assert cpu.host is None


def test_an_unknown_host_placement_names_both_options(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    with pytest.raises(ValueError, match="host='local'"):

        @let.function(device=cpu, host="somewhere")
        def noop() -> None:
            return None


def test_core_count_comes_from_the_instance_not_the_declaration(cpu: letify.Instance) -> None:
    # cpus is reported by discovery, so there is nothing for a declaration to ask for.
    assert not hasattr(cpu, "with_cpus")
    assert "cpus" in {field for field in cpu.__slots__}


def test_local_provider_is_persistent(let: letify.Launcher) -> None:
    assert let.providers.local.persistent is True


def test_unknown_provider_names_what_is_declared(let: letify.Launcher) -> None:
    with pytest.raises(letify.UnknownProvider, match="Declared: local"):
        let.provider("nope")


def test_unknown_device_lists_what_exists(let: letify.Launcher) -> None:
    with pytest.raises(letify.UnknownInstance, match="does not offer"):
        let.providers.local.H100  # noqa: B018


# -- invocation ----------------------------------------------------------------


def test_a_sync_call_returns_its_value(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host="remote")
    def double(x: int) -> int:
        return x * 2

    assert double(x=21) == 42


def test_a_call_needs_no_scope(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host="remote")
    def double(x: int) -> int:
        return x * 2

    # No context manager anywhere. The runtime starts on the call and dies after it.
    assert double(x=1) == 2
    assert let.pool.live == []


def test_a_space_fans_out_to_one_call_per_point(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote", concurrency=2)
    def identity(lr: float, bs: int) -> tuple[float, int]:
        return lr, bs

    results = identity(letify.grid(lr=[1e-4, 3e-4], bs=[16, 32]))
    assert len(results) == 4
    assert set(results) == {(1e-4, 16), (1e-4, 32), (3e-4, 16), (3e-4, 32)}


def test_an_async_declaration_is_awaitable(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host="remote")
    async def double(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    assert asyncio.run(double(x=4)) == 8


def test_an_async_space_can_be_iterated_as_it_completes(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote", concurrency=3)
    async def square(n: int) -> int:
        return n * n

    async def run() -> list[int]:
        return [r async for r in square(letify.grid(n=[1, 2, 3]))]

    assert sorted(asyncio.run(run())) == [1, 4, 9]


def test_local_runs_the_body_in_this_process(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host="remote")
    def double(x: int) -> int:
        return x * 2

    assert double.local(3) == 6


# -- the persistent worker -----------------------------------------------------


def test_a_kept_value_stays_in_the_runtime(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host="remote", keep_remote=True, warm=True)
    def build() -> dict[str, list[int]]:
        return {"weights": [1, 2, 3]}

    @let.function(device=cpu, host="remote", warm=True)
    def total(model: dict[str, list[int]]) -> int:
        return sum(model["weights"])

    handle = build()
    assert isinstance(handle, letify.Handle)
    # Resolving it in a later call is what the persistent worker exists for.
    assert total(model=handle) == 6
    let.release()


def test_a_large_argument_is_sent_once(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(device=cpu, host="remote", warm=True)
    def size(payload: bytes) -> int:
        return len(payload)

    big = b"x" * 200_000
    assert size(payload=big) == 200_000
    assert size(payload=big) == 200_000

    runtime = let.pool.live[0]
    stat = runtime.stat()
    # One blob, not two, even though the argument was passed twice.
    assert stat["blobs"] == 1
    let.release()


def test_the_worker_is_one_process_across_calls(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote", warm=True)
    def noop() -> None:
        return None

    noop()
    first = let.pool.live[0].stat()["pid"]
    noop()
    assert let.pool.live[0].stat()["pid"] == first
    let.release()


def test_files_written_into_a_runtime_survive_between_calls(
    let: letify.Launcher, cpu: letify.Instance, tmp_path
) -> None:
    @let.function(device=cpu, host="remote", warm=True)
    def read(path: str) -> str:
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    @let.function(device=cpu, host="remote", warm=True)
    def touch() -> None:
        return None

    touch()
    runtime = let.pool.live[0]
    target = str(tmp_path / "materialized.txt")
    runtime.put_bytes(b"from the store", target)
    assert read(path=target) == "from the store"
    payload, digest = runtime.get_bytes(target)
    assert payload == b"from the store"
    assert digest
    let.release()


# -- failure -------------------------------------------------------------------


def test_a_remote_exception_arrives_with_its_traceback(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote", retries=0)
    def boom() -> None:
        raise ValueError("intentional")

    with pytest.raises(letify.RemoteError) as caught:
        boom()

    assert "intentional" in str(caught.value)
    assert "ValueError" in caught.value.remote_traceback


def test_a_handle_from_another_runtime_is_refused(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote")
    def consume(value: object) -> object:
        return value

    stranger = letify.Handle(runtime="elsewhere", object_id="00", type_name="dict")
    with pytest.raises(letify.HandleScopeError, match="belongs to runtime"):
        consume(value=stranger)


def test_modal_refuses_a_local_host(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text('[m]\nkind = "modal"\n', encoding="utf-8")
    let = letify.Launcher(path, home=False, announce=False)
    modal = let.providers.m
    with pytest.raises(letify.UnsupportedMode, match="no device to forward"):
        modal.check_mode(modal.H100.on_host("local"))


# -- release rule --------------------------------------------------------------


def test_a_runtime_dies_when_its_call_finishes(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote")
    def noop() -> None:
        return None

    noop()
    assert let.pool.live == []


def test_a_warm_declaration_keeps_its_runtime(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote", warm=True)
    def noop() -> None:
        return None

    noop()
    assert len(let.pool.live) == 1
    noop()
    assert len(let.pool.live) == 1
    assert let.release()
    assert let.pool.live == []


def test_warm_declarations_on_one_device_share_a_runtime(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote", warm=True)
    def first() -> int:
        return 1

    @let.function(device=cpu, host="remote", warm=True)
    def second() -> int:
        return 2

    first()
    second()
    # Pooling is by instance and environment, so two declarations need no block to
    # share a session.
    assert len(let.pool.live) == 1
    let.release()


def test_a_sweep_holds_one_set_of_runtimes(let: letify.Launcher, cpu: letify.Instance) -> None:
    limited = letify.Launcher(home=False, announce=False, max_runtimes=2)
    here = limited.providers.local.CPU

    @limited.function(device=here, host="remote", concurrency=4)
    async def slow(n: int) -> int:
        await asyncio.sleep(0.05)
        return n

    async def run() -> list[int]:
        results = await slow(letify.grid(n=[1, 2, 3, 4]))
        assert len(limited.pool.live) <= 2
        return results

    assert sorted(asyncio.run(run())) == [1, 2, 3, 4]
    assert limited.pool.live == []


# -- configuration -------------------------------------------------------------


def test_a_hyphen_in_an_alias_is_rejected(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text('[colab-a]\nkind = "colab"\n', encoding="utf-8")
    with pytest.raises(letify.ConfigError, match="not a Python identifier"):
        letify.Launcher(path, home=False)


def test_a_reserved_alias_is_rejected(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text('[devices]\nkind = "colab"\n', encoding="utf-8")
    with pytest.raises(letify.ConfigError, match="reserved"):
        letify.Launcher(path, home=False)


def test_a_secret_is_read_from_the_environment(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".letify"
    path.write_text(
        '[lab]\nkind = "shell"\naddress = "h"\naccess_token_env = "LETIFY_TEST_TOKEN"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "secret-value")
    let = letify.Launcher(path, home=False, announce=False)
    assert let.config.providers["lab"].secret("access_token") == "secret-value"


def test_declaration_order_sets_any_priority(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text(
        '[first]\nkind = "shell"\naddress = "a"\ngpus = ["A100"]\n'
        '[second]\nkind = "shell"\naddress = "b"\ngpus = ["A100"]\n',
        encoding="utf-8",
    )
    let = letify.Launcher(path, home=False, announce=False)
    assert let.resolve(let.providers.any.A100).provider.alias == "first"


def test_shell_defaults_to_ephemeral_and_can_be_overridden(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text(
        '[a]\nkind = "shell"\naddress = "a"\n'
        '[b]\nkind = "shell"\naddress = "b"\npersistent = true\n',
        encoding="utf-8",
    )
    let = letify.Launcher(path, home=False, announce=False)
    assert let.providers.a.persistence == "ephemeral"
    assert let.providers.b.persistence == "persistent"


def test_colab_reports_its_round_trip_rather_than_refusing(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text('[c]\nkind = "colab"\naccount = "someone@example.com"\n', encoding="utf-8")
    let = letify.Launcher(path, home=False, announce=False)
    colab = let.providers.c
    # Slow is not a reason to refuse, so the provider carries a number to warn with.
    assert colab.has_fast_path is False
    assert colab.expected_round_trip_ms
    assert "G4" in colab.instances
    assert colab.RTX_PRO_6000.gpu == "G4"


# -- store ---------------------------------------------------------------------


def test_the_store_addresses_by_content(tmp_path) -> None:
    from letify.store import FilesystemBackend, Store

    store = Store(FilesystemBackend(tmp_path))
    first = store.put_bytes(b"same contents")
    second = store.put_bytes(b"same contents")
    assert first.digest == second.digest
    assert store.get_bytes(first.digest) == b"same contents"


def test_the_store_skips_what_it_already_holds(tmp_path) -> None:
    from letify.store import FilesystemBackend, Store

    store = Store(FilesystemBackend(tmp_path))
    payload = tmp_path / "data.bin"
    payload.write_bytes(b"x" * 100)
    store.put_file(payload)
    upload, held = store.plan_upload([payload])
    assert upload == []
    assert len(held) == 1


def test_refs_point_at_blobs(tmp_path) -> None:
    from letify.store import FilesystemBackend, Store

    store = Store(FilesystemBackend(tmp_path))
    info = store.put_bytes(b"checkpoint")
    store.point("ckpt/run-1", info.digest)
    assert store.resolve("ckpt/run-1") == info.digest
    assert store.resolve("ckpt/missing") is None


def test_a_tree_packs_into_one_blob_and_unpacks(tmp_path) -> None:
    from letify.store import FilesystemBackend, Store

    source = tmp_path / "env"
    (source / "nested").mkdir(parents=True)
    (source / "a.txt").write_text("a", encoding="utf-8")
    (source / "nested" / "b.txt").write_text("b", encoding="utf-8")

    store = Store(FilesystemBackend(tmp_path / "store"))
    info = store.put_tree(source, key="env/abc")
    assert store.resolve("env/abc") == info.digest

    restored = store.fetch_tree(info.digest, tmp_path / "out")
    assert (restored / "env" / "nested" / "b.txt").read_text(encoding="utf-8") == "b"


# -- remoting ------------------------------------------------------------------


def test_the_efficiency_formula_matches_the_documented_numbers() -> None:
    from letify.remoting import efficiency

    # A 0.5 s NVFP4 micro step with three synchronizations at a 150 ms round trip.
    assert round(efficiency(0.5, 3, 150.0), 2) == 0.53
    # The same step once synchronization is down to one per optimizer step of eight.
    assert round(efficiency(4.0, 1, 150.0), 2) == 0.96


def test_forwarding_reports_what_it_needs() -> None:
    from letify.remoting import probe

    capability = probe()
    # The shim is a Rust component, so a source checkout says plainly that it is absent.
    assert isinstance(capability.usable, bool)
    assert capability.explain()
