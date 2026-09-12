"""Tests that exercise the real code path through the local provider.

Nothing here is mocked. The local provider ships the same serialized call through
the same driver script that a remote runtime would, so a passing test means the
protocol, the pool and the scope all work rather than that a stub returned what it
was told to.
"""

from __future__ import annotations

import asyncio

import pytest

import letify


@pytest.fixture
def let() -> letify.Launcher:
    # home=False keeps the developer's own accounts out of the test run.
    return letify.Launcher(home=False)


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
    left = letify.grid(lr=[1e-4, 3e-4])
    right = letify.grid(lr=[3e-4, 1e-3])
    assert len(left | right) == 3


def test_instance_carries_provider_and_placement(cpu: letify.Instance) -> None:
    assert cpu.provider.alias == "local"
    assert cpu.placement == "local"
    assert cpu(cpus=8).cpus == 8
    # An override produces a new value rather than mutating the registered one.
    assert cpu.cpus is None


def test_local_provider_is_persistent(let: letify.Launcher) -> None:
    assert let.providers.local.persistent is True


def test_unknown_provider_names_what_is_declared(let: letify.Launcher) -> None:
    with pytest.raises(letify.UnknownProvider, match="Declared: local"):
        let.provider("nope")


def test_unknown_instance_lists_what_exists(let: letify.Launcher) -> None:
    with pytest.raises(letify.UnknownInstance, match="does not offer"):
        let.providers.local.H100  # noqa: B018


# -- invocation ----------------------------------------------------------------


def test_a_sync_call_returns_its_value(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu)
    def double(x: int) -> int:
        return x * 2

    with let.run():
        assert double(x=21) == 42


def test_a_call_outside_the_scope_is_refused(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu)
    def noop() -> None:
        return None

    with pytest.raises(letify.NotRunning, match="outside a run scope"):
        noop()


def test_a_space_fans_out_to_one_call_per_point(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu, concurrency=2)
    def identity(lr: float, bs: int) -> tuple[float, int]:
        return lr, bs

    with let.run():
        results = identity(letify.grid(lr=[1e-4, 3e-4], bs=[16, 32]))

    assert len(results) == 4
    assert set(results) == {(1e-4, 16), (1e-4, 32), (3e-4, 16), (3e-4, 32)}


def test_an_async_declaration_is_awaitable(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu)
    async def double(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    async def run() -> int:
        with let.run():
            return await double(x=4)

    assert asyncio.run(run()) == 8


def test_an_async_space_can_be_iterated_as_it_completes(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(gpu=cpu, concurrency=3)
    async def square(n: int) -> int:
        return n * n

    async def run() -> list[int]:
        with let.run():
            return [r async for r in square(letify.grid(n=[1, 2, 3]))]

    assert sorted(asyncio.run(run())) == [1, 4, 9]


def test_local_runs_the_body_without_a_runtime(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu)
    def double(x: int) -> int:
        return x * 2

    # No scope is open, so this proves the body ran in this process.
    assert double.local(3) == 6


# -- failure -------------------------------------------------------------------


def test_a_remote_exception_arrives_with_its_traceback(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(gpu=cpu, retries=0)
    def boom() -> None:
        raise ValueError("intentional")

    with let.run(), pytest.raises(letify.RemoteError) as caught:
        boom()

    assert "intentional" in str(caught.value)
    assert "ValueError" in caught.value.remote_traceback


def test_a_handle_from_another_runtime_is_refused(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(gpu=cpu)
    def consume(value: object) -> object:
        return value

    stranger = letify.Handle(runtime="elsewhere", object_id="00", type_name="dict")
    with let.run(), pytest.raises(letify.HandleScopeError, match="belongs to runtime"):
        consume(value=stranger)


def test_keep_remote_returns_a_handle(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu, keep_remote=True)
    def build() -> dict[str, int]:
        return {"a": 1}

    with let.run():
        handle = build()

    assert isinstance(handle, letify.Handle)
    assert handle.type_name == "dict"


# -- pool and scope ------------------------------------------------------------


def test_the_scope_tears_every_runtime_down(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu)
    def noop() -> None:
        return None

    with let.run():
        noop()
        assert let.pool.live

    assert let.pool.live == []


def test_runtimes_are_reused_within_a_scope(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu)
    def noop() -> None:
        return None

    with let.run():
        noop()
        first = [r.name for r in let.pool.live]
        noop()
        assert [r.name for r in let.pool.live] == first


def test_the_pool_honours_its_runtime_limit(cpu: letify.Instance) -> None:
    let = letify.Launcher(home=False, max_runtimes=2)
    cpu_here = let.providers.local.CPU

    @let.function(gpu=cpu_here, concurrency=4)
    async def slow(n: int) -> int:
        await asyncio.sleep(0.05)
        return n

    async def run() -> list[int]:
        with let.run():
            results = await slow(letify.grid(n=[1, 2, 3, 4]))
            assert len(let.pool.live) <= 2
            return results

    assert sorted(asyncio.run(run())) == [1, 2, 3, 4]


def test_nested_scopes_only_tear_down_once(let: letify.Launcher, cpu: letify.Instance) -> None:
    @let.function(gpu=cpu)
    def noop() -> None:
        return None

    with let.run():
        noop()
        with let.run():
            noop()
        # The inner scope closed, but the outer one still holds the runtime.
        assert let.pool.live

    assert let.pool.live == []


# -- configuration -------------------------------------------------------------


def test_a_hyphen_in_an_alias_is_rejected(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text('[colab-a]\nkind = "colab"\n', encoding="utf-8")
    with pytest.raises(letify.ConfigError, match="not a Python identifier"):
        letify.Launcher(path, home=False)


def test_a_reserved_alias_is_rejected(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text('[any]\nkind = "colab"\n', encoding="utf-8")
    with pytest.raises(letify.ConfigError, match="reserved"):
        letify.Launcher(path, home=False)


def test_a_secret_is_read_from_the_environment(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".letify"
    path.write_text(
        '[lab]\nkind = "shell"\naddress = "h"\naccess_token_env = "LETIFY_TEST_TOKEN"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "secret-value")
    let = letify.Launcher(path, home=False)
    assert let.config.providers["lab"].secret("access_token") == "secret-value"


def test_declaration_order_sets_any_priority(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text(
        '[first]\nkind = "shell"\naddress = "a"\ngpus = ["A100"]\n'
        '[second]\nkind = "shell"\naddress = "b"\ngpus = ["A100"]\n',
        encoding="utf-8",
    )
    let = letify.Launcher(path, home=False)
    resolved = let.resolve(let.providers.any.A100)
    assert resolved.provider.alias == "first"


def test_shell_defaults_to_ephemeral_and_can_be_overridden(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text(
        '[a]\nkind = "shell"\naddress = "a"\n'
        '[b]\nkind = "shell"\naddress = "b"\npersistent = true\n',
        encoding="utf-8",
    )
    let = letify.Launcher(path, home=False)
    assert let.providers.a.persistence == "ephemeral"
    assert let.providers.b.persistence == "persistent"
    # Storage decides the default placement, so persistence flips it.
    assert let.providers.a.default_cpu_placement == "local"
    assert let.providers.b.default_cpu_placement == "remote"


def test_colab_refuses_call_forwarding(tmp_path) -> None:
    path = tmp_path / ".letify"
    path.write_text('[c]\nkind = "colab"\naccount = "someone@example.com"\n', encoding="utf-8")
    let = letify.Launcher(path, home=False)
    colab = let.providers.c
    # Ephemeral storage but no fast path, so shipping is the default here.
    assert colab.default_cpu_placement == "remote"
    with pytest.raises(letify.UnsupportedMode, match="does not support cpu='local'"):
        colab.start(colab.G4(cpu="local"), letify.Env(), name="test")


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
