"""The declaration surface: what a function needs and where it belongs.

These tests pin the parts of a declaration that are decided before anything runs: the
environment key, the placement of device and host and lifetime, and the shape of a
declared search space. Anything that needs a live session is in test_runtime.py or
test_core.py instead.

Spec sections pinned here: "Declaration surface", "The three placements", "Invocation",
"Fan-out", "Environment" and "Module shipping".
"""

from __future__ import annotations

from pathlib import Path

import pytest

import letify
from letify.declare.env import Env
from letify.declare.instance import AnyInstance, Host, Instance
from letify.declare.sweep import Sweep, grid, zip_


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    path = tmp_path / "uv.lock"
    path.write_text("version = 1\n", encoding="utf-8")
    return path


# -- Spec: Environment ---------------------------------------------------------


def test_an_env_key_is_stable_for_the_same_declaration() -> None:
    # Pooling and the environment archive cache both key on this, so two declarations
    # that say the same thing have to produce the same key.
    assert Env().key == Env().key
    assert Env().key != Env().pip_install("torch").key


def test_the_env_key_changes_with_the_lock_file_contents(lock: Path) -> None:
    env = Env(lock=str(lock))
    first = env.key
    lock.write_text("version = 1\nchanged = true\n", encoding="utf-8")
    # A changed lock file is a different environment, so it must not reuse the archive
    # built from the old one.
    assert Env(lock=str(lock)).key != first


def test_a_missing_lock_file_says_so_rather_than_failing(tmp_path: Path) -> None:
    # A project with no uv.lock still has a usable environment declaration.
    assert Env(lock=str(tmp_path / "absent.lock")).lock_digest == "nolock"


def test_the_lock_digest_is_the_hash_of_the_file(lock: Path) -> None:
    env = Env(lock=str(lock))
    assert env.lock_digest != "nolock"
    assert env.lock_digest == Env(lock=str(lock)).lock_digest


@pytest.mark.parametrize(
    "refine",
    [
        lambda env: env.pip_install("torch"),
        lambda env: env.run("apt-get install -y git"),
        lambda env: env.vars(HF_HOME="/opt/cache"),
    ],
)
def test_each_refinement_produces_a_new_environment(refine) -> None:
    base = Env()
    refined = refine(base)
    assert refined is not base
    assert refined.key != base.key


def test_refinements_accumulate_rather_than_replacing() -> None:
    env = Env().pip_install("torch").pip_install("trl", "peft")
    assert env.packages == ("torch", "trl", "peft")
    env = env.run("nvidia-smi").run("uv --version")
    assert env.commands == ("nvidia-smi", "uv --version")


def test_variables_are_recorded_in_a_stable_order() -> None:
    # The key hashes them, so two declarations written in different orders have to agree.
    assert Env().vars(B="2", A="1").variables == (("A", "1"), ("B", "2"))
    assert Env().vars(A="1", B="2").key == Env().vars(B="2", A="1").key


def test_the_order_packages_were_added_in_does_not_change_the_key() -> None:
    assert Env().pip_install("trl", "peft").key == Env().pip_install("peft", "trl").key


def test_a_declared_interpreter_is_part_of_the_environment() -> None:
    assert Env(python="3.12").key != Env(python="3.11").key


def test_from_lock_and_the_constructor_are_the_same_declaration(lock: Path) -> None:
    assert Env.from_lock(str(lock)) == Env(lock=str(lock))


# -- Spec: Module shipping -----------------------------------------------------


def test_ship_records_the_modules_that_travel_with_the_call() -> None:
    env = Env().ship("my_project").ship("helpers")
    assert env.ship_modules == ("my_project", "helpers")


def test_shipping_a_module_changes_the_env_key() -> None:
    assert Env().ship("my_project").key != Env().key


# -- Spec: The three placements ------------------------------------------------


def test_the_host_defaults_to_this_process(cpu: letify.Instance) -> None:
    assert cpu.placement is Host.local
    assert cpu.host is None


def test_an_instance_has_no_placement_of_its_own() -> None:
    # Where the host code runs is said by the declaration's host and nowhere else, so there is
    # one place to read to know where a function runs.
    assert not hasattr(Instance, "on_host")
    assert not hasattr(AnyInstance, "on_host")


def test_host_accepts_the_plain_lowercase_string() -> None:
    assert Host("remote") is Host.remote
    assert Host.local == "local"


def test_the_two_host_placements_are_named_values_on_the_package() -> None:
    # Two named values say everything a declaration needs, so the enum class that holds them
    # is not one more public name to learn.
    assert letify.local is Host.local
    assert letify.remote is Host.remote
    assert letify.remote == "remote"
    assert not hasattr(letify, "Host")
    assert "Host" not in letify.__all__
    assert {"local", "remote"} <= set(letify.__all__)


def test_a_declaration_takes_the_named_placement(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host=letify.remote)
    def noop() -> None:
        return None

    assert noop.device.placement is letify.remote


def test_the_accelerator_name_falls_back_from_gpu_to_tpu_to_the_cpu(let: letify.Launcher) -> None:
    provider = let.providers.local
    assert Instance(provider, gpu="H100").accelerator == "H100"
    assert Instance(provider, tpu="v5e1").accelerator == "v5e1"
    assert Instance(provider).accelerator == "cpu"


def test_the_pool_key_names_provider_accelerator_placement_and_purchase(
    let: letify.Launcher,
) -> None:
    # Spec "Pooling": the pool key is the instance key joined with the environment key,
    # so anything that makes two instances non-interchangeable belongs in it.
    provider = let.providers.local
    on_demand = Instance(provider, gpu="H100")._placed("remote")
    assert on_demand.key == "local:H100:remote:x1:ondemand"
    assert Instance(provider, gpu="H100", spot=True)._placed("remote").key.endswith(":spot")
    assert on_demand.key != Instance(provider, gpu="H100")._placed("local").key


def test_the_core_count_comes_from_the_instance_not_the_declaration(cpu: letify.Instance) -> None:
    # Core count and memory arrive with the shape the provider registered, so there is
    # nothing for a declaration to ask for.
    assert "cpus" in set(cpu.__slots__)
    assert not hasattr(cpu, "with_cpus")


def test_an_instance_names_itself_by_provider_accelerator_and_placement(
    cpu: letify.Instance,
) -> None:
    # This string is what an error message about a failed call shows.
    assert repr(cpu) == "<Instance local:cpu host=local>"


def test_a_request_without_a_provider_carries_only_the_accelerator() -> None:
    request = AnyInstance(accelerator="G4")
    assert request._placed(None) is request
    assert request._placed("remote").host is Host.remote
    assert repr(request) == "<AnyInstance G4>"


# -- Spec: Fan-out -------------------------------------------------------------


def test_grid_takes_the_product_of_its_axes() -> None:
    space = grid(lr=[1e-4, 3e-4], bs=[16, 32])
    assert len(space) == 4
    assert {(p["lr"], p["bs"]) for p in space} == {
        (1e-4, 16),
        (1e-4, 32),
        (3e-4, 16),
        (3e-4, 32),
    }


def test_zip_pairs_the_axes_position_by_position() -> None:
    space = zip_(lr=[1e-4, 3e-4], bs=[16, 32])
    assert [(p["lr"], p["bs"]) for p in space] == [(1e-4, 16), (3e-4, 32)]


def test_zip_refuses_axes_of_unequal_length_and_names_the_lengths() -> None:
    with pytest.raises(ValueError, match=r"lr=3, bs=2"):
        zip_(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32])


def test_a_scalar_axis_stays_fixed_across_the_space() -> None:
    assert [p["bs"] for p in grid(lr=[1e-4, 3e-4], bs=32)] == [32, 32]
    assert [p["bs"] for p in zip_(lr=[1e-4, 3e-4], bs=32)] == [32, 32]


def test_a_string_axis_is_one_value_rather_than_its_characters() -> None:
    # Otherwise model="gpt" would silently become a three point sweep.
    assert [p["model"] for p in grid(model="gpt")] == ["gpt"]


def test_a_space_over_nothing_has_no_points() -> None:
    assert len(grid()) == 0
    assert len(zip_()) == 0


def test_two_spaces_combine_and_drop_duplicate_points() -> None:
    combined = grid(lr=[1e-4, 3e-4]) | grid(lr=[3e-4, 1e-3])
    assert [p["lr"] for p in combined] == [1e-4, 3e-4, 1e-3]


def test_with_fixed_adds_arguments_constant_across_every_point() -> None:
    space = grid(lr=[1e-4, 3e-4]).with_fixed(bs=32)
    assert all(point["bs"] == 32 for point in space)
    assert len(space) == 2


def test_a_point_overrides_a_fixed_argument_of_the_same_name() -> None:
    assert [p["lr"] for p in grid(lr=[1e-4]).with_fixed(lr=9.0)] == [1e-4]


def test_a_space_names_its_size_and_its_axes() -> None:
    # What a user sees when they print a space before spending money on it.
    assert repr(grid(lr=[1e-4, 3e-4], bs=[16])) == "<Sweep 2 points over ['bs', 'lr']>"


def test_a_space_is_a_finite_set_of_keyword_combinations() -> None:
    space = Sweep(({"lr": 1e-4},))
    assert list(space) == [{"lr": 1e-4}]
    assert len(space) == 1


# -- Spec: Invocation, values refused at declaration time ----------------------


def test_an_unrecognized_host_placement_names_both_options(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    with pytest.raises(ValueError, match=r"host='local'.*host='remote'"):

        @let.function(device=cpu, host="somewhere")
        def noop() -> None:
            return None


def test_how_long_a_session_lives_is_not_a_declaration_argument(let, cpu) -> None:
    # Keeping sessions is a with let.keep_alive() block around the calls, not a property of one
    # function, so the declaration refuses the argument rather than ignoring it.
    assert not hasattr(letify, "Lifetime")
    with pytest.raises(TypeError):

        @let.function(device=cpu, host=letify.remote, lifetime="process")
        def noop() -> None:
            return None


def test_only_one_search_space_may_be_passed_per_call(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    # Two would make the point count the product of two arguments rather than something
    # visible in one place.
    @let.function(device=cpu, host="remote")
    def train(lr: float, bs: int) -> None:
        return None

    with pytest.raises(TypeError, match="only one search space"):
        train(grid(lr=[1e-4]), grid(bs=[16]))


def test_local_runs_the_body_in_the_calling_process(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    # For testing a body with no provider at all. No runtime is started.
    @let.function(device=cpu, host="remote")
    def double(x: int) -> int:
        return x * 2

    assert double.local(3) == 6
    assert let.pool.live == []


def test_a_declaration_keeps_the_name_and_docstring_of_the_function(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    # A declared function is called like any other, so it has to look like the one that
    # was written.
    @let.function(device=cpu, host="remote")
    def train(lr: float) -> float:
        """Train the model."""
        return lr

    assert train.__name__ == "train"
    assert train.__doc__ == "Train the model."
    assert repr(train) == "<Function train sync on <Instance local:cpu host=remote>>"


def test_an_async_declaration_is_recognized_at_the_def_site(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    # Whether a call blocks is decided by the def, not by the call site.
    @let.function(device=cpu, host="remote")
    async def fetch() -> int:
        return 1

    @let.function(device=cpu, host="remote")
    def compute() -> int:
        return 1

    assert fetch.is_async is True
    assert compute.is_async is False


def test_a_declaration_is_registered_with_the_launcher_that_made_it(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    @let.function(device=cpu, host="remote")
    def noop() -> None:
        return None

    assert noop in let.functions
    assert noop.device.placement is Host.remote


# -- Spec: Inventory, a device count on the instance ----------------------------


def test_one_device_is_the_default_and_multiplication_asks_for_more(let) -> None:
    # Two cards in one session is a property of the shape being asked for, not a separate
    # argument, so it travels with the value the declaration already carries.
    one = let.providers.local.CPU
    assert one.devices == 1
    assert (one * 2).devices == 2
    assert (3 * one).devices == 3
    # A value, so the original is untouched and can be reused.
    assert one.devices == 1


def test_the_device_count_is_part_of_the_pool_key(let) -> None:
    # A session holding two cards is not interchangeable with one holding one.
    one = let.providers.local.CPU
    assert one.key != (one * 2).key
    assert (one * 2).key == (one * 2).key


def test_asking_for_no_devices_or_a_fraction_is_refused(let) -> None:
    one = let.providers.local.CPU
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least one device"):
            one * bad
    with pytest.raises(TypeError):
        one * 1.5


def test_a_multi_device_instance_says_so_when_printed(let) -> None:
    assert "x2" in repr(let.providers.local.CPU * 2)
