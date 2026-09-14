"""Elice machines driven through Elice's own ``eci`` command.

Spec sections pinned here: "Elice machines", "Price type" and "Spot preemption".

The ``eci`` binary is the stand-in from conftest: a real executable on PATH answering from
a state file, so the command line, the environment and the JSON letify parses are real.
Only the SSH step that installs the key is replaced, because it needs a live machine.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeResponse, provider_of

import letify
from letify.config.secrets import account_directory
from letify.declare.instance import Instance
from letify.errors import SpotPreempted
from letify.providers import elice as elice_module
from letify.providers.elice import Elice, generate_password
from letify.providers.local import Local


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(elice_module, "POLL_SECONDS", 0)


@pytest.fixture
def account(isolated_home, fake_eci, tmp_path: Path):
    """Build an Elice provider over the fake eci, with the key install recorded."""
    key = tmp_path / "id_test"
    key.write_text("private", encoding="utf-8")
    (tmp_path / "id_test.pub").write_text("ssh-ed25519 AAAA test", encoding="utf-8")

    def build(alias: str = "elice_a100", **options):
        values = {"zone_id": "zone-1", "access_token": "token-1", "key": str(key), **options}
        provider = provider_of(Elice, alias, **values)
        provider.authorized = []
        provider.authorize_key = lambda address, password: provider.authorized.append(
            (address, password)
        )
        return provider

    return build


def runtime_on(provider: Elice, instance: Instance) -> SimpleNamespace:
    """What stop and diagnose read from a runtime: its machine and its instance."""
    return SimpleNamespace(
        name="letify-a100-1", external_id=provider._pending_machine, instance=instance
    )


# -- Spec: Elice machines ------------------------------------------------------


def test_a_missing_eci_names_the_install_command(account, patch_which) -> None:
    provider = account()
    patch_which(elice_module, False)
    with pytest.raises(letify.ProviderUnavailable, match=r"eci-cli/main/scripts/install\.sh"):
        provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")


def test_every_eci_command_carries_the_token_in_the_environment_and_never_as_an_argument(
    account, fake_eci
) -> None:
    provider = account(endpoint="https://portal.gov.elice.cloud/api")
    provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    assert fake_eci.calls
    for call in fake_eci.calls:
        assert "token-1" not in call["argv"]
        assert call["env"]["ECI_API_TOKEN"] == "token-1"
        assert call["env"]["ECI_API_ENDPOINT"] == "https://portal.gov.elice.cloud/api"
        assert call["env"]["ECI_ZONE_ID"] == "zone-1"
        assert call["env"]["ECI_CONFIG"] == str(account_directory("elice_a100") / "eci.yaml")


def test_a_missing_machine_is_launched_with_a_generated_password_kept_owner_only(
    account, fake_eci
) -> None:
    provider = account()
    provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")

    launch = next(c["argv"] for c in fake_eci.calls if c["argv"][:3] == ["compute", "vm", "launch"])
    assert launch[launch.index("--name") + 1] == "letify-elice-a100"
    assert launch[launch.index("--instance-type") + 1] == "G-A100-1"
    assert "--wait" in launch
    assert "--price-type" not in launch
    assert "--image" not in launch and "--size-gib" not in launch

    password_file = account_directory("elice_a100") / "machine_password"
    password = password_file.read_text(encoding="utf-8")
    assert launch[launch.index("--password") + 1] == password
    if sys.platform != "win32":
        assert password_file.stat().st_mode & 0o777 == 0o600
    assert provider._pending_machine == "letify-elice-a100"
    assert provider.address == "203.0.113.1"
    # The key goes on right after the launch, with the password read from its file.
    assert provider.authorized == [("203.0.113.1", password)]


def test_an_account_image_and_disk_size_reach_the_launch(account, fake_eci) -> None:
    provider = account(image="Ubuntu 22.04 LTS", disk_gib=100)
    provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    launch = next(c["argv"] for c in fake_eci.calls if c["argv"][:3] == ["compute", "vm", "launch"])
    assert launch[launch.index("--image") + 1] == "Ubuntu 22.04 LTS"
    assert launch[launch.index("--size-gib") + 1] == "100"


def test_the_instance_type_is_the_one_whose_devices_match_the_declaration(
    account, fake_eci
) -> None:
    provider = account()
    provider.create_session(Instance(provider, gpu="A100") * 2, "letify-a100x2-1")
    launch = next(c["argv"] for c in fake_eci.calls if c["argv"][:3] == ["compute", "vm", "launch"])
    assert launch[launch.index("--instance-type") + 1] == "G-A100-2"


def test_a_cpu_instance_takes_the_cpu_type_with_the_fewest_cores(account, fake_eci) -> None:
    provider = account()
    provider.create_session(Instance(provider, gpu=None), "letify-cpu-1")
    launch = next(c["argv"] for c in fake_eci.calls if c["argv"][:3] == ["compute", "vm", "launch"])
    assert launch[launch.index("--instance-type") + 1] == "C-2"


def test_an_accelerator_no_instance_type_offers_lists_what_is_offered(account) -> None:
    provider = account()
    with pytest.raises(letify.ProviderUnavailable, match="G-A100-1"):
        provider.create_session(Instance(provider, gpu="H100"), "letify-h100-1")


def test_a_second_start_reuses_the_machine_by_name_and_starts_it(account, fake_eci) -> None:
    provider = account()
    instance = Instance(provider, gpu="A100")
    provider.create_session(instance, "letify-a100-1")
    provider.stop(runtime_on(provider, instance))

    again = account()
    again.create_session(instance, "letify-a100-2")
    commands = fake_eci.commands()
    assert commands.count("compute vm launch") == 1
    assert commands[-1].startswith("compute vm get") or "compute vm start letify-elice-a100" in (
        commands
    )
    assert "compute vm start letify-elice-a100" in commands
    assert fake_eci.read()["vms"][0]["status"] == "started"
    # A machine letify did not just launch already has the key.
    assert again.authorized == []


def test_a_machine_in_transition_is_waited_for_before_it_is_started(account, fake_eci) -> None:
    fake_eci.set(
        vms=[
            {
                "id": "vm-1",
                "name": "letify-elice-a100",
                "status": "stopping",
                "pricing_type": "ondemand",
                "public_ip": "203.0.113.9",
            }
        ],
        transitions={"letify-elice-a100": ["stopping", "idle"]},
    )
    provider = account()
    provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    commands = fake_eci.commands()
    start = commands.index("compute vm start letify-elice-a100")
    assert commands[:start].count("compute vm get letify-elice-a100") >= 2
    assert provider.address == "203.0.113.9"


def test_a_declared_machine_that_is_not_listed_is_refused(account) -> None:
    provider = account(machine_id="my-vm")
    with pytest.raises(letify.ProviderUnavailable, match="my-vm"):
        provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")


def test_a_declared_machine_is_started_and_never_launched(account, fake_eci) -> None:
    fake_eci.set(
        vms=[
            {
                "id": "vm-7",
                "name": "my-vm",
                "status": "idle",
                "pricing_type": "ondemand",
                "public_ips": [{"ip": "198.51.100.7"}],
            }
        ]
    )
    provider = account(machine_id="vm-7")
    provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    assert "compute vm launch" not in fake_eci.commands()
    assert "compute vm start my-vm" in fake_eci.commands()
    assert provider.address == "198.51.100.7"


def test_stopping_a_session_stops_the_machine_and_deletes_nothing(account, fake_eci) -> None:
    provider = account()
    instance = Instance(provider, gpu="A100")
    provider.create_session(instance, "letify-a100-1")
    provider.stop(runtime_on(provider, instance))
    commands = fake_eci.commands()
    assert commands[-1] == "compute vm stop letify-elice-a100"
    assert not any("delete" in command for command in commands)
    assert fake_eci.read()["vms"][0]["status"] == "idle"


def test_a_machine_another_runtime_is_on_is_not_stopped(account, fake_eci) -> None:
    provider = account()
    instance = Instance(provider, gpu="A100")
    provider.create_session(instance, "letify-a100-1")
    first = runtime_on(provider, instance)
    provider.create_session(instance, "letify-a100-2")
    second = runtime_on(provider, instance)
    provider.stop(first)
    assert "compute vm stop letify-elice-a100" not in fake_eci.commands()
    provider.stop(second)
    assert fake_eci.commands()[-1] == "compute vm stop letify-elice-a100"


def test_a_failed_stop_says_how_to_stop_by_hand_and_does_not_raise(
    account, fake_eci, capsys
) -> None:
    provider = account()
    instance = Instance(provider, gpu="A100")
    provider.create_session(instance, "letify-a100-1")
    fake_eci.set(fail={"compute vm stop": {"stderr": "Error: 503 upstream"}})
    provider.stop(runtime_on(provider, instance))
    err = capsys.readouterr().err
    assert "letify: could not stop letify-elice-a100" in err
    assert "Run 'eci compute vm stop letify-elice-a100'" in err


def test_a_refused_token_says_so_and_where_a_token_is_issued(account, fake_eci) -> None:
    fake_eci.set(fail={"compute vm list": {"stderr": "Error: 403 Forbidden"}})
    provider = account()
    with pytest.raises(letify.RuntimeFailure) as caught:
        provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    message = str(caught.value)
    assert message.startswith("Elice refused the access token or it lacks permission")
    assert "User access token" in message
    assert "token-1" not in message


def test_a_failed_launch_never_shows_the_password(account, fake_eci) -> None:
    fake_eci.set(fail={"compute vm launch": {"stderr": "Error: quota exceeded"}})
    provider = account()
    with pytest.raises(letify.RuntimeFailure) as caught:
        provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    password = (account_directory("elice_a100") / "machine_password").read_text(encoding="utf-8")
    assert password not in str(caught.value)
    assert "--password ***" in str(caught.value)


def test_a_machine_without_a_public_ip_is_refused(account, fake_eci) -> None:
    fake_eci.set(
        vms=[{"id": "vm-1", "name": "letify-elice-a100", "status": "started"}],
    )
    provider = account()
    with pytest.raises(letify.ProviderUnavailable, match="no public IP"):
        provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")


@pytest.mark.parametrize("attempt", range(200))
def test_the_generated_password_meets_elice_rules(attempt: int) -> None:
    password = generate_password()
    assert len(password) == 20
    assert re.search(r"[A-Z]", password)
    assert re.search(r"[a-z]", password)
    assert re.search(r"[0-9]", password)
    assert re.search(r"[^A-Za-z0-9]", password)
    for a, b, c in zip(password, password[1:], password[2:], strict=False):
        assert not (ord(b) - ord(a) == ord(c) - ord(b) and abs(ord(b) - ord(a)) == 1)


# -- Spec: Price type ----------------------------------------------------------


def test_the_account_price_type_defaults_to_ondemand_and_refuses_others(account) -> None:
    assert account().price_type == "ondemand"
    assert account(price_type="spot").price_type == "spot"
    with pytest.raises(letify.ConfigError, match="price_type"):
        account(price_type="reserved").price_type  # noqa: B018


def test_priced_returns_a_copy_with_that_price_type_and_refuses_others() -> None:
    provider = provider_of(Local, "local")
    plain = Instance(provider, gpu="A100")
    spot = plain.priced("spot")
    assert spot.price_type == "spot" and plain.price_type is None
    assert spot.key != plain.key
    assert plain.priced("ondemand").price_type == "ondemand"
    with pytest.raises(ValueError, match="reserved"):
        plain.priced("reserved")


def test_a_spot_instance_launches_the_spot_machine_and_prints_its_price(
    account, fake_eci, capsys
) -> None:
    provider = account()
    provider.create_session(Instance(provider, gpu="A100").priced("spot"), "letify-a100-1")
    launch = next(c["argv"] for c in fake_eci.calls if c["argv"][:3] == ["compute", "vm", "launch"])
    assert launch[launch.index("--name") + 1] == "letify-elice-a100-spot"
    assert launch[launch.index("--price-type") + 1] == "spot"
    price_line = "letify: letify-elice-a100-spot: G-A100-1 spot at 900 KRW/hour"
    assert price_line in capsys.readouterr().err


def test_the_account_price_type_applies_when_the_instance_names_none(account, fake_eci) -> None:
    provider = account(price_type="spot")
    provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    assert "compute vm launch" in fake_eci.commands()
    assert fake_eci.read()["vms"][0]["name"] == "letify-elice-a100-spot"


def test_spot_on_a_cpu_type_is_refused_before_anything_is_created(account, fake_eci) -> None:
    provider = account()
    with pytest.raises(letify.UnsupportedMode, match="spot"):
        provider.create_session(Instance(provider, gpu=None).priced("spot"), "letify-cpu-1")
    assert "compute vm launch" not in fake_eci.commands()


def test_an_ondemand_launch_checks_quota_and_names_the_gap(account, fake_eci) -> None:
    fake_eci.set(
        org={"resource_quota": {"compute": {"devices": 8, "instance_types": {"G-A100-1": 0}}}}
    )
    provider = account()
    with pytest.raises(letify.ProviderUnavailable, match="ondemand quota for G-A100-1 is 0"):
        provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    assert "compute vm launch" not in fake_eci.commands()


def test_no_devices_in_the_quota_refuses_an_ondemand_accelerator(account, fake_eci) -> None:
    fake_eci.set(org={"resource_quota": {"compute": {"devices": 0}}})
    provider = account()
    with pytest.raises(letify.ProviderUnavailable, match='price_type = "spot"'):
        provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")


def test_a_spot_launch_does_not_read_quota(account, fake_eci) -> None:
    fake_eci.set(org={"resource_quota": {"compute": {"devices": 0}}})
    provider = account()
    provider.create_session(Instance(provider, gpu="A100").priced("spot"), "letify-a100-1")
    assert "org info" not in fake_eci.commands()
    assert "compute vm launch" in fake_eci.commands()


def test_an_unreadable_quota_does_not_refuse(account, fake_eci, capsys) -> None:
    fake_eci.set(fail={"org info": {"stderr": "Error: 500"}})
    provider = account()
    provider.create_session(Instance(provider, gpu="A100"), "letify-a100-1")
    assert "compute vm launch" in fake_eci.commands()
    assert "the ondemand quota could not be read, launching anyway" in capsys.readouterr().err


def test_a_declared_machine_with_other_pricing_is_refused(account, fake_eci) -> None:
    fake_eci.set(
        vms=[{"id": "vm-7", "name": "my-vm", "status": "idle", "pricing_type": "ondemand"}]
    )
    provider = account(machine_id="my-vm")
    with pytest.raises(letify.ConfigError, match="ondemand"):
        provider.create_session(Instance(provider, gpu="A100").priced("spot"), "letify-a100-1")


def test_spot_on_a_provider_without_spot_pricing_is_unsupported(tmp_path) -> None:
    from letify.declare.env import Env

    provider = provider_of(Local, "local")
    instance = Instance(provider, gpu=None).priced("spot")._placed("remote")
    with pytest.raises(letify.UnsupportedMode, match="no spot pricing"):
        provider.start(instance, Env(lock=str(tmp_path / "absent.lock")), name="local-1")


def test_the_usage_record_carries_the_price_type(isolated_home, fake_elice) -> None:
    from letify.providers.elice import ALLOCATION_PATH, PRICING_PATH

    fake_elice.answer("GET", ALLOCATION_PATH, FakeResponse(200, {"items": []}))
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(200, {"items": []}))
    provider = provider_of(
        Elice, "e", endpoint=fake_elice.endpoint, zone_id="z", access_token="t", price_type="spot"
    )
    row = provider.report_usage().to_dict()
    assert row["price_type"] == "spot"

    from letify import render

    assert "price type spot" in render.usage_block(row, render.Style(color=False, width=80))


def test_status_reports_the_price_type_of_each_runtime(let, cpu) -> None:
    @let.function(device=cpu._placed("remote"), host="remote")
    def nothing() -> None:
        return None

    with let.keep_alive():
        nothing()
        rows = let.status()["runtimes"]
    assert rows and rows[0]["price_type"] is None

    from letify import render

    text = render.status_text(
        {"live": 1, "busy": 0, "runtimes": [{**rows[0], "price_type": "spot"}]},
        render.Style(color=False, width=80),
    )
    assert "price spot" in text


# -- Spec: Spot preemption -----------------------------------------------------


def spot_session(account, fake_eci, **options):
    provider = account(**options)
    instance = Instance(provider, gpu="A100").priced("spot")
    provider.create_session(instance, "letify-a100-1")
    return provider, instance, runtime_on(provider, instance)


def test_a_spot_machine_elice_stopped_turns_the_failure_into_spot_preempted(
    account, fake_eci, capsys
) -> None:
    provider, _instance, runtime = spot_session(account, fake_eci)
    fake_eci.machine("letify-elice-a100-spot", status="idle")
    capsys.readouterr()
    original = letify.RuntimeFailure("the pipe is closed")

    diagnosed = provider.diagnose(runtime, original)

    assert isinstance(diagnosed, SpotPreempted)
    assert isinstance(diagnosed, letify.RuntimeLost)
    assert diagnosed.machine == "letify-elice-a100-spot"
    assert diagnosed.state == "idle"
    assert diagnosed.at > 0
    err = capsys.readouterr().err
    assert "letify: letify-elice-a100-spot was preempted by Elice (state idle)" in err
    assert "keeps its disk and public IP, which keep billing" in err
    assert "'eci compute vm delete letify-elice-a100-spot --cascade'" in err
    assert not any("delete" in command for command in fake_eci.commands())


def test_a_deleted_spot_machine_is_preempted_with_state_deleted(account, fake_eci, capsys) -> None:
    provider, _instance, runtime = spot_session(account, fake_eci)
    fake_eci.remove("letify-elice-a100-spot")
    diagnosed = provider.diagnose(runtime, letify.RuntimeFailure("gone"))
    assert isinstance(diagnosed, SpotPreempted)
    assert diagnosed.state == "deleted"
    assert "keep billing" not in capsys.readouterr().err


def test_a_spot_machine_still_started_leaves_the_failure_as_it_is(account, fake_eci) -> None:
    provider, _instance, runtime = spot_session(account, fake_eci)
    original = letify.RuntimeFailure("a network blip")
    assert provider.diagnose(runtime, original) is original


def test_an_ondemand_failure_is_left_as_it_is_without_asking_elice(account, fake_eci) -> None:
    provider = account()
    instance = Instance(provider, gpu="A100")
    provider.create_session(instance, "letify-a100-1")
    before = len(fake_eci.calls)
    original = letify.RuntimeFailure("the pipe is closed")
    assert provider.diagnose(runtime_on(provider, instance), original) is original
    assert len(fake_eci.calls) == before


def test_a_machine_letify_stopped_is_not_a_preemption(account, fake_eci) -> None:
    provider, _instance, runtime = spot_session(account, fake_eci)
    provider.stop(runtime)
    original = letify.RuntimeFailure("closed while stopping")
    assert provider.diagnose(runtime, original) is original


def test_a_failed_read_leaves_the_original_failure(account, fake_eci) -> None:
    provider, _instance, runtime = spot_session(account, fake_eci)
    fake_eci.set(fail={"compute vm get": {"stderr": "Error: 502"}})
    original = letify.RuntimeFailure("the pipe is closed")
    assert provider.diagnose(runtime, original) is original


def test_after_a_preemption_the_retry_uses_spot_again_by_default(account, fake_eci, capsys) -> None:
    provider, instance, runtime = spot_session(account, fake_eci)
    fake_eci.machine("letify-elice-a100-spot", status="idle")
    provider.diagnose(runtime, letify.RuntimeFailure("closed"))
    assert "letify: letify-elice-a100-spot preempted, retrying on spot" in capsys.readouterr().err

    provider.create_session(instance, "letify-a100-2")
    assert provider._pending_machine == "letify-elice-a100-spot"
    assert "compute vm start letify-elice-a100-spot" in fake_eci.commands()


def test_with_ondemand_fallback_the_retry_launches_the_ondemand_machine(
    account, fake_eci, capsys
) -> None:
    provider, instance, runtime = spot_session(account, fake_eci, spot_fallback="ondemand")
    fake_eci.remove("letify-elice-a100-spot")
    provider.diagnose(runtime, letify.RuntimeFailure("closed"))
    err = capsys.readouterr().err
    assert (
        "letify: letify-elice-a100-spot preempted, retrying on ondemand machine letify-elice-a100"
        in err
    )

    provider.create_session(instance, "letify-a100-2")
    assert provider._pending_machine == "letify-elice-a100"
    launches = [c["argv"] for c in fake_eci.calls if c["argv"][:3] == ["compute", "vm", "launch"]]
    assert launches[-1][launches[-1].index("--name") + 1] == "letify-elice-a100"
    assert "--price-type" not in launches[-1]


def test_spot_fallback_refuses_other_values(account) -> None:
    with pytest.raises(letify.ConfigError, match="spot_fallback"):
        account(spot_fallback="reserved").spot_fallback  # noqa: B018
