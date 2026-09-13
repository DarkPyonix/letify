"""Providers: what each account offers, how it is reached, and what it refuses.

Spec sections pinned here: "Provider model", "Provider properties", "Instances",
"Execution modes", "Transport" and "Packaging".

Local is exercised for real. Colab, Shell, Tunnel, Elice and Modal are driven with the
CLI, SSH, HTTP and client stand-ins from conftest, because each of those needs an
account, a second machine or a GPU. What is still real in those tests is the command
line, the request and the instance table letify builds, which is what a caller observes.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from conftest import FakeCompleted, FakeResponse, FakeSandbox, provider_of

import letify
from letify import providers
from letify import tools as tools_module
from letify.config.schema import ProviderConfig
from letify.declare.instance import Host, Instance
from letify.providers import colab as colab_module
from letify.providers import local as local_module
from letify.providers import shell as shell_module
from letify.providers import tunnel as tunnel_module
from letify.providers.base import Provider
from letify.providers.colab import ALIASES, Colab
from letify.providers.elice import (
    ALLOCATION_PATH,
    INSTANCE_TYPE_PATH,
    PRICING_PATH,
    VM_PATH,
    Elice,
)
from letify.providers.local import Local
from letify.providers.modal import Modal, SandboxChannel
from letify.providers.naming import gib_from_mib, normalize_gpu
from letify.providers.shell import Shell
from letify.providers.tunnel import DEFAULT_MTU, Tunnel
from letify.runtime.channel import OneShotChannel, PersistentChannel
from letify.store.backends.filesystem import FilesystemBackend

probe_module = import_module("letify.remoting.probe")
usage_module = import_module("letify.providers.usage")


@pytest.fixture
def fresh_device_names():
    """Forget the machine's GPU list, which is read once per process."""
    local_module._device_names.cache_clear()
    yield
    local_module._device_names.cache_clear()


@pytest.fixture
def elice(fake_httpx):
    return provider_of(
        Elice,
        "elice_a100",
        zone_id="zone-1",
        machine_id="machine-1",
        access_token="token-1",
    )


# -- Spec: Instances, name normalization ---------------------------------------


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("NVIDIA RTX PRO 6000 Blackwell", "RTX_PRO_6000"),
        ("NVIDIA A100-SXM4-80GB", "A100"),
        ("NVIDIA H100 PCIe", "H100"),
        ("NVIDIA L40S", "L40S"),
        ("Tesla T4", "Tesla_T4"),
        ("NVIDIA GeForce RTX 4090 Laptop GPU", "GeForce_RTX_4090"),
        ("NVIDIA RTX 4000 Ada Generation", "RTX_4000"),
    ],
)
def test_a_vendor_product_name_becomes_an_attribute_name(raw: str, label: str) -> None:
    assert normalize_gpu(raw) == label


def test_a_memory_size_is_reported_in_whole_gibibytes() -> None:
    assert gib_from_mib(" 81920 MiB") == 80
    assert gib_from_mib("16384") == 16
    # nvidia-smi reports this for a device that will not answer.
    assert gib_from_mib("[N/A]") is None


# -- Spec: Provider model ------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("local", Local),
        ("colab", Colab),
        ("modal", Modal),
        ("shell", Shell),
        ("ssh", Shell),
        ("tunnel", Tunnel),
        ("elice", Elice),
    ],
)
def test_a_configuration_entry_selects_its_provider_class(kind: str, cls: type) -> None:
    built = providers.build(ProviderConfig("p", kind, {}, 0))
    assert isinstance(built, cls)
    assert built.alias == "p"


def test_the_kind_field_is_read_without_regard_to_case() -> None:
    assert isinstance(providers.build(ProviderConfig("p", "Colab", {}, 0)), Colab)


def test_an_unknown_kind_lists_the_kinds_that_exist() -> None:
    with pytest.raises(letify.ConfigError, match="which is not known"):
        providers.build(ProviderConfig("p", "vastai", {}, 0))


# -- Spec: Packaging -----------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(providers.KINDS))
def test_a_provider_is_built_without_importing_its_optional_dependency(kind: str) -> None:
    # A missing provider package disables that provider and nothing else, so building one
    # must not reach for its client library.
    built = providers.build(ProviderConfig("p", kind, {}, 0))
    assert isinstance(built, Provider)


def test_a_provider_whose_client_is_absent_says_which_extra_installs_it(no_module) -> None:
    no_module("modal")
    with pytest.raises(letify.ProviderUnavailable, match=r"letify\[modal\]"):
        provider_of(Modal).client()


# -- Spec: Provider properties -------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "persistence", "fast_path", "persistent_channel"),
    [
        (Local, "persistent", True, True),
        (Modal, "persistent", False, True),
        (Colab, "ephemeral", False, True),
        (Shell, "ephemeral", True, True),
        (Tunnel, "ephemeral", True, True),
        (Elice, "persistent", True, True),
    ],
)
def test_each_provider_declares_the_properties_the_spec_table_gives_it(
    cls: type[Provider], persistence: str, fast_path: bool, persistent_channel: bool
) -> None:
    built = provider_of(cls)
    assert built.persistence == persistence
    assert built.has_fast_path is fast_path
    assert built.persistent_channel is persistent_channel


@pytest.mark.parametrize(
    ("cls", "backend"),
    [(Local, "filesystem"), (Modal, "modal"), (Colab, "gcs"), (Elice, "shell")],
)
def test_each_provider_names_the_store_backend_the_spec_table_gives_it(
    cls: type[Provider], backend: str
) -> None:
    assert provider_of(cls).store_backend() == backend


@pytest.mark.parametrize("cls", [Shell, Tunnel, Elice])
def test_a_shell_machine_stores_blobs_on_a_filesystem(cls: type[Provider]) -> None:
    # The spec table calls this backend filesystem; the provider names it "shell", which
    # the registry maps to the filesystem backend.
    from letify.store import backends

    assert backends.BACKENDS[provider_of(cls).store_backend()] is FilesystemBackend


def test_a_machine_that_keeps_its_disk_is_declared_persistent() -> None:
    # Ephemeral is the safe default, because assuming persistent fails outright when the
    # disk turns out to be wiped.
    assert provider_of(Shell, address="a").persistence == "ephemeral"
    assert provider_of(Shell, address="a", persistent=True).persistent is True
    assert provider_of(Shell, address="a", persistent=False).persistence == "ephemeral"


def test_a_store_backend_can_be_overridden_by_the_configuration() -> None:
    assert provider_of(Colab, store="modal").store_backend() == "modal"
    assert provider_of(Shell, address="a", store="gcs").store_backend() == "gcs"
    assert provider_of(Elice, zone_id="z", store="gcs").store_backend() == "gcs"


# -- Spec: Instances, discovery ------------------------------------------------


def test_discovery_is_lazy_and_then_cached(patch_run, patch_which) -> None:
    # A provider that must connect to enumerate its accelerators does so on first access,
    # never at import time.
    recorder = patch_run(
        shell_module, result=FakeCompleted(stdout="NVIDIA A100-SXM4-80GB, 81920\n")
    )
    provider = provider_of(Shell, "lab", address="gpu.example.edu")
    assert recorder.calls == []

    assert "A100" in provider.instances
    assert len(recorder.calls) == 1
    # Asked once, then remembered.
    assert "A100" in provider.instances
    assert len(recorder.calls) == 1
    # refresh asks again.
    provider.refresh()
    assert len(recorder.calls) == 2


def test_a_declared_gpu_list_is_trusted_without_connecting(patch_run) -> None:
    recorder = patch_run(shell_module)
    provider = provider_of(Shell, "lab", address="gpu.example.edu", gpus=["A100", "H100"])
    assert sorted(provider.instances) == ["A100", "H100"]
    assert recorder.calls == []


def test_an_accelerator_is_reached_by_attribute_in_either_case(let: letify.Launcher) -> None:
    provider = let.providers.local
    assert provider.CPU is provider.device("CPU")
    assert provider.cpu is provider.CPU


def test_an_accelerator_the_account_does_not_offer_lists_what_it_does(
    let: letify.Launcher,
) -> None:
    with pytest.raises(letify.UnknownInstance, match="does not offer 'H100'"):
        let.providers.local.H100  # noqa: B018


def test_a_private_attribute_is_never_treated_as_an_accelerator(let: letify.Launcher) -> None:
    with pytest.raises(AttributeError):
        let.providers.local._not_a_gpu  # noqa: B018


def test_a_discovery_failure_does_not_look_like_a_typo(elice) -> None:
    # Elice needs a zone to enumerate anything, so the reason has to reach the user
    # rather than being reported as a missing attribute.
    broken = provider_of(Elice, "elice_a100")
    with pytest.raises(AttributeError, match="is unavailable"):
        broken.A100  # noqa: B018
    # dir() over a provider that cannot enumerate still works.
    assert "volume" in dir(broken)


def test_the_accelerators_a_provider_offers_appear_in_dir(let: letify.Launcher) -> None:
    assert "CPU" in dir(let.providers.local)


# -- Spec: Instances, the local machine ----------------------------------------


def test_the_local_machine_always_offers_a_plain_cpu(let: letify.Launcher) -> None:
    assert let.providers.local.CPU.accelerator == "cpu"
    assert let.providers.local.prepares_env is False
    assert let.providers.local.needs_lease is False


def test_the_local_gpu_names_come_from_nvidia_smi(
    fresh_device_names, patch_run, patch_which
) -> None:
    patch_which(local_module, present=True)
    patch_run(
        local_module,
        result=FakeCompleted(
            stdout="NVIDIA RTX PRO 6000 Blackwell, 98304 MiB\nNVIDIA L4, 23034 MiB\n\n"
        ),
    )
    table = provider_of(Local, "local").instances
    assert table["RTX_PRO_6000"].vram_gb == 96
    assert table["L4"].vram_gb == 22
    assert table["RTX_PRO_6000"].gpu == "RTX_PRO_6000"


def test_the_gpu_list_is_read_once_per_process(fresh_device_names, patch_run, patch_which) -> None:
    # Asking nvidia-smi takes seconds on a laptop whose discrete GPU is asleep, and the
    # answer does not change while the process runs.
    patch_which(local_module, present=True)
    recorder = patch_run(local_module, result=FakeCompleted(stdout="NVIDIA L4, 23034 MiB\n"))
    provider = provider_of(Local, "local")
    assert provider.instances
    assert provider_of(Local, "local").instances
    assert len(recorder.calls) == 1
    # refresh is what asks the machine again.
    provider.refresh()
    assert len(recorder.calls) == 2


def test_a_machine_with_no_nvidia_smi_offers_only_a_cpu(
    fresh_device_names, patch_which, patch_run
) -> None:
    patch_which(local_module, present=False)
    recorder = patch_run(local_module)
    assert list(provider_of(Local, "local").instances) == ["CPU"]
    assert recorder.calls == []


def test_an_nvidia_smi_that_fails_is_treated_as_no_gpu(
    fresh_device_names, patch_which, patch_run
) -> None:
    patch_which(local_module, present=True)
    patch_run(local_module, result=FakeCompleted(returncode=9, stderr="driver not loaded"))
    assert list(provider_of(Local, "local").instances) == ["CPU"]


def test_an_nvidia_smi_that_cannot_be_run_is_treated_as_no_gpu(
    fresh_device_names, patch_which, patch_run
) -> None:
    patch_which(local_module, present=True)
    patch_run(local_module, error=OSError("cannot execute"))
    assert list(provider_of(Local, "local").instances) == ["CPU"]


def test_the_local_worker_runs_under_the_declared_interpreter(let: letify.Launcher) -> None:
    provider = provider_of(Local, "local", python="/usr/bin/python3.12")
    runtime = type("R", (), {"name": "letify-cpu-1"})()
    channel = provider.open_channel(runtime)
    assert isinstance(channel, PersistentChannel)
    assert channel.command[0] == "/usr/bin/python3.12"


# -- Spec: Transport, Colab ----------------------------------------------------


def test_colab_offers_its_fixed_accelerator_list_without_connecting(patch_run) -> None:
    # Whether one is free right now is decided when a runtime starts, because Colab does
    # not promise availability.
    recorder = patch_run(colab_module)
    provider = provider_of(Colab, "colab_a")
    table = provider.instances
    assert table["G4"].vram_gb == 96
    assert table["v5e1"].tpu == "v5e1"
    assert recorder.calls == []


def test_colab_accepts_the_vendor_name_for_the_card_it_calls_g4() -> None:
    provider = provider_of(Colab, "colab_a")
    assert provider.RTX_PRO_6000.gpu == "G4"
    assert ALIASES["A100_80GB"] == "A100"


def test_colab_reports_its_round_trip_rather_than_refusing() -> None:
    # Slow is not a reason to refuse, so the provider carries a number to warn with.
    provider = provider_of(Colab, "colab_a")
    assert provider.has_fast_path is False
    assert provider.expected_round_trip_ms == 175.0


COLAB_CLI = [
    "/usr/bin/uv",
    "tool",
    "run",
    "--python",
    "3.13",
    "--with",
    "jupyter-kernel-client<1",
    "--from",
    "google-colab-cli",
    "colab",
]


def test_a_missing_uv_says_how_to_get_it(isolated_home, patch_which) -> None:
    # The Colab CLI runs through uv, so uv is the one thing that has to be installed.
    patch_which(tools_module, present=False)
    provider = provider_of(Colab, "colab_a")
    assert provider.available() is False
    with pytest.raises(letify.ProviderUnavailable, match="uv was not found"):
        provider.create_session(Instance(provider, gpu="G4"), "letify-g4-1")


def test_creating_a_colab_session_names_the_accelerator_the_cli_accepts(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    recorder = patch_run(colab_module)
    provider = provider_of(Colab, "colab_a")
    provider.create_session(provider.RTX_PRO_6000, "letify-g4-1")
    assert recorder.command == [*COLAB_CLI, "new", "-s", "letify-g4-1", "--gpu", "G4"]


def test_creating_a_colab_session_for_a_tpu_asks_for_a_tpu(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    recorder = patch_run(colab_module)
    provider = provider_of(Colab, "colab_a")
    provider.create_session(provider.v5e1, "letify-v5e1-1")
    assert recorder.command == [*COLAB_CLI, "new", "-s", "letify-v5e1-1", "--tpu", "v5e1"]


def test_the_declared_account_reaches_the_cli(isolated_home, patch_which, patch_run) -> None:
    patch_which(tools_module, present=True)
    recorder = patch_run(colab_module)
    provider = provider_of(Colab, "colab_a", account="someone@example.com")
    provider.sessions()
    assert recorder.calls[-1]["env"]["COLAB_ACCOUNT"] == "someone@example.com"


def test_the_colab_cli_keeps_its_login_in_the_account_directory(
    isolated_home, patch_which, patch_run
) -> None:
    # The CLI stores its token under the home directory, so each alias gets its own home.
    patch_which(tools_module, present=True)
    recorder = patch_run(colab_module)
    provider_of(Colab, "colab_a").sessions()
    env = recorder.calls[-1]["env"]
    assert env["HOME"] == str(Path.home() / ".letify" / "accounts" / "colab_a")
    assert "COLAB_ACCOUNT" not in env


def test_the_sessions_an_account_holds_are_read_from_the_cli(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    listing = "NAME        STATE\n-------     -----\nletify-g4-1  running\nletify-t4-2  idle\n"
    patch_run(colab_module, result=FakeCompleted(stdout=listing))
    assert provider_of(Colab, "colab_a").sessions() == ["letify-g4-1", "letify-t4-2"]


def test_a_failing_cli_command_carries_the_command_and_the_error(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    patch_run(colab_module, result=FakeCompleted(returncode=2, stderr="not entitled\n"))
    provider = provider_of(Colab, "colab_a")
    with pytest.raises(letify.RuntimeFailure) as caught:
        provider.sessions()
    assert caught.value.stderr == "not entitled"
    assert caught.value.command == " ".join([*COLAB_CLI, "sessions"])


def test_stopping_a_session_that_is_already_gone_is_not_an_error(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    patch_run(colab_module, result=FakeCompleted(returncode=1, stderr="no such session"))
    provider = provider_of(Colab, "colab_a")
    runtime = type("R", (), {"name": "letify-g4-1"})()
    assert provider.stop(runtime) is None


def test_colab_is_reached_through_the_cli_websocket_bridge(isolated_home, patch_which) -> None:
    # An OpenSSH ProxyCommand over the CLI's bridge, which is an official path and needs
    # no tunnel.
    patch_which(tools_module, present=True)
    command = provider_of(Colab, "colab_a").ssh_command("echo hello")
    assert command[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert f"ProxyCommand={' '.join(COLAB_CLI)} ssh --proxy-mode" in command
    assert command[-2:] == ["colab", "echo hello"]


def test_the_persistent_colab_channel_starts_a_worker_over_that_bridge(
    isolated_home, patch_which
) -> None:
    patch_which(tools_module, present=True)
    provider = provider_of(Colab, "colab_a")
    assert provider.channel_kind == "ssh"
    assert provider.persistent_channel is True
    runtime = type("R", (), {"name": "letify-g4-1"})()
    channel = provider.open_channel(runtime)
    assert isinstance(channel, PersistentChannel)
    assert "python3 -u -c" in channel.command[-1]


def test_the_exec_fallback_keeps_nothing_between_calls(
    isolated_home, patch_which, patch_run
) -> None:
    # colab exec always works and needs nothing beyond the CLI, at the cost of a fresh
    # process per call.
    patch_which(tools_module, present=True)
    recorder = patch_run(colab_module)
    provider = provider_of(Colab, "colab_a", channel="exec")
    assert provider.persistent_channel is False
    runtime = type("R", (), {"name": "letify-g4-1"})()
    channel = provider.open_channel(runtime)
    assert isinstance(channel, OneShotChannel)

    channel.runner("print('hello')", 60)
    assert recorder.command == [*COLAB_CLI, "exec", "-s", "letify-g4-1"]
    assert recorder.calls[-1]["input"] == "print('hello')"


# -- Spec: Transport, a machine reached over SSH -------------------------------


def test_the_ssh_command_names_the_port_the_user_and_the_key() -> None:
    provider = provider_of(
        Shell,
        "lab",
        address="gpu.lab.example.edu",
        user="researcher",
        port=2222,
        key="~/.ssh/id_ed25519",
    )
    command = provider.ssh_command("uname -a")
    assert command[:3] == ["ssh", "-p", "2222"]
    assert "BatchMode=yes" in command
    assert "ServerAliveInterval=30" in command
    assert command[command.index("-i") + 1] == "~/.ssh/id_ed25519"
    assert command[-2:] == ["researcher@gpu.lab.example.edu", "uname -a"]


def test_a_jump_host_is_used_before_any_tunnel() -> None:
    # Order of preference is a direct address, then a jump host, then a tunnel.
    provider = provider_of(Shell, "lab", address="gpu.internal", jump="bastion.example.edu")
    command = provider.ssh_command()
    assert command[command.index("-J") + 1] == "bastion.example.edu"
    assert command[-1] == "gpu.internal"


def test_a_port_that_changes_on_every_restart_is_read_at_connection_time(patch_run) -> None:
    # Some hosts hand out a fresh port every time the machine starts.
    patch_run(shell_module, result=FakeCompleted(stdout="24601\n"))
    provider = provider_of(
        Shell, "lab", address="gpu.example.edu", port=22, port_command="elice port show"
    )
    assert provider.port == 24601


def test_a_port_command_that_answers_with_nonsense_falls_back_to_the_declared_port(
    patch_run,
) -> None:
    patch_run(shell_module, result=FakeCompleted(stdout="unavailable\n"))
    provider = provider_of(
        Shell, "lab", address="gpu.example.edu", port=2222, port_command="elice port show"
    )
    assert provider.port == 2222


def test_the_default_ssh_port_is_used_when_nothing_says_otherwise() -> None:
    assert provider_of(Shell, "lab", address="gpu.example.edu").port == 22


def test_a_machine_with_no_address_cannot_be_reached() -> None:
    with pytest.raises(letify.ProviderUnavailable, match="has no 'address' field"):
        provider_of(Shell, "lab").address  # noqa: B018


def test_the_gpus_a_machine_has_are_read_over_ssh(patch_run) -> None:
    patch_run(
        shell_module,
        result=FakeCompleted(stdout="NVIDIA A100-SXM4-80GB, 81920 MiB\n\nNVIDIA L4, 23034 MiB\n"),
    )
    table = provider_of(Shell, "lab", address="gpu.example.edu").instances
    assert table["A100"].vram_gb == 80
    assert table["L4"].vram_gb == 22


def test_a_machine_that_does_not_answer_cannot_list_its_gpus(patch_run) -> None:
    patch_run(shell_module, result=FakeCompleted(returncode=255, stderr="connection refused"))
    provider = provider_of(Shell, "lab", address="gpu.example.edu")
    with pytest.raises(letify.ProviderUnavailable, match="could not list GPUs"):
        provider.instances  # noqa: B018


def test_a_check_runs_one_command_to_confirm_the_machine_answers(patch_run) -> None:
    patch_run(shell_module, result=FakeCompleted(stdout="Linux gpu 6.8.0\nNVIDIA L4\n"))
    assert "Linux" in provider_of(Shell, "lab", address="gpu.example.edu").check()


def test_a_check_that_fails_names_the_command_it_tried(patch_run) -> None:
    patch_run(shell_module, result=FakeCompleted(returncode=255, stderr="permission denied"))
    provider = provider_of(Shell, "lab", address="gpu.example.edu")
    with pytest.raises(letify.RuntimeFailure) as caught:
        provider.check()
    assert caught.value.stderr == "permission denied"
    assert "ssh" in caught.value.command


def test_a_remote_worker_is_one_python_reading_framed_requests(patch_run) -> None:
    patch_run(shell_module)
    provider = provider_of(Shell, "lab", address="gpu.example.edu", python="python3.12")
    runtime = type("R", (), {"name": "letify-a100-1"})()
    channel = provider.open_channel(runtime)
    assert isinstance(channel, PersistentChannel)
    assert channel.command[-1].startswith("python3.12 -u -c ")
    assert "base64" in channel.command[-1]


# -- Spec: Transport, the tunnel -----------------------------------------------


def test_tailscale_is_the_default_transport_and_the_mtu_is_held_down() -> None:
    # Every mesh VPN in this class stalls bulk transfers silently above about 1400.
    provider = provider_of(Tunnel, "lab")
    assert provider.transport == "tailscale"
    assert provider.mtu == DEFAULT_MTU == 1280
    assert provider_of(Tunnel, "lab", mtu=1400).mtu == 1400


def test_a_machine_that_needs_no_tunnel_declares_none(patch_run) -> None:
    recorder = patch_run(tunnel_module)
    provider = provider_of(Tunnel, "lab", transport="none")
    assert provider.connect() is None
    assert recorder.calls == []


def test_an_unknown_transport_is_refused() -> None:
    with pytest.raises(letify.ProviderUnavailable, match="unknown transport 'wireguard'"):
        provider_of(Tunnel, "lab", transport="wireguard").connect()


def test_tailscale_has_to_be_installed_on_both_machines(patch_which) -> None:
    patch_which(tunnel_module, present=False)
    with pytest.raises(letify.ProviderUnavailable, match="Install Tailscale"):
        provider_of(Tunnel, "lab").connect()


def test_a_path_that_is_already_up_is_left_alone(patch_which, patch_run) -> None:
    patch_which(tunnel_module, present=True)
    recorder = patch_run(tunnel_module, result=FakeCompleted(stdout='{"BackendState": "Running"}'))
    provider_of(Tunnel, "lab").connect()
    # Status was asked, and nothing was brought up.
    assert recorder.commands == [["tailscale", "status", "--json"]]


def test_bringing_a_path_up_needs_an_auth_key_that_is_not_in_a_tracked_file(
    patch_which, patch_run
) -> None:
    patch_which(tunnel_module, present=True)
    patch_run(tunnel_module, result=FakeCompleted(stdout='{"BackendState": "NeedsLogin"}'))
    with pytest.raises(letify.ProviderUnavailable, match="or set auth_key_env"):
        provider_of(Tunnel, "lab").connect()


def test_a_path_is_brought_up_from_an_auth_key_with_no_prompt(
    patch_which, patch_run, monkeypatch
) -> None:
    patch_which(tunnel_module, present=True)
    monkeypatch.setenv("LETIFY_TS_KEY", "tskey-auth-1")
    recorder = patch_run(
        tunnel_module,
        result=lambda command: FakeCompleted(stdout='{"BackendState": "NeedsLogin"}'),
    )
    provider_of(Tunnel, "lab", auth_key_env="LETIFY_TS_KEY").connect()
    assert recorder.commands[-1] == ["tailscale", "up", "--auth-key=tskey-auth-1"]


def test_a_self_hosted_control_plane_is_passed_through(patch_which, patch_run, monkeypatch) -> None:
    patch_which(tunnel_module, present=True)
    monkeypatch.setenv("LETIFY_TS_KEY", "tskey-auth-1")
    recorder = patch_run(
        tunnel_module, result=FakeCompleted(stdout='{"BackendState": "NeedsLogin"}')
    )
    provider_of(
        Tunnel,
        "lab",
        auth_key_env="LETIFY_TS_KEY",
        login_server="https://headscale.example.edu",
    ).connect()
    assert "--login-server=https://headscale.example.edu" in recorder.commands[-1]


def test_a_path_that_will_not_come_up_says_what_tailscale_said(
    patch_which, patch_run, monkeypatch
) -> None:
    patch_which(tunnel_module, present=True)
    monkeypatch.setenv("LETIFY_TS_KEY", "tskey-auth-1")

    def answer(command: list[str]) -> FakeCompleted:
        if command[1] == "status":
            return FakeCompleted(stdout='{"BackendState": "NeedsLogin"}')
        return FakeCompleted(returncode=1, stderr="auth key expired")

    patch_run(tunnel_module, result=answer)
    provider = provider_of(Tunnel, "lab", auth_key_env="LETIFY_TS_KEY")
    with pytest.raises(letify.ProviderUnavailable, match="auth key expired"):
        provider.connect()


def test_a_status_command_that_cannot_be_run_is_not_a_running_path(patch_which, patch_run) -> None:
    patch_which(tunnel_module, present=True)
    patch_run(tunnel_module, error=OSError("no such binary"))
    with pytest.raises(letify.ProviderUnavailable, match="auth_key_env"):
        provider_of(Tunnel, "lab").connect()


def test_the_path_is_brought_up_once_per_process(patch_which, patch_run) -> None:
    patch_which(tunnel_module, present=True)
    recorder = patch_run(tunnel_module, result=FakeCompleted(stdout='{"BackendState": "Running"}'))
    provider = provider_of(Tunnel, "lab")
    provider.connect()
    provider.connect()
    assert len(recorder.calls) == 1


def test_frp_is_the_fallback_for_a_network_that_blocks_udp(
    patch_which, monkeypatch, tmp_path
) -> None:
    # A relayed Tailscale path keeps working but slowly, so frp over TLS 443 is the way
    # out of the relay.
    patch_which(tunnel_module, present=True)
    started: list[list[str]] = []
    monkeypatch.setattr(
        tunnel_module.subprocess, "Popen", lambda command, **kwargs: started.append(command)
    )
    config = tmp_path / "frpc.toml"
    config.write_text("serverPort = 443\n", encoding="utf-8")
    provider_of(Tunnel, "lab", transport="frp", frp_config=str(config)).connect()
    assert started == [["frpc", "-c", str(config)]]


def test_frp_needs_its_binary_and_its_configuration(patch_which) -> None:
    patch_which(tunnel_module, present=False)
    with pytest.raises(letify.ProviderUnavailable, match="'frpc'"):
        provider_of(Tunnel, "lab", transport="frp").connect()

    patch_which(tunnel_module, present=True)
    with pytest.raises(letify.ProviderUnavailable, match="needs an frp_config path"):
        provider_of(Tunnel, "lab", transport="frp").connect()


def test_a_diagnosis_names_the_two_usual_causes_of_a_stalled_transfer(
    patch_which, patch_run
) -> None:
    # A relayed path and an MTU above 1400 are the two usual causes.
    patch_which(tunnel_module, present=True)
    patch_run(tunnel_module, result=FakeCompleted(stdout="100.1.1.1 lab linux relay sea\n"))
    report = provider_of(Tunnel, "lab").diagnose()
    assert report["transport"] == "tailscale"
    assert report["mtu"] == 1280
    assert report["relayed"] is True
    assert "100.1.1.1" in report["status"]


def test_a_diagnosis_of_a_non_tailscale_path_reports_only_what_it_knows() -> None:
    assert provider_of(Tunnel, "lab", transport="frp").diagnose() == {
        "transport": "frp",
        "mtu": 1280,
    }


def test_a_diagnosis_says_so_when_the_tailscale_command_cannot_be_run(
    patch_which, patch_run
) -> None:
    patch_which(tunnel_module, present=True)
    patch_run(tunnel_module, error=OSError("no such binary"))
    assert "no such binary" in provider_of(Tunnel, "lab").diagnose()["error"]


# -- Spec: Provider model, Elice -----------------------------------------------


def test_elice_needs_a_zone_a_machine_and_a_token(fake_httpx) -> None:
    # letify allocates and releases a declared machine; it does not create one.
    with pytest.raises(letify.ProviderUnavailable, match="no 'zone_id' field"):
        provider_of(Elice, "elice_a100").zone_id  # noqa: B018
    with pytest.raises(letify.ProviderUnavailable, match="does not create it"):
        provider_of(Elice, "elice_a100", zone_id="z").machine_id  # noqa: B018
    with pytest.raises(letify.ProviderUnavailable, match="access_token_env"):
        provider_of(Elice, "elice_a100", zone_id="z").machines()


def test_the_elice_endpoint_can_be_pointed_elsewhere() -> None:
    assert provider_of(Elice, "e", endpoint="https://portal.example/api/").endpoint == (
        "https://portal.example/api"
    )
    assert provider_of(Elice, "e").endpoint.startswith("https://")


def test_elice_cannot_be_reached_without_httpx(no_module) -> None:
    no_module("httpx")
    with pytest.raises(letify.ProviderUnavailable, match=r"letify\[shell\]"):
        provider_of(Elice, "e", zone_id="z", access_token="t").machines()


def test_an_elice_request_carries_the_token_and_the_zone(elice, fake_httpx) -> None:
    fake_httpx.answer("GET", VM_PATH, FakeResponse(200, {"items": [{"id": "machine-1"}]}))
    assert elice.machines() == [{"id": "machine-1"}]
    assert fake_httpx.clients[0]["headers"]["Authorization"] == "Bearer token-1"
    assert fake_httpx.last["params"] == {"zone_id": "zone-1"}


def test_anything_other_than_a_two_hundred_is_a_failure(elice, fake_httpx) -> None:
    # This API answers 200 for every success.
    fake_httpx.answer("GET", VM_PATH, FakeResponse(403, {"message": "quota exceeded"}))
    with pytest.raises(letify.RuntimeFailure, match="returned 403: quota exceeded"):
        elice.machines()


def test_a_failure_with_no_json_body_carries_the_text(elice, fake_httpx) -> None:
    fake_httpx.answer("GET", VM_PATH, FakeResponse(502, None, text="<html>bad gateway</html>"))
    with pytest.raises(letify.RuntimeFailure, match="bad gateway"):
        elice.machines()


def test_a_response_may_be_a_bare_list_or_an_items_table(elice, fake_httpx) -> None:
    fake_httpx.answer("GET", VM_PATH, FakeResponse(200, [{"id": "machine-1"}]))
    assert elice.machines() == [{"id": "machine-1"}]


def test_the_instance_types_a_zone_offers_are_normalized(elice, fake_httpx) -> None:
    fake_httpx.answer(
        "GET",
        INSTANCE_TYPE_PATH,
        FakeResponse(
            200,
            {
                "items": [
                    {
                        "gpu_model": "NVIDIA A100-SXM4-80GB",
                        "cpu_count": 16,
                        "memory_gb": 128,
                        "gpu_memory_gb": 80,
                    },
                    {"name": "NVIDIA L40S"},
                    {"description": "no name at all"},
                ]
            },
        ),
    )
    table = elice.instances
    assert sorted(table) == ["A100", "L40S"]
    assert table["A100"].cpus == 16
    assert table["A100"].memory_gb == 128
    assert table["A100"].vram_gb == 80


def test_an_allocation_is_what_powers_a_declared_machine_on(elice, fake_httpx) -> None:
    # The virtual machine is the instance and the allocation is the runtime.
    fake_httpx.answer("POST", ALLOCATION_PATH, FakeResponse(200, {"id": "alloc-1"}))
    elice.create_session(Instance(elice, gpu="A100"), "letify-a100-1")
    assert elice._pending_allocation == "alloc-1"
    assert fake_httpx.last["json"] == {"zone_id": "zone-1", "machine_id": "machine-1"}


def test_an_organization_is_named_in_the_allocation_when_declared(fake_httpx) -> None:
    provider = provider_of(
        Elice, "e", zone_id="z", machine_id="m", access_token="t", organization_id="org-1"
    )
    fake_httpx.answer("POST", ALLOCATION_PATH, FakeResponse(200, {"allocation_id": "alloc-2"}))
    assert provider.allocate("m") == "alloc-2"
    assert fake_httpx.last["json"]["organization_id"] == "org-1"


def test_an_allocation_with_no_id_is_a_failure(elice, fake_httpx) -> None:
    fake_httpx.answer("POST", ALLOCATION_PATH, FakeResponse(200, {"state": "pending"}))
    with pytest.raises(letify.RuntimeFailure, match="did not return an allocation id"):
        elice.allocate("machine-1")


def test_releasing_an_allocation_stops_compute_billing(elice, fake_httpx) -> None:
    elice.release("alloc-1")
    assert fake_httpx.last["method"] == "DELETE"
    assert fake_httpx.last["path"] == f"{ALLOCATION_PATH}/alloc-1"


def test_releasing_an_allocation_that_is_already_gone_is_not_an_error(elice, fake_httpx) -> None:
    fake_httpx.answer(
        "DELETE", f"{ALLOCATION_PATH}/alloc-1", FakeResponse(404, {"message": "gone"})
    )
    assert elice.release("alloc-1") is None


def test_stopping_an_elice_runtime_releases_its_allocation(elice, fake_httpx) -> None:
    runtime = type("R", (), {"external_id": "alloc-1", "name": "letify-a100-1"})()
    elice.stop(runtime)
    assert fake_httpx.last["path"].endswith("alloc-1")


def test_a_runtime_with_no_allocation_has_nothing_to_release(elice, fake_httpx) -> None:
    runtime = type("R", (), {"external_id": None, "name": "letify-a100-1"})()
    assert elice.stop(runtime) is None
    assert fake_httpx.requests == []


def test_the_zone_price_list_includes_any_preemptible_option(elice, fake_httpx) -> None:
    fake_httpx.answer("GET", "/user/pricing", FakeResponse(200, [{"id": "p1", "spot": True}]))
    assert elice.pricing() == [{"id": "p1", "spot": True}]


def test_the_allocations_of_one_machine_can_be_listed(elice, fake_httpx) -> None:
    fake_httpx.answer("GET", ALLOCATION_PATH, FakeResponse(200, {"items": []}))
    assert elice.allocations("machine-1") == []
    assert fake_httpx.last["params"] == {"filter_machine_id": "machine-1"}
    assert elice.allocations() == []
    assert fake_httpx.last["params"] is None


def test_an_elice_gpu_list_may_be_declared_instead_of_asked_for(fake_httpx) -> None:
    provider = provider_of(Elice, "e", zone_id="z", access_token="t", gpus=["A100", "H100"])
    assert sorted(provider.instances) == ["A100", "H100"]
    assert fake_httpx.requests == []


# -- Spec: Execution modes -----------------------------------------------------


def test_modal_refuses_forwarding_because_there_is_no_device_to_forward_at() -> None:
    # A limit of the service, not a speed judgement.
    provider = provider_of(Modal, "m")
    with pytest.raises(letify.UnsupportedMode, match="no device to forward"):
        provider.check_mode(provider.H100._placed("local"))
    assert provider.check_mode(provider.H100._placed("remote")) is None


def test_a_provider_without_a_fast_path_warns_with_its_round_trip_and_then_tries(
    isolated_home, patch_which, monkeypatch
) -> None:
    # The warning carries the arithmetic. The refusal that follows is about letify-core
    # being absent from this machine, not about the latency.
    patch_which(tools_module, present=True)
    monkeypatch.setattr(probe_module, "core_path", lambda: None)
    provider = provider_of(Colab, "colab_a")
    with (
        pytest.warns(UserWarning, match="round trip of about 175 ms"),
        pytest.raises(letify.UnsupportedMode, match="letify-core"),
    ):
        provider.check_mode(provider.G4._placed("local"))


def test_a_provider_with_a_fast_path_does_not_warn(patch_run, recwarn) -> None:
    patch_run(shell_module)
    provider = provider_of(Shell, "lab", address="gpu.example.edu", gpus=["A100"])
    with pytest.raises(letify.UnsupportedMode, match="letify-core"):
        provider.check_mode(provider.A100._placed("local"))
    assert [w for w in recwarn if "round trip" in str(w.message)] == []


def test_shipping_the_function_needs_no_capability_on_this_machine(patch_run) -> None:
    patch_run(shell_module)
    provider = provider_of(Shell, "lab", address="gpu.example.edu", gpus=["A100"])
    assert provider.check_mode(provider.A100._placed("remote")) is None


def test_modal_translates_an_instance_into_the_name_its_api_expects() -> None:
    provider = provider_of(Modal, "m")
    assert provider.wire_name(provider.A100_80GB) == "A100-80GB"
    assert provider.wire_name(provider.RTX_PRO_6000) == "RTX-PRO-6000"
    assert provider.wire_name(provider.H100) == "H100"
    assert provider.RTX_PRO_6000.vram_gb == 96


# -- Spec: Channels, the Modal sandbox ----------------------------------------


def test_a_modal_sandbox_keeps_a_process_alive_for_framed_requests(fake_modal) -> None:
    # A sandbox rather than a function call, because without a living process there is no
    # object table for a handle to point at.
    fake = fake_modal()
    provider = provider_of(Modal, "m", app="study", timeout=1800)
    runtime = type("R", (), {"name": "letify-h100-1", "instance": provider.H100})()
    channel = provider.open_channel(runtime)

    assert isinstance(channel, SandboxChannel)
    assert channel.persistent is True
    assert fake.looked_up == ["study"]
    assert fake.created["gpu"] == "H100"
    assert fake.created["timeout"] == 1800
    assert fake.created["args"] == ["python3", "-u", "-"]
    assert fake.created["image"].packages == ("cloudpickle", "blake3")


def test_a_sandbox_channel_sends_the_worker_once_and_then_framed_requests() -> None:
    sandbox = FakeSandbox(outcomes=[{"ok": True, "value": 42}], logs=["epoch 1\n"])
    channel = SandboxChannel(sandbox, name="letify-h100-1")
    channel.start()
    channel.start()
    value, logs = channel.request({"op": "stat"})
    assert value == 42
    assert logs == "epoch 1\n"
    # The worker source went out exactly once.
    assert sandbox.text.count("_READY = ") == 1


def test_a_sandbox_call_carries_the_pickled_function(fake_modal) -> None:
    sandbox = FakeSandbox(outcomes=[{"ok": True, "value": 3}])
    channel = SandboxChannel(sandbox, name="letify-h100-1")
    assert channel.call(len, ([1, 2, 3],), {}) == (3, "")


def test_a_sandbox_that_stops_without_replying_is_a_protocol_error() -> None:
    channel = SandboxChannel(FakeSandbox(outcomes=[], logs=["Killed\n"]), name="letify-h100-1")
    with pytest.raises(letify.ProtocolError, match="died before it finished"):
        channel.request({"op": "stat"})


def test_closing_a_sandbox_channel_asks_the_worker_to_shut_down() -> None:
    sandbox = FakeSandbox()
    channel = SandboxChannel(sandbox, name="letify-h100-1")
    channel.close()
    assert "__LETIFY_SHUTDOWN__" in sandbox.text


def test_closing_a_sandbox_whose_pipe_is_already_gone_is_harmless() -> None:
    sandbox = FakeSandbox()
    sandbox.write_error = RuntimeError("the sandbox is gone")
    assert SandboxChannel(sandbox, name="letify-h100-1").close() is None


def test_stopping_a_modal_runtime_terminates_its_sandbox(fake_modal) -> None:
    sandbox = FakeSandbox()
    fake_modal(sandbox)
    provider = provider_of(Modal, "m")
    runtime = type("R", (), {"name": "letify-h100-1", "instance": provider.H100})()
    provider.open_channel(runtime)
    provider.stop(runtime)
    assert sandbox.terminated is True
    # A sandbox that is already gone is fine, and so is one that was never opened.
    assert provider.stop(runtime) is None


def test_a_sandbox_that_will_not_terminate_does_not_break_the_teardown(fake_modal) -> None:
    sandbox = FakeSandbox()
    sandbox.terminate_error = RuntimeError("already gone")
    fake_modal(sandbox)
    provider = provider_of(Modal, "m")
    runtime = type("R", (), {"name": "letify-h100-1", "instance": provider.H100})()
    provider.open_channel(runtime)
    assert provider.stop(runtime) is None


# -- Spec: Provider model, reaching providers from the launcher -----------------


def test_providers_are_reached_by_attribute_index_and_iteration(launcher_from) -> None:
    let = launcher_from('[lab]\nkind = "shell"\naddress = "a"\ngpus = ["A100"]\n')
    assert let.providers.lab.alias == "lab"
    assert let.providers["lab"].alias == "lab"
    assert [p.alias for p in let.providers] == ["lab", "local"]
    assert let.providers.aliases == ["lab", "local"]
    assert "lab" in dir(let.providers)
    assert repr(let.providers) == "<Providers lab, local>"


def test_the_reserved_names_answer_without_naming_a_provider(launcher_from) -> None:
    let = launcher_from('[lab]\nkind = "shell"\naddress = "a"\ngpus = ["A100"]\n')
    assert let.providers.any.G4.accelerator == "G4"
    assert repr(let.providers.any) == "<Providers.any>"
    assert let.providers.devices["lab"] == ["A100"]
    assert let.providers.active == {}


def test_a_provider_that_cannot_be_reached_reports_its_reason_in_the_device_table(
    launcher_from,
) -> None:
    # One broken entry must not hide the rest.
    let = launcher_from('[elice_a]\nkind = "elice"\n')
    table = let.providers.devices
    assert table["elice_a"][0].startswith("unavailable:")
    assert table["local"]


def test_a_request_without_a_provider_takes_the_first_one_that_offers_the_card(
    launcher_from,
) -> None:
    let = launcher_from(
        '[first]\nkind = "shell"\naddress = "a"\ngpus = ["A100"]\n'
        '[second]\nkind = "shell"\naddress = "b"\ngpus = ["A100"]\n'
    )
    resolved = let.resolve(let.providers.any.A100)
    assert resolved.provider.alias == "first"
    assert let.resolve(letify.AnyInstance("A100", host=Host.remote)).placement is Host.remote


def test_a_card_no_declared_provider_offers_says_where_it_looked(launcher_from) -> None:
    let = launcher_from('[lab]\nkind = "shell"\naddress = "a"\ngpus = ["A100"]\n')
    with pytest.raises(letify.UnknownInstance, match="Checked: lab, local"):
        let.resolve(let.providers.any.H200)


def test_an_instance_that_already_names_its_provider_resolves_to_itself(
    let: letify.Launcher, cpu: letify.Instance
) -> None:
    assert let.resolve(cpu) is cpu


def test_a_provider_is_built_once_and_remembered(let: letify.Launcher) -> None:
    assert let.provider("local") is let.provider("local")


def test_an_undeclared_alias_says_what_is_declared(let: letify.Launcher) -> None:
    with pytest.raises(letify.UnknownProvider, match="Declared: local"):
        let.provider("colab_a")


def test_a_private_attribute_on_the_provider_table_is_not_a_provider(
    let: letify.Launcher,
) -> None:
    with pytest.raises(AttributeError):
        let.providers._private  # noqa: B018
    with pytest.raises(AttributeError):
        let.providers.any._private  # noqa: B018


def test_a_provider_names_itself_by_alias_and_persistence(let: letify.Launcher) -> None:
    assert repr(let.providers.local) == "<Local local (persistent)>"


# -- Spec: Remaining usage ----------------------------------------------------


def test_a_provider_that_cannot_report_a_balance_says_so_instead_of_guessing() -> None:
    # A fabricated balance is worse than an absent one, because a researcher spends
    # against it. Every field but the alias, the unit and the source may be None.
    usage = provider_of(Modal, "m").usage()
    assert usage.alias == "m"
    assert usage.remaining is None
    assert usage.unit == "USD"
    assert "no workspace balance" in usage.source
    assert usage.unmetered is False


def test_this_machine_is_reported_as_unmetered_rather_than_unknown() -> None:
    # Nothing to ask and nothing to run out of are different answers.
    usage = provider_of(Local, "here").usage()
    assert usage.unmetered is True
    assert usage.remaining is None
    assert "bills nobody" in usage.source


def test_a_configured_command_supplies_the_balance_a_service_does_not_publish(
    isolated_home, patch_run, patch_which
) -> None:
    # The last number in the output is the remaining amount, so a command that prints a
    # sentence around it still works.
    patch_which(tools_module, present=True)
    provider = provider_of(
        Colab,
        "colab_a",
        usage_command="my-colab-units",
        usage_unit="compute units",
        usage_limit=100.0,
    )
    patch_run(usage_module, result=FakeCompleted(stdout="remaining: 42.5 units\n"))
    usage = provider.usage()
    assert usage.remaining == 42.5
    assert usage.unit == "compute units"
    assert usage.limit == 100.0
    assert usage.used == 57.5
    assert usage.source == "usage_command"


def test_a_usage_command_that_prints_no_number_reports_nothing_rather_than_zero(
    isolated_home, patch_run, patch_which
) -> None:
    patch_which(tools_module, present=True)
    provider = provider_of(Colab, "colab_a", usage_command="broken")
    patch_run(usage_module, result=FakeCompleted(stdout="quota service unreachable\n"))
    assert provider.usage().remaining is None


def test_a_usage_command_that_fails_does_not_take_the_table_down_with_it(
    isolated_home, patch_run, patch_which
) -> None:
    patch_which(tools_module, present=True)
    provider = provider_of(Colab, "colab_a", usage_command="broken")
    patch_run(usage_module, result=FakeCompleted(returncode=1, stderr="no such command"))
    usage = provider.usage()
    assert usage.remaining is None
    assert "exit 1" in usage.note


def test_elice_prices_what_is_running_now_from_the_zones_own_price_list(elice, fake_httpx) -> None:
    # Elice publishes no balance, so what it can answer honestly is the burn rate of the
    # allocations that exist and what they have cost so far.
    fake_httpx.answer(
        "GET",
        ALLOCATION_PATH,
        FakeResponse(200, {"items": [{"id": "alloc-1", "instance_type_id": "it-1"}]}),
    )
    fake_httpx.answer(
        "GET",
        PRICING_PATH,
        FakeResponse(200, {"items": [{"instance_type_id": "it-1", "price_per_hour": 975.0}]}),
    )
    usage = elice.usage()
    assert usage.unit == "KRW"
    assert usage.remaining is None
    assert usage.rate_per_hour == 975.0
    assert "price list" in usage.source


def test_elice_reports_a_zero_rate_when_nothing_is_allocated(elice, fake_httpx) -> None:
    # Nothing powered on costs nothing per hour, which is a number rather than a gap.
    fake_httpx.answer("GET", ALLOCATION_PATH, FakeResponse(200, {"items": []}))
    fake_httpx.answer("GET", PRICING_PATH, FakeResponse(200, {"items": []}))
    assert elice.usage().rate_per_hour == 0.0


def test_a_colab_account_directory_is_readable_by_its_owner_only(isolated_home) -> None:
    # The Colab CLI writes its token there with its own permissions, so the directory is
    # what keeps other users out.
    import sys

    from letify import tools

    tools.environment("colab_a")
    directory = Path.home() / ".letify" / "accounts" / "colab_a"
    if sys.platform != "win32":
        assert directory.stat().st_mode & 0o777 == 0o700
