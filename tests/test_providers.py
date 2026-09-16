"""Providers: what each account offers, how it is reached, and what it refuses.

Spec sections pinned here: "Provider model", "Provider properties", "Instances",
"Execution modes", "Transport" and "Packaging".

Local is exercised for real. Colab, Shell, Tunnel, Elice and Modal are driven with the
CLI, SSH, HTTP and client stand-ins from conftest, because each of those needs an
account, a second machine or a GPU. What is still real in those tests is the command
line, the request and the instance table letify builds, which is what a caller observes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import pytest
from conftest import FakeCompleted, FakeResponse, provider_of

import letify
from letify import providers
from letify import tools as tools_module
from letify.config.schema import ProviderConfig
from letify.declare.instance import Host, Instance
from letify.providers import colab as colab_module
from letify.providers import local as local_module
from letify.providers import modal as modal_module
from letify.providers import shell as shell_module
from letify.providers.base import Provider
from letify.providers.colab import ALIASES, Colab
from letify.providers.elice import (
    ALLOCATION_PATH,
    PRICING_PATH,
    Elice,
)
from letify.providers.local import Local
from letify.providers.modal import Adapter, Modal, SandboxChannel
from letify.providers.naming import gib_from_mib, normalize_gpu
from letify.providers.shell import Shell
from letify.providers.tunnel import Tunnel
from letify.runtime.channel import OneShotChannel, PersistentChannel
from letify.store.backends.filesystem import FilesystemBackend
from letify.transport import strategies as strategies_module
from letify.transport.rendezvous import ShellCommandRendezvous, TailcatRendezvous

probe_module = import_module("letify.remoting.probe")
usage_module = import_module("letify.providers.usage")


@pytest.fixture
def fresh_device_names():
    """Forget the machine's GPU list, which is read once per process."""
    local_module._device_names.cache_clear()
    yield
    local_module._device_names.cache_clear()


@pytest.fixture
def elice(fake_elice):
    return provider_of(
        Elice,
        "elice_a100",
        endpoint=fake_elice.endpoint,
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


def test_a_modal_provider_without_uv_says_uv_is_needed(isolated_home, patch_which) -> None:
    # Spec "Modal adapter": Modal's client runs in a uv environment, so uv is the one
    # thing this machine needs, and the error names it rather than a Python package.
    patch_which(tools_module, present=False)
    provider = provider_of(Modal, "m")
    runtime = type("R", (), {"name": "letify-h100-1", "instance": provider.H100})()
    with pytest.raises(letify.ProviderUnavailable, match="uv was not found"):
        provider.open_channel(runtime)


# -- Spec: Provider properties -------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "persistence", "fast_path", "persistent_channel"),
    [
        (Local, "persistent", True, True),
        (Modal, "persistent", False, True),
        (Colab, "ephemeral", False, True),
        (Shell, "ephemeral", True, True),
        (Tunnel, "ephemeral", True, True),
        (Elice, "ephemeral", True, True),
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


def test_a_colab_worker_started_over_ssh_is_given_the_driver_library_path(
    isolated_home,
) -> None:
    # Spec "Colab": only the kernel's environment names /usr/lib64-nvidia, so a worker
    # started over SSH without it sees no CUDA device on a session that holds a GPU.
    from conftest import FakeLink

    provider = provider_of(Colab, "colab_a")
    runtime = type("R", (), {"name": "letify-l4-1"})()
    provider.__dict__["_links"] = {"letify-l4-1": FakeLink("tcp_punch", 2)}
    channel = provider.open_channel(runtime)
    assert isinstance(channel, PersistentChannel)
    assert "LD_LIBRARY_PATH=/usr/lib64-nvidia" in " ".join(channel.command)


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


def test_colab_offers_a_cpu_instance_under_the_name_local_uses(isolated_home) -> None:
    provider = provider_of(Colab, "colab_a")
    assert provider.cpu is provider.CPU
    assert provider.CPU.gpu is None and provider.CPU.tpu is None
    assert provider.devices_of(provider.cpu.accelerator).accelerator == "cpu"


def test_creating_a_colab_session_for_the_cpu_asks_for_no_accelerator(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    recorder = patch_run(colab_module)
    provider = provider_of(Colab, "colab_a")
    provider.create_session(provider.cpu, "letify-cpu-1")
    assert recorder.command == [*COLAB_CLI, "new", "-s", "letify-cpu-1"]


def test_a_failing_cli_command_names_the_command_and_the_end_of_its_stderr(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    stderr = "".join(f"line {n}\n" for n in range(100)) + "not entitled\n"
    patch_run(colab_module, result=FakeCompleted(returncode=2, stderr=stderr))
    with pytest.raises(letify.RuntimeFailure) as caught:
        provider_of(Colab, "colab_a").sessions()
    message = str(caught.value)
    assert "colab sessions" in message
    assert "not entitled" in message and "line 99" in message
    assert "line 60\n" not in message


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


def test_an_account_with_no_colab_session_lists_none(isolated_home, patch_which, patch_run) -> None:
    # Spec "Colab": the CLI's own message line names no session.
    patch_which(tools_module, present=True)
    listing = "[colab] No active sessions found on server.\n"
    patch_run(colab_module, result=FakeCompleted(stdout=listing))
    assert provider_of(Colab, "colab_a").sessions() == []


def test_a_colab_session_listed_in_brackets_is_named_without_them(
    isolated_home, patch_which, patch_run
) -> None:
    # Spec "Colab": the listing line the CLI printed for a live CPU session.
    patch_which(tools_module, present=True)
    listing = (
        "[letify-cpu-514e8c] m-s-kkb-ase1a1-32hpj198ieqis | Hardware: CPU | Variant: DEFAULT\n"
    )
    patch_run(colab_module, result=FakeCompleted(stdout=listing))
    assert provider_of(Colab, "colab_a").sessions() == ["letify-cpu-514e8c"]


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


# -- Spec: Transport, Colab bulk transfer through the Jupyter file API ---------


@pytest.fixture
def small_parts(monkeypatch):
    """Parts and chunks a few bytes long, so a short payload crosses every boundary."""
    from letify.providers import colab_files

    monkeypatch.setattr(colab_files, "PART_BYTES", 16)
    monkeypatch.setattr(colab_files, "CHUNK_BYTES", 6)
    return colab_files


def local_exec(source: str, timeout: float | None = None) -> str:
    """What `colab exec` does, on this machine: run a program and return its output."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=timeout or 120
    )
    if result.returncode != 0:
        raise letify.RuntimeFailure("exec failed", stderr=result.stderr)
    return result.stdout


def colab_exec_channel(fake_jupyter, name: str = "letify-g4-1", workspace: Path | None = None):
    from conftest import write_colab_session

    write_colab_session("colab_a", name, fake_jupyter.url, fake_jupyter.token)
    # The VM here is this machine, so a root under the test's directory stands in for it.
    options = {"workspace": str(workspace)} if workspace is not None else {}
    provider = provider_of(Colab, "colab_a", channel="exec", **options)
    provider._exec = lambda session, source, timeout: local_exec(source, timeout)
    runtime = type("R", (), {"name": name})()
    return provider.open_channel(runtime)


def test_the_proxy_address_comes_from_the_colab_cli_session_state(isolated_home) -> None:
    from conftest import write_colab_session

    from letify.providers.colab_files import session_endpoint

    write_colab_session("colab_a", "letify-g4-1", "https://proxy.example/", "proxy-token-1")
    assert session_endpoint("colab_a", "letify-g4-1") == (
        "https://proxy.example",
        "proxy-token-1",
    )
    with pytest.raises(letify.RuntimeFailure, match=r"sessions\.json"):
        session_endpoint("colab_a", "letify-t4-9")


def test_an_upload_is_sent_in_parallel_parts_of_chunked_puts_and_joined(
    isolated_home, fake_jupyter, small_parts, tmp_path
) -> None:
    import base64 as b64

    channel = colab_exec_channel(fake_jupyter)
    payload = bytes(range(40))
    target = tmp_path / "vm" / "weights.bin"
    target.parent.mkdir()

    value, _ = channel.request(
        {"op": "put_file", "path": str(target), "payload": b64.b64encode(payload).decode()}
    )

    assert target.read_bytes() == payload
    assert value == {"path": str(target), "size": 40}
    # 40 bytes in parts of 16 is three parts; the joined file leaves no part behind.
    assert list(target.parent.iterdir()) == [target]
    parts = sorted({r["path"] for r in fake_jupyter.puts()})
    assert parts == [f"/api/contents{target}.letify-part-{n}" for n in range(3)]
    assert all(
        r["query"]["colab-runtime-proxy-token"] == "proxy-token-1" for r in fake_jupyter.puts()
    )
    assert all(r["header_token"] == "proxy-token-1" for r in fake_jupyter.puts())


def test_each_part_is_numbered_the_way_the_large_file_manager_expects(
    isolated_home, fake_jupyter, small_parts, tmp_path, monkeypatch
) -> None:
    from letify.providers.colab_files import ContentsTransfer

    sent: list[tuple[str, object]] = []
    original = ContentsTransfer._put_chunk

    def record(self, remote: str, piece: bytes, chunk: int) -> None:
        sent.append((remote.rsplit("-", 1)[-1], chunk))
        original(self, remote, piece, chunk)

    monkeypatch.setattr(ContentsTransfer, "_put_chunk", record)
    target = tmp_path / "vm.bin"
    ContentsTransfer(fake_jupyter.url, fake_jupyter.token).upload(bytes(20), str(target))
    # Parts of 16 and 4 bytes, in chunks of 6.
    assert sorted(sent) == [("0", -1), ("0", 1), ("0", 2), ("1", 1)]


def test_an_uploaded_archive_is_unpacked_by_the_join_program(
    isolated_home, fake_jupyter, small_parts, tmp_path
) -> None:
    import base64 as b64
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("site/marker.txt")
        info.size = 6
        archive.addfile(info, io.BytesIO(b"cached"))
    mount = tmp_path / "mount"
    channel = colab_exec_channel(fake_jupyter)
    channel.request(
        {
            "op": "put_file",
            "path": str(mount / "blobs" / "env.tar.gz"),
            "payload": b64.b64encode(buffer.getvalue()).decode(),
            "unpack": True,
            "target": str(mount),
        }
    )
    assert (mount / "site" / "marker.txt").read_text(encoding="utf-8") == "cached"


def test_a_download_reads_the_file_in_parallel_ranges(
    isolated_home, fake_jupyter, small_parts, tmp_path
) -> None:
    import base64 as b64

    source = tmp_path / "checkpoint.pt"
    source.write_bytes(bytes(range(35)))
    channel = colab_exec_channel(fake_jupyter)

    value, _ = channel.request({"op": "get_file", "path": str(source)})

    assert b64.b64decode(value["payload"]) == bytes(range(35))
    assert value["size"] == 35
    assert sorted(fake_jupyter.ranges()) == ["bytes=0-15", "bytes=16-31", "bytes=32-34"]


def test_a_directory_is_packed_on_the_vm_and_then_downloaded(
    isolated_home, fake_jupyter, small_parts, tmp_path
) -> None:
    import base64 as b64
    import io
    import tarfile

    site = tmp_path / "site"
    site.mkdir()
    (site / "a.txt").write_text("a", encoding="utf-8")
    channel = colab_exec_channel(fake_jupyter, workspace=tmp_path / "ws")

    value, _ = channel.request({"op": "pack_dir", "path": str(site)})

    with tarfile.open(fileobj=io.BytesIO(b64.b64decode(value["payload"])), mode="r:gz") as archive:
        assert "site/a.txt" in archive.getnames()


def test_the_pipeline_fallback_link_carries_bulk_transfer(isolated_home, fake_jupyter) -> None:
    from conftest import write_colab_session

    write_colab_session("colab_a", "letify-g4-1", fake_jupyter.url, fake_jupyter.token)
    provider = provider_of(Colab, "colab_a")
    runtime = type("R", (), {"name": "letify-g4-1"})()
    link = provider.fallback(runtime)()
    assert link.files is not None


def test_a_refused_proxy_request_names_the_path_and_the_status(
    isolated_home, fake_jupyter, tmp_path
) -> None:
    from letify.providers.colab_files import ContentsTransfer

    with pytest.raises(letify.RuntimeFailure, match="403"):
        ContentsTransfer(fake_jupyter.url, "wrong").download(str(tmp_path / "x.bin"))


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


@pytest.mark.parametrize("platform", ["posix", "nt"])
def test_ssh_commands_share_one_connection_and_prefer_fast_ciphers(
    isolated_home, monkeypatch, platform: str
) -> None:
    # Spec "SSH authentication": multiplexing where OpenSSH implements it, never on Windows.
    from letify.transport import sshopts
    from letify.transport.strategies import Target

    monkeypatch.setattr(sshopts, "WINDOWS", platform == "nt")
    # The pytest temporary directory is too long for a socket path, so the /tmp default is used.
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    provider = provider_of(Shell, "lab", address="gpu.lab.example.edu", user="researcher")
    for command in (
        provider.ssh_command("true"),
        Target(alias="lab", user="researcher").forwarded_ssh(2200, "true"),
    ):
        assert "Ciphers=^aes128-gcm@openssh.com,chacha20-poly1305@openssh.com" in command
        assert not any(part.startswith("Compression=yes") for part in command)
        tag = hashlib.sha256(b"lab").hexdigest()[:8]
        control = Path(f"/tmp/letify-{os.getuid()}") / f"{tag}-%C"
        if platform == "nt":
            assert not any(part.startswith("Control") for part in command)
        else:
            assert "ControlMaster=auto" in command
            assert "ControlPersist=60" in command
            assert f"ControlPath={control}" in command
            assert control.parent.is_dir()
            assert control.parent.stat().st_mode & 0o777 == 0o700


def control_options(command: list[str]) -> list[str]:
    return [part for part in command if part.startswith("Control")]


def test_a_control_socket_fits_the_socket_limit_for_a_very_long_home(
    tmp_path: Path, monkeypatch
) -> None:
    # Spec "SSH authentication": sockets live in a short per-user directory.
    from letify.transport import sshopts

    monkeypatch.setattr(sshopts, "WINDOWS", False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    home = tmp_path / ("h" * 150)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    command = sshopts.options("a-rather-long-account-alias-for-the-lab-machine")
    (path,) = [part[len("ControlPath=") :] for part in command if part.startswith("ControlPath=")]
    expanded = path.replace("%C", "0" * 40)
    assert len(expanded.encode()) + 17 < sshopts.SOCKET_LIMIT
    assert Path(path).parent == Path(f"/tmp/letify-{os.getuid()}")


def test_sharing_is_left_out_when_the_socket_path_would_exceed_the_limit(
    tmp_path: Path, monkeypatch
) -> None:
    from letify.transport import sshopts

    monkeypatch.setattr(sshopts, "WINDOWS", False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / ("r" * 120)))
    command = sshopts.options("lab")
    assert control_options(command) == []
    assert "Ciphers=^aes128-gcm@openssh.com,chacha20-poly1305@openssh.com" in command


def test_a_control_directory_that_is_a_symbolic_link_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    from letify.transport import sshopts

    monkeypatch.setattr(sshopts, "WINDOWS", False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (tmp_path / "letify").symlink_to(elsewhere)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(letify.ConfigError, match="symbolic link"):
        sshopts.options("lab")


def test_a_control_directory_open_to_other_users_is_refused(tmp_path: Path, monkeypatch) -> None:
    from letify.transport import sshopts

    monkeypatch.setattr(sshopts, "WINDOWS", False)
    (tmp_path / "letify").mkdir()
    (tmp_path / "letify").chmod(0o755)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(letify.ConfigError, match="permission"):
        sshopts.options("lab")


def test_a_control_directory_owned_by_another_user_is_refused(tmp_path: Path, monkeypatch) -> None:
    from letify.transport import sshopts

    monkeypatch.setattr(sshopts, "WINDOWS", False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(sshopts.os, "getuid", lambda: os.stat(tmp_path).st_uid + 1)
    with pytest.raises(letify.ConfigError, match="owned"):
        sshopts.options("lab")


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
    # Spec "Channels": the stub reads a byte count line and raw source, with no base64.
    assert "sys.stdin.buffer" in channel.command[-1]
    assert "base64" not in channel.command[-1]


# -- Spec: Transport, Connection strategies per provider --------------------------


def test_a_plain_shell_lists_the_four_default_strategies_in_rank_order() -> None:
    provider = provider_of(Shell, "lab", address="gpu.example.edu")
    assert [(s.name, s.rank) for s in provider.strategies()] == [
        ("direct_ssh", 1),
        ("tcp_punch", 2),
        ("tailcat", 3),
        ("fallback", 4),
    ]


def test_a_shell_with_no_rendezvous_is_reached_by_forward_ssh_alone(patch_run) -> None:
    recorder = patch_run(shell_module)
    provider = provider_of(Shell, "lab", address="gpu.example.edu")
    assert provider.rendezvous() is None
    assert provider.link().strategy == "direct_ssh"
    assert recorder.calls == []


def test_reverse_ssh_takes_rank_four_and_moves_the_fallback_to_five() -> None:
    provider = provider_of(
        Shell,
        "lab",
        address="gpu.example.edu",
        reverse_ssh={"address": "home.example.com", "port": 2222, "user": "me"},
    )
    assert [(s.name, s.rank) for s in provider.strategies()][-2:] == [
        ("reverse_ssh", 4),
        ("fallback", 5),
    ]


def test_an_account_with_a_tailcat_address_has_the_agent_as_its_rendezvous() -> None:
    provider = provider_of(Tunnel, "lab", tailcat="tcHome", tailcat_port=40123)
    rendezvous = provider.rendezvous()
    assert isinstance(rendezvous, TailcatRendezvous)
    assert (rendezvous.address, rendezvous.port) == ("tcHome", 40123)


def test_a_tunnel_with_no_address_and_no_tailcat_says_what_is_missing(isolated_home) -> None:
    with pytest.raises(letify.ProviderUnavailable) as caught:
        provider_of(Tunnel, "lab").link()
    assert "no address" in str(caught.value)
    assert "no rendezvous" in str(caught.value)


def test_colab_races_without_forward_ssh(isolated_home, patch_which) -> None:
    patch_which(tools_module, present=True)
    provider = provider_of(Colab, "colab_a")
    assert [s.name for s in provider.strategies()] == ["tcp_punch", "tailcat", "fallback"]


def test_the_colab_rendezvous_is_colab_exec_and_asks_for_an_ssh_server(
    isolated_home, patch_which, patch_run
) -> None:
    patch_which(tools_module, present=True)
    answer = FakeCompleted(stdout='LETIFY-ANSWER {"pong": true}\n')
    recorder = patch_run(colab_module, result=answer)
    provider = provider_of(Colab, "colab_a")
    runtime = type("R", (), {"name": "letify-g4-1"})()
    assert provider.rendezvous(runtime).exchange({"kind": "ping"}, 30) == {"pong": True}
    assert recorder.command == [*COLAB_CLI, "exec", "-s", "letify-g4-1"]
    assert '"start_sshd": true' in recorder.calls[-1]["input"]


def test_colab_falls_back_to_exec_when_nothing_else_connects(
    isolated_home, patch_which, patch_run
) -> None:
    # One shutil module backs every which, so only tailcat is hidden and uv stays found.
    patch_which(
        strategies_module,
        present=lambda name: None if name == "tailcat" else f"/usr/bin/{name}",
    )
    recorder = patch_run(colab_module)
    provider = provider_of(Colab, "colab_a", stun="127.0.0.1:9")
    runtime = type("R", (), {"name": "letify-g4-1"})()
    channel = provider.open_channel(runtime)
    assert isinstance(channel, OneShotChannel)
    channel.runner("print('hello')", 60)
    assert recorder.command == [*COLAB_CLI, "exec", "-s", "letify-g4-1"]


def test_elice_runs_its_remote_half_over_forward_ssh_to_the_started_machine() -> None:
    # Spec "Elice machines": a launched machine is reached as root unless the account names
    # another user.
    provider = provider_of(Elice, "e", zone_id="z", machine_id="m", address="gpu.elice.io")
    rendezvous = provider.rendezvous()
    assert isinstance(rendezvous, ShellCommandRendezvous)
    assert rendezvous.ssh("python3 -")[-2:] == ["ubuntu@gpu.elice.io", "python3 -"]
    assert provider_of(Elice, "e", zone_id="z", machine_id="m").rendezvous() is None


# -- Spec: Provider model, Elice -----------------------------------------------


def test_elice_needs_a_zone_and_a_token() -> None:
    # A machine is optional: without machine_id letify creates one on first use.
    with pytest.raises(letify.ProviderUnavailable, match="no 'zone_id' field"):
        provider_of(Elice, "elice_a100").zone_id  # noqa: B018
    assert provider_of(Elice, "elice_a100", zone_id="z").machine_id is None
    with pytest.raises(letify.ProviderUnavailable, match="access_token_env"):
        provider_of(Elice, "elice_a100", zone_id="z").pricing()


def test_the_elice_endpoint_can_be_pointed_elsewhere() -> None:
    assert provider_of(Elice, "e", endpoint="https://portal.example/api/").endpoint == (
        "https://portal.example/api"
    )
    assert provider_of(Elice, "e").endpoint.startswith("https://")


def test_elice_is_reached_with_the_standard_library_alone(no_module, fake_elice) -> None:
    # Spec "Packaging": Elice uses the standard library HTTP client, so no HTTP package
    # has to be installed.
    no_module("httpx", "requests")
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(200, []))
    provider = provider_of(Elice, "e", endpoint=fake_elice.endpoint, zone_id="z", access_token="t")
    assert provider.pricing() == []


def test_an_elice_request_carries_the_token(elice, fake_elice) -> None:
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(200, {"items": [{"id": "p1"}]}))
    assert elice.pricing() == [{"id": "p1"}]
    assert fake_elice.last["authorization"] == "Bearer token-1"


def test_anything_other_than_a_two_hundred_is_a_failure(elice, fake_elice) -> None:
    # This API answers 200 for every success.
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(403, {"message": "quota exceeded"}))
    with pytest.raises(letify.RuntimeFailure, match="returned 403: quota exceeded"):
        elice.pricing()


def test_a_failure_with_no_json_body_carries_the_text(elice, fake_elice) -> None:
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(502, None, text="<html>bad gateway</html>"))
    with pytest.raises(letify.RuntimeFailure, match="bad gateway"):
        elice.pricing()


def test_a_response_may_be_a_bare_list_or_an_items_table(elice, fake_elice) -> None:
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(200, [{"id": "p1"}]))
    assert elice.pricing() == [{"id": "p1"}]


def test_the_instance_types_eci_lists_are_normalized(isolated_home, fake_eci) -> None:
    # Spec "Elice machines": instance types come from eci instance-type list.
    provider = provider_of(Elice, "e", zone_id="zone-1", access_token="token-1")
    table = provider.instances
    assert sorted(table) == ["A100", "CPU"]
    assert table["A100"].cpus == 16
    assert fake_eci.commands() == ["instance-type list"]


def test_the_zone_price_list_includes_any_preemptible_option(elice, fake_elice) -> None:
    fake_elice.answer("GET", "/user/pricing", FakeResponse(200, [{"id": "p1", "spot": True}]))
    assert elice.pricing() == [{"id": "p1", "spot": True}]


def test_the_allocations_of_one_machine_can_be_listed(elice, fake_elice) -> None:
    fake_elice.answer("GET", ALLOCATION_PATH, FakeResponse(200, {"items": []}))
    assert elice.allocations("machine-1") == []
    assert fake_elice.last["params"] == {"filter_machine_id": "machine-1"}
    assert elice.allocations() == []
    assert fake_elice.last["params"] is None


def test_an_elice_gpu_list_may_be_declared_instead_of_asked_for(fake_elice) -> None:
    provider = provider_of(Elice, "e", zone_id="z", access_token="t", gpus=["A100", "H100"])
    assert sorted(provider.instances) == ["A100", "H100"]
    assert fake_elice.requests == []


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
    # The warning carries the arithmetic. The refusal that follows is about PyTorch being
    # absent from this process, not about the latency.
    patch_which(tools_module, present=True)
    monkeypatch.setitem(sys.modules, "torch", None)
    provider = provider_of(Colab, "colab_a")
    with (
        pytest.warns(UserWarning, match="round trip of about 175 ms"),
        pytest.raises(letify.UnsupportedMode, match="PyTorch"),
    ):
        provider.check_mode(provider.G4._placed("local"))


def test_a_provider_with_a_fast_path_does_not_warn(
    patch_run, recwarn, monkeypatch, tmp_path
) -> None:
    # Whether PyTorch is installed in the test environment must not decide this answer.
    monkeypatch.setitem(sys.modules, "torch", None)
    patch_run(shell_module)
    provider = provider_of(Shell, "lab", address="gpu.example.edu", gpus=["A100"])
    with pytest.raises(letify.UnsupportedMode, match="PyTorch"):
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


# -- Spec: Modal adapter -------------------------------------------------------


def modal_runtime(provider: Modal):
    return type("R", (), {"name": "letify-h100-1", "instance": provider.H100})()


def test_the_modal_adapter_runs_in_its_own_uv_environment_with_a_pinned_modal() -> None:
    command = tools_module.modal_adapter_command("/usr/bin/uv")
    assert command[:6] == ["/usr/bin/uv", "run", "--no-project", "--python", "3.12", "--with"]
    assert command[6] == "modal>=1.0,<2"
    assert command[7:9] == ["python", "-P"]
    assert Path(command[9]) == Path(modal_module.__file__).with_name("modal_adapter.py")


def test_a_sibling_modal_module_does_not_shadow_the_modal_package(tmp_path: Path) -> None:
    # The adapter sits next to letify/providers/modal.py. Run by path without -P, Python
    # puts that directory first on sys.path, so `import modal` would load the sibling.
    import os
    import shutil
    import sys

    providers = tmp_path / "providers"
    providers.mkdir()
    script = providers / "modal_adapter.py"
    shutil.copy(Path(modal_module.__file__).with_name("modal_adapter.py"), script)
    (providers / "modal.py").write_text("from . import nothing\n", encoding="utf-8")
    site = tmp_path / "site" / "modal"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("__version__ = 'the-real-package'\n", encoding="utf-8")

    command = tools_module.modal_adapter_command("/usr/bin/uv")
    command = command[command.index("python") :]
    command[0], command[-1] = sys.executable, str(script)
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site")}
    adapter = Adapter(command, env=env, name="m")
    try:
        assert adapter.request("hello") == {"modal": "the-real-package"}
    finally:
        adapter.close()


#: A stand-in ``modal`` package that logs how the adapter uses apps, so the real adapter
#: file can be run with no Modal account and no resource created.
STUB_MODAL = """
import contextlib, json, os

def _log(*event):
    with open(os.environ["STUB_MODAL_LOG"], "a", encoding="utf-8") as out:
        out.write(json.dumps(event) + "\\n")

class App:
    def __init__(self, name=None, **kwargs):
        self.name = name
    @classmethod
    def lookup(cls, name, create_if_missing=False, **kwargs):
        _log("lookup", name)
        return cls(name)
    @contextlib.contextmanager
    def run(self, **kwargs):
        _log("run_start", self.name)
        try:
            yield self
        finally:
            _log("run_stop", self.name)

class Image:
    @staticmethod
    def debian_slim():
        return Image()
    def pip_install(self, *packages):
        return self

class _Tunnel:
    host = "r7.modal.host"
    port = 443

class _Sandbox:
    object_id = None
    stdout = ()
    def terminate(self):
        _log("terminate")
    def tunnels(self, timeout=50):
        return {8765: _Tunnel()}

class Sandbox:
    @staticmethod
    def create(*args, app=None, **kwargs):
        _log("create", app.name, list(kwargs.get("encrypted_ports") or []))
        return _Sandbox()
"""


def test_the_modal_adapter_runs_sandboxes_in_an_ephemeral_app_it_stops_on_exit(
    tmp_path: Path,
) -> None:
    # Spec "Modal adapter": no deployed app is left on the account once letify stops.
    import json
    import os
    import sys

    site = tmp_path / "site" / "modal"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text(STUB_MODAL, encoding="utf-8")
    log = tmp_path / "modal.log"
    script = Path(modal_module.__file__).with_name("modal_adapter.py")
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site"), "STUB_MODAL_LOG": str(log)}
    adapter = Adapter([sys.executable, "-P", str(script)], env=env, name="m")
    fields = {"app": "study", "args": ["python3"], "packages": [], "gpu": None, "timeout": 60}
    first = adapter.request("create", **fields)["sandbox"]
    adapter.request("create", **fields)
    adapter.request("terminate", sandbox=first)
    adapter.close()

    events = [tuple(json.loads(line)) for line in log.read_text("utf-8").splitlines()]
    events = [e[:2] if e[0] == "create" else e for e in events]
    assert ("lookup", "study") not in events
    assert events[0] == ("run_start", "study")
    assert events[-1] == ("run_stop", "study")
    assert [e for e in events if e[0] == "run_start"] == [("run_start", "study")]
    assert events.count(("terminate",)) == 2


def test_the_modal_app_stops_when_letify_closes_the_adapter(isolated_home, fake_modal) -> None:
    provider = provider_of(Modal, "modal_lab", app="study")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    provider.stop(runtime)
    provider.adapter().close()
    assert fake_modal.app_events() == [["run_start", "study"], ["run_stop", "study"]]


def test_modal_offers_a_cpu_instance_under_the_name_local_uses(isolated_home) -> None:
    provider = provider_of(Modal, "modal_lab")
    assert provider.cpu is provider.CPU
    assert provider.CPU.gpu is None


def test_creating_a_modal_sandbox_for_the_cpu_asks_for_no_gpu(isolated_home, fake_modal) -> None:
    provider = provider_of(Modal, "modal_lab")
    runtime = type("R", (), {"name": "letify-cpu-1", "instance": provider.CPU})()
    provider.open_channel(runtime)
    try:
        [created] = fake_modal.requests("create")
        assert created["gpu"] is None
    finally:
        provider.stop(runtime)


def test_the_modal_adapter_imports_nothing_outside_the_standard_library_at_load() -> None:
    # It runs by file path in an environment that holds only modal, so neither letify nor
    # letify's own dependencies may be imported, and modal only when an op needs it.
    import ast
    import sys

    source = Path(modal_module.__file__).with_name("modal_adapter.py").read_text("utf-8")
    top: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import needs the letify package"
            top.add(str(node.module).split(".")[0])
    assert top <= set(sys.stdlib_module_names) | {"__future__"}, top


def test_the_real_modal_adapter_answers_a_missing_modal_as_unavailable() -> None:
    # The real adapter file, run in isolated mode so letify is not importable, in an
    # interpreter that has no modal: the reply must say unavailable, not crash.
    import sys

    script = Path(modal_module.__file__).with_name("modal_adapter.py")
    adapter = Adapter([sys.executable, "-I", str(script)], env=None, name="m")
    try:
        with pytest.raises(letify.ProviderUnavailable, match="modal"):
            adapter.request("hello")
    finally:
        adapter.close()


def test_the_modal_adapter_acts_as_the_account_in_the_account_directory(
    isolated_home, fake_modal, monkeypatch
) -> None:
    # A token in the caller's environment would override the account file, so two
    # accounts could not coexist. It is removed, and the account's modal.toml decides.
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-someone-else")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-someone-else")
    monkeypatch.setenv("MODAL_PROFILE", "someone-else")
    provider = provider_of(Modal, "modal_lab")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    provider.stop(runtime)

    env = fake_modal.env()
    expected = Path.home() / ".letify" / "accounts" / "modal_lab" / "modal.toml"
    assert env["MODAL_CONFIG_PATH"] == str(expected)
    assert "MODAL_TOKEN_ID" not in env
    assert "MODAL_TOKEN_SECRET" not in env
    assert "MODAL_PROFILE" not in env


def test_a_modal_sandbox_keeps_a_process_alive_for_framed_requests(
    isolated_home, fake_modal
) -> None:
    # A sandbox rather than a function call, because without a living process there is no
    # object table for a handle to point at.
    from letify.protocol.worker import BOOTSTRAP

    provider = provider_of(Modal, "m", app="study", timeout=1800)
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        assert isinstance(channel, SandboxChannel)
        assert channel.persistent is True
        [created] = fake_modal.requests("create")
        assert created["app"] == "study"
        assert created["gpu"] == "H100"
        assert created["timeout"] == 1800
        # Spec "Channels": not `python -`, which reads standard input to the end before
        # running anything, so the requests that follow would be compiled as source.
        assert created["args"] == ["python3", "-u", "-c", BOOTSTRAP]
        assert created["packages"] == ["cloudpickle", "blake3"]
    finally:
        provider.stop(runtime)


def test_a_sandbox_started_the_way_modal_starts_it_answers_requests(
    isolated_home, fake_modal
) -> None:
    # The stand-in runs the requested command as a real process, so the worker has to
    # come up from the bootstrap stub and answer framed requests over the adapter.
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        assert channel.call(len, ([1, 2, 3],), {})[0] == 3
        assert channel.call(sum, ([1, 2, 3],), {})[0] == 6
    finally:
        channel.close()
        provider.stop(runtime)


def test_a_sandbox_channel_sends_an_argument_larger_than_modal_stdin_buffer(
    isolated_home, fake_modal
) -> None:
    # Spec "Modal adapter", op write: a frame above Modal's 2 MiB stdin buffer goes out as
    # write requests of at most 1 MiB, so a 16 MiB argument arrives whole.
    import base64

    provider = provider_of(Modal, "m", data_channel=False)
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        payload = bytes(range(256)) * (16 * 1024 * 1024 // 256)
        assert channel.call(len, (payload,), {})[0] == len(payload)
        sizes = [len(base64.b64decode(r["data"])) for r in fake_modal.requests("write")]
        assert max(sizes) <= 1024 * 1024
    finally:
        channel.close()
        provider.stop(runtime)


def test_a_sandbox_channel_reads_a_result_longer_than_a_modal_output_line(
    isolated_home, fake_modal
) -> None:
    # Spec "Modal adapter": Modal splits a stdout line above 64 KiB, so a frame line stays
    # below that and a 1 MiB result arrives whole.
    import os

    provider = provider_of(Modal, "m", data_channel=False)
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        assert len(channel.call(os.urandom, (1 << 20,), {})[0]) == 1 << 20
    finally:
        channel.close()
        provider.stop(runtime)


def test_a_sandbox_channel_sends_the_worker_once_and_then_framed_requests(
    isolated_home, fake_modal
) -> None:
    import base64

    from letify.protocol.worker import SOURCE
    from letify.protocol.worker import source as worker_source

    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        channel.start()
        channel.start()
        channel.call(len, ("abc",), {})
        source = worker_source(text_frames=True).encode()
        written = b"".join(base64.b64decode(r["data"]) for r in fake_modal.requests("write"))
        # The worker source went out exactly once, behind its byte count line, with the flag
        # that makes the sandbox write its frames as base64 lines.
        assert written.count(b"%d\n" % len(source) + source) == 1
        assert SOURCE.encode() not in written
    finally:
        provider.stop(runtime)


def test_a_sandbox_that_stops_without_replying_is_a_protocol_error(
    isolated_home, fake_modal
) -> None:
    adapter = Adapter.for_account("m")
    try:
        created = adapter.request(
            "create",
            app="letify",
            args=["python3", "-c", "print('Killed')"],
            packages=[],
            gpu=None,
            timeout=60,
        )
        channel = SandboxChannel(adapter, created["sandbox"], name="letify-h100-1")
        with pytest.raises(letify.ProtocolError, match="died before it finished") as caught:
            channel.request({"op": "stat"})
        assert "Killed" in str(caught.value)
    finally:
        adapter.close()


def test_closing_a_sandbox_channel_asks_the_worker_to_shut_down(isolated_home, fake_modal) -> None:
    provider = provider_of(Modal, "m", data_channel=False)
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    channel.start()
    channel.close()
    provider.stop(runtime)
    import base64

    from letify.protocol import wire

    written = b"".join(base64.b64decode(r["data"]) for r in fake_modal.requests("write"))
    assert wire.HEADER.pack(wire.MAGIC, wire.SHUTDOWN, 0, 0, 0) in written


def test_closing_a_sandbox_whose_adapter_is_already_gone_is_harmless(
    isolated_home, fake_modal
) -> None:
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    provider.adapter().close()
    assert channel.close() is None


def test_stopping_a_modal_runtime_terminates_its_sandbox(isolated_home, fake_modal) -> None:
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    provider.stop(runtime)
    assert [r["sandbox"] for r in fake_modal.requests("terminate")] == ["sb-1"]
    # A sandbox that is already gone is fine, and so is one that was never opened.
    assert provider.stop(runtime) is None


def test_a_sandbox_that_will_not_terminate_does_not_break_the_teardown(
    isolated_home, fake_modal
) -> None:
    fake_modal.fail("terminate")
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    assert provider.stop(runtime) is None


def test_an_adapter_that_exits_is_an_infrastructure_failure_carrying_its_stderr(
    isolated_home, fake_modal
) -> None:
    # Retryable, because nothing of the user's code ran.
    fake_modal.exit_on("create")
    provider = provider_of(Modal, "m")
    with pytest.raises(letify.RuntimeFailure, match="crashed on create"):
        provider.open_channel(modal_runtime(provider))


def test_a_failure_the_adapter_reports_is_raised_with_its_message(
    isolated_home, fake_modal
) -> None:
    fake_modal.fail("create")
    provider = provider_of(Modal, "m")
    with pytest.raises(letify.RuntimeFailure, match="create was refused by the fake"):
        provider.open_channel(modal_runtime(provider))


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


def _modal_billing(monkeypatch, **summary: object) -> None:
    monkeypatch.setenv("FAKE_MODAL_BILLING", json.dumps(summary))


def test_modal_counts_the_months_metered_cost_against_the_monthly_credit(
    isolated_home, fake_modal, monkeypatch
) -> None:
    # Modal publishes the month's spend, not a balance, so the balance is the plan credit
    # minus that spend, renewed at the start of the next month.
    _modal_billing(
        monkeypatch,
        metered_cost="4.50000000",
        billed_cost="0",
        credits="-4.5",
        start=1788220800.0,
        end=1790812800.0,
    )
    usage = provider_of(Modal, "m").usage()
    assert usage.unit == "USD"
    assert usage.used == 4.5
    assert usage.limit == 30.0
    assert usage.remaining == 25.5
    assert usage.resets_at == 1790812800.0
    assert usage.note is None
    assert [r["op"] for r in fake_modal.requests()] == ["billing_summary"]


def test_a_declared_monthly_credit_replaces_the_starter_plan_credit(
    isolated_home, fake_modal, monkeypatch
) -> None:
    _modal_billing(monkeypatch, metered_cost="120", billed_cost="20", credits="-100", end=1.0)
    usage = provider_of(Modal, "m", monthly_credit=100.0).usage()
    assert usage.limit == 100.0
    # Spend past the credit leaves nothing, not a negative balance.
    assert usage.remaining == 0.0


def test_a_modal_billing_failure_is_a_note_rather_than_an_error(isolated_home, fake_modal) -> None:
    # FAKE_MODAL_BILLING is unset, so the adapter answers the op with a failure.
    usage = provider_of(Modal, "m").usage()
    assert usage.remaining is None
    assert "billing" in (usage.note or "")


def _colab_token(home: Path, alias: str, token_uri: str, expiry: str) -> Path:
    path = home / ".letify" / "accounts" / alias / ".config" / "colab-cli" / "token.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "token": "stale-token",
                "refresh_token": "refresh-1",
                "client_id": "client-1",
                "client_secret": "secret-1",
                "token_uri": token_uri,
                "expiry": expiry,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_colab_reads_the_compute_unit_balance_the_web_page_reads(
    isolated_home, fake_google, monkeypatch
) -> None:
    # An expired token is refreshed in memory and the CLI's file is left as it was.
    home = Path.home()
    token = _colab_token(home, "colab_a", f"{fake_google.endpoint}/token", "2000-01-01T00:00:00Z")
    before = token.read_text(encoding="utf-8")
    monkeypatch.setattr(colab_module, "CCU_INFO_URL", f"{fake_google.endpoint}/ccu-info")
    fake_google.answer("POST", "/token", FakeResponse(200, {"access_token": "fresh-token"}))
    fake_google.answer(
        "GET",
        "/ccu-info",
        FakeResponse(200, text=')]}\'\n{"currentBalance": 99.93, "consumptionRateHourly": 1.5}'),
    )
    usage = provider_of(Colab, "colab_a").usage()
    assert usage.unit == "compute units"
    assert usage.remaining == 99.93
    assert usage.rate_per_hour == 1.5
    assert fake_google.requests[0]["form"]["refresh_token"] == "refresh-1"
    assert fake_google.last["authorization"] == "Bearer fresh-token"
    assert token.read_text(encoding="utf-8") == before


def test_a_colab_account_that_never_signed_in_says_how_to(isolated_home) -> None:
    usage = provider_of(Colab, "colab_a").usage()
    assert usage.remaining is None
    assert "letify login colab colab_a" in (usage.note or "")


def test_elice_reads_the_remaining_credit_from_the_billing_api(fake_google) -> None:
    provider = provider_of(
        Elice,
        "e",
        endpoint=fake_google.endpoint,
        billing_endpoint=f"{fake_google.endpoint}/billing",
        organization="lab",
        access_token="token-1",
    )
    fake_google.answer(
        "GET", "/billing/stats", FakeResponse(200, {"total_credit_remaining_amount": "12345 KRW"})
    )
    usage = provider.usage()
    assert usage.unit == "KRW"
    assert usage.remaining == 12345.0
    stats = next(r for r in fake_google.requests if r["path"] == "/billing/stats")
    assert stats["authorization"] == "Bearer token-1"
    assert stats["org"] == "lab"


def test_elice_without_a_billing_endpoint_names_the_missing_field(elice, fake_elice) -> None:
    fake_elice.answer("GET", ALLOCATION_PATH, FakeResponse(200, {"items": []}))
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(200, {"items": []}))
    usage = elice.usage()
    assert usage.remaining is None
    assert "billing_endpoint" in (usage.note or "")


def test_a_machine_reached_by_ssh_has_no_quota() -> None:
    usage = provider_of(Shell, "lab", address="gpu.example.edu").usage()
    assert usage.unmetered is True
    assert "no quota" in usage.source
    assert provider_of(Tunnel, "t", address="10.0.0.1").usage().unmetered is True


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


def test_elice_prices_what_is_running_now_from_the_zones_own_price_list(elice, fake_elice) -> None:
    # Elice publishes no balance, so what it can answer honestly is the burn rate of the
    # allocations that exist and what they have cost so far.
    fake_elice.answer(
        "GET",
        ALLOCATION_PATH,
        FakeResponse(200, {"items": [{"id": "alloc-1", "instance_type_id": "it-1"}]}),
    )
    fake_elice.answer(
        "GET",
        PRICING_PATH,
        FakeResponse(200, {"items": [{"instance_type_id": "it-1", "price_per_hour": 975.0}]}),
    )
    usage = elice.usage()
    assert usage.unit == "KRW"
    assert usage.remaining is None
    assert usage.rate_per_hour == 975.0
    assert "price list" in usage.source


def test_elice_reports_a_zero_rate_when_nothing_is_allocated(elice, fake_elice) -> None:
    # Nothing powered on costs nothing per hour, which is a number rather than a gap.
    fake_elice.answer("GET", ALLOCATION_PATH, FakeResponse(200, {"items": []}))
    fake_elice.answer("GET", PRICING_PATH, FakeResponse(200, {"items": []}))
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


# -- Spec: Workspace root ------------------------------------------------------


def test_a_modal_sandbox_mounts_the_workspace_volume_at_the_workspace_root(
    isolated_home, fake_modal
) -> None:
    provider = provider_of(Modal, "m", app="study")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    try:
        [created] = fake_modal.requests("create")
        assert created["volumes"] == {"/letify": "study-workspace"}
    finally:
        provider.stop(runtime)


def test_a_modal_workspace_moves_the_volume_mount(isolated_home, fake_modal) -> None:
    provider = provider_of(Modal, "m", workspace="/data/letify")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    try:
        [created] = fake_modal.requests("create")
        assert created["volumes"] == {"/data/letify": "letify-workspace"}
    finally:
        provider.stop(runtime)


def test_a_colab_directory_is_packed_under_the_workspace_tmp(
    isolated_home, fake_jupyter, small_parts, tmp_path, monkeypatch
) -> None:
    import base64 as b64
    import io
    import tarfile

    root = tmp_path / "ws"
    monkeypatch.setattr(Colab, "workspace_root", property(lambda self: str(root)))
    site = tmp_path / "site"
    site.mkdir()
    (site / "a.txt").write_text("a", encoding="utf-8")
    channel = colab_exec_channel(fake_jupyter)
    seen: list[str] = []
    original = channel.files.get_file

    def record(path: str):
        seen.append(path)
        return original(path)

    channel.files.get_file = record

    value, _ = channel.request({"op": "pack_dir", "path": str(site)})

    assert Path(seen[0]).parent == root / "tmp"
    assert not Path(seen[0]).exists()
    with tarfile.open(fileobj=io.BytesIO(b64.b64decode(value["payload"])), mode="r:gz") as archive:
        assert "site/a.txt" in archive.getnames()


def test_a_check_reports_that_the_workspace_is_writable(patch_run) -> None:
    recorder = patch_run(
        shell_module,
        result=FakeCompleted(stdout="Linux gpu 6.8.0\nNVIDIA L4\nletify-workspace-ok\n"),
    )
    provider = provider_of(Shell, "lab", address="gpu.example.edu", workspace="/workspace/me")
    report = provider.check()
    assert "mkdir -p /workspace/me" in recorder.command[-1]
    assert "workspace /workspace/me: writable" in report
    assert "letify-workspace-ok" not in report


def test_a_check_reports_a_workspace_that_cannot_be_written(patch_run) -> None:
    patch_run(
        shell_module,
        result=FakeCompleted(
            stdout="Linux gpu\nletify-workspace-failed: mkdir: cannot create directory "
            "'/srv/x': Permission denied\n"
        ),
    )
    provider = provider_of(Shell, "lab", address="gpu.example.edu", workspace="/srv/x")
    report = provider.check()
    assert "workspace /srv/x: not writable: mkdir: cannot create directory" in report


# -- Spec: Provider model, Inventory: busy cards on a remote machine ----------------

REMOTE_UUIDS = "0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n3, GPU-ddd\n"


def owned_apps(apps: str, owners: dict[int, str] | None = None, login: str = "work") -> str:
    """The output of the owner script: the listing, each visible owner, then the login user.

    A pid missing from ``owners`` is owned by ``someone``, another user. A pid mapped to
    ``None`` is not visible in the namespace, so the script prints no owner for it.
    """
    table = dict(owners or {})
    lines = [line for line in apps.splitlines() if line.strip()]
    out = [*lines, "#owners"]
    for line in lines:
        pid = int(line.split(",")[1])
        owner = table.get(pid, "someone")
        if owner is not None:
            out.append(f"{pid} {owner}")
    out += ["#login", login]
    return "\n".join(out) + "\n"


def remote_smi(
    apps: str,
    *,
    uuids: str = REMOTE_UUIDS,
    returncode: int = 0,
    owners: dict[int, str] | None = None,
    login: str = "work",
    owner_returncode: int = 0,
):
    """Answer the two commands a busy check sends over SSH."""

    def answer(command: list[str]) -> FakeCompleted:
        remote = command[-1]
        if "--query-gpu=index,uuid" in remote:
            return FakeCompleted(returncode=returncode, stdout=uuids, stderr="nvidia-smi failed")
        if "--query-compute-apps=gpu_uuid,pid" in remote:
            code = returncode or owner_returncode
            return FakeCompleted(
                returncode=code,
                stdout="" if code else owned_apps(apps, owners, login),
                stderr="owner query failed",
            )
        return FakeCompleted(returncode=1, stderr=f"unexpected command {remote}")

    return answer


def indexed_shell(**table):
    return provider_of(Shell, "lab", address="gpu.example.edu", devices=dict(table))


def test_a_remote_card_with_another_users_process_is_skipped(patch_run) -> None:
    recorder = patch_run(shell_module, result=remote_smi("GPU-aaa, 4100\nGPU-bbb, 4200\n"))
    provider = indexed_shell(P100={"indices": "0-3"})
    assert provider.busy() == (0, 1)
    assert provider.reserve(provider.P100) == (2,)
    remote = [command[-1] for command in recorder.commands]
    assert any("--query-compute-apps=gpu_uuid,pid" in command for command in remote)
    assert all(command[0] == "ssh" for command in recorder.commands)


def test_a_remote_busy_reading_is_taken_again_at_every_reservation(patch_run) -> None:
    readings = iter(["", "GPU-aaa, 4100\n"])

    def answer(command: list[str]) -> FakeCompleted:
        if "--query-compute-apps" in command[-1]:
            return FakeCompleted(stdout=owned_apps(next(readings)))
        return FakeCompleted(stdout=REMOTE_UUIDS)

    patch_run(shell_module, result=answer)
    provider = indexed_shell(P100={"indices": "0-1"})
    assert provider.free("P100") == (0, 1)
    assert provider.free("P100") == (1,)


def test_a_remote_worker_of_this_client_does_not_make_its_card_busy(patch_run) -> None:
    patch_run(shell_module, result=remote_smi("GPU-aaa, 777\nGPU-bbb, 4200\n"))
    provider = indexed_shell(P100={"indices": "0-3"})
    provider.add_worker_pid(777)
    assert provider.busy() == (1,)
    provider.remove_worker_pid(777)
    assert provider.busy() == (0, 1)


def test_a_remote_busy_check_that_cannot_run_is_refused_rather_than_read_as_free(
    patch_run,
) -> None:
    patch_run(shell_module, result=remote_smi("", returncode=255))
    provider = indexed_shell(P100={"indices": "0-3"})
    with pytest.raises(letify.RuntimeFailure, match="busy check"):
        provider.reserve(provider.P100)


def test_a_remote_machine_whose_registered_cards_are_all_busy_reserves_nothing(
    patch_run,
) -> None:
    apps = "GPU-aaa, 1\nGPU-bbb, 2\nGPU-ccc, 3\nGPU-ddd, 4\n"
    patch_run(shell_module, result=remote_smi(apps))
    provider = indexed_shell(P100={"indices": "0-3"})
    assert provider.reserve(provider.P100) is None
    assert provider.last_busy == (0, 1, 2, 3)


def test_a_card_running_only_the_login_users_own_processes_is_shared(patch_run) -> None:
    apps = "GPU-aaa, 10\nGPU-aaa, 11\nGPU-bbb, 20\n"
    patch_run(shell_module, result=remote_smi(apps, owners={10: "work", 11: "work"}))
    provider = indexed_shell(P100={"indices": "0-1"})
    assert provider.busy() == (1,)
    assert provider.reserve(provider.P100) == (0,)


def test_a_card_with_one_process_of_another_user_is_busy(patch_run) -> None:
    apps = "GPU-aaa, 10\nGPU-aaa, 11\n"
    patch_run(shell_module, result=remote_smi(apps, owners={10: "work", 11: "alice"}))
    provider = indexed_shell(P100={"indices": "0-1"})
    assert provider.busy() == (0,)
    assert provider.last_busy_owners == {0: ("alice",)}


def test_a_process_whose_owner_is_not_visible_makes_its_card_busy(patch_run) -> None:
    patch_run(shell_module, result=remote_smi("GPU-bbb, 30\n", owners={30: None}))
    provider = indexed_shell(P100={"indices": "0-1"})
    assert provider.busy() == (1,)
    assert provider.last_busy_owners == {1: ("unknown",)}


def test_a_root_login_still_reads_root_owned_foreign_processes_as_busy(patch_run) -> None:
    apps = "GPU-aaa, 10\nGPU-bbb, 777\n"
    patch_run(shell_module, result=remote_smi(apps, owners={10: "root", 777: "root"}, login="root"))
    provider = indexed_shell(P100={"indices": "0-1"})
    provider.add_worker_pid(777)
    assert provider.busy() == (0,)
    assert provider.last_busy_owners == {0: ("root",)}


def test_an_owner_query_that_cannot_run_is_refused_rather_than_read_as_free(patch_run) -> None:
    patch_run(shell_module, result=remote_smi("GPU-aaa, 10\n", owner_returncode=127))
    provider = indexed_shell(P100={"indices": "0-1"})
    with pytest.raises(letify.RuntimeFailure, match="busy check"):
        provider.busy()


def test_owners_are_read_in_one_remote_command_with_the_listing(patch_run) -> None:
    recorder = patch_run(shell_module, result=remote_smi("GPU-aaa, 10\nGPU-bbb, 20\n"))
    indexed_shell(P100={"indices": "0-1"}).busy()
    remote = [command[-1] for command in recorder.commands]
    assert len(remote) == 2
    script = next(command for command in remote if "--query-compute-apps" in command)
    assert "stat -c %U" in script and "id -un" in script


def test_local_shares_a_card_running_only_the_current_users_processes(monkeypatch) -> None:
    import getpass

    from letify.runtime import telemetry

    me = getpass.getuser()
    asked: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> str:
        asked.append(command)
        if command == telemetry.UUID_COMMAND:
            return REMOTE_UUIDS
        return owned_apps("GPU-aaa, 10\nGPU-bbb, 20\n", {10: me}, login=me)

    monkeypatch.setattr(telemetry, "_run", run)
    provider = provider_of(Local, "box", devices={"P100": {"indices": "0-1"}})
    assert provider.busy() == (1,)
    assert asked[-1] == telemetry.OWNERS_COMMAND


@pytest.mark.parametrize("cls", [Tunnel, Elice])
def test_every_shell_kind_reads_busy_cards_on_its_machine(cls, patch_run, monkeypatch) -> None:
    patch_run(shell_module, result=remote_smi("GPU-ccc, 9\n"))
    provider = provider_of(cls, "lab", address="gpu.example.edu")
    link = type("L", (), {"ssh_command": lambda self, remote=None: ["ssh", "h", remote]})()
    monkeypatch.setattr(provider, "link", lambda runtime=None: link)
    assert provider.busy() == (2,)


# -- Modal: ending a sandbox while a request is blocked (spec "modal-abort") ----------


def test_a_modal_call_past_its_timeout_leaves_no_sandbox_running(isolated_home, fake_modal) -> None:
    import time

    from letify.errors import ProtocolError, RuntimeFailure

    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    channel.start()
    [pid] = fake_modal.sandbox_pids()
    started = time.monotonic()
    with pytest.raises((RuntimeFailure, ProtocolError)):
        channel.call(time.sleep, (600,), {}, timeout=2)
    assert time.monotonic() - started < 30
    assert not fake_modal.alive(pid)
    channel.close()
    provider.stop(runtime)


def test_a_modal_call_interrupted_while_blocked_leaves_no_sandbox_running(
    isolated_home, fake_modal
) -> None:
    import signal
    import time

    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    channel.start()
    [pid] = fake_modal.sandbox_pids()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        with pytest.raises(KeyboardInterrupt):
            channel.call(time.sleep, (600,), {})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    started = time.monotonic()
    channel.close()
    provider.stop(runtime)
    assert time.monotonic() - started < 30
    assert not fake_modal.alive(pid)


def test_a_request_on_an_adapter_out_of_step_fails_without_waiting(
    isolated_home, fake_modal
) -> None:
    import signal
    import time

    from letify.errors import RuntimeFailure

    # Spec "modal-abort": an adapter falls out of step when a request on its standard input
    # and output is interrupted, so the call has to travel there rather than over the data
    # channel, where an interrupted call leaves the adapter idle.
    provider = provider_of(Modal, "m", data_channel=False)
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    channel.start()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        with pytest.raises(KeyboardInterrupt):
            channel.call(time.sleep, (600,), {})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    started = time.monotonic()
    with pytest.raises(RuntimeFailure, match="out of step"):
        provider.adapter().request("hello")
    assert time.monotonic() - started < 5
    provider.stop(runtime)


def test_a_modal_sandbox_is_created_with_a_lifetime_and_an_idle_limit(
    isolated_home, fake_modal
) -> None:
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    try:
        [created] = fake_modal.requests("create")
        assert created["timeout"] == 3600
        assert created["idle_timeout"] == 600
    finally:
        provider.stop(runtime)


# -- Modal data channel ----------------------------------------------------------


def test_the_real_modal_adapter_exposes_encrypted_ports_and_resolves_their_tunnel(
    tmp_path: Path,
) -> None:
    # Spec "Modal adapter", ops create and tunnel.
    import json
    import os
    import sys

    site = tmp_path / "site" / "modal"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text(STUB_MODAL, encoding="utf-8")
    log = tmp_path / "modal.log"
    script = Path(modal_module.__file__).with_name("modal_adapter.py")
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site"), "STUB_MODAL_LOG": str(log)}
    adapter = Adapter([sys.executable, "-P", str(script)], env=env, name="m")
    try:
        fields = {"app": "study", "args": ["python3"], "packages": [], "gpu": None, "timeout": 60}
        sandbox = adapter.request("create", ports=[8765], **fields)["sandbox"]
        tunnel = adapter.request("tunnel", sandbox=sandbox, port=8765)
    finally:
        adapter.close()
    assert tunnel == {"host": "r7.modal.host", "port": 443, "tls": True}
    events = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    assert ["create", "study", [8765]] in events


def test_a_modal_sandbox_is_created_with_the_data_port_exposed(isolated_home, fake_modal) -> None:
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    try:
        [created] = fake_modal.requests("create")
        assert created["ports"] == [modal_module.DATA_PORT]
    finally:
        provider.stop(runtime)


def test_a_modal_sandbox_without_the_data_channel_exposes_no_port(
    isolated_home, fake_modal
) -> None:
    provider = provider_of(Modal, "m", data_channel=False)
    runtime = modal_runtime(provider)
    provider.open_channel(runtime)
    try:
        [created] = fake_modal.requests("create")
        assert created["ports"] == []
    finally:
        provider.stop(runtime)


def test_a_sandbox_channel_carries_frames_over_the_data_connection(
    isolated_home, fake_modal
) -> None:
    # Spec "Modal data channel": after hello, frames leave standard input and output, so a
    # result far above what Modal's stdout carried arrives with no further stdio request.
    import os

    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        channel.start()
        assert [r["port"] for r in fake_modal.requests("tunnel")] == [modal_module.DATA_PORT]
        reads = len(fake_modal.requests("read_until"))
        writes = len(fake_modal.requests("write"))
        payload = os.urandom(8 << 20)
        assert channel.call(len, (payload,), {})[0] == len(payload)
        assert len(channel.call(os.urandom, (32 << 20,), {})[0]) == 32 << 20
        assert len(fake_modal.requests("read_until")) == reads
        assert len(fake_modal.requests("write")) == writes
    finally:
        channel.close()
        provider.stop(runtime)


def test_worker_output_arrives_over_the_data_connection(isolated_home, fake_modal) -> None:
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    seen: list[bytes] = []
    channel.on_output = lambda stream, data: seen.append(data)
    try:
        channel.start()
        reads = len(fake_modal.requests("read_until"))
        channel.request({"op": "exec", "source": "print('hello from the sandbox', flush=True)"})
        assert b"hello from the sandbox" in b"".join(seen)
        assert len(fake_modal.requests("read_until")) == reads
    finally:
        channel.close()
        provider.stop(runtime)


def test_a_sandbox_whose_tunnel_fails_keeps_frames_on_standard_io(
    isolated_home, fake_modal, capfd
) -> None:
    # Spec "Modal data channel": a failure is reported in one line and stdio carries on,
    # without waiting for the worker's listen window to end.
    import time

    fake_modal.fail("tunnel")
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        started = time.monotonic()
        channel.start()
        assert channel.call(sum, ([1, 2, 3],), {})[0] == 6
        assert time.monotonic() - started < 30
        err = capfd.readouterr().err
        assert "the data channel did not open" in err
        assert "frames stay on standard input and output" in err
    finally:
        channel.close()
        provider.stop(runtime)


def test_switching_the_interpreter_reopens_the_data_channel(isolated_home, fake_modal) -> None:
    import sys

    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        channel.start()
        channel.switch_interpreter(sys.executable)
        assert channel.call(len, (b"x" * (4 << 20),), {})[0] == 4 << 20
        assert len(fake_modal.requests("tunnel")) == 2
    finally:
        channel.close()
        provider.stop(runtime)


def _listening_worker():
    """A local worker whose frame reader was asked to listen, its port, and its token."""
    import socket
    import sys

    from letify.protocol.worker import BOOTSTRAP
    from letify.runtime.channel import PersistentChannel

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    channel = PersistentChannel([sys.executable, "-u", "-c", BOOTSTRAP], name="w")
    channel.start()
    token = "ab" * 32
    channel.request({"op": "listen", "port": port, "token": token, "wait": 20})
    return channel, port, token


def _connect(port: int):
    import socket
    import time

    for _ in range(100):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=10)
        except OSError:
            time.sleep(0.05)
    raise AssertionError("the worker never listened")


def test_a_data_connection_with_the_wrong_token_is_closed_and_the_right_one_gets_hello() -> None:
    # Spec "Modal data channel", step 4, through the real worker.
    from letify.protocol import wire

    channel, port, token = _listening_worker()
    try:
        intruder = _connect(port)
        intruder.sendall(b"LETIFY-DATA " + b"cd" * 32 + b"\n")
        assert intruder.recv(16) == b""
        intruder.close()
        good = _connect(port)
        good.sendall(b"LETIFY-DATA " + token.encode() + b"\n")
        receiver = wire.Receiver(good.recv_into)
        kind, _stream, _value = receiver.next_event()
        assert kind == wire.HELLO
        good.close()
    finally:
        channel.close()


class _WatchedTls:
    """An identity cipher over two memory buffers that records two threads inside it at once."""

    def __init__(self, incoming, outgoing):
        import threading

        self.incoming = incoming
        self.outgoing = outgoing
        self.overlapped = False
        self._inside = 0
        self._count = threading.Lock()

    def _enter(self) -> None:
        import time

        with self._count:
            self._inside += 1
            if self._inside > 1:
                self.overlapped = True
        time.sleep(0.0005)

    def _leave(self) -> None:
        with self._count:
            self._inside -= 1

    def do_handshake(self) -> None:
        return None

    def write(self, data) -> int:
        self._enter()
        try:
            self.outgoing.write(bytes(data))
            return memoryview(data).nbytes
        finally:
            self._leave()

    def read(self, size, buffer) -> int:
        import ssl

        self._enter()
        try:
            data = self.incoming.read(size)
            if not data:
                raise ssl.SSLWantReadError("no data")
            buffer[: len(data)] = data
            return len(data)
        finally:
            self._leave()


def test_the_tls_state_is_never_used_by_two_threads_at_once() -> None:
    # Spec "Modal data channel": a writing thread and the reading thread share one TLS
    # object, which OpenSSL does not allow, so the channel serializes every use of it.
    import os
    import socket
    import ssl
    import threading

    from letify.providers.modal import TlsStream

    near, far = socket.socketpair()

    def echo() -> None:
        while True:
            data = far.recv(1 << 16)
            if not data:
                break
            far.sendall(data)
        far.close()

    threading.Thread(target=echo, daemon=True).start()
    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    tls = _WatchedTls(incoming, outgoing)
    stream = TlsStream(near, tls, incoming, outgoing)
    payload = os.urandom(4 << 20)
    received = bytearray()

    def read() -> None:
        buffer = bytearray(1 << 20)
        while len(received) < len(payload):
            count = stream.recv_into(memoryview(buffer))
            if not count:
                break
            received.extend(buffer[:count])

    reader = threading.Thread(target=read)
    reader.start()
    view = memoryview(payload)
    while view:
        view = view[stream.send(view[: 256 << 10]) :]
    reader.join(60)
    near.close()
    assert bytes(received) == payload
    assert tls.overlapped is False


def test_a_tls_data_connection_carries_overlapping_requests_during_large_transfers(
    isolated_home, fake_modal, tls_proxy
) -> None:
    # Spec "Modal data channel": frames through a TLS-terminating proxy, as Modal's encrypted
    # port is, while a second thread keeps sending requests during every transfer.
    import os
    import threading

    fake_modal.tunnel_to("localhost", tls_proxy.start(modal_module.DATA_PORT), tls=True)
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    channel.tls_context = tls_proxy.context
    stop = threading.Event()
    replies: list[object] = []
    errors: list[BaseException] = []

    def overlap() -> None:
        while not stop.is_set():
            try:
                replies.append(channel.request({"op": "stat"}, timeout=60)[0])
            except BaseException as exc:
                errors.append(exc)
                return

    try:
        channel.start()
        reads = len(fake_modal.requests("read_until"))
        thread = threading.Thread(target=overlap, daemon=True)
        thread.start()
        for _ in range(3):
            assert channel.call(len, (os.urandom(16 << 20),), {}, timeout=60)[0] == 16 << 20
            assert len(channel.call(os.urandom, (16 << 20,), {}, timeout=60)[0]) == 16 << 20
        stop.set()
        thread.join(60)
        assert errors == []
        assert replies
        assert len(fake_modal.requests("read_until")) == reads
    finally:
        stop.set()
        channel.close()
        provider.stop(runtime)


# -- parallel data streams: spec "Parallel data streams" ----------------------------------


def _count_connections(monkeypatch) -> list[object]:
    import socket

    opened: list[object] = []
    real = socket.create_connection

    def counting(address, *args, **kwargs):
        opened.append(address)
        return real(address, *args, **kwargs)

    monkeypatch.setattr(modal_module.socket, "create_connection", counting)
    return opened


def test_a_modal_channel_opens_one_connection_per_data_stream(
    isolated_home, fake_modal, monkeypatch
) -> None:
    import os

    opened = _count_connections(monkeypatch)
    provider = provider_of(Modal, "m", data_streams=3)
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        channel.start()
        assert len(opened) == 3
        payload = os.urandom(24 << 20)
        assert channel.call(len, (payload,), {})[0] == len(payload)
        assert channel.call(bytes, (payload,), {})[0] == payload
    finally:
        channel.close()
        provider.stop(runtime)


def test_the_data_streams_default_is_four(isolated_home, fake_modal, monkeypatch) -> None:
    opened = _count_connections(monkeypatch)
    provider = provider_of(Modal, "m")
    runtime = modal_runtime(provider)
    channel = provider.open_channel(runtime)
    try:
        channel.start()
        assert len(opened) == 4
    finally:
        channel.close()
        provider.stop(runtime)


@pytest.mark.parametrize("value", [0, 17, "4", True])
def test_a_data_streams_value_out_of_range_is_refused(isolated_home, fake_modal, value) -> None:
    provider = provider_of(Modal, "m", data_streams=value)
    runtime = modal_runtime(provider)
    with pytest.raises(letify.ConfigError, match="data_streams"):
        provider.open_channel(runtime)


def test_a_lane_whose_index_is_taken_is_closed_and_hello_waits_for_every_lane() -> None:
    # Spec "Parallel data streams", step 1, through the real worker.
    import socket
    import sys

    from letify.protocol import wire
    from letify.protocol.worker import BOOTSTRAP
    from letify.runtime.channel import PersistentChannel

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    channel = PersistentChannel([sys.executable, "-u", "-c", BOOTSTRAP], name="w")
    channel.start()
    token = "ab" * 32
    channel.request({"op": "listen", "port": port, "token": token, "wait": 20, "streams": 2})
    try:
        lane0 = _connect(port)
        lane0.sendall(b"LETIFY-DATA " + token.encode() + b"\n")
        again = _connect(port)
        again.sendall(b"LETIFY-DATA " + token.encode() + b"\n")
        assert again.recv(16) == b""
        again.close()
        lane1 = _connect(port)
        lane1.sendall(b"LETIFY-DATA " + token.encode() + b" 1\n")
        stream = wire.Striped([lane0.send, lane1.send], [lane0.recv_into, lane1.recv_into])
        kind, _stream, _value = wire.Receiver(stream.recv_into).next_event()
        assert kind == wire.HELLO
        lane0.close()
        lane1.close()
    finally:
        channel.close()


def test_every_provider_start_accepts_the_keywords_the_pool_passes() -> None:
    # Spec "Provider model": the pool starts a runtime with name, volumes and held, so an
    # override that drops one fails only when a live session starts.
    import inspect

    from letify.providers import base, colab, elice, local, modal, shell, tunnel

    wanted = {"name", "volumes", "held"}
    classes = {base.Provider}
    for module in (colab, elice, local, modal, shell, tunnel):
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, base.Provider):
                classes.add(value)
    for cls in classes:
        parameters = inspect.signature(cls.start).parameters
        missing = wanted - parameters.keys()
        assert not missing, f"{cls.__name__}.start lacks {sorted(missing)}"
