"""CUDA call forwarding: the arithmetic, the capability probe and the loader.

Spec sections pinned here: "Efficiency model", "Execution modes", "letify-core" and
"Loading". The interception itself is Rust under letify-core/ and is not covered by this
suite; what is covered is the Python part that finds it, measures the link and refuses
clearly when forwarding cannot run.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest
from conftest import FakeCompleted

import letify
from letify.remoting import loader
from letify.remoting.capability import LATENCY_BUDGET_MS, Capability
from letify.remoting.loader import Injection, core_directory, inject, preload_command
from letify.remoting.probe import CORE_NAMES, core_path, efficiency, ping, probe, require

# letify.remoting re-exports a function named probe, which shadows the submodule of the
# same name, so the module object is taken from the import system directly.
probe_module = import_module("letify.remoting.probe")

WINDOWS_PING = """
Pinging gpu.lab.example.edu [10.0.0.5] with 32 bytes of data:
Reply from 10.0.0.5: bytes=32 time=148ms TTL=52
Reply from 10.0.0.5: bytes=32 time=151ms TTL=52
Reply from 10.0.0.5: bytes=32 time=150ms TTL=52

Ping statistics for 10.0.0.5:
    Minimum = 148ms, Maximum = 151ms, Average = 149ms
"""

LINUX_PING = """
PING gpu.lab.example.edu (10.0.0.5) 56(84) bytes of data.
64 bytes from 10.0.0.5: icmp_seq=1 ttl=52 time=148 ms
64 bytes from 10.0.0.5: icmp_seq=2 ttl=52 time=151 ms
rtt min/avg/max/mdev = 148.0/149.6/151.2/1.3 ms
"""


@pytest.fixture
def built_core(tmp_path: Path, monkeypatch) -> Path:
    """Stand in for a built letify-core library, which is a Rust build product."""
    library = tmp_path / "lib" / CORE_NAMES.get(sys.platform, "libletify_driver.so")
    library.parent.mkdir(parents=True)
    library.write_bytes(b"not really a shared library")
    monkeypatch.setenv("LETIFY_CORE_PATH", str(library))
    return library


# -- Spec: Efficiency model ----------------------------------------------------


def test_the_efficiency_formula_matches_the_documented_numbers() -> None:
    # A 0.5 s NVFP4 micro step with three host synchronizations at a 150 ms round trip.
    assert round(efficiency(0.5, 3, 150.0), 2) == 0.53
    # The same work once synchronization is down to one per optimizer step of eight.
    assert round(efficiency(4.0, 1, 150.0), 2) == 0.96


def test_a_faster_gpu_makes_forwarding_worse() -> None:
    # T shrinks while the round trip does not, so the same step on a slower card keeps
    # more of its throughput.
    l4_in_bf16 = efficiency(1.8, 3, 150.0)
    assert round(l4_in_bf16, 2) == 0.80
    assert l4_in_bf16 > efficiency(0.5, 3, 150.0)


def test_decoding_token_by_token_fails_at_any_useful_latency() -> None:
    # A decode step is 2 ms to 3 ms and synchronizes once or twice per token, so the round
    # trip sets the ceiling whatever the card is.
    assert efficiency(0.0025, 2, 150.0) < 0.01
    assert efficiency(0.0025, 2, 10.0) < 0.15


def test_a_link_with_no_latency_costs_nothing() -> None:
    assert efficiency(0.5, 3, 0.0) == 1.0


# -- Spec: letify-core, the capability probe ------------------------------------


def test_forwarding_needs_both_the_local_shim_and_the_remote_agent() -> None:
    assert Capability(core=True, agent=True, round_trip_ms=5.0, platform="linux").usable
    assert not Capability(core=False, agent=True, round_trip_ms=5.0, platform="linux").usable
    assert not Capability(core=True, agent=False, round_trip_ms=5.0, platform="linux").usable


def test_a_ready_capability_says_so() -> None:
    assert Capability(core=True, agent=True, round_trip_ms=5.0, platform="linux").explain() == (
        "ready"
    )


def test_a_missing_shim_says_there_is_nothing_to_intercept_the_driver_with() -> None:
    explanation = Capability(core=False, agent=True, round_trip_ms=5.0, platform="linux").explain()
    assert "letify-core is not installed" in explanation


def test_a_missing_agent_names_the_remote_machine() -> None:
    explanation = Capability(core=True, agent=False, round_trip_ms=5.0, platform="linux").explain()
    assert "agent is not installed on the remote machine" in explanation


def test_an_unmeasured_link_says_so_rather_than_guessing() -> None:
    capability = Capability(core=True, agent=True, round_trip_ms=None, platform="linux")
    assert "has not been measured" in capability.explain()
    assert capability.costly is False


def test_a_long_round_trip_is_reported_as_costly_without_being_refused() -> None:
    # Speed is not a reason to refuse, so this only decides whether letify says something.
    capability = Capability(core=True, agent=True, round_trip_ms=150.0, platform="linux")
    assert capability.costly is True
    assert capability.usable is True
    assert "each host synchronization pays it" in capability.explain()
    assert Capability(True, True, LATENCY_BUDGET_MS, "linux").costly is False


def test_a_probe_of_this_machine_reports_what_is_present() -> None:
    capability = probe()
    assert capability.platform == sys.platform
    # With no host named there is nothing to reach, so only the shim decides.
    assert capability.agent is True
    assert capability.round_trip_ms is None
    assert isinstance(capability.usable, bool)
    assert capability.explain()


def test_forwarding_is_refused_when_there_is_nothing_to_run(monkeypatch) -> None:
    # The refusal names the way out: a platform wheel, or ship the function instead.
    monkeypatch.delenv("LETIFY_CORE_PATH", raising=False)
    monkeypatch.setattr(probe_module, "core_path", lambda: None)
    with pytest.raises(letify.UnsupportedMode) as caught:
        require()
    message = str(caught.value)
    assert "letify-core" in message
    assert "host='remote'" in message


def test_forwarding_is_allowed_once_the_shim_is_there(built_core) -> None:
    capability = require()
    assert capability.usable is True
    assert capability.core is True


# -- Spec: letify-core, where the library lives --------------------------------


def test_the_library_takes_the_name_of_the_driver_it_replaces() -> None:
    # Being found before the real driver is the whole mechanism.
    assert CORE_NAMES["win32"] == "nvcuda.dll"
    assert CORE_NAMES["linux"] == "libcuda.so.1"


def test_a_built_library_is_found_through_the_environment(built_core) -> None:
    assert core_path() == built_core
    assert core_directory() == built_core.parent


def test_an_environment_override_pointing_nowhere_is_ignored(monkeypatch, tmp_path) -> None:
    override = tmp_path / "never-built.so"
    monkeypatch.setenv("LETIFY_CORE_PATH", str(override))
    # Falls through to the packaged location instead of trusting the override.
    assert core_path() != override


def test_a_library_installed_beside_the_package_is_found(monkeypatch) -> None:
    monkeypatch.delenv("LETIFY_CORE_PATH", raising=False)
    name = CORE_NAMES.get(sys.platform, "libletify_driver.so")
    library = Path(probe_module.__file__).resolve().parent / "lib" / name
    # letify-core may or may not be built in this checkout, and both are honest states.
    placed = not library.exists()
    if placed:
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(b"not really a shared library")
    try:
        assert core_path() == library
        assert core_directory() == library.parent
    finally:
        if placed:
            library.unlink()


def test_the_macos_library_takes_the_name_build_py_installs() -> None:
    assert CORE_NAMES["darwin"] == "libletify_driver.dylib"


def test_an_agent_bundled_in_the_wheel_is_found_before_path(
    monkeypatch, tmp_path, patch_which
) -> None:
    agent = tmp_path / probe_module.AGENT_NAME
    agent.write_bytes(b"not really an executable")
    monkeypatch.setattr(probe_module, "LIB_DIR", tmp_path)
    patch_which(probe_module, present=False)
    assert probe_module.agent_path() == agent
    assert probe(remote=True).agent is True


def test_an_agent_on_path_is_used_when_none_is_bundled(monkeypatch, tmp_path, patch_which) -> None:
    monkeypatch.setattr(probe_module, "LIB_DIR", tmp_path)
    patch_which(probe_module, present=["letify-agent"])
    assert probe_module.agent_path() == Path("/usr/bin/letify-agent")


# -- Spec: Efficiency model, measuring the round trip --------------------------


def test_the_round_trip_is_the_fastest_reply_the_machine_gave(patch_run) -> None:
    patch_run(probe_module, result=FakeCompleted(stdout=WINDOWS_PING))
    assert ping("gpu.lab.example.edu") == 148.0


def test_the_round_trip_is_read_from_either_ping_dialect(patch_run) -> None:
    patch_run(probe_module, result=FakeCompleted(stdout=LINUX_PING))
    assert ping("gpu.lab.example.edu") == 148.0


def test_a_host_that_does_not_answer_has_no_round_trip(patch_run) -> None:
    patch_run(probe_module, result=FakeCompleted(returncode=1, stdout="Request timed out.\n"))
    assert ping("gpu.lab.example.edu") is None


def test_a_ping_that_cannot_be_run_has_no_round_trip(patch_run) -> None:
    patch_run(probe_module, error=OSError("ping is not installed"))
    assert ping("gpu.lab.example.edu") is None


def test_a_reply_with_no_numbers_in_it_has_no_round_trip(patch_run) -> None:
    patch_run(probe_module, result=FakeCompleted(stdout="Destination host unreachable.\n"))
    assert ping("gpu.lab.example.edu") is None


def test_naming_a_host_makes_the_probe_measure_it(patch_run, patch_which) -> None:
    patch_which(probe_module, present=["letify-agent"])
    patch_run(probe_module, result=FakeCompleted(stdout=LINUX_PING))
    capability = probe("gpu.lab.example.edu")
    # A named host is measured rather than assumed. What the measurement reads out of the
    # ping output is pinned separately above.
    assert capability.round_trip_ms is not None
    assert capability.agent is True


def test_a_host_with_no_agent_on_it_cannot_serve_forwarding(patch_run, patch_which) -> None:
    patch_which(probe_module, present=False)
    patch_run(probe_module, result=FakeCompleted(stdout=LINUX_PING))
    assert probe("gpu.lab.example.edu").agent is False


# -- Spec: Loading -------------------------------------------------------------


def test_nothing_can_be_injected_before_letify_core_is_built(monkeypatch) -> None:
    monkeypatch.delenv("LETIFY_CORE_PATH", raising=False)
    monkeypatch.setattr(loader, "core_path", lambda: None)
    outcome = inject()
    assert bool(outcome) is False
    assert "platform" in outcome.instructions
    assert "host='remote'" in outcome.instructions
    assert repr(outcome) == "<Injection active=False>"


def test_the_agent_to_forward_to_is_recorded_for_the_shim(monkeypatch, built_core) -> None:
    monkeypatch.delenv("LETIFY_AGENT", raising=False)
    inject(agent="gpu.lab.example.edu:7654")
    import os

    assert os.environ["LETIFY_AGENT"] == "gpu.lab.example.edu:7654"


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows loader search order")
def test_on_windows_letify_puts_the_library_at_the_front_of_the_search_order(
    monkeypatch, built_core
) -> None:
    import os

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("PATH", "C:\\existing")
    outcome = inject()
    assert bool(outcome) is True
    # PATH as well, because some loaders consult it for dependent libraries.
    assert os.environ["PATH"].startswith(str(built_core.parent))


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows loader search order")
def test_injecting_after_torch_is_imported_says_it_is_too_late(monkeypatch, built_core) -> None:
    # The real driver may already be resolved, and a silently ineffective injection would
    # look like forwarding.
    monkeypatch.setitem(sys.modules, "torch", object())
    outcome = inject()
    assert bool(outcome) is False
    assert "before importing torch" in outcome.instructions


def test_on_linux_letify_reports_the_command_rather_than_pretending(
    monkeypatch, built_core
) -> None:
    # LD_PRELOAD cannot be set from inside a running process for libraries already
    # resolved, so this hands the caller the exact command.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    outcome = inject(agent="gpu:7654")
    assert bool(outcome) is False
    assert f"LD_PRELOAD={built_core}" in outcome.instructions
    assert "LETIFY_AGENT=gpu:7654" in outcome.instructions


def test_on_linux_an_already_preloaded_library_needs_no_action(monkeypatch, built_core) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LD_PRELOAD", str(built_core))
    assert bool(inject()) is True


def test_the_preload_command_is_reported_for_the_platform_it_runs_on(monkeypatch, built_core):
    monkeypatch.setattr(sys, "platform", "linux")
    command = preload_command("train.py", agent="gpu:7654")
    assert command.startswith(f"LD_PRELOAD={built_core}")
    assert command.endswith("python train.py")

    monkeypatch.setattr(sys, "platform", "win32")
    assert "inject()" in preload_command("train.py")


def test_the_preload_command_says_letify_core_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(loader, "core_path", lambda: None)
    assert "platform wheel" in preload_command()


def test_an_injection_that_did_nothing_is_false() -> None:
    assert not Injection(False, "do this instead")
    assert Injection(True)
