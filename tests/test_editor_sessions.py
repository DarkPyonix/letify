"""Provider session discovery and safe adapter queries, as specified in SPEC.md."""

from __future__ import annotations

import json

from conftest import FakeCompleted

from letify import tools
from letify.cli import main
from letify.providers import colab


def test_sessions_lists_colab_outside_the_current_process_pool(
    isolated_home, patch_which, patch_run, capsys
) -> None:
    (isolated_home / ".letify" / "config.toml").write_text('[lab]\nkind = "colab"\n')
    patch_which(tools, present=True)
    recorder = patch_run(
        colab, result=FakeCompleted(stdout="[live-cpu] id | Hardware: CPU | Variant: DEFAULT\n")
    )
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runtimes"] == []
    recorder.calls.clear()
    assert main(["sessions", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert next(row for row in rows if row["alias"] == "lab") == {
        "alias": "lab",
        "kind": "colab",
        "sessions": ["live-cpu"],
        "reason": None,
    }
    assert (
        next(row for row in rows if row["alias"] == "local")["reason"]
        == "session discovery is not supported"
    )
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["command"][-1] == "sessions"


def test_sessions_preserves_a_failed_provider(
    isolated_home, patch_which, patch_run, capsys
) -> None:
    (isolated_home / ".letify" / "config.toml").write_text('[lab]\nkind = "colab"\n')
    patch_which(tools, present=True)
    patch_run(colab, result=FakeCompleted(returncode=1, stderr="query failed"))
    assert main(["sessions", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    failed = next(row for row in rows if row["alias"] == "lab")
    assert failed["sessions"] == []
    assert "unavailable" in failed
    assert next(row for row in rows if row["alias"] == "local")["sessions"] == []


def test_read_only_adapter_commands_cannot_sync_the_project() -> None:
    for tool, path in [
        (tools.MODAL, tools.MODAL_ADAPTER),
        (tools.KAGGLE_KERNEL, tools.KAGGLE_ADAPTER),
    ]:
        command = tools.script_command(tool, "/usr/bin/uv", path)
        assert "--frozen" in command
        assert "--no-sync" in command
        assert "--no-project" in command
        assert command.index("--no-sync") < command.index("python")
