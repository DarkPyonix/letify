"""How every command prints, for a person and with ``--json``.

Pins spec "Command line": the output conventions (marks, tables, fields, errors, the log
prefix) and each command's layout. The renderer is tested directly at a fixed style, and
the commands through ``main(argv)``, whose captured streams are not terminals and are UTF-8.
"""

from __future__ import annotations

import io
import json

import pytest

from letify import render
from letify.cli import main

PLAIN = render.Style(width=80, color=False, unicode=True)
ASCII = render.Style(width=80, color=False, unicode=False)
COLOUR = render.Style(width=80, color=True, unicode=True)


# -- Spec: Command line, output conventions ------------------------------------


@pytest.mark.parametrize(
    ("kind", "utf8", "ascii", "code"),
    [("ok", "✓", "+", "\x1b[32m"), ("fail", "✗", "x", "\x1b[31m"), ("warn", "!", "!", "\x1b[33m")],
)
def test_a_mark_has_a_symbol_an_ascii_fallback_and_a_colour(
    kind: str, utf8: str, ascii: str, code: str
) -> None:
    assert render.mark(kind, PLAIN) == utf8
    assert render.mark(kind, ASCII) == ascii
    assert render.mark(kind, COLOUR) == f"{code}{utf8}\x1b[0m"


def test_a_table_aligns_columns_to_the_longest_cell() -> None:
    out = render.table(["ALIAS", "KIND"], [["lab", "shell"], ["dept_gpu", "local"]], PLAIN)
    assert out == "ALIAS     KIND\nlab       shell\ndept_gpu  local\n"


def test_a_table_header_is_bold_only_with_colour() -> None:
    assert render.table(["A"], [["x"]], COLOUR).startswith("\x1b[1mA\x1b[0m\n")


def test_fields_align_their_names() -> None:
    out = render.fields([("platform", "linux"), ("round trip", "1.0 ms")], PLAIN)
    assert out == "platform    linux\nround trip  1.0 ms\n"


def test_an_error_reaching_the_command_line_is_one_marked_line_on_stderr(
    isolated_home, capsys
) -> None:
    assert main(["check", "nosuch"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("✗ ")
    assert captured.err.count("\n") == 1


def test_a_log_line_keeps_its_content_and_only_dims_the_prefix_on_a_terminal(
    monkeypatch,
) -> None:
    from letify.transport import announce

    class Terminal(io.StringIO):
        encoding = "utf-8"

        def isatty(self) -> bool:
            return True

    plain = io.StringIO()
    monkeypatch.setattr("sys.stderr", plain)
    announce.say("connecting to box")
    assert plain.getvalue() == "letify: connecting to box\n"

    terminal = Terminal()
    monkeypatch.setattr("sys.stderr", terminal)
    monkeypatch.delenv("NO_COLOR", raising=False)
    announce.say("connecting to box")
    assert terminal.getvalue() == "\x1b[2mletify:\x1b[0m connecting to box\n"


# -- Spec: Command line, commands ----------------------------------------------


def _declare(project, text: str) -> None:
    (project / ".letify" / "config.toml").write_text(text, encoding="utf-8")


def test_providers_prints_a_table_and_json_on_request(isolated_home, capsys) -> None:
    _declare(isolated_home, '[lab]\nkind = "shell"\naddress = "a"\npersistent = true\n')
    assert main(["providers"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["ALIAS", "KIND", "PERSISTENCE"]
    assert lines[1].split() == ["lab", "shell", "persistent"]
    assert main(["providers", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {"alias": "lab", "kind": "shell", "persistence": "persistent"} in rows


def test_an_unbuildable_provider_is_marked_in_the_providers_table(isolated_home, capsys) -> None:
    _declare(isolated_home, '[odd]\nkind = "vastai"\n')
    assert main(["providers"]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("odd"))
    assert "✗ unavailable: " in line


def test_devices_prints_a_table(isolated_home, capsys) -> None:
    _declare(isolated_home, '[lab]\nkind = "shell"\naddress = "a"\ngpus = ["A100", "H100"]\n')
    assert main(["devices"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["PROVIDER", "ACCELERATORS"]
    assert any(line.startswith("lab ") and line.endswith("A100, H100") for line in lines)


def test_status_with_no_runtime_says_so_and_lists_the_inventory(isolated_home, capsys) -> None:
    _declare(
        isolated_home,
        '[lab]\nkind = "shell"\naddress = "a"\n[lab.devices]\nP100 = { indices = "0-1" }\n',
    )
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "0 live, 0 busy" in out.splitlines()[0]
    assert "no live session in this process" in out
    assert any(line.split() == ["lab", "P100", "0/2", "0,", "1"] for line in out.splitlines())
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["live"] == 0


def test_a_runtime_block_shows_link_uptime_and_the_cost_against_the_credit() -> None:
    status = {
        "name": "proj",
        "live": 1,
        "busy": 1,
        "devices": {"lab": {"P100": {"count": 2, "reserved": 1, "indices": [0, 1]}}},
        "runtimes": [
            {
                "name": "run-1",
                "provider": "lab",
                "accelerator": "P100",
                "devices": [0, 1],
                "placement": "remote",
                "busy": True,
                "idle_seconds": 180.0,
                "uptime_seconds": 3720.0,
                "link": "forward-ssh",
                "rtt_ms": 42.0,
                "usage": {
                    "unit": "compute units",
                    "remaining": 50.0,
                    "limit": 100.0,
                    "rate_per_hour": 2.0,
                },
            }
        ],
    }
    assert render.status_text(status, ASCII) == (
        "proj  1 live, 1 busy\n"
        "\n"
        "run-1  lab.P100  busy\n"
        "  cards 0, 1  host remote  link forward-ssh, 42.0 ms\n"
        "  up 1 h 2 min  idle 3 min\n"
        "  about 2.07 compute units so far at 2.00 compute units/hour\n"
        "  [" + "#" * 20 + "-" * 20 + "] 50% used\n"
        "  50.00 compute units left of 100.00\n"
        "\n"
        "PROVIDER  ACCELERATOR  RESERVED  INDICES\n"
        "lab       P100         1/2       0, 1\n"
    )


def test_a_runtime_block_leaves_out_what_is_not_known() -> None:
    runtime = {
        "name": "run-2",
        "provider": "modal_lab",
        "accelerator": "A10G",
        "devices": [],
        "placement": "remote",
        "busy": False,
        "idle_seconds": 30.0,
        "uptime_seconds": 90.0,
        "link": None,
        "rtt_ms": None,
    }
    status = {"name": "p", "live": 1, "busy": 0, "devices": {}, "runtimes": [runtime]}
    assert render.status_text(status, ASCII) == (
        "p  1 live, 0 busy\n\nrun-2  modal_lab.A10G  idle\n  host remote\n  up 1 min  idle 0 min\n"
    )


def test_probe_prints_a_verdict_with_aligned_fields_and_json_on_request(
    isolated_home, capsys
) -> None:
    assert main(["probe"]) == 0
    out = capsys.readouterr().out
    first = out.splitlines()[0]
    assert first[:2] in {"✓ ", "✗ ", "! "}
    assert "forwarding" in first
    assert any(line.startswith("platform ") for line in out.splitlines())
    assert main(["probe", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["platform"] and "usable" in report


def test_efficiency_prints_json_on_request(capsys) -> None:
    assert main(["efficiency", "0.5", "3", "150", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["efficiency"] == pytest.approx(0.526, abs=1e-3)


def test_logout_of_an_undeclared_alias_is_a_marked_failure(isolated_home, capsys) -> None:
    assert main(["logout", "nosuch"]) == 1
    assert capsys.readouterr().err.startswith("✗ nosuch is not declared in ")


def test_login_marks_the_declaration_and_the_reference(isolated_home, capsys) -> None:
    argv = ["login", "shell", "lab", "--address", "a", "--no-input", "--skip-key-install"]
    code = main(argv)
    out = capsys.readouterr()
    if code != 0:  # pragma: no cover - the shell login needs ssh on this host
        pytest.skip(f"shell login could not complete here: {out.err.strip()}")
    lines = out.out.splitlines()
    assert lines[-2].startswith("✓ lab declared in ")
    assert lines[-1].startswith("✓ lab referenced in ")
