"""Terminal rendering of usage blocks.

Pins spec "Remaining usage": the block layout, the gauge, the relative reset time, the
further allowances in ``resources``, and the colour, encoding and width rules.
"""

from __future__ import annotations

import io

import pytest

from letify import render
from letify.providers.local import Local
from letify.providers.usage import Usage

# 2026-09-14 06:00 UTC, fixed so a reset time prints the same relative text every run.
NOW = 1789365600.0
PLAIN = render.Style(width=60, color=False, unicode=True)


def _row(**fields: object) -> dict:
    base = {"alias": "a", "kind": "k", "source": "s"}
    base.update(fields)
    return base


# -- Spec: Remaining usage -----------------------------------------------------


def test_a_block_with_a_limit_prints_a_gauge_the_amount_and_the_reset() -> None:
    row = _row(
        alias="kaggle",
        kind="kaggle",
        unit="GPU hours",
        remaining=12.0,
        limit=30.0,
        resets_at=NOW + 4 * 86400 + 6 * 3600,
    )
    assert render.usage_blocks([row], PLAIN, now=NOW) == (
        "kaggle  kaggle\n"
        "  [████████████████████████░░░░░░░░░░░░░░░░] 60% used\n"
        "  12.0 GPU hours left of 30.0\n"
        "  resets in 4 d 6 h (2026-09-18 12:00 UTC)\n"
    )


def test_an_unknown_limit_prints_the_balance_without_a_percentage() -> None:
    row = _row(alias="colab_a", kind="colab", unit="compute units", remaining=99.934)
    out = render.usage_blocks([row], PLAIN, now=NOW)
    assert out == "colab_a  colab\n  99.93 compute units left, limit unknown\n"
    assert "%" not in out


def test_a_known_rate_prints_the_burn_and_how_long_the_balance_lasts() -> None:
    row = _row(unit="compute units", remaining=60.0, limit=100.0, rate_per_hour=2.0)
    out = render.usage_blocks([row], PLAIN, now=NOW)
    assert "  2.00 compute units/hour running now, about 1 d 6 h at this rate\n" in out


def test_an_unmetered_machine_prints_no_quota_and_nothing_else() -> None:
    row = _row(alias="lab", kind="shell", unit="hours", unmetered=True, note="ignored")
    assert render.usage_blocks([row], PLAIN, now=NOW) == "lab  shell\n  no quota, unmetered\n"


def test_a_row_with_no_figure_prints_not_reported_and_its_note() -> None:
    row = _row(alias="e", kind="elice", unit="KRW", note="set billing_endpoint")
    assert render.usage_blocks([row], PLAIN, now=NOW) == (
        "e  elice\n  not reported\n  set billing_endpoint\n"
    )


def test_an_unavailable_provider_prints_its_reason() -> None:
    row = {"alias": "odd", "unavailable": "unknown kind vastai"}
    assert render.usage_blocks([row], PLAIN, now=NOW) == "odd\n  unavailable: unknown kind vastai\n"


def test_blocks_are_separated_by_one_blank_line() -> None:
    rows = [_row(alias="x", unmetered=True), _row(alias="y", unmetered=True)]
    out = render.usage_blocks(rows, PLAIN, now=NOW)
    assert out == "x  k\n  no quota, unmetered\n\ny  k\n  no quota, unmetered\n"


def test_a_further_allowance_prints_its_own_named_gauge_under_the_account() -> None:
    row = _row(
        alias="kaggle",
        kind="kaggle",
        unit="GPU hours",
        remaining=15.0,
        limit=30.0,
        resources=[
            {"name": "TPU", "unit": "TPU hours", "remaining": 18.0, "limit": 20.0, "used": 2.0}
        ],
    )
    out = render.usage_blocks([row], render.Style(width=40, color=False, unicode=False), now=NOW)
    assert out == (
        "kaggle  kaggle\n"
        "  [#############-------------] 50% used\n"
        "  15.0 GPU hours left of 30.0\n"
        "  TPU [##--------------------] 10% used\n"
        "      18.0 TPU hours left of 20.0\n"
    )


def test_a_reset_in_the_past_is_printed_as_due() -> None:
    row = _row(unit="USD", remaining=1.0, limit=30.0, resets_at=NOW - 60)
    assert "  reset due (2026-09-14 05:59 UTC)\n" in render.usage_blocks([row], PLAIN, now=NOW)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(45 * 60, "45 min"), (3 * 3600 + 12 * 60, "3 h 12 min"), (4 * 86400 + 6 * 3600, "4 d 6 h")],
)
def test_a_relative_time_uses_the_two_largest_units(seconds: float, expected: str) -> None:
    assert render.relative(seconds) == expected


def test_the_gauge_is_ascii_when_the_output_is_not_utf8() -> None:
    row = _row(unit="USD", remaining=15.0, limit=30.0)
    style = render.Style(width=40, color=False, unicode=False)
    assert "[#############-------------] 50% used" in render.usage_blocks([row], style)


@pytest.mark.parametrize(("width", "cells"), [(10, 16), (60, 40), (200, 40)])
def test_the_gauge_width_follows_the_terminal_within_its_range(width: int, cells: int) -> None:
    row = _row(unit="USD", remaining=15.0, limit=30.0)
    out = render.usage_blocks([row], render.Style(width=width, color=False, unicode=False))
    gauge = out.splitlines()[1]
    assert gauge.index("]") - gauge.index("[") - 1 == cells


@pytest.mark.parametrize(
    ("remaining", "code"), [(20.0, "\x1b[32m"), (10.0, "\x1b[33m"), (3.0, "\x1b[31m")]
)
def test_the_gauge_colour_follows_the_remaining_share(remaining: float, code: str) -> None:
    row = _row(unit="USD", remaining=remaining, limit=30.0)
    out = render.usage_blocks([row], render.Style(width=60, color=True, unicode=True))
    assert code in out.splitlines()[1]


def test_without_colour_no_escape_sequence_is_written() -> None:
    row = _row(unit="USD", remaining=3.0, limit=30.0, rate_per_hour=1.0, note="n")
    assert "\x1b" not in render.usage_blocks([row], PLAIN, now=NOW)


class _Stream(io.StringIO):
    def __init__(self, *, tty: bool, encoding: str) -> None:
        super().__init__()
        self._tty = tty
        self._encoding = encoding

    def isatty(self) -> bool:
        return self._tty

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding


def test_colour_needs_a_terminal_and_no_NO_COLOR(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert render.Style.for_stream(_Stream(tty=True, encoding="utf-8")).color
    assert not render.Style.for_stream(_Stream(tty=False, encoding="utf-8")).color
    monkeypatch.setenv("NO_COLOR", "1")
    assert not render.Style.for_stream(_Stream(tty=True, encoding="utf-8")).color
    monkeypatch.setenv("NO_COLOR", "")
    assert render.Style.for_stream(_Stream(tty=True, encoding="utf-8")).color


def test_block_characters_need_a_utf8_stream() -> None:
    assert render.Style.for_stream(_Stream(tty=False, encoding="UTF-8")).unicode
    assert not render.Style.for_stream(_Stream(tty=False, encoding="ascii")).unicode


def test_a_record_carries_its_further_allowances_in_its_dictionary() -> None:
    usage = Usage(alias="a", kind="k", unit="GPU hours", source="s")
    assert usage.to_dict()["resources"] == []


class _Balance(Local):
    """A local provider that answers with a balance and no allowance, as Colab does."""

    def report_usage(self) -> Usage:
        return Usage(
            alias=self.alias, kind=self.kind, unit="compute units", source="s", remaining=40.0
        )


def test_a_configured_plan_limit_fills_a_limit_the_service_did_not_state() -> None:
    from conftest import provider_of

    usage = provider_of(_Balance, "c", usage_limit=100).usage()
    assert (usage.limit, usage.used, usage.remaining) == (100.0, 60.0, 40.0)
