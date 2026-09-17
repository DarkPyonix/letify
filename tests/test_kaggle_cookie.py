"""The Kaggle cookie credential: parsing and expiry.

Spec "Kaggle account", the credential bullets: the cookie is the whole account, and its
expiry is read from the CLIENT-TOKEN JWT's exp claim rather than guessed.
"""
from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

import pytest

from letify.providers import kaggle


def make_client_token(exp_iso: str) -> str:
    """An alg:none JWT like Kaggle's CLIENT-TOKEN, carrying one exp claim."""
    def part(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    header = part({"alg": "none", "typ": "JWT"})
    payload = part({"sub": "irack000", "exp": exp_iso})
    return f"{header}.{payload}."


def make_cookie(exp_iso: str | None = "2026-10-17T07:27:13.9739045Z",
                with_client_token: bool = True) -> str:
    jar = {"ka_sessionid": "abc123", "XSRF-TOKEN": "xtok", "__Host-KAGGLEID": "kid"}
    if with_client_token and exp_iso is not None:
        jar["CLIENT-TOKEN"] = make_client_token(exp_iso)
    return "; ".join(f"{name}={value}" for name, value in jar.items())


def test_parse_cookie_splits_name_value_pairs() -> None:
    jar = kaggle.parse_cookie("a=1; b=2; c=3")
    assert jar == {"a": "1", "b": "2", "c": "3"}


def test_parse_cookie_keeps_equals_inside_a_value() -> None:
    jar = kaggle.parse_cookie("t=ab.cd==; x=1")
    assert jar["t"] == "ab.cd=="


def test_cookie_expiry_reads_the_client_token_exp_claim() -> None:
    expiry = kaggle.cookie_expiry(make_cookie("2026-10-17T07:27:13.9739045Z"))
    assert expiry == datetime(2026, 10, 17, 7, 27, 13, tzinfo=UTC)


def test_cookie_days_left_counts_from_now() -> None:
    now = datetime(2026, 10, 7, 7, 27, 13, tzinfo=UTC)
    left = kaggle.cookie_days_left(make_cookie("2026-10-17T07:27:13Z"), now=now)
    assert round(left) == 10


def test_an_expired_cookie_has_days_left_at_or_below_zero() -> None:
    now = datetime(2026, 11, 1, tzinfo=UTC)
    assert kaggle.cookie_days_left(make_cookie("2026-10-17T07:27:13Z"), now=now) < 0


def test_a_cookie_without_a_client_token_is_refused() -> None:
    with pytest.raises(ValueError):
        kaggle.cookie_expiry(make_cookie(with_client_token=False))


def test_a_cookie_missing_a_required_name_is_refused() -> None:
    with pytest.raises(ValueError):
        kaggle.require_cookie_shape("CLIENT-TOKEN=" + make_client_token("2026-10-17T07:27:13Z"))


def test_a_well_formed_cookie_passes_the_shape_check() -> None:
    # returns the parsed jar and does not raise
    jar = kaggle.require_cookie_shape(make_cookie())
    assert jar["ka_sessionid"] == "abc123"
