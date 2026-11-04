"""letify-link is opt-in in 2.x.

Spec "letify-link": with no ``link`` option the connection path is that of 2.10.x, and with
``link = "letify-link"`` the account carries its channel over letify-link.
"""

from __future__ import annotations

import pytest
from conftest import provider_of

from letify.providers.colab import Colab
from letify.providers.kaggle import Kaggle
from letify.providers.shell import Shell
from letify.transport import pipeline as pipeline_module
from letify.transport.linkchannel import PlainSshLink


@pytest.fixture
def raced(monkeypatch):
    """Record the strategies a Pipeline is built with, and connect to nothing."""
    seen: list[list[str]] = []

    class Recorder:
        def __init__(self, strategies, **options):
            seen.append([s.name for s in strategies])

        def connect(self):
            return object()

    monkeypatch.setattr(pipeline_module, "Pipeline", Recorder)
    return seen


def test_a_shell_account_without_the_switch_races_the_2_10_strategies(isolated_home, raced) -> None:
    provider = provider_of(Shell, "lab", address="gpu.example.edu", port_mapping=False)
    provider.link()
    assert raced == [["direct_ssh", "tcp_punch", "tailcat", "quic", "fallback"]]


def test_a_shell_account_with_the_switch_off_by_name_races_the_same(isolated_home, raced) -> None:
    provider = provider_of(Shell, "lab", address="gpu.example.edu", link="ssh", port_mapping=False)
    provider.link()
    assert raced == [["direct_ssh", "tcp_punch", "tailcat", "quic", "fallback"]]


def test_a_shell_account_with_the_switch_on_races_nothing(isolated_home, raced) -> None:
    provider = provider_of(Shell, "lab", address="gpu.example.edu", link="letify-link")
    assert provider.uses_letify_link is True
    assert isinstance(provider.link(), PlainSshLink)
    assert raced == []


def test_a_shell_account_with_the_switch_on_tries_letify_link(isolated_home, tmp_path) -> None:
    from letify.errors import ProviderUnavailable

    missing = tmp_path / "nowhere" / "letify-link"
    provider = provider_of(
        Shell, "lab", address="gpu.example.edu", link="letify-link", link_binary=str(missing)
    )
    with pytest.raises(ProviderUnavailable, match="letify-link"):
        provider.link_remote(None)


def test_kaggle_and_colab_without_the_switch_keep_tailcat_in_the_race(isolated_home) -> None:
    kaggle = provider_of(Kaggle, "k")
    assert [s.name for s in kaggle.strategies()] == ["tcp_punch", "tailcat", "quic", "fallback"]
    assert kaggle.uses_letify_link is False
    colab = provider_of(Colab, "c")
    assert "tailcat" in [s.name for s in colab.strategies()]
    assert colab.uses_letify_link is False


def test_the_switch_is_read_from_the_account_option_link(isolated_home) -> None:
    assert provider_of(Kaggle, "k", link="letify-link").uses_letify_link is True
    assert provider_of(Colab, "c", link="letify-link").uses_letify_link is True
