"""QUIC over a punched UDP pair is removed.

Spec "Connection strategies": the ranked list has forward SSH, connect back, TCP hole
punching, reverse SSH and the provider fallback, and no QUIC entry. The ``letify-quic``
binary, its crate and its Python module are not in the tree. letify-link carries UDP paths.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest
from conftest import provider_of

from letify.providers.colab import Colab
from letify.providers.kaggle import Kaggle
from letify.providers.shell import Shell

ROOT = Path(__file__).resolve().parent.parent


def test_the_transport_package_has_no_quic_module() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("letify.transport.quic")


def test_the_strategy_module_has_no_quic_strategy() -> None:
    from letify.transport import strategies

    assert not hasattr(strategies, "QuicUDP")
    assert "QuicUDP" not in strategies.__all__


@pytest.mark.parametrize("cls", [Shell, Kaggle, Colab])
def test_no_ssh_strategy_list_has_a_quic_entry(isolated_home, cls) -> None:
    options = {"address": "gpu.example.edu"} if cls is Shell else {}
    provider = provider_of(cls, "box", link="ssh", **options)
    assert "quic" not in [s.name for s in provider.strategies()]


def test_the_rendezvous_does_not_prepare_a_carrier() -> None:
    from letify.transport import rendezvous

    assert not hasattr(rendezvous.CommandRendezvous, "prepare_quic")
    assert not hasattr(rendezvous, "QUIC_PREPARE_TIMEOUT")


def test_the_wheel_extraction_helpers_name_only_the_sidecar() -> None:
    from letify import install

    assert not hasattr(install, "remote_quic_source")
    assert hasattr(install, "remote_link_source")


def test_the_remote_half_of_a_punch_has_no_udp_carrier_helpers() -> None:
    from letify.transport import nat

    assert not hasattr(nat, "udp_stun_mapping")
    assert not hasattr(nat, "punch_udp")


def test_the_workspace_has_no_letify_quic_crate() -> None:
    members = tomllib.loads((ROOT / "letify-core" / "Cargo.toml").read_text())["workspace"][
        "members"
    ]
    assert "letify-quic" not in members
    assert not (ROOT / "letify-core" / "letify-quic").exists()


def test_the_build_script_ships_only_the_sidecar() -> None:
    source = (ROOT / "letify-core" / "build.py").read_text()
    assert "letify-quic" not in source
    assert "letify-link" in source
