"""Replicas: the same body on several runtimes, reducing between themselves.

Spec "Replicas", "Reducing between replicas", "How often to synchronize". Claims N9, N10.
"""

from __future__ import annotations

import pytest

import letify


def test_distributed_is_the_third_placement() -> None:
    """Spec "The three placements": host takes a third value, as a string enum member."""
    from letify.declare.instance import Host

    assert letify.distributed is Host.distributed
    assert letify.distributed == "distributed"
    assert sorted(member.value for member in Host) == ["distributed", "local", "remote"]


def test_rank_and_world_size_refuse_to_answer_outside_a_distributed_call() -> None:
    """Spec "Replicas": both raise rather than answering 0 and 1.

    A body that reads them under host=remote has a bug, and a default would hide it.
    """
    with pytest.raises(letify.UnsupportedMode) as rank_failure:
        letify.rank()
    assert "distributed" in str(rank_failure.value)
    with pytest.raises(letify.UnsupportedMode) as world_failure:
        letify.world_size()
    assert "distributed" in str(world_failure.value)
