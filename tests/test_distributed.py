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


def declared(launcher_from, devices, host=letify.distributed):
    """One declaration over ``devices``, so the validation is what is under test."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')
    picked = [getattr(let.providers, name).CPU for name in devices]
    device = picked if isinstance(devices, list) else picked[0]

    @let.function(device=device, host=host)
    def body() -> int:
        return 1

    return body


def test_a_device_list_declares_one_replica_each(launcher_from) -> None:
    """Spec "Replicas": each entry is one replica, in the order the list names."""
    body = declared(launcher_from, ["a", "b"])
    assert [instance.provider.alias for instance in body.replicas] == ["a", "b"]


def test_one_instance_is_one_replica(launcher_from) -> None:
    """Spec "Replicas": a declaration does not have to change to run on one runtime."""
    let = launcher_from('[a]\nkind = "local"\n')

    @let.function(device=let.providers.a.CPU, host=letify.distributed)
    def body() -> int:
        return 1

    assert [instance.provider.alias for instance in body.replicas] == ["a"]


def test_two_replicas_of_one_instance_are_refused(launcher_from) -> None:
    """Spec "Replicas": that would ask one account for twice its devices."""
    let = launcher_from('[a]\nkind = "local"\n')
    twice = [let.providers.a.CPU, let.providers.a.CPU]
    with pytest.raises(ValueError) as failure:
        @let.function(device=twice, host=letify.distributed)
        def body() -> int:
            return 1

    message = str(failure.value)
    assert "twice" in message or "once" in message
    assert "a" in message


def test_a_device_list_without_the_distributed_placement_is_refused(launcher_from) -> None:
    """Spec "Replicas": device takes a list only there, so the two cannot disagree."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')
    pair = [let.providers.a.CPU, let.providers.b.CPU]
    with pytest.raises(ValueError) as failure:
        @let.function(device=pair, host=letify.remote)
        def body() -> int:
            return 1

    assert "host=letify.distributed" in str(failure.value)


def test_an_empty_device_list_is_refused(launcher_from) -> None:
    """Spec: a replica list with nothing in it is a declaration that says nothing."""
    let = launcher_from('[a]\nkind = "local"\n')
    with pytest.raises(ValueError):
        @let.function(device=[], host=letify.distributed)
        def body() -> int:
            return 1


def test_local_steps_is_the_first_count_that_makes_a_reduction_cheap_enough() -> None:
    """Spec "How often to synchronize": the smallest count under share of the step.

    Worked from measurements in the spec: 100 MB of parameters and a 0.2 s step takes
    4.54 s to reduce at the 21 MiB/s a near Kaggle runtime gives and 28.05 s at the
    3.4 MiB/s a far one gives, against the 0.05 s a quarter of the step allows, so 91 and
    561. The answer is per runtime rather than per job.
    """
    from letify.declare.replicas import steps_for

    MIB = 1 << 20
    assert steps_for(param_bytes=100_000_000, step_seconds=0.2, rate_bps=21 * MIB) == 91
    assert steps_for(param_bytes=100_000_000, step_seconds=0.2, rate_bps=3.4 * MIB) == 561
    # A link fast enough reduces every step, which is what DDP does.
    assert steps_for(param_bytes=1_000_000, step_seconds=1.0, rate_bps=100 * MIB) == 1
    # share is the fraction of the step a reduction may take, so a larger one allows fewer.
    assert steps_for(
        param_bytes=100_000_000, step_seconds=0.2, rate_bps=21 * MIB, share=1.0
    ) == 23


def test_local_steps_refuses_a_link_that_was_never_measured() -> None:
    """Spec: a guess here is a guess about the whole run's shape."""
    from letify.declare.replicas import steps_for

    with pytest.raises(letify.UnsupportedMode) as failure:
        steps_for(param_bytes=1, step_seconds=1.0, rate_bps=None)
    assert "probe" in str(failure.value)


def test_calling_a_distributed_declaration_says_it_is_not_finished(launcher_from) -> None:
    """Spec "Replicas": the placement is declared and the runner is not written yet.

    Accepting the declaration and then running on the first replica alone would be worse
    than refusing: the user asked for several and would be billed for one, silently.
    """
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def body() -> int:
        return 1

    with pytest.raises(letify.UnsupportedMode) as failure:
        body()
    message = str(failure.value)
    assert "cannot run them yet" in message
    assert "2 replicas" in message
    assert "letify.remote" in message
