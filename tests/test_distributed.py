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


def test_every_replica_runs_the_body_and_knows_which_one_it_is(launcher_from) -> None:
    """Spec "Replicas": the same body once per device, and the values come back in rank order."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def who() -> tuple[int, int]:
        import letify as inner

        return inner.rank(), inner.world_size()

    assert who() == [(0, 2), (1, 2)]


def test_one_replica_still_comes_back_as_a_list(launcher_from) -> None:
    """Spec "Replicas": a single instance is one replica, so the shape does not change."""
    let = launcher_from('[a]\nkind = "local"\n')

    @let.function(device=let.providers.a.CPU, host=letify.distributed)
    def who() -> tuple[int, int]:
        import letify as inner

        return inner.rank(), inner.world_size()

    assert who() == [(0, 1)]


def test_a_replica_that_fails_fails_the_call(launcher_from) -> None:
    """Spec "Replicas": a replica that cannot run fails the call, as one remote call does."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def breaks() -> int:
        import letify as inner

        if inner.rank() == 1:
            raise ValueError("the second replica refused")
        return 0

    with pytest.raises(letify.RemoteError) as failure:
        breaks()
    assert "the second replica refused" in str(failure.value)


def test_a_replica_does_not_need_torch_on_this_machine(launcher_from, no_module) -> None:
    """Spec "Replicas": the placement is not local, so nothing asks for PyTorch here.

    The replicas were handed to the provider without the declared placement folded in,
    so their placement read as the default, local, and `check_mode` asked for PyTorch on
    the client. It passed on a machine that has torch and failed on CI, which does not.
    """
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def who() -> int:
        import letify as inner

        return inner.rank()

    no_module("torch")
    assert who() == [0, 1]
    assert [instance.placement for instance in who.replicas] == [
        letify.distributed,
        letify.distributed,
    ]


# -- all_reduce ---------------------------------------------------------------


def test_all_reduce_outside_a_distributed_call_is_refused() -> None:
    """Spec "Reducing between replicas": there is nothing to reduce with."""
    with pytest.raises(letify.UnsupportedMode) as failure:
        letify.all_reduce([1.0, 2.0])
    assert "distributed" in str(failure.value)


def test_all_reduce_averages_in_place_and_returns_the_count(launcher_from) -> None:
    """Spec "Reducing between replicas": in place, and the return value is how many."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def reduce_one() -> tuple[int, list[float]]:
        import array

        import letify as inner

        def average(arrays):
            columns = zip(*arrays, strict=True)
            return array.array("f", [sum(column) / len(arrays) for column in columns])

        values = array.array("f", [float(inner.rank()), 10.0])
        count = inner.all_reduce(values, op=average)
        return count, list(values)

    assert reduce_one() == [(2, [0.5, 10.0]), (2, [0.5, 10.0])]


def test_the_provided_reductions_average_and_add(launcher_from) -> None:
    """Spec "Reducing between replicas": letify.mean and letify.sum, where numpy is here.

    They import numpy on the runtime, which letify does not depend on and CI does not
    install, so this is the one test in the file that needs it. The transport itself is
    pinned without it.
    """
    pytest.importorskip("numpy")
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed)
    def both() -> tuple[list[float], list[float]]:
        import numpy

        import letify as inner

        averaged = numpy.array([float(inner.rank()), 10.0], dtype="float32")
        inner.all_reduce(averaged, op=inner.mean)
        added = numpy.array([float(inner.rank()) + 1.0], dtype="float32")
        inner.all_reduce(added, op=inner.sum)
        return averaged.tolist(), added.tolist()

    assert both() == [([0.5, 10.0], [3.0]), ([0.5, 10.0], [3.0])]


def test_an_arbitrary_callable_is_the_reduction(launcher_from) -> None:
    """Spec "Reducing between replicas": ``op`` is a callable, so no option is needed."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def weighted() -> list[float]:
        import array

        import letify as inner

        def first_only(arrays):
            return arrays[0]

        values = array.array("f", [float(inner.rank())])
        inner.all_reduce(values, op=first_only)
        return list(values)

    assert weighted() == [[0.0], [0.0]]


def test_a_cuda_tensor_is_refused_and_names_cpu(launcher_from) -> None:
    """Spec "Reducing between replicas": the copy to host is the user's to write."""
    let = launcher_from('[a]\nkind = "local"\n')

    @let.function(device=let.providers.a.CPU, host=letify.distributed)
    def on_the_card() -> str:
        import letify as inner

        class Pretend:
            is_cuda = True

        try:
            inner.all_reduce(Pretend())
        except inner.UnsupportedMode as failure:
            return str(failure)
        return "it was accepted"

    assert ".cpu()" in on_the_card()[0]


def test_min_peers_proceeds_with_fewer_and_reports_how_many(launcher_from) -> None:
    """Spec "Reducing between replicas": a replica missing a round is the normal case."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def uneven() -> int:
        import array

        import letify as inner

        values = array.array("f", [1.0])
        if inner.rank() == 1:
            return 0
        return inner.all_reduce(values, op=lambda arrays: arrays[0], min_peers=1, timeout=5.0)

    assert uneven() == [1, 0]


def test_a_round_that_cannot_reach_min_peers_is_an_infrastructure_failure(
    launcher_from,
) -> None:
    """Spec "Reducing between replicas": the call may be retried on it."""
    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def never_meets() -> str:
        import array

        import letify as inner

        values = array.array("f", [1.0])
        if inner.rank() == 1:
            return "left early"
        try:
            inner.all_reduce(values, timeout=2.0)
        except inner.RuntimeFailure as failure:
            return type(failure).__name__
        return "it returned"

    assert never_meets() == ["ReductionFailed", "left early"]


def test_the_reduced_bytes_never_cross_the_client(launcher_from) -> None:
    """Spec "Reducing between replicas", claim N10: the client brokers and nothing more."""
    from letify.runtime import broker

    let = launcher_from('[a]\nkind = "local"\n\n[b]\nkind = "local"\n')

    @let.function(
        device=[let.providers.a.CPU, let.providers.b.CPU], host=letify.distributed
    )
    def one_round() -> int:
        import array

        import letify as inner

        return inner.all_reduce(
            array.array("f", [0.0] * 4096), op=lambda arrays: arrays[0]
        )

    before = broker.brokered_bytes()
    assert one_round() == [2, 2]
    # The array is 4096 float32, so 16384 bytes crossed between the replicas. What the
    # client carried for that round is two joins and two tables.
    assert broker.brokered_bytes() - before < 2048
