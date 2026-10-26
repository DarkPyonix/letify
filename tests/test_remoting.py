"""CUDA call forwarding: the efficiency arithmetic.

Spec section pinned here: "Efficiency model". The forwarding itself is covered by
test_device.py.
"""

from __future__ import annotations

from letify.remoting import efficiency

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
