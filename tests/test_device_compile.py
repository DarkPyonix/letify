"""Compiling captured steps on the runtime, over a real device worker subprocess.

The worker runs the executor on CPU tensors, so Inductor compiles for CPU here. Each test
pins what spec "Compilation" decides: the request that enables compilation, the segments
and scalars the worker compiles, the fallback to eager replay and the note a failed
compilation leaves. Graphs are kept tiny because an Inductor compilation takes seconds.
"""

from __future__ import annotations

import sys
import warnings

import pytest

torch = pytest.importorskip("torch", reason="PyTorch forwarding tests need torch importable")

from letify.remoting import device as forwarding  # noqa: E402

LEARNING_RATES = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
WARMUP = 4


@pytest.fixture(scope="module")
def inductor_cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp("inductor"))


@pytest.fixture
def run_loop(inductor_cache, monkeypatch):
    """Train a tiny model on a fresh worker and return ``(losses, weights, stats)``."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", inductor_cache)
    monkeypatch.setenv("LETIFY_COMPILE_STEPS", "1")

    def run(compile_it: bool, **compile_arguments):
        connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
        try:
            with connected.activate(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.manual_seed(0)
                weight = (torch.randn(6, 6) * 0.3).cuda().requires_grad_()
                data = torch.randn(5, 6).cuda()
                if compile_it:
                    torch.compile(lambda: None, **compile_arguments)
                losses = []
                for index, rate in enumerate(LEARNING_RATES):
                    loss = (data @ weight).tanh().pow(2).mean()
                    loss.backward()
                    with torch.no_grad():
                        weight.sub_(weight.grad * rate)
                    weight.grad = None
                    losses.append(loss.detach())
                    if index == WARMUP - 1:
                        connected.call("letify.compile_stats", True)
                stats = connected.call("letify.compile_stats", True)
                return (
                    torch.stack([loss.cpu() for loss in losses]),
                    weight.detach().cpu(),
                    stats,
                )
        finally:
            connected.close()

    return run


def test_a_compiled_step_computes_what_eager_replay_computes(run_loop) -> None:
    eager_losses, eager_weight, _ = run_loop(False)
    losses, weight, stats = run_loop(True)
    assert stats["runs"] > 0
    assert torch.allclose(losses, eager_losses, rtol=1e-5, atol=1e-6)
    assert torch.allclose(weight, eager_weight, rtol=1e-5, atol=1e-6)


def test_a_step_is_compiled_once_and_a_varying_scalar_does_not_recompile(run_loop) -> None:
    _, _, stats = run_loop(True)
    assert stats["compiled"] == 1
    assert stats["failed"] == 0
    assert stats["runs"] >= len(LEARNING_RATES) - WARMUP - 2


def test_without_torch_compile_steps_replay_eagerly(run_loop) -> None:
    _, _, stats = run_loop(False)
    assert stats == {"compiled": 0, "failed": 0, "runs": 0, "eager": 0}


def test_a_step_whose_compilation_fails_runs_eagerly(run_loop, capsys) -> None:
    eager_losses, eager_weight, _ = run_loop(False)
    losses, weight, stats = run_loop(True, options={"letify_no_such_option": True})
    assert stats["failed"] == 1
    assert stats["runs"] == 0
    assert stats["eager"] == 1
    assert "could not be compiled" in capsys.readouterr().err
    assert torch.allclose(losses, eager_losses)
    assert torch.allclose(weight, eager_weight)


def test_an_adamw_loop_with_its_bias_corrections_computes_what_eager_replay_computes(
    inductor_cache, monkeypatch
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", inductor_cache)
    monkeypatch.setenv("LETIFY_COMPILE_STEPS", "1")

    def train(compile_it: bool):
        connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
        try:
            with connected.activate(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(6, 8), torch.nn.GELU(), torch.nn.Linear(8, 2)
                ).cuda()
                if compile_it:
                    model.compile()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
                data = torch.randn(5, 6).cuda()
                losses = []
                for index in range(14):
                    loss = model(data).pow(2).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach())
                    if index in (5, 8):
                        connected.call("letify.compile_stats", True)
                stats = connected.call("letify.compile_stats", True)
                return torch.stack([loss.cpu() for loss in losses]), stats
        finally:
            connected.close()

    eager, _ = train(False)
    compiled, stats = train(True)
    assert stats["runs"] > 0
    assert stats["failed"] == 0
    assert torch.allclose(compiled, eager, rtol=1e-4, atol=1e-6)
