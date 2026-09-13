"""The training body, written as if the GPU were local, because to this code it is.

Nothing in this file knows about letify. It imports torch, reads its arguments, trains and
returns a dictionary. That is the point of the declaration: the site where it runs is
decided outside the code that runs there, so this same function is what you debug locally
and what runs on a rented card.

This module is sent by value, which is why the scripts declare
``letify.Env().ship("recipe")``. The machine on the other end installs what the lock file
names and has no copy of this project, so a function imported from here has to travel
inside the call.

The arithmetic this example exists for: NVFP4 halves the weight memory of BF16 and roughly
doubles the tensor core throughput on a Blackwell card, which is what makes a 7B model
fine-tunable on one 96 GiB RTX PRO 6000 instead of two 80 GiB H100s.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

#: Where a runtime keeps this run's files. Under the volume mount, so a checkpoint written
#: here is one call away from the store.
WORKSPACE = "/opt/letify/run"


def environment() -> dict[str, Any]:
    """Report what the machine actually is, without assuming torch is installed.

    Run first, on purpose. A missing driver, a card that is not the one that was paid for
    or a torch built without Blackwell support all fail here in seconds rather than
    twenty minutes into a sweep.
    """
    found: dict[str, Any] = {
        "hostname": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
    }
    try:
        import torch
    except ImportError as exc:
        found["torch"] = None
        found["reason"] = str(exc)
        return found

    found["torch"] = torch.__version__
    found["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        return found

    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    found["device"] = torch.cuda.get_device_name(index)
    found["capability"] = f"{capability[0]}.{capability[1]}"
    found["vram_gb"] = round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 1)
    # NVFP4 tensor cores arrive with compute capability 10.0. Below that the same code
    # runs and the quantized path falls back, which costs speed rather than correctness.
    found["nvfp4"] = capability[0] >= 10
    return found


def count_syncs(steps: int = 3) -> dict[str, Any]:
    """Count how many times one training step reads a value back to the host.

    That count is the only term in the forwarding efficiency formula that depends on the
    training code rather than on the network, and it is measurable before any GPU is
    rented. Efficiency against a direct run is ``T / (T + k * RTT)``, where ``T`` is GPU
    time per step and ``k`` is what this returns.
    """
    import torch

    if not torch.cuda.is_available():
        return {"syncs": None, "reason": "no CUDA device to measure on"}

    seen: list[str] = []
    torch.cuda.set_sync_debug_mode("warn")
    try:
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tensor = torch.randn(256, 256, device="cuda")
            for _ in range(steps):
                result = tensor @ tensor
                # A comparison the host acts on is a synchronization, and this is the one
                # most training loops make without noticing: a finite check per step.
                if not torch.isfinite(result).all().item():
                    break
            seen = [str(warning.message) for warning in caught]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return {"syncs": round(len(seen) / steps, 2), "warnings": seen[:3]}


def train(
    *,
    lr: float,
    rank: int,
    steps: int = 60,
    batch: int = 8,
    sequence: int = 512,
    hidden: int = 2048,
    quantize: bool = True,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Train a LoRA adapter over a synthetic batch and return what it cost.

    Synthetic data on purpose: the scenario being demonstrated is the infrastructure, and a
    real dataset would make the example depend on a download, a tokenizer and a licence.
    Swap the two lines that build ``inputs`` and ``targets`` for a dataloader and the rest
    of this function is what you would actually run.

    The adapter is written to a single file rather than a directory, because a volume
    absorbs one file per checkpoint name.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    base = torch.nn.Linear(hidden, hidden, bias=False, device=device)
    base.weight.requires_grad_(False)
    if quantize and device == "cuda":
        # Stand-in for the real quantized kernel: the weight makes a round trip through a
        # narrow type and comes back to the compute dtype, which is the accuracy cost of
        # weight-only quantization without needing the Blackwell kernel to demonstrate it.
        # The adapter stays in float32, which is the memory shape NVFP4 training has.
        base.weight.data = base.weight.data.to(torch.float8_e4m3fn).to(torch.float32)

    down = torch.nn.Parameter(torch.zeros(rank, hidden, device=device))
    up = torch.nn.Parameter(torch.randn(hidden, rank, device=device) * 0.01)
    torch.nn.init.kaiming_uniform_(down, a=5**0.5)

    start_step = 0
    if resume_from:
        state = torch.load(resume_from, map_location=device)
        down.data.copy_(state["down"])
        up.data.copy_(state["up"])
        start_step = int(state.get("step", 0))

    optimizer = torch.optim.AdamW([down, up], lr=lr)
    inputs = torch.randn(batch, sequence, hidden, device=device)
    targets = torch.randn(batch, sequence, hidden, device=device)

    began = time.perf_counter()
    losses: list[float] = []
    for step in range(start_step, start_step + steps):
        hidden_states = base(inputs) + (inputs @ down.T) @ up.T
        loss = torch.nn.functional.mse_loss(hidden_states, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # Read back once every ten steps rather than every step. Each read is one network
        # round trip when the host is here and the card is elsewhere, so this is the single
        # change that moves forwarding efficiency from about half to nearly all of a direct
        # run at a 150 ms link.
        if step % 10 == 0 or step == start_step + steps - 1:
            losses.append(round(loss.item(), 5))

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - began

    os.makedirs(WORKSPACE, exist_ok=True)
    adapter = f"{WORKSPACE}/adapter-r{rank}-lr{lr:g}.pt"
    torch.save(
        {"down": down.detach().cpu(), "up": up.detach().cpu(), "step": start_step + steps},
        adapter,
    )

    return {
        "lr": lr,
        "rank": rank,
        "device": device,
        "resumed_at": start_step,
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "loss_curve": losses,
        "seconds": round(elapsed, 2),
        "step_seconds": round(elapsed / max(steps, 1), 4),
        "trainable_params": down.numel() + up.numel(),
        "adapter": adapter,
        "peak_vram_gb": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None
        ),
    }


def summarize(results: list[dict[str, Any]]) -> str:
    """One table of the sweep, best first, for printing next to the run."""
    ranked = sorted(results, key=lambda row: row["final_loss"] or float("inf"))
    lines = [f"{'rank':>5} {'lr':>9} {'loss':>9} {'s/step':>8} {'peak GiB':>9}"]
    for row in ranked:
        peak = row["peak_vram_gb"]
        lines.append(
            f"{row['rank']:>5} {row['lr']:>9.2e} {row['final_loss']:>9.4f} "
            f"{row['step_seconds']:>8.3f} {('-' if peak is None else f'{peak:>9.2f}')}"
        )
    return "\n".join(lines)


def write_report(results: list[dict[str, Any]], path: str) -> None:
    """Keep the numbers next to the run, because a printed table is gone tomorrow."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
