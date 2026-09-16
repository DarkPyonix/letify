# 3️⃣ Choosing the execution mode

> Ship the loop or forward the PyTorch operators. When each wins, with the arithmetic.

[← Providers](02-providers.md) · [Guides](README.md) · [Next: Environments and data →](04-environments-and-data.md)

---

## The two modes

| | 📦 Function shipping | 🔌 PyTorch forwarding |
|---|---|---|
| Written as | `host="remote"` | `host="local"` |
| Where Python runs | on the remote machine | in your process |
| Where data has to be | on the remote machine | on your machine |
| What crosses the network | the function, once | every PyTorch operator, queued and sent in batches |
| What that costs | one transfer | one round trip per host synchronization |

Neither name appears in your code. You say where the CPU side of the work lives, and letify picks the mechanism.

```python
@let.function(device=colab.G4, host="remote")                    # provider default
@let.function(device=lab.A100, host="remote")      # ship the loop
@let.function(device=lab.A100, host="local")       # forward PyTorch operators
```

## The formula

```
efficiency = T / (T + n × d + k × RTT)
```

- `T` is GPU time per step
- `n` is how many PyTorch operators the step dispatches, and `d` is what dispatching one costs in your process
- `k` is how many times in that step the host reads a value back from the device
- `RTT` is the network round trip

`n × d` is paid in your process whatever the link, so a step made of many small operators loses more than one made of a few large ones.

Function shipping runs the loop on the remote machine, so those reads are local there and both terms vanish. Forwarding pays `k × RTT` on every read and `n × d` on every step.

## Where the numbers land

RTX PRO 6000 with NVFP4, a 0.5 s micro step, `k = 3`:

| Round trip | Fine-tuning | Decoding, batch 1 |
|---|---|---|
| 20 ms | 89% | 43 tok/s |
| 150 ms | **53%** | 6.5 tok/s |
| 450 ms | 27% | 2 tok/s |

Function shipping is about 99% at every one of those round trips, and hundreds of tokens per second for decoding.

### Two surprises worth internalizing

**A faster GPU makes forwarding worse.** `T` shrinks and `RTT` does not. The same step on an L4 in bf16 takes 1.8 s and reaches 80% where the RTX PRO 6000 reaches 53%.

**For decoding, the card stops mattering.** Throughput is bounded near `1000 / (k × RTT)` tokens per second. At 150 ms an L4 and an RTX PRO 6000 both land near 5 to 6 tokens per second. Paying for the faster card buys nothing in that mode.

## Where `k` comes from

`k ≈ 3` for a default Hugging Face training step:

| Synchronization | How often | Why |
|---|---|---|
| `logging_nan_inf_filter` | every micro step | evaluates `torch.isnan(loss)` as a Python bool |
| SDPA attention mask check | every forward | `torch.all(attention_mask == 1)` decides the causal path |
| Loss or gradient norm logging | every logging step | `.item()` |
| Gradient scaler | every step under fp16 | checks for infinities |

A read does not have to be a synchronization. `loss.detach().to("cpu", non_blocking=True)` returns a CPU tensor at once and the loop keeps going, and the value is waited for only where it is used, for example when it is printed a step later. In `async def` code, `await letify.fetch(loss)` does the same without blocking the event loop.

Other things that force a synchronization: `nonzero()`, boolean mask indexing, `unique()`, and mixture of experts routing, which reads per-expert token counts back to the host. Any operation whose output shape the host has to know is a synchronization.

### Measure it instead of guessing

```python
import torch

torch.cuda.set_sync_debug_mode("warn")
# run exactly one real training step and count the warnings
```

This does not depend on where the GPU is, so it can be measured on any CUDA device, including a Colab runtime. Under `host="local"`, `letify.remoting.device.current_client().stats` counts it for you: `round_trips` is `k` and `ops` is `n`. With `k` measured and the round trip measured, the formula gives you a real answer instead of an estimate.

```bash
letify probe gpu.lab.example.edu     # round trip, and whether forwarding is possible
```

## Tuning `k` down

Forwarding is not inherently 53%. That is 53% **with default settings**. Three changes leave about one synchronization per optimizer step, which reaches roughly 96%:

1. **Turn off the NaN filter.** `logging_nan_inf_filter=False` in `TrainingArguments`.
2. **Remove the mask check.** Use fixed length packing and pass `attention_mask=None` so the causal path is taken without inspecting the mask.
3. **Log at the accumulation boundary.** Set `logging_steps` so loss and gradient norm are read once per optimizer step rather than once per micro step.

Turning off the NaN filter alone stops at 92%, because the mask check remains.

This is tuning on your training code, which is why letify does not do it for you and why shipping stays the default.

## Per-step data under `host="local"`

Under `host="local"` the data stays on your machine, so every batch crosses the network to the GPU. A link carries only so many bytes per second, and letify cannot change that. Two parallel streams over ssh from a university client to lab_docker carried 101 MiB/s in total, and one carried 91 MiB/s. When the batches are large, that transfer, not the GPU, sets the step time.

Send the smallest form of the batch, and do the rest of the preprocessing on the GPU. For images, that means the decoded `uint8` pixels rather than normalized `float32`, which is 4 times fewer bytes:

```python
class Images(torch.utils.data.Dataset):
    def __getitem__(self, i):
        return decode(i), label(i)            # uint8, 3 x H x W, no ToTensor() or Normalize()

mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

for images, labels in loader:
    images = images.cuda(non_blocking=True).float().div_(255).sub_(mean).div_(std)
    labels = labels.cuda(non_blocking=True)
```

Measured on ResNet-50, bf16, against training directly on the same RTX PRO 5000: at batch 64, `float32` batches run at 13% of direct speed and `uint8` batches at 52%. At batch 256 it is 14% and 52%. The losses are identical to direct training in both cases, because the same normalization runs on the same GPU. `pin_memory=True` does not apply under `host="local"`: this process has no CUDA to pin with, and letify copies the batch before sending it anyway.

When even `uint8` batches are too large for the link, the data belongs next to the GPU. Use `host="remote"` with the dataset in a volume on the runtime.

## How the default is chosen

It is not chosen. `host` defaults to `"local"`, and nothing derives it from the provider.

That default is the least surprising one: your code and your data are already on this
machine, so borrowing a GPU should not require moving them. Shipping the function is the
optimization a heavy loop opts into.

What a provider does decide is whether it can serve a mode at all, and what it costs there.

| Provider | `host="local"` | Why |
|---|---|---|
| `Local` | not applicable | the device is already here, nothing crosses a network |
| `Colab` | allowed, with a warning | the control path crosses a Google frontend, so about 175 ms |
| `Modal` | refused | it exposes function calls into a container, with no device to forward at |
| `Shell`, `Tunnel` | allowed | a machine reached directly has a short round trip |
| `Elice` | allowed | same |

The warning carries the expected round trip, so you see what the choice costs before the
run rather than after it.

## No silent fallback

```python
@let.function(device=colab.G4, host="local")
def train(lr): ...
```

```
UnsupportedMode: Colab does not support host='local'. Forwarding CUDA calls over
the Colab control path costs one round trip of about 150 ms per host
synchronization, which leaves roughly half the throughput for fine-tuning and a
few percent for token by token decoding. Use host='remote' so the loop runs
inside the runtime.
```

letify raises rather than taking the slower path. A silent downgrade turns a four times slowdown into a mystery, and a mystery costs more than an exception.

## Choosing in practice

**Ship the loop** for training, evaluation, batch inference, and anything where the work is a loop you can hand over whole. That is nearly everything, which is why it is the default.

**Forward the operators** when the code is interactive and human paced, when the data has to stay on your machine, and when the machine is close. Exploratory notebook work is the honest use case: a person cannot feel 20 ms per cell, and the data stays where it already is.

**Never forward a generation loop.** Ship the whole `generate` call instead and you get the card's real speed.

## A note on what is implemented

Both modes work. PyTorch forwarding supports PyTorch only: code written for `"cuda"` runs unchanged with any local PyTorch build, and the same torch major.minor version has to be in the project on both sides. It forwards to one device per session and has no CUDA streams, events or graphs. Measured numbers against a direct run are in [NETWORK.md](../NETWORK.md#pytorch-forwarding-on-dept_gpu), and the remaining gaps are at the end of [docs/SPEC.md](../SPEC.md).

---

[← Providers](02-providers.md) · [Guides](README.md) · [Next: Environments and data →](04-environments-and-data.md)
