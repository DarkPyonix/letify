# 3️⃣ Choosing the execution mode

> Ship the loop or forward the CUDA calls. When each wins, with the arithmetic.

[← Providers](02-providers.md) · [Guides](README.md) · [Next: Environments and data →](04-environments-and-data.md)

---

## The two modes

| | 📦 Function shipping | 🔌 Call forwarding |
|---|---|---|
| Written as | `host="remote"` | `host="local"` |
| Where Python runs | on the remote machine | in your process |
| Where data has to be | on the remote machine | on your machine |
| What crosses the network | the function, once | every CUDA call |
| What that costs | one transfer | one round trip per host synchronization |

Neither name appears in your code. You say where the CPU side of the work lives, and letify picks the mechanism.

```python
@let.function(device=colab.G4, host="remote")                    # provider default
@let.function(device=lab.A100, host="remote")      # ship the loop
@let.function(device=lab.A100, host="remote", host="local")       # forward CUDA calls
```

## The formula

```
efficiency = T / (T + k × RTT)
```

- `T` is GPU time per step
- `k` is how many times in that step the host reads a value back from the device
- `RTT` is the network round trip

Function shipping runs the loop on the remote machine, so those reads are local there and `k × RTT` vanishes. Forwarding pays it on every one.

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

Other things that force a synchronization: `nonzero()`, boolean mask indexing, `unique()`, and mixture of experts routing, which reads per-expert token counts back to the host. Any operation whose output shape the host has to know is a synchronization.

### Measure it instead of guessing

```python
import torch

torch.cuda.set_sync_debug_mode("warn")
# run exactly one real training step and count the warnings
```

This does not depend on where the GPU is, so it can be measured on any CUDA device, including a Colab runtime. With `k` measured and the round trip measured, the formula gives you a real answer instead of an estimate.

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
@let.function(device=colab.G4, host="remote", host="local")
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

**Forward the calls** when the code is interactive and human paced, when the data has to stay on your machine, and when the machine is close. Exploratory notebook work is the honest use case: a person cannot feel 20 ms per cell, and the data stays where it already is.

**Never forward a generation loop.** Ship the whole `generate` call instead and you get the card's real speed.

## A note on what is implemented

Function shipping works today. Call forwarding is currently a capability probe: `letify.remoting.probe()` reports whether a layer 3 tunnel is possible, whether the letify-core is present and what the round trip is, and `require()` raises when forwarding would not pay off. The forwarding client itself is not written. See the known gaps at the end of [docs/SPEC.md](../SPEC.md).

---

[← Providers](02-providers.md) · [Guides](README.md) · [Next: Environments and data →](04-environments-and-data.md)
