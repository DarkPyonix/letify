<div align="center">

# ✨ letify

### Declarations that become infrastructure.

**Say what your function needs. Run it on the GPU you can afford.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-black)](LICENSE)
[![Providers](https://img.shields.io/badge/providers-Colab%20%7C%20Modal%20%7C%20SSH%20%7C%20Local-6C5CE7)](#-providers)
[![Pure Python](https://img.shields.io/badge/pure-python-2ECC71)](pyproject.toml)

[Quickstart](#-quickstart) · [Why](#-why-this-exists) · [Providers](#-providers) · [Sweeps](#-sweeps) · [Docs](docs/) · [한국어](docs/locales/README_ko.md)

</div>

---

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_a

@let.function(gpu=colab.G4, concurrency=3)
def train(lr, bs):
    import torch
    ...
    return {"loss": loss}

with let.run():
    print(train(lr=1e-4, bs=32))
```

No session to create. No environment to install. No files to upload. No machine left running when you close the lid. 🎉

---

## 💡 Why this exists

The same GPU costs wildly different amounts depending on how you rent it.

| Same card, different door | Per hour |
|---|---|
| 🥇 Colab credits | **~975 KRW** |
| 💸 Modal | ~4,070 KRW |

That is a factor of four for identical silicon. For anyone paying out of pocket, it decides how many experiments get run.

The catch is that the cheap door is a notebook: no persistent disk, eviction at any moment, and whichever accelerator happens to be free. So you spend your session re-installing packages, re-downloading weights, and babysitting a browser tab.

**letify makes the cheap door behave like the expensive one.** You declare; it handles sessions, environments, caching and teardown.

<table>
<tr><th width="50%">😖 Without letify</th><th width="50%">😌 With letify</th></tr>
<tr valign="top"><td>

```python
# open a browser, pick a GPU, hope it's free
!pip install -q torch transformers  # 4 min
!gdown ...                           # 18 min
!git clone https://github.com/me/repo
%cd repo
# ... finally start working
# ... session dies, start over
```

</td><td>

```python
@let.function(gpu=colab.G4, volumes=[cache])
def train(lr, bs):
    ...

with let.run():
    train(lr=1e-4, bs=32)
```

</td></tr>
</table>

---

## 📦 Installation

```bash
uv add letify                 # core, no provider dependencies
uv add "letify[colab]"        # Google Colab
uv add "letify[modal]"        # Modal
uv add "letify[shell]"        # SSH, tunnels, Elice Cloud
uv add "letify[gcs]"          # Google Cloud Storage cache
uv add "letify[s3]"           # S3 compatible cache
uv add "letify[all]"          # everything
```

Pure Python. No compiled extension, no wheel to build, no toolchain. A provider whose package is missing simply reports itself unavailable, and the rest keeps working.

---

## 🚀 Quickstart

### 1. Declare your accounts once

Put accounts in `~/.letify`, so they belong to your machine and never to the repository.

```toml
[colab_a]
kind = "colab"
account = "you@example.com"

[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true
```

> 🔐 Secrets are referenced, never written. Use `access_token_env = "MY_TOKEN"` or `access_token_keyring = "service/user"`.

### 2. Look around

```bash
$ letify providers
colab_a   colab   ephemeral   cpu=remote
lab_a100  shell   persistent  cpu=remote
local     local   persistent  cpu=local

$ letify gpus
{
  "colab_a":  ["A100", "G4", "H100", "L4", "T4", "v5e1", "v6e1"],
  "lab_a100": ["A100"],
  "local":    ["CPU", "GeForce_RTX_4050"]
}
```

### 3. Declare and run

```python
import letify

let = letify.Launcher()
env = letify.Env()                      # reads uv.lock
colab = let.providers.colab_a
cache = colab.volume("hf-cache")        # survives the session

@let.function(gpu=colab.G4, env=env, volumes=[cache], concurrency=3)
def train(lr, bs):
    ...
    return {"loss": loss}

with let.run():
    print(train(lr=1e-4, bs=32))
```

That is the whole program. 🍰

---

## 🧭 The one idea

**You declare resources. letify picks the mechanism.**

There are two ways to use a remote GPU, and they have wildly different performance depending on where you and the GPU are.

| | 📦 Function shipping | 🔌 CUDA call forwarding |
|---|---|---|
| What moves | your whole loop, once | every CUDA call, always |
| Cost | one transfer | one round trip per host sync |
| Fine-tuning at 150 ms | **~99%** | 50 to 57% |
| Decoding at 150 ms | **hundreds of tok/s** | 2 to 7 tok/s |

You never name either one. You say where the CPU side of the work lives:

```python
@let.function(gpu=colab.G4)                   # default: loop runs remotely
@let.function(gpu=lab.A100(cpu="local"))      # Python stays here, CUDA calls go there
```

And the default comes from your provider, because **storage decides**. If a provider's disk outlives a session, your data is already there and shipping the loop is natural. If it does not, keeping state local makes more sense, but only when the link is fast enough to afford it.

> ⚠️ letify **never** silently falls back to the slower mode. Ask for something a provider cannot serve and you get an exception with the arithmetic in the message, not a run that mysteriously takes four times as long.

<details>
<summary><b>📐 The formula, if you like formulas</b></summary>

Efficiency against running directly on the machine is:

```
efficiency = T / (T + k × RTT)
```

where `T` is GPU time per step and `k` is how many times per step the host reads a value back from the device.

A default Hugging Face training step has `k ≈ 3`: the trainer's NaN filter, the attention mask check, and logging. With a 0.5 s NVFP4 micro step at 150 ms round trip, that is 53%.

Counter-intuitive consequence: **a faster GPU makes forwarding worse**, because `T` shrinks and `RTT` does not. The same step on an L4 takes 1.8 s and reaches 80%.

Measure your own `k` with `torch.cuda.set_sync_debug_mode("warn")`. See [docs/NETWORK.md](docs/NETWORK.md).

</details>

---

## 🌍 Providers

```
Provider
├── 💻 Local      your machine            persistent
├── ☁️  Modal      serverless GPU          persistent
└── 🐚 Shell      any machine over SSH    ephemeral by default
    ├── 📓 Colab   via the official CLI
    ├── 🕳️  Tunnel  Tailscale or frp, for NAT
    └── 🇰🇷 Elice   Elice Cloud, allocated by API
```

| | Storage | Best for |
|---|---|---|
| 💻 `Local` | persistent | your own GPU, and testing everything else |
| ☁️ `Modal` | persistent | production serving, reproducible images |
| 📓 `Colab` | ephemeral | cheap batch work, sweeps, NVFP4 on `G4` |
| 🐚 `Shell` | overridable | lab and university servers |
| 🕳️ `Tunnel` | overridable | a machine behind NAT you cannot port-forward |
| 🇰🇷 `Elice` | persistent | Korean GPU cloud, per-second billing |

**Multiple accounts are first class.** Each configuration entry is one account, and entries of the same kind coexist. Two Colab accounts means twice the concurrent sessions.

```python
a = let.providers.colab_a
b = let.providers.colab_b

@let.function(gpu=a.G4)
def train(lr): ...

@let.function(gpu=b.L4)      # different account, same program
def evaluate(ckpt): ...
```

**Or do not pick at all:**

```python
@let.function(gpu=let.providers.any.A100)   # first declared provider that has one
def train(lr): ...
```

---

## 🔭 Sweeps

Fan-out is a **declared space**, not a `.map()` call. Passing a space where a scalar is expected says that argument varies.

```python
space = letify.grid(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32])   # 6 points
pairs = letify.zip(lr=[1e-4, 3e-4], bs=[16, 32])          # 2 points
both  = letify.grid(lr=[1e-4]) | letify.grid(lr=[1e-3])   # union
```

Then consume it with the language you already know. 🐍

```python
@let.function(gpu=colab.G4, concurrency=3)
async def train(lr, bs):
    ...

with let.run():
    results = await train(space)              # list, in input order

    async for r in train(space):              # streamed, as each finishes
        print(r)
```

> 🧵 **Sync or async is declared at the `def`, not at the call.** A plain `def` blocks. An `async def` gives you a coroutine, so `await` and `asyncio.gather` work exactly as they always do. letify adds no future type of its own, and there is no `.remote()`, `.spawn()` or `.map()` to remember.

---

## 💾 Caching that actually helps

A **volume** is a content addressed blob store. Contents are named by their hash, and mutable names live in a separate tiny namespace, exactly like Git objects and refs.

```python
cache = colab.volume("hf-cache")

@let.function(gpu=colab.G4, volumes=[cache])
def train(lr): ...
```

Why this shape:

| | 🐌 Two-way file sync | ⚡ Content addressed |
|---|---|---|
| Concurrent writers | last one wins, work is lost | cannot collide, by construction |
| Already transferred? | compare size and timestamps | holding the hash **is** the proof |
| 50k small files | 50k round trips | one packed archive, one transfer |

The payoff is where it hurts most, which is session start:

| Pulling a 20 GB model cache | Time |
|---|---|
| 🐢 From a lab server over 100 Mbit/s | ~27 min |
| 🚶 From the Hugging Face hub | 3 to 5 min |
| 🚀 From a bucket next to the runtime | **40 to 60 s** |

All of that time is billed as GPU time. That is why an ephemeral provider with a volume attached behaves like a persistent one.

---

## 💸 Your bill cannot run away

Three layers, and you only have to think about the first.

```python
with let.run():          # 1️⃣ leaving this tears every session down
    train(lr=1e-4)
```

2️⃣ **Idle timeout.** A session nobody uses inside an open scope is torn down anyway.

3️⃣ **Heartbeat lease.** The session holds a deadline that this process keeps renewing. Kill your script, lose your laptop, crash your kernel: the GPU shuts itself down. The grace period is long enough that a flaky connection does not kill a training run.

> 🚫 There is deliberately **no detached mode**. A detached run whose remote side gets preempted loses its results. Instead, the local process stays the owner, and durability comes from checkpoints in the store.

---

## 🧪 Test without a GPU, without mocks

The `local` provider runs the same serialized call through the same driver script a remote runtime would. Your tests exercise the real path.

```python
def test_train_returns_a_loss():
    let = letify.Launcher(home=False)

    @let.function(gpu=let.providers.local.CPU)
    def train(lr):
        return {"loss": 1.0 / lr}

    with let.run():
        assert train(lr=2.0)["loss"] == 0.5
```

---

## 🛠️ CLI

```bash
letify providers      # who is declared, persistence, default placement
letify gpus           # what each one offers
letify status         # what is running right now
letify check lab      # does this machine answer?
letify probe lab      # is it close enough for call forwarding?
```

---

## 📚 Documentation

| | |
|---|---|
| 📖 [PROJECT.md](PROJECT.md) | The full feature set and API surface |
| 🎯 [docs/INTENT.md](docs/INTENT.md) | Goals, claims, constraints, open questions |
| 📐 [docs/SPEC.md](docs/SPEC.md) | The design as it stands, decision by decision |
| 🧩 [docs/COMPONENT.md](docs/COMPONENT.md) | Every class, and the vocabulary |
| 🌐 [docs/NETWORK.md](docs/NETWORK.md) | Transports, latency measurements, tunnel choices |
| 🧭 [docs/guide/](docs/guide/) | Task-oriented guides |
| 🇰🇷 [docs/locales/README_ko.md](docs/locales/README_ko.md) | 한국어 |

---

## 🚧 Status

Alpha, and honest about it. What works today:

✅ Declarations, sync and async, sweeps, pooling, scopes and the lease
✅ The call protocol, content addressed storage, configuration and secrets
✅ The `Local` and `Colab` providers, with 31 tests over the real code path

Not finished yet:

🚧 The persistent session process, so a `Handle` cannot yet be resolved by a later call
🚧 CUDA call forwarding, which is currently a capability probe rather than a client
🚧 `Modal` and `Elice`, written to each published interface but not yet run live

The full list is at the end of [docs/SPEC.md](docs/SPEC.md).

---

## 🤝 Contributing

Performance work in this repository uses [ResearchTree](https://darkpyonix.github.io/researchtree/): one branch is one experiment, one pull request is its lab note. Read [CLAUDE.md](CLAUDE.md) before opening one.

---

<div align="center">

**MIT licensed.** Built for people who pay for their own GPUs. 🔬

</div>
