<div align="center">

# ✨ letify

### Declarations that become infrastructure.

**Say what your function needs. Run it on the GPU you can afford.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-black)](LICENSE)
[![Providers](https://img.shields.io/badge/providers-Colab%20%7C%20Modal%20%7C%20SSH%20%7C%20Local-6C5CE7)](#-providers)
[![letify-core](https://img.shields.io/badge/letify--core-rust-DEA584?logo=rust&logoColor=white)](letify-core/)

[Quickstart](#-quickstart) · [Why](#-why-this-exists) · [Providers](#-providers) · [Sweeps](#-sweeps) · [Docs](docs/) · [한국어](docs/locales/README_ko.md)

</div>

---

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_a

@let.function(device=colab.G4, host="remote", concurrency=3)
def train(lr, bs):
    import torch
    ...
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

No session to create. No environment to install. No files to upload. No scope to open, and no machine left running when you close the lid. 🎉

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
@let.function(device=colab.G4, host="remote",
              volumes=[cache])
def train(lr, bs):
    ...

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

The Python package is pure Python. A provider whose package is missing simply reports itself unavailable, and the rest keeps working.

One optional piece is native. `host="local"` needs [letify-core](letify-core/), a Rust workspace that stands in for the CUDA driver, built with `python letify-core/build.py`. If you only ship functions to remote machines you never need it.

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
colab_a   colab   ephemeral   channel=persistent
lab_a100  shell   persistent  channel=persistent
local     local   persistent  channel=persistent

$ letify devices
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

@let.function(device=colab.G4, host="remote", env=env, volumes=[cache], concurrency=3)
def train(lr, bs):
    ...
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

That is the whole program. 🍰

---

## 🧭 What a declaration says

Three words, and none of them is a mechanism.

```python
@let.function(
    device=colab.G4,       # where the accelerator is, with provider and account
    host="remote",         # where the host code runs
    lifetime="call",       # how long the session lives
)
```

**`device`** carries the provider, the account and the accelerator in one value, because those are one decision. Core count and memory come with the shape the provider registered, so there is nothing to ask for.

**`host`** is the CUDA word for the CPU side. `"local"`, the default, keeps Python and the libraries in this process and forwards only CUDA calls. `"remote"` ships the function to the machine that holds the device.

**`lifetime`** is how long the session lives. `"call"`, the default, ends it with the call, counting a search space as one call. `"process"` keeps it so a run of separate calls does not pay session start each time.

<details>
<summary><b>📐 Which host to pick, with the arithmetic</b></summary>

| | 📦 `host="remote"` | 🔌 `host="local"` |
|---|---|---|
| What moves | your whole loop, once | every CUDA call |
| Cost | one transfer | one round trip per host synchronization |
| Fine-tuning at 150 ms | **~99%** | 53% default, ~96% tuned |
| Decoding at 150 ms | **hundreds of tok/s** | 2 to 7 tok/s |

Efficiency against a direct run is `T / (T + k × RTT)`, where `T` is GPU time per step and `k` is how many times per step the host reads a value back from the device.

A default Hugging Face training step has `k ≈ 3`: the trainer's NaN filter, the SDPA attention mask check, and logging. With a 0.5 s NVFP4 micro step at 150 ms that is 53%. Turn the NaN filter off, remove the mask check with fixed length packing, and log at the gradient accumulation boundary, and it is about 96%.

Counter-intuitive consequence: **a faster GPU makes forwarding worse**, because `T` shrinks and `RTT` does not. The same step on an L4 takes 1.8 s and reaches 80%.

Decoding is the case that stays bad. Throughput is bounded near `1000 / (k × RTT)` tokens per second, so the card stops mattering. Ship the whole `generate` call instead.

Measure your own `k` with `torch.cuda.set_sync_debug_mode("warn")` and your round trip with `letify probe`. See [docs/NETWORK.md](docs/NETWORK.md).

</details>

> ⚠️ letify **never** silently changes the mode. Ask for something a provider cannot serve and you get an exception naming the reason. Ask for something slow and you get a warning with the numbers, and then it runs, because the choice is yours.

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

@let.function(device=a.G4, host="remote")
def train(lr): ...

@let.function(device=b.L4, host="remote")      # different account, same program
def evaluate(ckpt): ...
```

**Or do not pick at all:**

```python
@let.function(device=let.providers.any.A100, host="remote")
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
@let.function(device=colab.G4, host="remote", concurrency=3)
async def train(lr, bs):
    ...

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

@let.function(device=colab.G4, host="remote", volumes=[cache])
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

## 🔗 Values that stay put

A session is one living process, so a value can stay in it.

```python
@let.function(device=colab.G4, host="remote", lifetime="process", keep_remote=True)
def build_model():
    return load_model()          # 14 GB, stays on the remote machine

@let.function(device=colab.G4, host="remote", lifetime="process")
def evaluate(model, batch):
    return model(batch)          # the handle resolves in place

model = build_model()            # a Handle, not 14 GB
evaluate(model=model, batch=...)
```

Large arguments are content addressed too. Pass the same tensor to ten calls and it crosses the network once, because the runtime is asked by digest whether it already holds it.

A handle names the session that holds it. Passing one to a different session raises rather than quietly copying the object across, since that would be an unrequested transfer of everything it points at.

---

## 💸 Your bill cannot run away

Nothing has to be torn down by hand.

**A call ends its own session.** That is the default, and a sweep counts as one call, so six points start one set of sessions and end them once.

**`lifetime="process"` is the opt in**, for a run of separate calls that would otherwise pay session start each time. The idle reaper takes it once it stops being used.

**The lease is the backstop.** The session holds a deadline that this process keeps renewing. Kill your script, lose your laptop, crash your kernel: the GPU shuts itself down. The grace period is long enough that a flaky connection does not kill a training run.

> 🚫 There is deliberately **no detached mode**. A detached run whose remote side gets preempted loses its results. Instead, the local process stays the owner, and durability comes from checkpoints in the store.

---

## 🧪 Test without a GPU, without mocks

The `local` provider starts the same worker behind the same framed protocol a remote runtime would. Your tests exercise the real path.

```python
def test_train_returns_a_loss():
    let = letify.Launcher(home=False)

    @let.function(device=let.providers.local.CPU, host="remote")
    def train(lr):
        return {"loss": 1.0 / lr}

    assert train(lr=2.0)["loss"] == 0.5
```

---

## 🛠️ CLI

```bash
letify providers              # who is declared, storage, channel kind
letify devices                # what each one offers
letify status                 # what is running right now
letify check lab              # does this machine answer?
letify probe lab              # is host="local" worth using here?
letify efficiency 0.5 3 150   # the formula, from measured terms
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
| 🦀 [letify-core/](letify-core/) | The Rust workspace behind `host="local"` |
| 🧭 [docs/guide/](docs/guide/) | Task-oriented guides |
| 🇰🇷 [docs/locales/README_ko.md](docs/locales/README_ko.md) | 한국어 |

---

## 🚧 Status

Alpha, and honest about it. What works today:

✅ Declarations, sync and async, sweeps, pooling, session lifetimes and the lease
✅ Persistent sessions: handles resolve in later calls, large arguments travel once
✅ Content addressed storage, configuration and secrets
✅ The `Local` and `Colab` providers
✅ `letify-core`, verified on a real GPU: the agent opens the driver, the local driver forwards an allocation and a copy in both directions, and the bytes match

Not finished yet:

🚧 `letify-driver` covers the entry points a PyTorch process needs to start up and run one kernel. Anything else names itself and returns `CUDA_ERROR_NOT_SUPPORTED`, so a real run prints the list of what to build next
🚧 `Modal` and `Elice` follow each published interface but have not been run against the live services
🚧 Unified memory cannot be forwarded at all, so a paged optimizer needs `host="remote"`

The full list is at the end of [docs/SPEC.md](docs/SPEC.md).

---

## 🤝 Contributing

Spec driven and test driven: settle [docs/SPEC.md](docs/SPEC.md), write the failing test, then write the code. Performance work uses [ResearchTree](https://darkpyonix.github.io/researchtree/), where one branch is one experiment and one pull request is its lab note. Read [CLAUDE.md](CLAUDE.md) before opening one.

---

<div align="center">

**MIT licensed.** Built for people who pay for their own GPUs. 🔬

</div>
