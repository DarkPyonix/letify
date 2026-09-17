<div align="center">

# ✨ letify

### Declarations that become infrastructure.

**Say what your function needs. Run it on the GPU you can afford.**

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-black)](LICENSE)
[![Providers](https://img.shields.io/badge/providers-Colab%20%7C%20Modal%20%7C%20SSH%20%7C%20Local-6C5CE7)](#-providers)
[![letify-core](https://img.shields.io/badge/letify--core-rust-DEA584?logo=rust&logoColor=white)](letify-core/)

[Quickstart](#-quickstart) · [Why](#-why-declare-instead-of-connect) · [Providers](#-providers) · [Docs](docs/) · [한국어](docs/locales/README_ko.md)

</div>

---

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_pro_plus

@let.function(device=colab.G4, host=letify.remote)
def train(lr, bs):
    import torch
    loss = ...                          # your training loop
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

No session to create. No environment to install. No files to upload. Nothing to tear down, and no machine left running when you close the lid. 🎉

---

## 💡 Why declare instead of connect

Renting a GPU normally means going to it. You open a notebook or an SSH session, rebuild your
environment there, copy your data across, and work inside somebody else's machine for as long
as it lives. The training is the small part. The rest is infrastructure work that produces no
results.

Declaring inverts that. You say what a function needs and where it belongs; the session,
the environment, the transfer and the teardown are arranged for you.

### 🧠 You never leave the environment you were working in

Your editor, your debugger, your notes, your data and your git history stay where they are.
The declaration sends one function to the card and brings the result back, so the remote GPU
is a detail of one function rather than a place you move into.

That continuity is the point. You are not two people, one of whom lives in a browser tab with
a different Python and no working directory. There is nothing to keep in sync and nothing to
copy back before the session dies.

```python
@let.function(device=colab.G4, host=letify.remote)
def train(lr, bs):
    ...

train(lr=1e-4, bs=32)     # the same file you were already editing
```

### ⚡ There is no infrastructure step

Nothing is turned on, and nothing is left running by accident. A call starts the session it
needs and ends it when the work is done, so forgetting to stop a GPU is not a mistake you can
make. Neither is installing packages: the environment comes from the `uv.lock` you already
have, cached so the second session does not pay for it.

The failure mode this removes is the expensive one. A forgotten instance bills overnight, and
a session you spent twenty minutes preparing dies with everything in it.

### 📈 Nothing is tied to one server

A declaration names an accelerator shape, not a machine. So the same code runs on a Colab
runtime, a lab box over SSH, an Elice allocation or this laptop by changing one line of
configuration, and when one account runs out you add another rather than rewriting anything.

That is also how the work scales sideways. Capacity is what the provider entry declares it
has, so calls made at the same time spread across every card available to you, across
accounts and across machines:

```toml
[colab_pro.devices]
G4 = { count = 2 }            # two sessions on this account

[lab_a100.devices]
A100 = { indices = "0-3" }    # four cards in the shared box are ours
```

Six configurations then run six ways at once if six cards exist, on hardware that never had
to be the same hardware. Nothing about the declaration changes when the pool grows.

### 💸 What that makes affordable

The same card costs wildly different amounts depending on which door you rent it through.

| Same card, different door | Per hour |
|---|---|
| 🥇 Colab credits | **~975 KRW** |
| 💸 Modal | ~4,070 KRW |

A factor of four for identical silicon. The cheap door is a notebook, though: no persistent
disk, eviction at any moment, and whichever accelerator happens to be free. Everything above
is what makes that door usable, so the cheapest option stops being the inconvenient one.

<table>
<tr><th width="50%">😖 Going to the GPU</th><th width="50%">😌 Declaring it</th></tr>
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
@let.function(device=colab.G4, host=letify.remote,
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
uv add letify
```

That is the whole install, for every provider. letify installs only cloudpickle and blake3. It needs [uv](https://docs.astral.sh/uv/) on the machine, because provider tools such as the Colab CLI and the Modal client run through uv in their own environments, not in your `.venv`.

The wheel carries [letify-core](letify-core/), the Rust component behind `host=letify.local` that stands in for the CUDA driver. Wheels are built for Linux x86_64 and aarch64, Windows x86_64 and arm64, and macOS arm64 and x86_64. On another platform, pip falls back to the source distribution, which has no letify-core, and `host=letify.local` is refused.

---

## 🚀 Quickstart

### 1. Declare your accounts once

Run `letify login` once per account, from your project directory.

```bash
letify login colab colab_pro_plus
letify login shell lab_a100
```

Each command writes two files. The account goes to `~/.letify/config.toml`, which belongs to your machine and never to the repository:

```toml
[colab_pro_plus]
kind = "colab"
account = "you@example.com"

[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true
```

The project's `.letify/config.toml` gets only the alias, which is what makes the account usable in this project:

```toml
[colab_pro_plus]

[lab_a100]
```

An account with `global = true` in the home file needs no project line, and neither does `local`.

> 🔐 Secrets never go in `config.toml`. A field such as `access_token` comes from the environment variable named by `access_token_env`, or from the file `~/.letify/accounts/<alias>/access_token` that `letify login` writes with owner only permissions.

### 2. Look around

```bash
$ letify providers
colab_pro_plus       colab      ephemeral
lab_a100             shell      persistent
local                local      persistent
```

### 3. Declare and run

```python
import letify

let = letify.Launcher()
env = letify.Env()                      # reads uv.lock
colab = let.providers.colab_pro_plus

@let.function(device=colab.G4, host=letify.remote, env=env)
def train(lr, bs):
    loss = ...                          # your training loop
    return {"loss": loss}

print(train(lr=1e-4, bs=32))
```

That is the whole program. 🍰

---

## 🧭 What a declaration says

Two words, and neither of them is a mechanism.

```python
@let.function(
    device=colab.G4,       # where the accelerator is, with provider and account
    host=letify.remote,    # where the host code runs
)
```

**`device`** carries the provider, the account and the accelerator in one value, because those are one decision. Core count and memory come with the shape the provider registered, so there is nothing to ask for.

**`host`** is the CUDA word for the CPU side. `letify.local` keeps Python and the libraries in this process and forwards only CUDA calls. `letify.remote` ships the function to the machine that holds the device. The strings `"local"` and `"remote"` are the same values.

Leave `host` out and you get `letify.local`. When the device is far away, forwarding every CUDA call is slow, so the call warns with the expected efficiency and then runs. For a remote GPU, `host=letify.remote` is usually what you want.

How long a session lives is not a declaration argument. A call ends its session. `with let.keep_alive():` keeps sessions for the length of a block, so a run of separate calls does not pay session start each time.

<details>
<summary><b>📐 Which host to pick, with the arithmetic</b></summary>

| | 📦 `host=letify.remote` | 🔌 `host=letify.local` |
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
├── 🏅 Kaggle     Kaggle account          ephemeral, host=remote only
└── 🐚 Shell      any remote machine      ephemeral by default
    ├── 📓 Colab   via the official CLI
    ├── 🕳️  Tunnel  a machine behind NAT
    └── 🇰🇷 Elice   Elice Cloud, allocated by API
```

| | Storage | Best for |
|---|---|---|
| 💻 `Local` | persistent | your own GPU, and testing everything else |
| ☁️ `Modal` | persistent | production serving, reproducible images |
| 📓 `Colab` | ephemeral | cheap batch work, NVFP4 on `G4` |
| 🐚 `Shell` | overridable | lab and university servers |
| 🕳️ `Tunnel` | overridable | a machine behind NAT you cannot port-forward |
| 🇰🇷 `Elice` | persistent | Korean GPU cloud, per-second billing |
| 🏅 `Kaggle` | ephemeral | free weekly GPU hours, `host=letify.remote` only |

**Kaggle.** Make an API token at kaggle.com under Settings, API, then run `letify login kaggle kaggle_a`. The token is asked for without echo, and a `kaggle.json` path works too. letify keeps it in `~/.letify/accounts/kaggle_a/` and checks it with a read-only `kaggle quota` call. `letify usage kaggle_a` prints the GPU hours left this week. A Kaggle declaration must say `host=letify.remote`: Kaggle forbids tunnels, so `host=letify.local` is a type error and raises when the decorator runs.

One more step, and it is the only part of Kaggle that needs a person: start a session in the Kaggle editor with Run, Kaggle Jupyter Server, copy its Colab Compatible URL, and run `letify login kaggle kaggle_a --connect '<URL>'`. That URL exists only in the editor. Kaggle's API can start and stop a session, but it returns no address for one, and the proxy that fronts a session accepts only the token the editor issues. A call on an account without one is refused and tells you this.

After that, Kaggle is like any other provider: letify builds your declared environment on the session with `uv sync`, enters a workspace root, ships files, and checks the worker's interpreter. It records the session's GPUs and runs calls there until Kaggle ends the session, after 20 minutes idle or 12 hours. Then it raises an error telling you to register a new URL, and it never keeps a session alive.

**letify finds the fastest way in.** For any `Shell`, letify tries several ways to reach the machine at once and keeps the fastest one that works:

1. SSH straight to the machine's address
2. TCP hole punching, for two machines that are both behind NAT
3. UDP hole punching with [Tailcat](https://github.com/tailscale/tailcat), then SSH over it
4. The provider's own path, such as `colab exec` and the Colab file API

A lower number wins unless it is far slower than the fastest one that connected. The winner is remembered per account and per network, so the next connection starts with it. A plain machine behind NAT needs letify installed and `letify client shell connect` run on it once, so letify can reach it. When that machine's SSH server is also published to the outside under another port, such as Docker's `-p 30501:8022`, run `letify client shell connect --ssh-port 8022 --public-address <host> --public-port 30501`: SSH straight to the address then dials port 30501, while hole punching and Tailcat keep using 8022. On an existing account, `letify login tunnel <alias> --address <host> --public-port 30501` sets the same. Colab and Elice never need that step: their provider layer creates and opens the machine through the provider's own API, so that layer takes the place of `letify client shell connect`. Modal is reached through its own API and is not part of this.

**Multiple accounts are first class.** Each configuration entry is one account, and entries of the same kind coexist. Two Colab accounts means twice the concurrent sessions.

```python
a = let.providers.colab_pro_plus
b = let.providers.colab_pro

@let.function(device=a.G4, host=letify.remote)
def train(lr): ...

@let.function(device=b.L4, host=letify.remote)      # different account, same program
def evaluate(ckpt): ...
```

**Or do not pick at all:**

```python
@let.function(device=let.providers.any.A100, host=letify.remote)   # the first declared provider with an A100
def train(lr): ...
```

---

## ⚡ Many calls at once

Declare the function with `async def`, and a call gives you a coroutine. Run as many as you like with `asyncio.gather`, inside `with let.keep_alive():` so they share sessions instead of each starting its own.

```python
import asyncio

@let.function(device=colab.G4, host=letify.remote)
async def train(lr, bs):
    loss = ...                          # your training loop
    return {"loss": loss}

async def main():
    with let.keep_alive():
        return await asyncio.gather(*(
            train(lr=lr, bs=bs) for lr in (1e-4, 3e-4, 1e-3) for bs in (16, 32)
        ))

results = asyncio.run(main())           # six results, in the order they were asked for
```

The calls run on as many cards as the account declares, and a call waits for a card when all of them are busy. Nothing on the declaration sets the width.

> 🧵 **Sync or async is declared at the `def`, not at the call.** A plain `def` blocks and returns its value. An `async def` returns a coroutine, so `await`, `asyncio.gather` and `asyncio.as_completed` work exactly as they always do. letify adds no future type of its own, and there is no `.remote()`, `.spawn()` or `.map()` to remember.

---

## 💾 Caching that actually helps

A **volume** is a content addressed blob store. Contents are named by their hash, and mutable names live in a separate tiny namespace, exactly like Git objects and refs.

```python
project = colab.volume("my-project")    # a copy of what this project needs, kept in a bucket

@let.function(device=colab.G4, host=letify.remote, volumes=[project])
def train(lr): ...
```

The runtime pulls the volume straight from the bucket, not through your machine. It uses a short-lived token borrowed from your own login, so no credential is left on the remote side.

Training data needs no declaration at all. Pass a `pathlib.Path`, or read one from a global, and letify sends the files it names as content addressed blobs. The body receives a path on the runtime with the same layout. A persistent machine keeps the blobs on its own disk, so the second session uploads 0 bytes. An ephemeral account with `bucket = "<name>"` uploads each file to the bucket once, and every later runtime downloads it from there. The runtime's copy is kept within a budget, 50 GiB by default or `data_cache_gib` on the account, and `letify cache` shows or clears it. The call does not wait for the dataset: letify derives the read order from the call itself, sends only the first wave of that order before the call, and keeps sending the rest while the call runs, reordered by what the runtime observes the body reading. Listings and file sizes are answered from the manifest from the first step, so only reading a file that has not arrived waits.

Results come back the same way. A `Path` that does not exist yet, or a directory, is also an output location: what the body creates or changes there is copied to the local path when the call returns, and a file the local copy already matches is not sent. So checkpoints and logs saved under `Path("runs/exp1")` are in your project after the call.

```python
DATA = Path("data/imagenet-subset")

@let.function(device=lab.A100, host=letify.remote)
def train(lr):
    for file in DATA.iterdir(): ...   # already on the runtime's disk
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

A session is one living process, so a value can stay in it. `letify.session_cache` builds a value the first time a session asks for it and hands back the same object on every later call in that session.

```python
def load_model():
    return ...                   # 14 GB, loaded once per session

@let.function(device=colab.G4, host=letify.remote)
def evaluate(batch):
    model = letify.session_cache("model", load_model)
    return model(batch)

with let.keep_alive():               # the session outlives each call
    evaluate(batch=first)            # loads the model
    evaluate(batch=second)           # reuses it
```

The value lives until its session ends. Every session keeps its own, so it does not matter which session a call lands on, and calls made at the same time on two cards each load one copy. Outside a runtime, for example when you test the body locally, it is an ordinary in-process cache with the same behaviour.

A plain global dictionary in your script does not do this. The function is sent to the runtime with copies of the script's globals on every call, so such a cache starts empty each time.

Large arguments are content addressed. Pass the same tensor to ten calls and it crosses the network once, because the runtime is asked by digest whether it already holds it.

---

## 💸 Your bill cannot run away

Nothing has to be torn down by hand.

**A call ends its own session.** That is the default.

**`with let.keep_alive():` is the opt in**, for a run of separate calls that would otherwise pay session start each time. Calls made at the same time inside it, for example with `asyncio.gather`, run on as many cards as the account declares and wait for one when all are busy. Leaving the block ends every idle session. Nothing ends one on a timer.

**Devices that cannot be allocated raise.** A call that asks for cards held by an idle kept session, by another process or beyond what the account declares raises `letify.InsufficientDevices` at once, instead of waiting for a device nothing will free.

**The lease is the backstop.** The session holds a deadline that this process keeps renewing. Kill your script, lose your laptop, crash your kernel, and the worker exits on its own, which frees the card. The grace period is long enough that a flaky connection does not kill a training run. Whether it also stops the billing depends on what the provider charges for: [docs/guide/06-cost.md](docs/guide/06-cost.md) says which providers are covered and which are not.

> 🚫 There is deliberately **no detached mode**. A detached run whose remote side gets preempted loses its results. Instead, the local process stays the owner, and durability comes from checkpoints in the store.

---

## 🧪 Test without a GPU, without mocks

The `local` provider starts the same worker behind the same framed protocol a remote runtime would. Your tests exercise the real path.

```python
def test_train_returns_a_loss():
    let = letify.Launcher(home=False)

    @let.function(device=let.providers.local.CPU, host=letify.remote)
    def train(lr):
        return {"loss": 1.0 / lr}

    assert train(lr=2.0)["loss"] == 0.5
```

---

## 🛠️ CLI

```bash
letify login shell lab        # declare an account, and reference it here
letify logout lab             # take the account off this machine
letify client shell connect   # run on a remote machine behind NAT, so letify can reach it
letify setup tailcat          # install tailcat or eci from its publisher ahead of time
letify providers              # who is declared, storage, channel kind
letify devices                # what each one offers
letify status                 # what is running right now
letify usage                  # what is left on each account
letify utilization            # how busy each instance's GPU is
letify check lab              # does this machine answer?
letify probe lab              # is host=letify.local worth using here?
letify efficiency 0.5 3 150   # the formula, from measured terms
```

Add `--json` to `usage`, `utilization` or `status` for output a program can read. The VS Code extension in [letify-ext/](letify-ext/) uses it to show quota and GPU activity in the status bar.

---

## 📚 Documentation

| | |
|---|---|
| 🧪 [examples/](examples/) | Working scenarios, starting with LoRA runs on a rented card |
| 📖 [PROJECT.md](PROJECT.md) | The full feature set and API surface |
| 🎯 [docs/INTENT.md](docs/INTENT.md) | Goals, claims, constraints, open questions |
| 📐 [docs/SPEC.md](docs/SPEC.md) | The design as it stands, decision by decision |
| 🧩 [docs/COMPONENT.md](docs/COMPONENT.md) | Every class, and the vocabulary |
| 🌐 [docs/NETWORK.md](docs/NETWORK.md) | Transports, latency measurements, tunnel choices |
| 🦀 [letify-core/](letify-core/) | The Rust workspace behind `host=letify.local` |
| 🧭 [docs/guide/](docs/guide/) | Task-oriented guides |
| 🇰🇷 [docs/locales/README_ko.md](docs/locales/README_ko.md) | 한국어 |

---

## 🚧 Status

Alpha, and honest about it. What works today:

✅ Declarations, sync and async, pooling, session lifetimes and the lease
✅ Persistent sessions: a session cache keeps values between calls, large arguments travel once
✅ Content addressed storage, configuration and secrets
✅ The `Local` and `Colab` providers
✅ `letify-core`, verified on a real GPU: the agent opens the driver, the local driver forwards an allocation and a copy in both directions, and the bytes match

Not finished yet:

🚧 `letify-driver` covers the entry points a PyTorch process needs to start up and run one kernel. Anything else names itself and returns `CUDA_ERROR_NOT_SUPPORTED`, so a real run prints the list of what to build next
🚧 `Modal` and `Elice` follow each published interface but have not been run against the live services
🚧 Unified memory cannot be forwarded at all, so a paged optimizer needs `host=letify.remote`

The full list is at the end of [docs/SPEC.md](docs/SPEC.md).

---

## 🤝 Contributing

Spec driven and test driven: settle [docs/SPEC.md](docs/SPEC.md), write the failing test, then write the code. Performance work uses [ResearchTree](https://darkpyonix.github.io/researchtree/), where one branch is one experiment and one pull request is its lab note. Read [CLAUDE.md](CLAUDE.md) before opening one.

---

<div align="center">

**Apache 2.0 licensed.** Built for people who pay for their own GPUs. 🔬

</div>
