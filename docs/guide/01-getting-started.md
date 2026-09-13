# 1️⃣ Getting started

> From nothing to a function running on a remote GPU. About ten minutes.

[← Guides](README.md) · [Next: Providers →](02-providers.md)

---

## Install

```bash
uv add letify
```

This one install covers every provider. You also need [uv](https://docs.astral.sh/uv/) installed, because letify runs provider tools such as the Colab CLI through uv, outside your `.venv`.

You do not need a provider at all to follow this guide. The `local` provider always exists, so you can try the whole flow on your own machine first.

## Try it with no account

```python
import letify

let = letify.Launcher()

@let.function(device=let.providers.local.CPU, host="remote")
def add(a, b):
    return a + b

print(add(a=2, b=3))      # 5
```

That already went through the real path: the function was serialized, sent to a separate process, executed under the driver script, and its result decoded. Only the distance was missing.

## Declare an account

The easiest way is `letify login`. It writes the account to `~/.letify/config.toml` and names the alias in the project's `.letify/config.toml`.

```bash
letify login colab colab_a
```

Accounts live in `~/.letify/config.toml`, not in your repository, because they belong to your machine. The entry looks like this.

```toml
[colab_a]
kind = "colab"
account = "you@example.com"
```

For a lab server:

```toml
[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
```

> 🔐 Never write a token into `config.toml`. `letify login` stores it in `~/.letify/accounts/<alias>/`. Use `access_token_env = "MY_TOKEN"` to read it from an environment variable instead.

The project's `.letify/config.toml` names the accounts the project uses. An empty table is enough, and `letify login` writes it for you.

```toml
[colab_a]
[lab_a100]
```

An account whose home entry has `global = true` needs no project table, and neither does `local`.

The alias, `colab_a` here, has to be a Python identifier, because you reach providers by attribute. `colab-a` is rejected with a message telling you to use `colab_a`.

## Check that it works

```bash
$ letify providers
colab_a   colab   ephemeral   host=remote
lab_a100  shell   ephemeral   host=local
local     local   persistent  host=local

$ letify devices
{
  "colab_a": ["A100", "G4", "H100", "L4", "T4", "v5e1", "v6e1"],
  "lab_a100": ["A100"],
  "local": ["CPU"]
}
```

For an SSH machine, confirm it answers:

```bash
$ letify check lab_a100
Linux gpu 6.8.0 x86_64
NVIDIA A100-SXM4-80GB
```

## Your first remote function

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_a

@let.function(device=colab.G4, host="remote")
def check():
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
    }

print(check())
```

If `capability` comes back as `(12, 0)`, you have a Blackwell card, which is what `G4` is. That is the check to run before assuming NVFP4 will work.

## Add your environment

`Env` reads `uv.lock`, which resolves for every platform, so one lock file drives a Linux runtime from a Windows or macOS client.

```python
env = letify.Env()                        # uv.lock
env = env.pip_install("flash-attn")       # things the lock file does not carry
env = env.vars(HF_HOME="/opt/cache")      # environment variables in the runtime

@let.function(device=colab.G4, host="remote", env=env)
def train(lr, bs): ...
```

## Add a cache so restarts stop hurting

This is the single change that makes a short session usable.

```python
cache = colab.volume("hf-cache")

@let.function(device=colab.G4, host="remote", env=env, volumes=[cache])
def train(lr, bs): ...
```

A twenty gigabyte model cache takes 27 minutes from a lab server over a 100 Mbit/s link and 40 to 60 seconds from a bucket next to the runtime. All of that is billed as GPU time. See [Environments and data](04-environments-and-data.md).

## Run several configurations

```python
@let.function(device=colab.G4, host="remote", env=env)
async def train(lr, bs):
    ...
    return {"lr": lr, "bs": bs, "loss": loss}

configs = [dict(lr=lr, bs=bs) for lr in (1e-4, 3e-4, 1e-3) for bs in (16, 32)]

async def main():
    with let.keep_alive():
        for finished in asyncio.as_completed([train(**c) for c in configs]):
            print(await finished)

asyncio.run(main())
```

Each configuration is one call. The six calls run as wide as the account has cards for, which the provider entry declares and nothing on the declaration repeats, and `keep_alive` lets later calls reuse the sessions earlier ones started. Three cards means the six finish in roughly a third of the wall clock time. See [Concurrency and capacity](05-concurrency.md).

## What to know before going further

**A call needs nothing around it.** The session starts on the call and ends when the call finishes, so there is no scope to open and nothing to tear down. Wrap calls in `with let.keep_alive():` when a run of separate calls should share one session.

**Your script has to stay alive.** There is no detached mode. If the local process exits, the remote session shuts itself down within the lease grace period. For a long run, write checkpoints to a volume so a restart resumes. See [Cost control](06-cost.md).

**Colab needs a paid plan for accelerators,** and the remote control features letify uses are permitted while the compute unit balance is positive.

---

[← Guides](README.md) · [Next: Providers and accounts →](02-providers.md)
