# PROJECT

> The feature set and the public API of letify, as agreed. For the reasoning behind each decision see [docs/SPEC.md](docs/SPEC.md); for the vocabulary see [docs/COMPONENT.md](docs/COMPONENT.md).

## What letify is

A declarative infrastructure library for Python. You declare what a function needs and where it belongs, and letify arranges the session, the environment, the data and the teardown.

```python
import letify

let = letify.Launcher()
env = letify.Env()

colab = let.providers.colab_a

@let.function(gpu=colab.G4, env=env, concurrency=3)
def train(lr, bs):
    ...

with let.run():
    train(lr=1e-4, bs=32)
```

## Features

### Providers

Four kinds, three of them remote. A provider object is one account on one kind of infrastructure, declared once in a configuration file and reached by alias.

| Provider | Kind | Storage | Session created by |
|---|---|---|---|
| `Local` | `local` | persistent | a subprocess of this machine |
| `Modal` | `modal` | persistent | the Modal Python API |
| `Colab` | `colab` | ephemeral | `colab new` through the official CLI |
| `Shell` | `shell` | ephemeral, overridable | SSH |
| `Tunnel` | `tunnel` | ephemeral, overridable | Tailscale or frp, then SSH |
| `Elice` | `elice` | persistent | an allocation on the Elice Cloud API |

`Shell` is the base for every machine letify reaches by running commands on it. `Colab`, `Tunnel` and `Elice` are its subclasses and differ only in how the connection is obtained.

Multiple accounts are supported for every provider. Each configuration entry is one account, and several entries of the same kind coexist, which is how a user with more than one Colab account gets more concurrent sessions.

### Two execution modes, chosen for you

**Function shipping** sends the whole loop to the remote machine. **Call forwarding** keeps Python local and forwards only CUDA calls.

You never name a mode. You say where the CPU side runs with `cpu="remote"` or `cpu="local"` on the instance, and the default comes from whether the provider's storage outlives a runtime and whether the link is fast enough. Asking for a mode a provider cannot serve raises with the arithmetic in the message instead of running slowly.

### Runtime pooling

Runtimes are pooled by instance and environment, so the second call through a declaration pays nothing for provider boot, environment installation or the first data transfer. All three are billed as GPU time, which is the reason the pool exists.

### Three-layer lifetime

An explicit `with let.run():` scope, an idle timeout inside it, and a heartbeat lease that makes a session terminate itself if this process stops renewing. The lease has a grace period, so a brief network drop does not kill a run while a crashed script cannot leave a GPU billing.

There is no detached mode. The local process stays the owner and durability comes from checkpoints in the store.

### Content addressed storage

A volume is a blob store where contents are named by their hash and mutable names live in a separate namespace of refs. Concurrent sessions cannot overwrite each other, a transfer that already happened is skipped by name, and a tree of small files is packed into one archive so that thousands of round trips become one.

Backends are `filesystem`, `gcs`, `s3` and `modal`, chosen by the provider rather than stated in the declaration.

### Declared search spaces

`grid` and `zip` build a space; passing one where a scalar is expected declares that the argument varies. Results are consumed with the language's own protocols: `await` gives input order, `async for` gives completion order.

### Environment from a uv lock file

`Env` is keyed by the hash of `uv.lock`. One lock file resolves for every platform, so a Windows or macOS client drives a Linux runtime. Packages in the lock file are installed remotely by name; the project's own code travels with the call.

## Public API

Everything a user needs is on `letify` itself.

### `letify.Launcher`

```python
Launcher(
    config=None,        # path to a .letify file; defaults to the project and home files
    name=None,          # session and app name; defaults to the pyproject project name
    max_runtimes=3,     # how many sessions may exist at once
    idle_timeout=600,   # seconds a runtime may sit unused inside an open scope
    stream_logs=True,   # print remote stdout to stderr
    home=True,          # read ~/.letify
)
```

| Member | Does |
|---|---|
| `let.providers.<alias>` | Return the provider declared under that alias |
| `let.providers.any.<GPU>` | Request an accelerator without naming a provider |
| `let.providers.gpus` | Accelerators every provider offers |
| `let.providers.active` | Providers that currently hold a runtime |
| `let.provider(alias)` | Same as attribute access, for a computed alias |
| `let.function(...)` | Declare a function, returning a decorator |
| `let.run()` | Context manager in which sessions may exist |
| `let.runtime(instance, env)` | Start one runtime now instead of on first call |
| `let.grid`, `let.zip` | Build a search space |
| `let.reap_idle()` | Tear down runtimes past the idle timeout |
| `let.status()` | What is running, and for how long |

### `let.function`

```python
@let.function(
    gpu=colab.G4,       # an Instance, carrying provider, account, accelerator, placement
    env=env,            # an Env; defaults to Env()
    volumes=[cache],    # volumes to attach
    concurrency=3,      # runtimes this declaration may occupy at once
    timeout=3600,       # seconds one call may take
    retries=1,          # retries on infrastructure failure, never on user code failure
    keep_remote=False,  # return a Handle instead of the value
)
def train(lr, bs): ...
```

Sync or async is taken from the `def`. A plain `def` blocks; an `async def` returns an awaitable that is also async-iterable.

| Call form | Returns |
|---|---|
| `train(lr=1e-4)` on a `def` | the value |
| `train(space)` on a `def` | a list, in input order |
| `await train(lr=1e-4)` on an `async def` | the value |
| `await train(space)` | a list, in input order |
| `async for r in train(space)` | results as they complete |
| `train.local(lr=1e-4)` | runs the body in this process |

### `letify.Env`

```python
env = letify.Env()                       # uv.lock
env = letify.Env.from_lock("other.lock")
env = env.pip_install("flash-attn")      # packages the lock file does not carry
env = env.run("apt-get install -y git")  # commands after installation
env = env.vars(HF_HOME="/opt/cache")     # environment variables in the runtime
env = env.ship("mypkg")                  # send this module by value, overriding inference
```

### Instances

```python
colab.G4                          # registered shape
colab.G4(cpu="local", cpus=8)     # refined copy
colab.gpu("G4", cpus=8)           # same, for a computed name
colab.machines if hasattr(...)    # provider specific extras
let.providers.any.A100            # deferred provider choice
```

Colab accepts `T4`, `L4`, `G4`, `A100`, `H100`, plus the TPUs `v5e1` and `v6e1`. `G4` is the RTX PRO 6000 Blackwell part, and `RTX_PRO_6000` is accepted as an alias for it.

### Volumes

```python
cache = colab.volume("hf-cache")                    # backend from the provider
cache = colab.volume("hf-cache", bucket="my-bucket")

cache.cached_env(env)                # digest of a prebuilt environment archive
cache.cache_env(env, "/opt/venv")    # pack one and remember it
cache.put_checkpoint("run-1", path)  # store and move the ref
cache.fetch_checkpoint("run-1", target)
```

### Spaces

```python
letify.grid(lr=[1e-4, 3e-4], bs=[16, 32])   # 4 points
letify.zip(lr=[1e-4, 3e-4], bs=[16, 32])    # 2 points
letify.grid(lr=[1e-4]) | letify.grid(lr=[3e-4])
space.with_fixed(epochs=3)
```

### Errors

| Error | Means | Retryable |
|---|---|---|
| `ConfigError` | The configuration file is malformed or names an unknown kind | no |
| `ProviderUnavailable` | A provider's optional dependency or setting is missing | no |
| `UnknownProvider`, `UnknownInstance` | No such alias or accelerator | no |
| `NotRunning` | Called outside `with let.run():` | no |
| `RuntimeFailure`, `RuntimeLost` | The session misbehaved | yes |
| `ProtocolError` | The remote process died before producing a result | yes |
| `RemoteError` | The shipped function raised; carries the remote traceback | no |
| `HandleScopeError` | A handle from one runtime was passed to another | no |
| `UnsupportedMode` | The requested execution mode cannot work here | no |

### Command line

```bash
letify providers          # declared providers, persistence and default placement
letify gpus               # accelerators each provider offers
letify status             # open scopes and live runtimes
letify check <alias>      # confirm a machine answers
letify probe <host>       # whether call forwarding is worth using
```

## Configuration

`~/.letify` holds accounts, the project's `.letify` holds defaults, and neither holds a secret.

```toml
[defaults]
name = "nvfp4"
max_runtimes = 3

[colab_a]
kind = "colab"
account = "someone@example.com"

[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true

[elice_a100]
kind = "elice"
zone_id = "..."
machine_id = "..."
access_token_env = "ELICE_ACCESS_TOKEN"
```

An alias must be a Python identifier, because providers are reached by attribute. `any`, `gpus` and `active` are reserved. Declaration order sets the priority for `let.providers.any`.

A credential is referenced, never written: `<name>_env` names an environment variable, `<name>_keyring` names a keyring entry as `service/user`.

## Installation

```bash
uv add letify                # core only
uv add "letify[colab]"       # Google Colab
uv add "letify[modal]"       # Modal
uv add "letify[shell]"       # SSH, tunnel and Elice
uv add "letify[gcs]"         # Google Cloud Storage blob store
uv add "letify[s3]"          # S3 compatible blob store
uv add "letify[all]"         # everything
```

No provider dependency is imported at package import time, so a provider whose package is absent reports itself unavailable and everything else keeps working.

## What is not implemented yet

Stated plainly so nobody builds on a promise. The same list, with detail, is at the end of [docs/SPEC.md](docs/SPEC.md).

- The persistent session process is missing, so a `Handle` cannot yet be resolved by a later call.
- Call forwarding is a capability probe, not a client. The driver shim is not written.
- The Modal and Elice providers follow each service's published interface but have not been run against the live services.
- Volume materialization inside a runtime is a stub beyond unpacking a cached environment archive.
