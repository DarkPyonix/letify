# PROJECT

> The feature set and the public API of letify. For why each decision is what it is, see [docs/SPEC.md](docs/SPEC.md); for the vocabulary, see [docs/COMPONENT.md](docs/COMPONENT.md).

## What letify is

A declarative infrastructure library for Python. You declare what a function needs and where it belongs, and letify arranges the session, the environment, the data and the teardown.

```python
import letify

let = letify.Launcher()
colab = let.providers.colab_a

@let.function(device=colab.G4, host="remote")
def train(lr, bs):
    ...

train(lr=1e-4, bs=32)
```

## Features

### Placements, no mechanisms

A declaration says where the accelerator is and where the host code runs. It never names a transport or a mode. How long a session lives is the `keep_alive` block around the calls.

| Argument | Says | Default |
|---|---|---|
| `device` | which provider, account, accelerator and how many of it | required |
| `host` | where the host code runs: `"local"` or `"remote"` | `"local"` |

Core count and memory are not arguments. They arrive with the shape the provider registered, and a provider offering several sizes registers them as separate shapes. `lab.A100 * 2` is the same shape taking two cards.

There is no width argument and no session ceiling. A provider entry declares what the account has, and that inventory is the only thing bounding how much runs at once:

```toml
[lab_a100.devices]
A100 = { indices = "0-3" }   # four cards in a shared box are ours

[colab_pro.devices]
G4 = { count = 2 }           # two concurrent sessions on this account
```

`indices` is for a machine letify shares with other people: it takes only those registered indices that are actually free when a session starts, and sets the session's visible devices so the code inside sees its cards as 0 upward. `count` is for a provider that assigns the device itself.

### Providers

Four kinds, three of them remote. A provider object is one account on one kind of infrastructure, declared once in a configuration file and reached by alias.

| Provider | Kind | Storage | Session created by |
|---|---|---|---|
| `Local` | `local` | persistent | a subprocess of this machine |
| `Modal` | `modal` | persistent | a Modal sandbox |
| `Colab` | `colab` | ephemeral | `colab new` through the official CLI |
| `Shell` | `shell` | ephemeral, overridable | SSH |
| `Tunnel` | `tunnel` | ephemeral, overridable | Tailscale or frp, then SSH |
| `Elice` | `elice` | persistent | an allocation on the Elice Cloud API |

`Shell` is the base for every machine letify reaches by running commands on it. `Colab`, `Tunnel` and `Elice` are its subclasses and differ only in how the connection is obtained.

Multiple accounts are supported for every provider. Each configuration entry is one account, and several entries of the same kind coexist, which is how a user with more than one Colab account gets more concurrent sessions.

### Persistent sessions

A session is one worker process kept alive behind a framed pipe, which is what makes three things work.

A value a declared body stores with `letify.session_cache("model", load_model)` stays in the session's worker process, so a later call in the same session gets it without building it again. Each session has its own cache, and a one-shot channel keeps nothing between calls. A large argument is hashed and offered by digest, so the same tensor passed to ten calls crosses the network once. Files written into a session survive between calls, so a volume can materialize an environment archive or a checkpoint.

### Session pooling

Runtimes are pooled by instance and environment, so two declarations that agree on both share a session with nothing said about it. Provider boot, environment installation and the first data transfer are all billed as GPU time, which is the reason the pool exists.

### Teardown with nothing to call

A call ends its own session. `with let.keep_alive():` keeps sessions for the length of a block, and they end when the outermost block exits. Nothing ends a session on a timer, because a timer would overrule the block that asked to keep it.

A call whose devices cannot be allocated raises `letify.InsufficientDevices` at once instead of waiting. It waits only while a session in this process that holds that accelerator is serving a call.

A heartbeat lease covers the one case nothing else can: a process killed outright says nothing to anybody. The worker exits on its own, which frees the card. Whether that also stops the billing depends on what the provider charges for, and letify states that per provider rather than implying a guarantee it cannot make. See [docs/SPEC.md](docs/SPEC.md).

There is no release call, no shutdown call and no detached mode. Durability comes from checkpoints in the store rather than from a session that outlives you.

### Content addressed storage

A volume is a blob store where contents are named by their hash and mutable names live in a separate namespace of refs. Concurrent sessions cannot overwrite each other, a transfer that already happened is skipped by name, and a tree of small files is packed into one archive so that thousands of round trips become one.

Backends are `filesystem`, `gcs` and `modal`, chosen by the provider rather than stated in the declaration.

### Concurrency from repeated calls

Many configurations are many calls. Inside `with let.keep_alive():`, `asyncio.gather` over an `async def` declaration runs them at once and returns results in call order, and `asyncio.as_completed` yields them as they finish. The inventory bounds how many run; a call beyond it waits for a card.

### Environment from a uv lock file

`Env` is keyed by the hash of `uv.lock`. One lock file resolves for every platform, so a Windows or macOS client drives a Linux runtime. Packages in the lock file are installed remotely by name; the project's own code travels with the call.

### CUDA call forwarding

`host="local"` keeps Python here and forwards only driver calls, through [letify-core](letify-core/). Only a call whose result the host reads waits for an answer, so the round trip count is the number of host synchronizations rather than the number of calls.

## Public API

Everything a user needs is on `letify` itself.

### `letify.Launcher`

```python
Launcher(
    config=None,        # a .letify directory or its config.toml; defaults to the project and home ones
    name=None,          # session and app name; defaults to the pyproject project name
    stream_logs=True,   # print remote stdout to stderr
    announce=True,      # say when a session starts, because that is when money starts
    home=True,          # read ~/.letify/config.toml
)
```

| Member | Does |
|---|---|
| `let.providers.<alias>` | Return the provider declared under that alias |
| `let.providers.any.<DEVICE>` | Request an accelerator without naming a provider |
| `let.providers.devices` | Accelerators every provider offers |
| `let.providers.active` | Providers that currently hold a session |
| `let.providers.aliases` | Declared aliases, in configuration order |
| `let.provider(alias)` | Same as attribute access, for a computed alias |
| `let.function(...)` | Declare a function, returning a decorator |
| `with let.keep_alive():` | Keep sessions alive between calls for the length of the block |
| `let.status()` | How many sessions are live and busy against the ceiling, and what each one is |
| `let.usage(alias=None)` | What is left on each account, or why it is not reported |
| `let.utilization(alias=None)` | How busy each instance's accelerator is right now |

### `let.function`

```python
@let.function(
    device=colab.G4,       # an Instance, carrying provider, account and accelerator
    host="remote",         # letify.local forwards CUDA calls, letify.remote ships the function
    env=env,               # an Env; defaults to Env()
    volumes=[cache],       # volumes to attach
    timeout=None,          # seconds one call may take; no default deadline
    retries=1,             # retries on infrastructure failure, never on user code failure
)
def train(lr, bs): ...
```

Sync or async is taken from the `def`. A plain `def` blocks; an `async def` returns a plain coroutine.

| Call form | Returns |
|---|---|
| `train(lr=1e-4)` on a `def` | the value |
| `await train(lr=1e-4)` on an `async def` | the value |
| `await asyncio.gather(*(train(lr=x) for x in lrs))` | a list, in call order |
| `train.local(lr=1e-4)` | runs the body in this process |

### `letify.Env`

```python
env = letify.Env()                       # uv.lock
env = letify.Env.from_lock("other.lock")
env = env.pip_install("flash-attn")      # packages the lock file does not carry
env = env.run("apt-get install -y git")  # commands after installation
env = env.vars(HF_HOME="/opt/cache")     # environment variables in the session
env = env.ship("mypkg")                  # send this module by value, overriding inference
```

### Instances

```python
colab.G4                          # registered shape
colab.device("G4")                # same, for a computed name
colab.instances                   # everything this account offers
colab.refresh()                   # ask the provider again
let.providers.any.A100            # deferred provider choice
```

Colab accepts `T4`, `L4`, `G4`, `A100`, `H100`, plus the TPUs `v5e1` and `v6e1`. `G4` is the RTX PRO 6000 Blackwell part, and `RTX_PRO_6000` is accepted as an alias for it.

### Volumes

```python
cache = colab.volume("hf-cache")                    # backend from the provider
cache = colab.volume("hf-cache", bucket="my-bucket")

cache.cached_env(env)                  # digest of a prebuilt environment archive
cache.cache_env(env, "/opt/venv")      # pack one from this machine
cache.cache_env_from(runtime, env, path)   # pack one installed inside a session
cache.put_checkpoint("run-1", path)    # store and move the ref
cache.absorb(runtime, path, "run-1")   # pull one out of a session
cache.fetch_checkpoint("run-1", target)
cache.resume(runtime, "run-1", path)   # put the newest one inside a session
```

### Types

| Name | Is |
|---|---|
| `Launcher`, `Providers` | the entry point and its provider view |
| `Instance`, `AnyInstance` | an accelerator shape, and a deferred one |
| `local`, `remote` | the two values `host` takes |
| `Env` | an environment declaration |
| `Volume` | a content addressed store on a provider |
| `session_cache` | a value built once per session, keyed by name |
| `Blob`, `RemoteFile` | references to something that lives in a session |
| `Function` | what the decorator returns |

### Errors

| Error | Means | Retryable |
|---|---|---|
| `ConfigError` | The configuration file is malformed or names an unknown kind | no |
| `ProviderUnavailable` | A provider's optional dependency or setting is missing | no |
| `UnknownProvider`, `UnknownInstance` | No such alias or accelerator | no |
| `RuntimeFailure`, `RuntimeLost` | The session misbehaved | yes |
| `ProtocolError` | The remote process died before producing a result | yes |
| `RemoteError` | The shipped function raised; carries the remote traceback | no |
| `UnsupportedMode` | The requested mode cannot work here | no |
| `InsufficientDevices` | The devices a call asks for cannot be allocated; names what holds them | no |

### Command line

```bash
letify login <kind> [alias]   # declare an account at home, reference it here
letify logout <alias>         # remove the account from this machine
letify providers              # declared providers, storage, channel kind
letify devices                # accelerators each provider offers
letify status                 # live sessions and what they are costing
letify usage [alias]          # what is left on each account
letify utilization [alias]    # how busy each instance's accelerator is
letify check <alias>          # confirm a machine answers
letify probe <host>           # whether host="local" is worth using
letify efficiency 0.5 3 150   # expected share of a direct run
letify stubs                  # write typings/letify_providers.pyi for editor completion
```

## Configuration

Configuration is a directory. `~/.letify/config.toml` holds accounts, the project's `.letify/config.toml` holds defaults and the aliases it uses, and neither holds a secret.

An account in `~/.letify/config.toml` is available in a project only when the project's `.letify/config.toml` names its alias, even as an empty `[colab_pro]` table, when the home entry sets `global = true`, or when it is `local`. A named alias takes every home setting and the project's own fields override them one by one. With no project `.letify/` at all, only global accounts and `local` exist.

`letify login <kind> [alias]` writes both. The account goes to `~/.letify/config.toml`, which belongs to the machine, and any credential it collects goes to `~/.letify/accounts/<alias>/`. `letify logout <alias>` deletes that account directory. The project file gets the alias as an empty table and nothing else, which is safe to commit and tells a teammate which accounts the repository needs. An account already declared is not asked for again, so `letify login` in a second repository writes only the table. An alias the project names that this machine does not have is a configuration error naming the command that fixes it.

SSH authenticates by key, because letify opens sessions with `ssh -o BatchMode=yes`: a session is started by the pool in the background, with nobody present to answer a password prompt. So `letify login shell` generates an ed25519 key if there is none, asks for the password once to install it, drops the password, and confirms the key works before declaring the alias. It then runs `nvidia-smi` on the machine once, asks which indices of each accelerator letify may use, and writes them to `[<alias>.devices]` in the home file, so editor completion knows the names. `--indices NAME=SPEC` chooses them in a script, `--detect-devices` asks an already declared machine again, and a machine where `nvidia-smi` does not answer is logged in with no table. Nothing about the password is written to a file or the environment. A machine whose administrator forbids key authentication can use `--auth password`, which reads the password from `~/.letify/accounts/<alias>/password` and drives `sshpass`; it is refused on Windows, where that tool does not exist.

`letify login modal <alias>` runs Modal's own `modal token new` through `uv tool run`, with `MODAL_CONFIG_PATH` pointing at `~/.letify/accounts/<alias>/modal.toml`, and approves in the browser. The Modal provider and the `modal` volume backend never import `modal` in the letify process: they drive a small adapter, `letify/providers/modal_adapter.py`, run with `uv run --no-project --with "modal>=1.0,<2"` as that account and spoken to in JSON lines.

```toml
[defaults]
name = "nvfp4"

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

An alias must be a Python identifier, because providers are reached by attribute. `any`, `devices` and `active` are reserved. Declaration order sets the priority for `let.providers.any`.

Where a service publishes no balance, an entry names a command that prints one: `usage_command` is run when `letify usage` asks, `usage_unit` names what it counts and `usage_limit` gives the ceiling. The last number in the output is read as the remaining amount.

A credential never enters `config.toml`. It is resolved in order: the environment variable named by `<name>_env`, then the file `~/.letify/accounts/<alias>/<name>` (mode 0600, written by `letify login`), then a literal value. The OS keyring is not used.

Provider types for editor completion are written to `typings/letify_providers.pyi` at the project root, automatically when the configuration loads or on demand with `letify stubs`. `[tool.letify] typings = "<path>"` in `pyproject.toml` moves it.

## Installation

```bash
uv add letify
```

That is the only install. `letify` installs cloudpickle and blake3 and nothing else. Provider tools run out of process through uv, never in the user's `.venv`, so uv must be installed. letify finds it from the `UV` environment variable, then `PATH`, and raises a clear error if it is absent.

`host="local"` additionally needs [letify-core](letify-core/) built with `python letify-core/build.py`, which requires a Rust toolchain.

## How this project is built

Spec driven and test driven. [docs/SPEC.md](docs/SPEC.md) is the source of truth, a change edits it before the code, and every test traces to a section of it. The rules are in [CLAUDE.md](CLAUDE.md).

## What is not implemented yet

Stated plainly so nobody builds on a promise. The same list, with detail, is at the end of [docs/SPEC.md](docs/SPEC.md).

- `letify-driver` covers the entry points a PyTorch process needs to start up and run one kernel. Anything else names itself and returns `CUDA_ERROR_NOT_SUPPORTED`.
- `Modal` and `Elice` follow each service's published interface but have not been run against the live services.
- Whether `ssh -L` works over `colab ssh --proxy-mode` is unverified.
- Persistence detection by marker file is a decision in the spec, not yet code.
