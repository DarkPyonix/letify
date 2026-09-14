# 2️⃣ Providers and accounts

> Declaring where your GPUs are, including several accounts of the same kind.

[← Getting started](01-getting-started.md) · [Guides](README.md) · [Next: Execution modes →](03-execution-modes.md)

---

## The model

A provider object is **one account on one kind of infrastructure**. You declare it once in a configuration file and reach it by alias.

```python
let = letify.Launcher()

colab = let.providers.colab_a      # one Google account
lab   = let.providers.lab_a100     # one machine
here  = let.providers.local        # always exists, needs no declaration
```

An accelerator is then an attribute of the provider, and that value carries everything a declaration needs.

```python
@let.function(device=colab.G4, host="remote")
def train(lr): ...
```

`colab.G4` is one value holding the provider, the account, the accelerator and where the CPU side runs. Those four are one decision, which is why they are one argument.

## Where configuration lives

letify keeps its state in two `.letify` directories with different jobs.

| Path | Holds | Tracked by Git? |
|---|---|---|
| `~/.letify/config.toml` | Every account this machine has: kind and connection details, never a secret | No, it is outside the repository |
| `~/.letify/accounts/<alias>/` | That account's credentials, owner only | No |
| `<project>/.letify/config.toml` | Project defaults, and the aliases of the accounts the project uses | Yes |

The home file is the set of accounts on this machine. The project file chooses from it. An account is available in a project only when one of these holds:

1. The project file names it. An empty table is enough.
2. The home entry sets `global = true`.
3. It is `local`, which is always available.

```toml
# .letify/config.toml in the project, committed
[colab_a]         # use the home account colab_a with all of its settings
[lab_a100]
```

A field set in the project table overrides the home entry's field. Someone else can clone your repository and run `letify login` for the aliases it names.

## Secrets

A credential never appears in either `config.toml`. `letify login` asks for it and stores it.

```bash
letify login elice elice_a100      # writes ~/.letify/accounts/elice_a100/access_token
letify logout elice_a100           # removes the account and deletes its directory
```

A field such as `access_token` is resolved in this order:

1. The environment variable named by `access_token_env`.
2. The file `~/.letify/accounts/<alias>/access_token`, created with mode 0600.
3. A literal value in the home entry, meant only for non secret defaults.

```toml
[elice_a100]
kind = "elice"
access_token_env = "ELICE_ACCESS_TOKEN"        # optional, overrides the stored file
```

The OS keyring is not used.

## 📓 Colab

```toml
[colab_a]
kind = "colab"
account = "you@example.com"
```

Uses the official Colab CLI, so there is no tunnel and no terms risk. Sessions are created with `colab new` and destroyed with `colab stop`, and a background daemon keeps the runtime from idling out without a browser tab.

| | |
|---|---|
| Accelerators | `T4`, `L4`, `G4`, `A100`, `H100`, and the TPUs `v5e1`, `v6e1` |
| `G4` is | RTX PRO 6000 Blackwell, 96 GB. `RTX_PRO_6000` is accepted as an alias |
| Storage | **Ephemeral.** A runtime change gives a new machine with an empty disk |
| Execution mode | Function shipping only |
| Cache backend | Google Cloud Storage |

Two things to know. Accelerators require a Pro or Pro plus entitlement, and the remote control features are permitted while the compute unit balance stays positive. An exhausted balance reverts the account to the free tier policy, which disallows them.

`host="local"` raises here. The control path crosses a Google frontend, so the round trip is 150 ms to 200 ms from Korea, which leaves about half the throughput for fine-tuning and a few percent for decoding. The message says exactly that.

## 🐚 Shell, for a lab or university server

```toml
[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true
```

| Option | Does |
|---|---|
| `address`, `user`, `port`, `key` | Standard SSH connection details |
| `jump` | A jump host, passed to `ssh -J` |
| `persistent` | Declares that the disk survives between sessions |
| `gpus` | A list, to skip connecting during discovery |
| `port_command` | A command that prints the current port, for hosts that reassign it |
| `store` | Override the cache backend |
| `workspace` | The one directory letify writes under on the machine: the project `.venv`, volume data and temporary files. Default `~/.letify-runtime`. Home file only |

**Persistence defaults to ephemeral,** which is the pessimistic choice on purpose. Assuming ephemeral when the disk actually survives only costs time, because letify rebuilds the environment and the work still succeeds. Assuming persistent when the disk is wiped fails outright.

Set `persistent = true` once you know the machine keeps its home directory, which is the normal case for a lab server. That flips the default execution mode to function shipping.

Declaring `gpus` avoids an SSH connection at import time:

```toml
gpus = ["A100", "A100", "A100", "A100"]
```

Some servers allow writes only under a given directory, such as `/workspace`. Right after the key works, `letify login shell` asks where letify may write, and checks over SSH that it can create and write that directory without root. A path that fails writes nothing and names the error.

```
$ letify login shell lab_a100 --address gpu.lab.example.edu
Workspace root on gpu.lab.example.edu [~/.letify-runtime]: /workspace/researcher/letify
```

In a script, pass `--workspace /workspace/researcher/letify` with `--no-input`. To move an account that is already declared, run the login again with `--workspace`, and the new path is checked before it is written. `letify check lab_a100` prints `workspace <path>: writable` or the reason it is not.

`letify login shell` writes the devices table for you. Right after the key works, it runs `nvidia-smi` on the machine once, shows what it found and asks which cards letify may use. A blank answer takes them all.

```
$ letify login shell lab_a100 --address gpu.lab.example.edu
A100: 4 cards, indices 0-3 (80 GB each)
RTX_PRO_6000: 2 cards, indices 4-5 (96 GB each)
Indices letify may use for A100 [0-3]: 0,1
Indices letify may use for RTX_PRO_6000 [4-5]:
```

```toml
[lab_a100.devices]
A100 = { indices = "0-1" }
RTX_PRO_6000 = { indices = "4-5" }
```

In a script, `--no-input` records every card, and `--indices A100=0-1` (repeatable) narrows one name. When `nvidia-smi` is missing or finds no GPU, the login still succeeds, nothing is recorded, and letify asks the machine at first use. An account that is already declared is not asked again; `letify login shell lab_a100 --detect-devices` asks the machine again and replaces the table once you confirm. The table is ordinary TOML, so you can also edit it by hand.

## 🕳️ Tunnel, for a machine behind NAT

For a machine that cannot accept an inbound connection. letify reaches it over [Tailcat](https://github.com/tailscale/tailcat), which needs no account and no server of your own. SSH still does the work.

Setup is two commands, one on each machine. Both machines need `tailcat`, and the remote one needs letify and an SSH server.

**1. On the remote machine**, start the agent:

```bash
letify client shell connect --name home_box
```

If `tailcat` or an SSH server is missing, the command prints the exact install steps for that machine and exits. Otherwise it prints one command to run on your own machine:

```
letify login tunnel home_box --connect eyJ0YWlsY2F0Ijoi...
```

Keep the agent running, for example inside `tmux`. It carries every connection, and a restart gives it a new address, so you would log in again with the new token.

**2. On your own machine**, run the printed command:

```bash
letify login tunnel home_box --connect eyJ0YWlsY2F0Ijoi...
```

It installs your SSH key over Tailcat (you type the remote password once), confirms the key, checks the workspace root and records the GPUs, as `letify login shell` does. Then `letify check home_box` confirms the machine answers. The account in `~/.letify/config.toml` looks like this, with no address:

```toml
[home_box]
kind = "tunnel"
tailcat = "tc..."
tailcat_port = 40123
user = "researcher"
port = 22
key = "~/.ssh/id_letify"
```

Try the simpler paths first. A direct address, then a jump host, then this. Campus machines often allow one of the first two. Details and measurements are in [docs/NETWORK.md](../NETWORK.md).

## ☁️ Modal

```toml
[modal_lab]
kind = "modal"
```

Sign in once per account:

```bash
letify login modal modal_lab
```

letify asks for an optional Modal profile, which names the Modal workspace to sign in to (`--profile` in a script), then runs Modal's own `modal token new` through uv. It prints a link; approve it in the browser. The token is written to `~/.letify/accounts/modal_lab/modal.toml`, so two Modal accounts can live on one machine. You never install `modal` yourself, on `PATH` or in your project's `.venv`. uv is the only requirement.

Storage is persistent because a Modal volume is mounted from outside the container, so function shipping is the default and no separate cache tier is needed. letify's own files in the sandbox, the project `.venv` included, live under `/letify`, where a Modal volume named `<app>-workspace` is mounted, so the next sandbox finds them. `--workspace PATH` at login moves that mount.

`host="local"` raises. Modal exposes function calls into a container, not a device to forward operators to.

Modal's client runs in its own uv environment, with Modal pinned to `>=1.0,<2`, in a small adapter process that letify starts on the first call. The first start downloads Modal into uv's cache. A token in `MODAL_TOKEN_ID` or `MODAL_TOKEN_SECRET` in your shell is ignored, because the account's `modal.toml` decides which account acts.

## 🇰🇷 Elice

```toml
[elice_a100]
kind = "elice"
zone_id = "00000000-0000-0000-0000-000000000000"
machine_id = "00000000-0000-0000-0000-000000000000"
address = "..."
user = "elicer"
key = "~/.ssh/elice.pem"
access_token_env = "ELICE_ACCESS_TOKEN"
```

Targets Elice Cloud Infrastructure, which has a published REST API. letify powers the machine on and off by creating and deleting an allocation, which maps exactly onto a session, so a call's own lifetime follows Elice's.

**letify does not create the machine.** Declare the virtual machine once in the console or with Terraform and put its id in `machine_id`. letify allocates and releases it.

Two costs to remember. Compute bills by the second while allocated, and block storage keeps billing while the machine is stopped. A forgotten machine costs money with no allocation running.

Elice's other product, Run Box, is console driven and has no API. You can still use it by declaring it as a plain `shell` with its tunnel address and port.

## 💻 Local

Always available, no declaration needed. Runs a declared function in a subprocess of this machine with no transfer at all.

```python
here = let.providers.local
here.CPU                      # always present
here.GeForce_RTX_4050         # whatever nvidia-smi reports
```

Two uses. If you have a GPU, use it through the same declarations as everything else. If you do not, your tests still exercise the real path rather than a mock.

## 🔀 Several accounts

Each configuration entry is one account, and entries of the same kind coexist. This is how you get more concurrent sessions.

```toml
[colab_a]
kind = "colab"
account = "first@example.com"

[colab_b]
kind = "colab"
account = "second@example.com"
```

```python
a = let.providers.colab_a
b = let.providers.colab_b

@let.function(device=a.G4, host="remote")
def train(lr): ...

@let.function(device=b.L4, host="remote")
def evaluate(ckpt): ...
```

Training on one account while evaluation runs on another is a common arrangement, because it keeps a long run from blocking a quick check.

## 🎲 Not choosing a provider

```python
@let.function(device=let.providers.any.A100, host="remote")
def train(lr): ...
```

Picks the first declared provider that registered an `A100`. Priority is the order the entries appear in the configuration file, so put your preferred provider first.

## 🔍 Inspecting

```python
let.providers.aliases          # ['colab_a', 'lab_a100', 'local']
let.providers.devices             # accelerators per provider
let.providers.active           # providers currently holding a runtime
let.providers.colab_a.instances
let.providers.lab_a100.refresh()   # ask the machine again
```

`let.providers.active` is the quickest answer to what is costing money right now.

```bash
letify providers
letify devices
letify status
letify check lab_a100
```

---

[← Getting started](01-getting-started.md) · [Guides](README.md) · [Next: Execution modes →](03-execution-modes.md)
