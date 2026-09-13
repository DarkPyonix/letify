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

Two files, merged, with different jobs.

| File | Holds | Tracked by Git? |
|---|---|---|
| `~/.letify` | Accounts, addresses, keys, tokens | No, it is outside the repository |
| `.letify` in the project | Defaults that are safe to share | Yes |

The project file refines what the home file declared, so someone else can clone your repository and run it under their own accounts.

```toml
# .letify in the project, committed
[defaults]
name = "nvfp4"
```

## Secrets

A credential is referenced, never written.

```toml
[elice_a100]
kind = "elice"
access_token_env = "ELICE_ACCESS_TOKEN"        # an environment variable
# access_token_keyring = "elice/researcher"    # or an OS keyring entry
```

Resolution order is the environment variable, then the keyring, then a literal value. A literal is only appropriate in `~/.letify`, which is not tracked.

For the keyring, install the extra: `uv add "letify[keyring]"`.

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

**Persistence defaults to ephemeral,** which is the pessimistic choice on purpose. Assuming ephemeral when the disk actually survives only costs time, because letify rebuilds the environment and the work still succeeds. Assuming persistent when the disk is wiped fails outright.

Set `persistent = true` once you know the machine keeps its home directory, which is the normal case for a lab server. That flips the default execution mode to function shipping.

Declaring `gpus` avoids an SSH connection at import time:

```toml
gpus = ["A100", "A100", "A100", "A100"]
```

## 🕳️ Tunnel, for a machine behind NAT

For a machine that cannot accept an inbound connection. The tunnel builds the path; SSH still does the work.

```toml
[home_box]
kind = "tunnel"
transport = "tailscale"
address = "home-box"               # the Tailscale machine name
user = "researcher"
auth_key_env = "TS_AUTHKEY"
mtu = 1280
persistent = true
```

Tailscale is the default because it needs no server of your own, authenticates from an auth key without a prompt, and carries any TCP port. When UDP is blocked it relays over TCP 443, which keeps working but slowly.

If your network blocks UDP and the relay is too slow, switch transports:

```toml
transport = "frp"
frp_config = "~/.config/frp/frpc.toml"
```

> ⚠️ **Leave the MTU low.** Every mesh VPN in this class shows the same failure above roughly 1400: the tunnel comes up, small commands work, and bulk transfers stall silently. 1280 always works.

Try the simpler paths first. A direct address, then a jump host, then this. Campus machines often allow one of the first two. Details and measurements are in [docs/NETWORK.md](../NETWORK.md).

## ☁️ Modal

```toml
[modal_lab]
kind = "modal"
```

Reads its credentials the way the Modal client does. Storage is persistent because a Modal volume is mounted from outside the container, so function shipping is the default and no separate cache tier is needed.

`host="local"` raises. Modal exposes function calls into a container, not a device to forward calls at.

The Modal package is imported lazily. Without it, this provider reports itself unavailable and every other provider keeps working.

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
