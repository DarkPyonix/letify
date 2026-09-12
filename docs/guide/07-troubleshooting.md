# 7️⃣ Troubleshooting

> Each error letify raises, what it actually means, and what to do.

[← Cost control](06-cost.md) · [Guides](README.md)

---

## Reading an error

letify distinguishes two kinds of failure, and the distinction decides whether a retry helps.

| Kind | Errors | Retried? |
|---|---|---|
| The infrastructure misbehaved | `RuntimeFailure`, `RuntimeLost`, `ProtocolError` | yes, on a fresh runtime |
| Your code raised | `RemoteError` | no, a retry reproduces it |
| You asked for something impossible | `ConfigError`, `UnsupportedMode`, `HandleScopeError`, `UnknownProvider`, `UnknownInstance` | no |
| Missing dependency or setting | `ProviderUnavailable` | no |

---

## `NotRunning`

```
NotRunning: train was called outside a run scope. Wrap the call in
`with let.run():` so letify knows when to start and stop paying for a session.
```

Sessions exist only inside the scope. This is deliberate: the point where money starts and stops should be visible in the code.

```python
with let.run():
    train(lr=1e-4)
```

---

## `ProviderUnavailable`

```
provider 'modal' is unavailable: the modal package is not installed.
Install it with: uv add "letify[modal]"
```

The message names the extra to install. If it names a missing setting instead, add that field to the configuration.

```
provider 'elice' is unavailable: elice_a100 needs an access token. Set
access_token_env or access_token_keyring so the token stays out of tracked files
```

---

## `ConfigError`

**An alias with a hyphen.**

```
.letify: alias 'colab-a' is not a Python identifier, so let.providers.colab-a
cannot work. Try 'colab_a'.
```

Providers are reached by attribute, so an alias has to be a valid identifier.

**A reserved alias.**

```
.letify: 'any' is reserved. Pick another alias, because let.providers.any
already means something else.
```

`any`, `gpus` and `active` are taken.

**A missing kind.** Every entry needs `kind`. Valid values are `local`, `colab`, `modal`, `shell`, `ssh`, `tunnel`, `elice`.

---

## `UnknownProvider` and `UnknownInstance`

```
no provider is declared under 'colab_b'. Declared: colab_a, lab_a100, local.
Add it to .letify, or to ~/.letify if it carries credentials.
```

```
colab_a does not offer 'B200'. Available: A100, G4, H100, L4, T4, v5e1, v6e1
```

Both messages list what exists. `letify providers` and `letify gpus` show the same thing from the shell.

---

## `UnsupportedMode`

```
Colab does not support cpu='local'. Forwarding CUDA calls over the Colab control
path costs one round trip of about 150 ms per host synchronization, which leaves
roughly half the throughput for fine-tuning and a few percent for token by token
decoding. Use cpu='remote' so the loop runs inside the runtime.
```

This is letify refusing to take a slower path silently. Use `cpu="remote"`, or move that work to a provider with a fast path, which is `Shell`, `Tunnel` or `Elice`. See [Execution modes](03-execution-modes.md).

---

## `RemoteError`

Your function raised on the remote machine. The remote traceback is attached.

```python
try:
    train(lr=1e-4)
except letify.RemoteError as exc:
    print(exc.remote_traceback)
```

Not retried, because a retry reproduces it. Debug it locally first where you can:

```python
train.local(lr=1e-4)      # run the body in this process
```

Or declare the same function on a `Local` provider so the whole path is exercised without a remote machine.

**Common causes worth knowing.** An import that exists locally but is not in your lock file, which `Env.pip_install` or a lock file update fixes. An out of memory error, which appears as a `RemoteError` if PyTorch raised it and as a `ProtocolError` if the process was killed. A path that exists on your machine and not in the runtime, which usually means data that should be in a volume.

---

## `ProtocolError`

```
the runtime produced no result marker, so the process died before it finished.
The usual causes are an out of memory kill, a preempted session, or a crash
below Python.
--- last remote output ---
...
```

The remote process died without returning. The tail of its output is included, which is normally enough to tell which case it was.

| Cause | Signal | What to do |
|---|---|---|
| Out of memory kill | the output stops mid-step, no Python traceback | reduce the batch, enable gradient checkpointing, or take a larger card |
| Preempted session | happens at an unremarkable point | checkpoint to a volume so a rerun resumes |
| Crash below Python | a CUDA or driver message in the tail | usually a version mismatch in the environment |

letify discards the runtime and retries once by default, because the session is at fault.

---

## `RuntimeFailure` and `RuntimeLost`

```
train failed after 2 attempt(s) on <Instance colab_a:G4 cpu=remote>: ...
```

The session could not be reached or a command failed. `RuntimeFailure` carries the command and the remote stderr:

```python
except letify.RuntimeFailure as exc:
    print(exc.command)
    print(exc.stderr)
```

**If it is Colab:** confirm the CLI works on its own with `colab sessions`. Check that your compute unit balance is positive, since an exhausted balance reverts the account to free tier policies that disallow the features letify uses. An accelerator may simply not be available, which is not promised and is more likely for `G4` because Blackwell supply is tight.

**If it is a Shell machine:** `letify check <alias>`. Then try the SSH command by hand. A host that reassigns its port on restart needs `port_command` in the configuration.

**If it is a Tunnel:** `let.providers.<alias>.diagnose()` reports the transport, the MTU, and whether the path is relayed.

---

## `HandleScopeError`

```
<Handle dict a1b2c3 on colab_a:G4...> belongs to runtime 'X' but the call targets
'Y'. Route the call to the owning runtime, or return the value to this process
before passing it on.
```

A handle is a pointer into one process and one CUDA context. letify refuses to resolve one across runtimes rather than copying the whole object over the network without being asked.

Send the call to the runtime that owns the object, or have the first call return the value instead of a handle.

> 🚧 Handles are returned today, but resolving one in a later call needs the persistent session process, which is not implemented yet. See the known gaps at the end of [docs/SPEC.md](../SPEC.md).

---

## Slow, not broken

### Session start takes twenty minutes

You have no volume attached. Add one.

```python
cache = colab.volume("hf-cache")

@let.function(gpu=colab.G4, volumes=[cache])
def train(lr): ...
```

Pulling 20 GB from a bucket next to the runtime is 40 to 60 seconds against 27 minutes from a remote origin. See [Environments and data](04-environments-and-data.md).

### Training is much slower than the same code run directly

If `cpu="local"`, you are forwarding CUDA calls and paying a round trip per host synchronization. Measure both terms:

```bash
letify probe gpu.lab.example.edu
```

```python
import torch
torch.cuda.set_sync_debug_mode("warn")     # count warnings in one step
```

Then `efficiency = T / (T + k × RTT)`. If the answer is poor, switch to `cpu="remote"` or reduce `k`. See [Execution modes](03-execution-modes.md).

If `cpu="remote"` and it is still slow, the loop itself is slow. It is running on the remote machine with no letify overhead per step, so profile it as you would locally.

### Bulk transfers stall on a tunnel, while small commands work

This is the MTU symptom, and it is the single most common tunnel problem. Lower it.

```toml
[home_box]
kind = "tunnel"
mtu = 1280
```

Then check whether the path is relayed, because a relayed Tailscale path has been measured as low as 2.2 Mbit/s across continents:

```python
let.providers.home_box.diagnose()
```

If it is relayed and UDP is blocked on your network, switch to `transport = "frp"` on TLS port 443. See [docs/NETWORK.md](../NETWORK.md).

### A sweep is not running in parallel

Check three things. `concurrency` on the declaration, `max_runtimes` on the launcher, and how many runtimes actually came up:

```python
print(len(let.pool.live))
```

If fewer came up than you asked for, the provider refused them. For Colab that limit is undocumented and moves with tier, credit balance and demand. Find it with cheap `L4` sessions. See [Sweeps](05-sweeps.md).

---

## Things that are not bugs

**`import letify` works with no providers installed.** That is the design. A provider whose package is missing reports itself unavailable and the rest keeps working.

**`let.providers.gpus` shows `unavailable: ...` for one provider.** Also the design. One broken entry should not hide the others.

**A GPU name is normalized.** `NVIDIA RTX PRO 6000 Blackwell` becomes `RTX_PRO_6000` so it can be an attribute. On Colab the same card is `G4`, which is what the CLI calls it, and `RTX_PRO_6000` is accepted as an alias.

**A Shell provider defaults to ephemeral even for a machine that keeps its disk.** Deliberate. Guessing ephemeral wrongly costs time; guessing persistent wrongly fails. Declare `persistent = true` once you know.

---

## Filing a bug

Include the output of:

```bash
letify --version
letify providers
letify gpus
```

Plus the full traceback. For a `RemoteError`, include `exc.remote_traceback`; for a `RuntimeFailure`, include `exc.command` and `exc.stderr`. Redact account names and addresses, and never paste a token.

---

[← Cost control](06-cost.md) · [Guides](README.md)
