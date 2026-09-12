# Specification

> The current design of letify. Decisions only. Derivations and measurements live in the experiment pull requests, which each section links to when one exists.

## Declaration surface

> A user declares resources and lets letify choose the mechanism.

The public surface is five names: `Launcher`, `Env`, the `function` decorator it carries, `grid` and `zip`. Everything else is reached through a provider object.

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

The decorator takes `gpu`, `env`, `volumes`, `concurrency`, `timeout`, `retries` and `keep_remote`. It does not take a transport, a mode or a provider, because `gpu` already carries all three.

### Invocation

> Calling a declared function runs it. Blocking behaviour is declared at the `def` site.

A declared function is called like any other function. There is no second verb such as `.remote()`: a decorator that wraps a `def` and then needs another call to run it has moved the declaration out of the declaration.

A plain `def` blocks and returns its value. An `async def` returns an awaitable, so `await` and `asyncio.gather` work as they do for any coroutine and letify contributes no future type of its own.

`Function.local()` runs the body in the calling process. It exists for testing a body without any provider; the preferred way to run locally is a `Local` provider, which keeps the production code path.

### Fan-out

> Passing a space where a scalar is expected declares that the argument varies.

`grid(**axes)` is the Cartesian product of its axes. `zip(**axes)` pairs them position by position and rejects axes of unequal length. Two spaces combine with `|`, which drops duplicate points. A scalar axis stays fixed across the space.

A space is consumed by the language's own protocols. `await` on an async declaration collects results in input order; `async for` yields them as they complete. A sync declaration returns a list in input order.

`concurrency` on the declaration bounds how many runtimes one declaration may occupy at once. It belongs to the declaration rather than to a call, because it describes the infrastructure the declaration is allowed to use.

## Provider model

> A provider object is one account on one kind of infrastructure. It registers the instances it offers, owns storage, and creates runtimes.

```
Provider (abstract)
├── Local                 persistent
├── Modal                 persistent
└── Shell                 ephemeral by default, SSH transport
    ├── Colab             session created by the Colab CLI
    ├── Tunnel            network path built first, then SSH
    └── Elice             machine allocated through the Elice Cloud API
```

`Shell` is named for the shared ability, which is running a command on a remote machine, rather than for SSH, which is only its default transport.

A provider is built from one entry in the configuration file and reached by attribute on `let.providers`. Three attribute names there are reserved: `any` for a request that does not name a provider, `gpus` for the registered accelerators of every provider, and `active` for the providers that currently hold a runtime.

### Persistence

> Persistence means storage outlives a runtime. It is a property of the provider object, not of its class.

| Provider | Default | Reason |
|---|---|---|
| `Local` | persistent | The machine's own disk. |
| `Modal` | persistent | A volume is mounted from outside the container. |
| `Colab` | ephemeral | A runtime change gives a new virtual machine with an empty disk. |
| `Shell` and subclasses | ephemeral | A machine's disk policy is not knowable in advance. |

A configuration entry overrides the default with `persistent = true`. The default for an unknown machine is the pessimistic one: assuming ephemeral costs time, because letify rebuilds the environment each runtime and the work still succeeds, while assuming persistent fails outright when the disk turns out to be wiped.

letify may detect persistence by writing a marker file and looking for it in a later runtime, but detection only advises. It never changes the execution mode on its own, because that would change the performance characteristics of a run without the user asking.

### Instances

> An `Instance` is one accelerator shape on one provider account, carrying the CPU placement with it.

`colab.G4` is an `Instance`. It holds the provider, the accelerator name, where the Python side runs, the core count and whether the instance is preemptible. Calling it refines it: `colab.G4(cpu="local", cpus=8)`.

Because an instance carries its provider, `gpu=colab.G4` fixes provider, account, accelerator and placement in one argument. `let.providers.any.G4` defers the provider choice to the first declared provider that registers a matching accelerator, in configuration order.

Instance discovery is lazy and cached. A provider that must connect to enumerate its GPUs does so on first access, never at import time, and a configuration entry may list `gpus` to skip the connection entirely.

## Execution modes

> Two modes exist. The user names resources; letify picks the mode.

**Function shipping** (`cpu="remote"`) serializes the declared function with cloudpickle and runs it inside the runtime. The whole loop executes there, so the loop's host synchronizations never cross the network.

**Call forwarding** (`cpu="local"`) keeps Python and the libraries in the local process and forwards only CUDA driver calls. Local data and the local environment stay in place, at the cost of one network round trip at every point where the host reads a value back from the device.

### Mode selection

> Storage decides first, then whether a low-latency path exists.

| Provider state | Default placement | Mode |
|---|---|---|
| persistent | `remote` | function shipping |
| ephemeral with a volume attached | `remote` | function shipping |
| ephemeral, no volume, fast path available | `local` | call forwarding |
| ephemeral, no volume, no fast path | `remote` | function shipping |

An ephemeral provider with no volume has nothing on it that survives, so keeping state local is the cheaper arrangement where the link allows it. Forwarding also needs only a driver and a daemon on the remote side, which is the least setup a brand new machine can require.

`Colab` sets `has_fast_path = False`, so it never selects forwarding. Asking for `cpu="local"` there raises `UnsupportedMode` with the arithmetic in the message rather than running slowly.

### Efficiency model

> Efficiency against a direct run is `T / (T + k * RTT)`, where `T` is GPU time per step and `k` is host synchronizations per step.

Numbers for an RTX PRO 6000 with NVFP4, a 0.5 s micro step, at a 150 ms round trip:

| Workload | Function shipping | Call forwarding, default settings | Call forwarding, tuned |
|---|---|---|---|
| LoRA fine-tuning | about 99 percent | 50 to 57 percent | about 96 percent |
| Decode, batch 1 | hundreds of tokens per second | 2 to 7 tokens per second | unchanged |
| Evaluation, teacher forcing | about 99 percent | about 99 percent | about 99 percent |

`k` is about three for a default Hugging Face training step: the trainer's NaN filter every step, the attention mask check every forward, and logging or the gradient scaler. Tuning means turning the NaN filter off, removing the mask check with fixed length packing, and moving logging to the gradient accumulation boundary, which leaves about one synchronization per optimizer step.

A faster GPU makes forwarding worse, because `T` shrinks while `RTT` does not. The same step on an L4 in bf16 takes 1.8 s and reaches about 80 percent where the RTX PRO 6000 reaches 53 percent.

Decoding fails at any useful latency. A decode step for 4-bit weights on an RTX PRO 6000 is 2 ms to 3 ms and synchronizes once or twice per token, so throughput is bounded near `1000 / (k * RTT)` tokens per second regardless of the card.

## Call protocol

> A call is a serialized function plus arguments, executed by a driver script that prints its outcome between two markers.

The local side pickles `(function, args, kwargs, keep_remote)` with cloudpickle, base64 encodes it, and embeds it in a driver script. The driver deserializes, resolves handles, runs the call, and writes the outcome as base64 between `__LETIFY_RESULT_BEGIN__` and `__LETIFY_RESULT_END__`, so a result can be found in a stream that also carries the user's prints.

Absence of the marker is not a protocol quirk, it means the remote process died. letify reports that as `ProtocolError` naming the likely causes: an out of memory kill, a preempted session, or a crash below Python.

### Handles

> A value may stay in the runtime. The caller receives a reference scoped to that runtime.

A declaration with `keep_remote=True` registers its return value in the runtime's object table and returns a `Handle`. Passing a handle to a later call on the same runtime resolves it in place, so a model stays on the remote machine instead of being copied back and forth.

A handle carries the key of the runtime that owns it. Passing it to a call on another runtime raises `HandleScopeError` rather than materializing the object, because a handle is a pointer into one process and one CUDA context and resolving it across that boundary would mean an unrequested transfer of the whole object.

### Argument addressing

> Large arguments are named by the hash of their contents, so the same value travels once.

An argument above the inline limit is hashed and sent only when the runtime does not already hold that digest. Hashing is not a bottleneck at any link speed involved: blake3 runs at gigabytes per second where a home uplink runs at megabytes per second.

### Callbacks

> A side effect reported to the local process is one way.

A streamer, a logger or a progress callback in shipped code sends to the local process without waiting for a reply. Waiting would put one round trip inside the token loop, which is the worst thing available to this design.

### Failure and retry

> Infrastructure failure may be retried. User code failure never is. Neither falls back to a slower path.

`RuntimeFailure` and `ProtocolError` mean the session misbehaved, so the runtime is discarded and the call is retried on a fresh one up to `retries` times. `RemoteError` means the shipped function raised, and it propagates with the remote traceback attached.

letify never falls back to local execution or to a slower mode when the declared one is unavailable. A silent downgrade turns a four times slowdown into a mystery, so the failure is explicit.

## Runtimes

> A runtime is one live session and the only object that costs money.

Everything above a runtime is declaration. Creating one is when a provider actually powers something on; destroying one is when the charge stops.

A runtime boots in three steps: install the declared environment, attach volumes, and start the lease. Installation is skipped where the machine already runs in the environment, which is the local provider.

### Pooling

> Runtimes are pooled by instance and environment, so the second call through a declaration pays nothing for setup.

The pool key is the instance key joined with the environment key. Two declarations that agree on both share runtimes. A pool holds at most `max_runtimes` runtimes; a call that finds every slot taken waits for one of its own key to free rather than starting a new session.

`max_runtimes` defaults to 3 and is a placeholder. The concurrent session limit of a Colab account is undocumented and moves with tier, credit balance and demand.

### Lifetime

> Three layers, and the middle one protects the bill.

1. **Scope.** Leaving `with let.run():` tears every runtime down. Nested scopes are allowed and only the outermost tears down.
2. **Idle timeout.** A runtime unused for longer than `idle_timeout` inside an open scope is torn down.
3. **Heartbeat lease.** The local process renews a deadline inside the session every 30 s, and the session terminates itself if the deadline passes. The grace period is 300 s, so a brief network drop does not kill a training run, while a crashed or killed local process cannot leave a GPU billing.

There is no detached execution. A detached run whose remote side is preempted would lose its results, so the local process stays the owner and durability comes from checkpoints in the store.

## Storage

> A volume is a content addressed blob store on whichever backend a provider has. It is what makes an ephemeral provider behave like a persistent one.

Attaching a volume to an ephemeral provider moves the environment archive and the model cache into storage that outlives the runtime. A twenty gigabyte cache takes about 27 minutes to pull from a lab server over a 100 Mbit/s link, three to five minutes from the Hugging Face hub, and 40 to 60 seconds from a bucket inside the same infrastructure as the runtime. That difference is billed as GPU time.

### Content addressed layout

> Blobs are immutable and named by their hash. Mutable names live in a separate, tiny namespace of refs.

```
blobs/<first two hex characters>/<digest>
refs/<name>
```

Immutability buys two things. Concurrent writers cannot conflict, because different contents get different names, where a two way synchronization loses one writer's changes to the other. And nothing is verified twice, because holding a digest is proof of holding the contents.

Refs carry the mutable part, in the way Git keeps branch names apart from objects. A ref is a few dozen bytes, so a last writer wins race on one is harmless and both blobs survive it. letify reserves `env/<env key>` and `ckpt/<name>`; the rest of the namespace belongs to the user.

### Blob granularity

> The unit of a blob is a decision. Large files stand alone; trees of small files are packed.

A model shard is already large, so one file is one blob. An environment is tens of thousands of small files, so the whole tree is packed into one archive keyed by the hash of its lock file. That is where the speedup is: tens of thousands of round trips become one.

A content addressed store does not by itself reduce the bytes of a first transfer. What it improves is the metadata exchange, which becomes a single manifest read instead of one request per file, and repeat transfers, which are skipped by name. Neither it nor a synchronization tool sends deltas within a changed file.

### Backends

| Backend | Used by | Note |
|---|---|---|
| `filesystem` | `Local`, `Shell` | A directory. The local machine can be the origin others pull from. |
| `gcs` | `Colab` | A Colab runtime is a Compute Engine virtual machine, so this is an internal transfer. Use a multi-region bucket, because runtime placement is not selectable. |
| `s3` | `Elice` and anything S3 compatible | Elice Data Hub speaks the S3 API. |
| `modal` | `Modal` | A Modal volume, mounted beside the container. |

## Environment

> An `Env` is a declaration keyed by the hash of a uv lock file, not a built image.

`Env()` reads `uv.lock`. `pip_install`, `run`, `vars` and `ship` refine it and return a new value. The key is a hash of the lock file contents together with the refinements, so two declarations that agree share a pooled runtime and a cached archive.

A uv lock file resolves for every platform uv supports, so one lock file drives a Linux runtime from a Windows or macOS client. A `pip freeze` list does not, because it carries platform specific pins.

### Module shipping

> Modules in the lock file are installed remotely by name. Modules that are not travel with the call.

A package the lock file names is installed in the runtime and referenced by name. A package it does not name, such as the project's own code or an editable install, has to be sent by value, because the remote side either lacks it or holds an older copy. letify infers the split from the lock file; `Env.ship()` overrides the inference.

## Transport

> A provider reaches its machine by the shortest path available. A tunnel is the last resort.

Order of preference: a direct SSH address, then a jump host, then a tunnel. Campus machines often accept one of the first two, which removes the tunnel's setup and its failure modes.

`Tunnel` builds the path and then uses the parent class's SSH path for everything else. Tailscale is the default because it needs no server of the user's own, authenticates from an auth key without a prompt, and carries any TCP port. frp on TLS port 443 is the fallback for a network that blocks UDP, where Tailscale keeps working but falls to a relay whose throughput has been measured as low as 2.2 Mbit/s across continents.

MTU is held at 1280 to 1400. Every mesh VPN in this class shows the same failure above that: the connection works, small commands work, and bulk transfers stall silently.

Colab is reached through the Colab CLI: `colab new` and `colab stop` for the session, `colab exec` for commands, and `colab ssh --proxy-mode` as an OpenSSH ProxyCommand bridge. That is an official path, so it carries no terms risk, and it needs no tunnel. Network details and the measurements behind these choices are in [NETWORK.md](NETWORK.md).

## Configuration

> Accounts live in the home file, project defaults live in the repository file, secrets live in neither.

`~/.letify` holds accounts and connection details, which belong to the machine. The project's `.letify` holds defaults that are safe to commit. The project file refines what the home file declared, so a repository can be cloned by someone else and run under their own accounts.

An alias must be a Python identifier, because providers are reached by attribute access. `any`, `gpus` and `active` are reserved. Declaration order sets the priority for `let.providers.any`.

A credential field is never a literal in a tracked file. `<name>_env` names an environment variable and `<name>_keyring` names a keyring entry as `service/user`, both resolved when used.

```toml
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
zone_id = "00000000-0000-0000-0000-000000000000"
machine_id = "00000000-0000-0000-0000-000000000000"
access_token_env = "ELICE_ACCESS_TOKEN"
```

## Packaging

> The base install carries no provider dependency. Each provider is an extra.

`letify` alone installs cloudpickle and blake3. `letify[colab]`, `letify[modal]`, `letify[shell]`, `letify[gcs]`, `letify[s3]` and `letify[all]` add what a provider needs. No provider dependency is imported at package import time, so a provider whose package is absent reports itself unavailable and everything else keeps working.

## Known gaps

> Implemented and unimplemented, stated plainly so nobody builds on a promise.

- **A persistent session process is not implemented.** Each call currently runs in a fresh remote process, so the object table that backs handles does not survive between calls. `keep_remote=True` returns a handle, and resolving one in a later call needs the session daemon.
- **Call forwarding is a capability probe, not a client.** `letify.remoting` reports what forwarding would need and refuses when it is missing. The driver shim itself is not written.
- **Modal and Elice runtimes are not exercised against the live services.** Their provider code follows each service's published interface, and the Elice paths come from Elice's own Terraform provider, but neither has been run end to end.
- **The Colab data channel is unverified.** Whether `ssh -L` works over `colab ssh --proxy-mode` is an open decision in [INTENT.md](INTENT.md).
- **Volume materialization inside a runtime is a stub.** A volume's mount point is created and a cached environment archive is unpacked when one exists, but the blob store is not yet mounted or mirrored into the runtime.
