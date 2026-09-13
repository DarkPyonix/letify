# Specification

> The current design of letify. Decisions only. Derivations and measurements live in the experiment pull requests, which each section links to when one exists.
>
> This file is the source of truth. Code follows it, and every test traces to a section here. See "How this project is built" in [CLAUDE.md](../CLAUDE.md).

## Declaration surface

> A declaration places three things and names no mechanism.

The public surface is five names: `Launcher`, `Env`, the `function` decorator it carries, `grid` and `zip`. Everything else is reached through a provider object.

```python
import letify

let = letify.Launcher()
env = letify.Env()

colab = let.providers.colab_a

@let.function(device=colab.G4, host="remote", env=env)
def train(lr, bs):
    ...

train(lr=1e-4, bs=32)
```

The decorator takes `device`, `host`, `env`, `volumes`, `timeout`, `retries` and `keep_remote`. `timeout` has no default: a deadline letify invented would end a two hour training run at whatever hour it guessed, which is letify deciding how long the user's own work is allowed to take. It takes no transport, no mode and no width: the three placements below settle where the work runs, and how much can run at once is the provider's inventory rather than a number on the declaration.

### The two placements <!-- id: the-three-placements -->

> `device` says where the accelerator is and `host` says where the host code runs. Both are said in the declaration and nowhere else.

`device` carries the provider, the account, the accelerator and how many of it one session takes, because those are one decision. `colab.G4` is such a value, and so is `lab.A100 * 2` for a run that trains across two cards. Core count and memory are not arguments: they arrive with the shape the provider registered, and a provider that offers several sizes registers them as separate shapes.

`host` is the CUDA word for the CPU side, paired with the device the declaration already placed. `"local"`, the default, keeps Python and the libraries in this process and forwards only CUDA calls. `"remote"` ships the declared function to the machine that holds the device.

`host` takes `letify.local` or `letify.remote`, which are the two members of a string enum defined next to `Instance`. The enum class itself is not part of the public surface: two named values say everything a declaration needs, and a class at the top of the package was one more name to learn for the same two choices. Because the members are strings, `host="remote"` is the same value. An unrecognized value raises at declaration time with both options named.

An instance carries no placement of its own. `colab.G4` says which card, and only the declaration's `host` says where the host code runs, so there is one place to read to know where a function runs. How long a session lives is not a declaration argument either: it is the `keep_alive` block around the calls, described under Lifetime.

### Invocation

> Calling a declared function runs it. Blocking behaviour is declared at the `def` site.

A declared function is called like any other. There is no second verb such as `.remote()`: a decorator that wraps a `def` and then needs another call to run it has moved the declaration out of the declaration.

A plain `def` blocks and returns its value. An `async def` returns a coroutine when no space is passed, so the standard library accepts it wherever a coroutine is expected, and an awaitable that is also async-iterable when a space is passed.

`Function.local()` runs the body in the calling process. It exists for testing a body with no provider; the preferred way to run locally is a `Local` provider, which keeps the production code path.

### Fan-out

> Passing a space where a scalar is expected declares that the argument varies.

`grid(**axes)` is the Cartesian product of its axes. `zip(**axes)` pairs them position by position and rejects axes of unequal length. Two spaces combine with `|`, which drops duplicate points. A scalar axis stays fixed across the space. `with_fixed(**kw)` adds arguments constant across every point.

A space is consumed by the language's own protocols. `await` collects results in input order; `async for` yields them as they complete. A sync declaration returns a list in input order.

Only one space may be passed per call. Two would make the point count the product of two arguments rather than something visible in one place.

A space runs as wide as the provider has devices for, and no wider. Nothing on the declaration bounds it, because a bound there would be a second statement of the same fact: the inventory already says how many cards exist, and a point that cannot reserve one waits for a point that can to finish.

A declaration taking two cards halves the width on a four card machine, which is arithmetic rather than policy. That is also why a width knob could not work: with `device=lab.A100 * 2`, a number that says how many runtimes may exist says nothing about how many cards they need.

## Provider model

> A provider object is one account on one kind of infrastructure. It registers the instances it offers, owns storage, opens the channel, and starts and stops the session.

```
Provider (abstract)
├── Local                 persistent
├── Modal                 persistent
└── Shell                 ephemeral by default, reached through the connection pipeline
    ├── Colab             session created by the Colab CLI, rendezvous over colab exec
    ├── Tunnel            a machine behind NAT, rendezvous through letify client shell connect
    └── Elice             machine allocated through the Elice Cloud API, rendezvous over its API
```

`Shell` is named for the shared ability, which is running a command on a remote machine, rather than for SSH, which is only its default transport.

A provider is built from one entry in the configuration file and reached by attribute on `let.providers`. Three attribute names there are reserved: `any` for a request that does not name a provider, `devices` for the registered accelerators of every provider, and `active` for the providers that currently hold a runtime.

### Provider properties

> Four class attributes set every behaviour that differs between providers, and none of them is a user-facing switch.

`persistence` says whether storage outlives a runtime.

`has_fast_path` says whether the machine is close enough for CUDA call forwarding to pay off. It does not gate the mode: a declaration that asks for forwarding over a long link gets a warning carrying the arithmetic and then runs.

`persistent_channel` says whether a worker process can be kept alive behind a pipe.

`needs_lease` says whether a session can outlive this process and keep billing.

| Provider | Persistence | Fast path | Channel | Store backend |
|---|---|---|---|---|
| `Local` | persistent | yes | persistent | `filesystem` |
| `Modal` | persistent | no | persistent | `modal` |
| `Colab` | ephemeral | no | persistent, or one-shot by configuration | `gcs` |
| `Shell` | ephemeral, overridable | yes | persistent | `filesystem` |
| `Tunnel` | ephemeral, overridable | yes | persistent | `filesystem` |
| `Elice` | persistent | yes | persistent | `filesystem` |

`Shell` and its subclasses default to ephemeral because a machine's disk policy is not knowable in advance. Assuming ephemeral costs time, since letify rebuilds the environment each runtime and the work still succeeds; assuming persistent fails outright when the disk turns out to be wiped. A configuration entry overrides it with `persistent = true`.

### Remaining usage

> Every provider is asked the same question, and a provider that cannot answer says so instead of guessing.

`Provider.usage()` returns a `Usage` record: the alias, the unit the account is metered in, how much is left, how much is spent, the ceiling, the hourly rate of what is running now, when the figure was taken, and where it came from. Every field except the alias, the unit and the source may be `None`, because a missing number is information and a fabricated one is not.

A provider reports what its service actually publishes:

| Provider | Unit | Remaining | Comes from |
|---|---|---|---|
| `Local` | hours | unmetered | nothing to ask; this machine bills nobody |
| `Elice` | KRW | not published | live allocations priced from the zone price list, which gives the rate and the spend, not the balance |
| `Colab` | compute units | not published | the CLI has no balance command; the figure is in the web console |
| `Modal` | USD | not published | the SDK exposes no workspace balance |
| `Shell`, `Tunnel` | hours | not published | a machine letify only runs commands on has no account behind it |

Where the service publishes nothing, a configuration entry supplies the number itself:

```toml
[colab_a]
kind = "colab"
usage_command = "my-colab-units"   # prints the remaining amount
usage_unit = "compute units"
usage_limit = 100.0
```

The last number in the command's output is read as the remaining amount. This exists because the alternative is letify inventing an endpoint, and a wrong balance is worse than an absent one. The command runs only when usage is asked for, never during a call.

`letify usage` prints one row per declared provider, and `letify usage <alias>` one provider. A provider whose optional dependency or setting is missing is reported as unavailable rather than skipped, so the table always lists every alias.

### Inventory

> A provider entry declares which accelerators the account can get and how many of each. That inventory is the only thing that bounds how much runs at once.

Three facts force this, and no launcher-level number can express any of them.

A Colab account's available accelerators depend on its state: the tier, and whether the compute unit balance is positive. So the kinds are per account, and they change without letify being told.

A shared department machine holds several cards in one box, and which indices are free moves with whoever else is logged in. So an entry names the indices it may use, `indices = "0-3"`, and at the moment a session starts letify takes only those registered indices that are actually free on the machine. A card another person is already computing on is skipped, not fought over.

A run can take more than one card. `device=lab.A100 * 2` asks for two, and on a four card machine that is two concurrent sessions rather than four. A number bounding how many sessions may exist cannot say that, which is the plain reason such a number is not in the declaration.

| Field | Means | For |
|---|---|---|
| `count` | How many of this accelerator the account can hold at once | A provider that assigns the device itself, such as Colab or Modal |
| `indices` | Which device indices on the machine letify may use, as `"0-3"` or `[0, 1, 6]` | A machine letify shares with other people, where it sets the visible devices itself |

An entry with `indices` has a count: the number of indices. An entry with neither is one of that accelerator.

Which registered indices are free is read with `nvidia-smi` at reservation time, not cached, because the answer changes while a run is queued. A card is taken as busy when another process is computing on it. Nothing else on the machine is inspected, and letify never kills anything.

A reserved session sets the visible devices for its own process, so the training code sees its cards as 0 upward and needs to know nothing about which physical indices it was given.

### Instances

> An `Instance` is one accelerator shape on one provider account.

`colab.G4` is an `Instance`. It holds the provider, the accelerator name, the host placement, how many devices one session takes, and the core count, memory and VRAM the provider reported. `n * instance` returns a copy taking `n` devices. An instance has no method that changes where the host code runs, because that is the declaration's `host`.

The device count is part of the pool key, because a session holding two cards is not interchangeable with one holding one.

Because an instance carries its provider, `device=colab.G4` fixes provider, account and accelerator in one argument. `let.providers.any.G4` defers the provider choice to the first declared provider that registers a matching accelerator, in configuration order.

Instance discovery is lazy and cached. A provider that must connect to enumerate its accelerators does so on first access, never at import time, and a configuration entry may list `gpus` to skip the connection. `refresh()` asks again.

`Local` reads its accelerator names once per process, because asking `nvidia-smi` takes seconds on a laptop whose discrete GPU is asleep and the answer does not change while the process runs.

Accelerator names are normalized so they can be attributes. `NVIDIA RTX PRO 6000 Blackwell` becomes `RTX_PRO_6000`. Colab calls the same card `G4`, which is what its CLI accepts, and accepts `RTX_PRO_6000` as an alias for it.

### GPU utilization

> How hard each declared instance's accelerator is working right now, read from the machine that owns it.

`letify utilization` reports one row per instance: the provider alias, the accelerator, and for each physical device its utilization percentage, memory used against memory total, temperature and power draw. `nvidia-smi --query-gpu` is the single source, because it is the only reading present on every machine letify reaches and it needs no framework loaded.

Where the reading comes from depends on where the device is. An instance on the local provider is read by running `nvidia-smi` here. An instance on a remote provider is read inside its live session, by shipping the same reader function through the ordinary call protocol, so no new channel and no new remote dependency is involved.

An instance with no live session reports no devices and says why, because starting a session to measure its load would cost money and change the answer. A machine without `nvidia-smi` reports no devices with that as the reason. Neither is an error: the table lists every declared instance either way.

The reading is taken at the moment it is asked for and carries no history. A load that has to be watched over time belongs in the caller's own loop, not in a CLI that shells out to `nvidia-smi` per poll.

## Execution modes

> Two modes exist. `host` picks between them and nothing derives it.

**Function shipping** (`host="remote"`) serializes the declared function with cloudpickle and runs it inside the runtime. The whole loop executes there, so its host synchronizations never cross the network.

**Call forwarding** (`host="local"`) keeps Python and the libraries in the local process and forwards only CUDA driver calls. Local data and the local environment stay in place, at the cost of one network round trip at every point where the host reads a value back from the device.

A provider refuses a mode only when it cannot serve it. `Modal` refuses `host="local"` because it exposes function calls into a container and there is no device to forward at. A provider without a fast path warns with its expected round trip and then runs, because the choice belongs to whoever wrote the declaration.

### Efficiency model

> Efficiency against a direct run is `T / (T + k * RTT)`, where `T` is GPU time per step and `k` is host synchronizations per step.

Numbers for an RTX PRO 6000 with NVFP4, a 0.5 s micro step, at a 150 ms round trip:

| Workload | Function shipping | Forwarding, default settings | Forwarding, tuned |
|---|---|---|---|
| LoRA fine-tuning | about 99 percent | 53 percent | about 96 percent |
| Decode, batch 1 | hundreds of tokens per second | 2 to 7 tokens per second | unchanged |
| Evaluation, teacher forcing | about 99 percent | about 99 percent | about 99 percent |

`k` is about three for a default Hugging Face training step: the trainer's NaN filter every step, the SDPA attention mask check every forward, and logging or the gradient scaler. Tuning means turning the NaN filter off, removing the mask check with fixed length packing, and moving logging to the gradient accumulation boundary, which leaves about one synchronization per optimizer step.

A faster GPU makes forwarding worse, because `T` shrinks while `RTT` does not. The same step on an L4 in bf16 takes 1.8 s and reaches about 80 percent where the RTX PRO 6000 reaches 53 percent.

Decoding fails at any useful latency. A decode step for 4-bit weights on an RTX PRO 6000 is 2 ms to 3 ms and synchronizes once or twice per token, so throughput is bounded near `1000 / (k * RTT)` tokens per second regardless of the card.

`letify.remoting.efficiency(step_seconds, syncs, round_trip_ms)` computes this, and `letify efficiency` exposes it on the command line.

## Channels

> A channel is how letify talks to a runtime, and which kind a provider offers decides what letify can do there.

A **persistent channel** keeps one worker process alive behind a pipe. Requests are framed lines, so the object table, the blob table and anything written to disk all survive between calls.

A **one-shot channel** can only run a command and collect its output. Every call starts a fresh process, so nothing persists. It exists because some transports offer nothing more, and it refuses the operations that need persistence rather than pretending.

Both hand back the user's own stdout separately from the outcome, because they share one stream.

The worker source cannot be sent on standard input as a script, because `python -` reads to end of file before compiling anything and the pipe has to stay open for requests. A small bootstrap stub passed with `-c` reads a length-prefixed base64 blob, executes it, and leaves standard input where it was.

## Call protocol

> A call is a serialized function plus arguments, and the outcome comes back on the same channel.

The local side pickles `(function, args, kwargs)` with cloudpickle and sends it as a framed request. Framing is one base64 line per message, which survives an SSH channel, a WebSocket bridge and a plain pipe without any of them mangling it.

On a one-shot channel the call travels inside a driver script that prints its outcome between `__LETIFY_RESULT_BEGIN__` and `__LETIFY_RESULT_END__`, so it can be found in a stream that also carries the user's prints. Absence of the marker is not a protocol quirk: it means the remote process died, and letify reports that as `ProtocolError` naming the likely causes.

An `async def` body is awaited on the remote side, so it runs to completion there and can use `await` internally.

### Handles

> A value may stay in the runtime. The caller receives a reference scoped to that session.

A declaration with `keep_remote=True` registers its return value in the runtime's object table and returns a `Handle`. Passing a handle to a later call on the same runtime resolves it in place, so a model stays on the remote machine instead of being copied back and forth.

A handle names the live runtime that holds it, not the pool key, because two runtimes can share a key and an object lives in only one of them. Passing it to a call on another runtime raises `HandleScopeError` rather than materializing the object, since resolving it across that boundary would mean an unrequested transfer of everything it points at.

`keep_remote=True` on a one-shot channel fails with its reason, because there is no process for the handle to point at once the call returns.

### Argument addressing

> Large arguments are named by the hash of their contents, so the same value travels once.

An argument above 64 KB is pickled and hashed, the runtime is asked which digests it already holds, and only the rest is sent. A later call carrying the same value sends a `Blob` reference instead of the bytes.

Hashing is not a bottleneck at any link speed involved: blake3 runs at gigabytes per second where a home uplink runs at megabytes per second. blake2b from the standard library is the fallback.

### Failure and retry

> Infrastructure failure may be retried. User code failure never is. Neither falls back to a slower path.

`RuntimeFailure` and `ProtocolError` mean the session misbehaved, so the runtime is discarded and the call is retried on a fresh one up to `retries` times. `RemoteError` means the shipped function raised, and it propagates with the remote traceback attached.

letify never falls back to local execution or to a slower mode when the declared one is unavailable. A silent downgrade turns a four times slowdown into a mystery.

## Sessions

> A runtime is one live session and the only object that costs money.

Everything above a runtime is declaration. Creating one is when a provider actually powers something on; shutting it down is when the charge stops.

A runtime boots in four steps: open the channel, arm the lease, install the declared environment, attach volumes. Installation is skipped where the machine already runs in the environment, which is the local provider.

### Pooling

> Runtimes are pooled by instance and environment, so the second call through a declaration pays nothing for setup.

The pool key is the instance key joined with the environment key. Two declarations that agree on both share runtimes, which is why nothing has to be said for two functions on one device to reuse a session.

How many sessions may exist is the provider's inventory and nothing else. Starting one reserves the devices its instance asks for. A call that cannot reserve them waits only while a session in this process that holds that accelerator is serving a call, because that session gives its devices back when the call finishes; a sweep wider than the inventory relies on exactly this. In every other case the devices cannot be allocated, and the call raises `InsufficientDevices` at once, naming what holds them: an idle session that a `keep_alive` block is keeping, a card another process is computing on, or a request for more devices than the account declares. Waiting there would never end, because nothing letify is running would free a device. `InsufficientDevices` is not retried, since a retry asks for the same devices from the same inventory. There is no ceiling on the launcher: a number there would be a guess about hardware the provider entry already describes, and when the two disagreed the smaller would win silently.

A session is never a value the caller holds. Pooling, reuse and teardown are decided from the declaration and the `keep_alive` block around it, so there is no call that starts a session, none that returns one, and none that takes one. `Runtime` exists, and letify hands it to a provider and to a volume, but it does not appear in anything a user writes.

### Lifetime

> A session ends with the call that needed it. `with let.keep_alive():` keeps sessions for the length of a block. Nothing else decides.

1. **The call.** A session ends when the call that started it finishes. A search space counts as one call, so a sweep starts its runtimes once and releases them once.
2. **A `keep_alive` block.** Inside `with let.keep_alive():` a session is not ended when its call finishes, so the next call in the block that matches its instance and environment reuses it and pays no session start, which is minutes on Colab. When the block exits, every idle session ends; one still serving a call ends when that call finishes. Blocks nest, and only the outermost exit ends anything.
3. **The lease.** The local process renews a deadline inside the session every 30 seconds, and the worker exits on its own if the deadline passes. The grace period is 300 seconds, so a brief network drop does not kill a training run.

Keeping is a block rather than a declaration argument because it describes a stretch of the caller's program, not a property of one function: the same function is kept in one script and not in another. A block also has a visible end, so no session outlives the code that asked for it.

Nothing is torn down by hand and nothing is torn down on a timer. There is no release call and no shutdown call on the public surface, and everything left goes at process exit.

The lease is the one exception, and it is not a timer on the work: it covers the moment a process is killed outright, which is the one moment nothing can be told to anybody. `SIGKILL`, the out of memory killer and a power cut all run no code at all, so a session that only ends when asked would never be asked.

What the lease actually does is exit the worker process, which releases the occupancy. Whether that stops the billing depends on what the provider charges for, and infrastructure cannot choose to switch itself off: something that owns it has to. So the guarantee is per provider and letify states it rather than implying one.

| Provider | What is billed | Killed local process |
|---|---|---|
| `Local` | nothing | the subprocess dies with its parent |
| `Shell`, `Tunnel` | nothing; the card is occupied | the worker exits, so the card frees |
| `Modal` | the sandbox | **guaranteed.** A deadline is set when the sandbox is created and Modal enforces it |
| `Colab` | the runtime | not guaranteed. Colab's own idle policy is what ends it |
| `Elice` | the allocation | **not guaranteed.** An allocation bills until something issues the delete |

Where it is not guaranteed, the preferred answer is a deadline at creation time, because the platform outlives the caller. Modal takes one and letify sets it. Whether the Elice allocation API takes one is unverified.

Where the platform takes none, the intended bound is reconciliation: the next letify process asks the provider what is running under this project's name and ends what nothing is watching. That is not immediate, and it is **not implemented yet**, so today an Elice allocation left by a killed machine bills until somebody deletes it. It is listed under Known gaps.

There is no detached execution. A detached run whose remote side is preempted would lose its results, so the local process stays the owner and durability comes from checkpoints in the store.

### Status reporting

> What is running, counted rather than described, with no internal bookkeeping in it.

`Launcher.status()` answers three questions: how many sessions exist, how many are serving a call, and what each one is. `live` and `busy` are counts, and `devices` reports each provider's inventory against what is reserved, so a reader can see at a glance whether a call is waiting for a card. `runtimes` describes each session: its name, provider, accelerator, the device indices it holds, placement, whether it is busy and how long it has been idle.

Nothing internal is reported. The pool holds a guard so that one invocation does not restart a session between the points of a sweep, and whether that guard is currently open is a fact about the pool's implementation rather than about what is running. A field among counts that looks like a count and is actually a boolean is worse than no field, because it is read as a count.

`status()` describes this process only. A session started by a different process is not in it, since the pool lives in the process that owns it. What a machine itself is doing is a different question, answered by `letify utilization`.

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

A content addressed store does not by itself reduce the bytes of a first transfer. What it improves is the metadata exchange, which becomes a single manifest read instead of one request per file, and every repeat transfer, which is skipped by name. Neither it nor a synchronization tool sends deltas within a changed file.

Extraction checks every member's path against the destination before unpacking, so an archive cannot write outside it.

### Materializing into a runtime

> A volume writes into a runtime through its channel, not by asking the runtime to reach the bucket.

That works with every backend and needs no credentials on the far side, at the cost of the bytes passing through the local process. `Volume.resume()` puts the newest checkpoint for a name inside the runtime, which is what makes a preempted session cheap to restart. `Volume.absorb()` pulls one back out, and `cache_env_from()` packs an environment installed inside a runtime so the next session skips the installation.

These take the declaration, not a session. A declaration already says which provider, which accelerator, which environment and which volumes, so which session is letify's answer to work out and not a value for the caller to carry. The alternative was tried and is worse: a caller holding a session has to have asked for it with the same instance the declaration uses, and the declaration folds the host placement into that instance, so asking with the bare one silently starts a second session that holds none of the first one's files. On one machine that still passes, because both sessions see the same disk. On a rented one it fails, which makes it the worst kind of defect: it works in the test and breaks where the money is.

So nothing hands a session to the caller. There is no call that returns one, no argument that takes one, and no way to hold the wrong one.

Moving a file through a session only works while that session exists, so `absorb`, `resume` and `cache_env_from` are called inside `with let.keep_alive():`. Outside one they raise `UnsupportedMode`, because the session they would use ends as soon as they return and a resumed checkpoint would vanish before the call that needs it.

### Backends

| Backend | Used by | Note |
|---|---|---|
| `filesystem` | `Local`, `Shell`, `Elice` | A directory. The local machine can be the origin others pull from. On Elice it sits on the machine's own disk. |
| `gcs` | `Colab` | A Colab runtime is a Compute Engine virtual machine, so this is an internal transfer. Use a multi-region bucket, because runtime placement is not selectable. The client is the standard library HTTP client against the Cloud Storage JSON API. |
| `modal` | `Modal` | A Modal volume, mounted beside the container. |

Every backend answers "which of these digests are missing" with one listing rather than one request per digest, because object level requests are billed and add latency.

## Environment

> An `Env` is a declaration keyed by the hash of a uv lock file, not a built image.

`Env()` reads `uv.lock`. `pip_install`, `run`, `vars` and `ship` refine it and return a new value. The key is a hash of the lock file contents together with the refinements, so two declarations that agree share a pooled runtime and a cached archive.

A uv lock file resolves for every platform uv supports, so one lock file drives a Linux runtime from a Windows or macOS client. A `pip freeze` list does not, because it carries platform specific pins.

### Module shipping

> Modules in the lock file are installed remotely by name. Modules that are not travel with the call.

A package the lock file names is installed in the runtime and referenced by name. A package it does not name, such as the project's own code or an editable install, has to be sent by value, because the remote side either lacks it or holds an older copy. `Env.ship()` overrides the inference.

## Transport

> A `Shell` reaches its machine through a connection pipeline: several strategies are tried at once, the fastest acceptable one wins, and the winner is cached per account.

`Modal` and `Local` are not part of this. Modal is reached through its own API, and Local starts its worker as a child process.

The measurements behind the order and the rules below are in [NETWORK.md](NETWORK.md#connection-pipeline-measurements).

### Connection strategies <!-- id: connection-strategies -->

> Four default strategies, ranked. A lower rank wins whenever it is acceptable.

| Rank | Strategy | Needs |
|---|---|---|
| 1 | Forward SSH to the machine's address | an address the user's machine can reach |
| 2 | TCP hole punching | a rendezvous, and NATs on both sides that allow a simultaneous open |
| 3 | UDP hole punching with Tailcat, then SSH over it | a rendezvous, and UDP in both directions |
| 4 | Provider fallback | the provider's own path, such as `colab exec` and the Colab file API |

Forward SSH is first because it needs no remote agent and costs one connection attempt, and when it works it is a kernel TCP connection. TCP hole punching comes before UDP because a punched TCP connection is also kernel TCP and is not subject to UDP rate limits. Tailcat comes next because its NAT traversal succeeds more often, but it runs in user space and a network that limits UDP limits it too. The provider fallback is last because it is the slowest.

A strategy whose needs are not met is skipped, not attempted. A `Shell` with no rendezvous has only rank 1.

Each stage is its own object behind a small interface, so a strategy can be added, removed or reordered by changing the subclass's list:

| Object | Owns |
|---|---|
| `Rendezvous` | sending a command to the remote side and exchanging addresses with it |
| `Strategy` | one way to connect: checking its needs, attempting, and returning a `Link` |
| `Link` | an established connection, used by both the worker channel and bulk transfer |
| `Probe` | a short measurement of a `Link` |
| `Pipeline` | running strategies, choosing one, and consulting the cache |
| `LinkCache` | remembering the winning strategy |

A `Shell` subclass differs from its parent only in its `Rendezvous` and its strategy list.

### Choosing a link <!-- id: choosing-a-link -->

> All applicable strategies start together. Among those that connect and pass the probe, the lowest rank is chosen, unless it is far slower than the fastest.

Strategies are raced rather than tried in turn, so a strategy that times out does not delay the others. Once the first strategy connects, the pipeline waits a grace period of 2 s for lower ranked strategies before choosing.

Every connected strategy is probed: 30 round trips, then 2 s of transfer in each direction. A strategy whose throughput in either direction is below 25% of the fastest connected strategy in that direction is rejected. The lowest ranked strategy that remains is chosen. When only one strategy connects, it is chosen without comparison.

Strategies that lose are closed, including one that connects after the choice is made.

When only one strategy is applicable there is nothing to choose between, so it is used directly: it is not raced, not probed and not cached, and a failure surfaces at its first use. When a race of two or more ends with one connected strategy, that strategy is still probed so the cache has throughput to compare against, but a failed probe does not reject it.

A `Link` that cannot carry the probe, such as the provider fallback, is left out of the throughput comparison. It is chosen only when no probed strategy remains, and it is never written to the cache, so the next connection races again. A probe that fails on a strategy in a race of two or more rejects that strategy. When no strategy connects, the error names every strategy with the reason it failed or was skipped.

### Link cache <!-- id: link-cache -->

> The winning strategy is remembered per account and per network, so the next connection starts with it alone.

The cache lives in `~/.letify/accounts/<alias>/link.json`. It records the strategy, the probe results, and a network fingerprint: the local machine's public IP address and the name of its default route interface.

On the next connection the cached strategy is attempted alone. When it connects and its probe is at least 50% of the cached throughput in both directions, it is used. Otherwise, or when the fingerprint differs, the full race runs and the cache is rewritten. A fingerprint whose public IP address could not be learned matches nothing, so the race runs.

The public IP address is learned from the same STUN servers the punch uses. The default route interface is read from `/proc/net/route` on Linux, `route -n get default` on macOS and `Get-NetRoute` on Windows.

### Rendezvous <!-- id: rendezvous -->

> Hole punching needs a way to start a program on the remote side and swap addresses. For Colab and Elice the provider layer fills that role; a plain machine behind NAT runs `letify client shell connect`.

| Provider | Rendezvous |
|---|---|
| `Colab` | the provider layer: `colab new` creates the runtime and `colab exec` runs the remote half |
| `Elice` | the provider layer: the Elice Cloud API creates and allocates the machine, and the remote half runs over forward SSH to it |
| `Tunnel`, and a plain `Shell` behind NAT | the remote agent started by `letify client shell connect`, reached over Tailcat |

Colab and Elice never need `letify client shell connect`. Their create and open step is what puts letify's remote half on the machine.

`letify client shell connect` is run once on a plain machine by its user. It starts the remote agent on a port the operating system chooses, starts `tailcat serve <agent port>` in front of it, and prints the Tailcat address and the agent port. The user writes both into the account once, as `tailcat = "<address>"` and `tailcat_port = <agent port>`. The agent needs letify installed on that machine.

A connection to the agent is told apart by its first bytes: `SSH-` is spliced to the machine's SSH server, `LETIFY-RDV ` is followed by one JSON request line and answered with one JSON line, and `LETIFY-PROBE` is followed by the probe. For such an account the pipeline connects over Tailcat first, runs `tailcat <address> <agent port>` to exchange the TCP punch mapping and start time over that link, and then races as specified: the Tailcat link is the rank 3 candidate, and when TCP punching passes the probe it takes over.

Both sides learn their public mapping from STUN servers reached over TCP on port 443, because networks that restrict outbound ports usually still allow 443. A punch starts at a time both sides agree on through the rendezvous. Each side connects from its bound port to the other's mapping and listens on the same port, so whichever direction's SYN arrives first completes the connection.

Every socket in a punch is bound with `SO_REUSEADDR`, and with `SO_REUSEPORT` where the platform has it, so the STUN connection, the listener and the connecting socket share one port. Both sides may complete a connection in each direction. The user's side takes the first connection that completes and writes a hello carrying a 16 byte token the two sides agreed on through the rendezvous. The remote side keeps the connection on which that hello arrives and closes the others.

A punched connection first answers the probe, then on request is spliced to the remote machine's SSH server, and SSH runs over a local forwarding port. A second SSH connection over the same link punches again.

A local port that letify binds for forwarding is chosen by the operating system, never fixed, because Windows reserves port ranges that vary by machine. The remote end of a reverse forward is chosen the same way, with `ssh -R 0:`.

The remote half of a punch, a Tailcat listener or a reverse forward is one standard library Python module sent as source, so a Colab or Elice machine needs Python and nothing else. Through the provider layer it runs as a detached process that prints its answer and outlives the command.

Tailcat is run as `tailcat serve <port>` on the remote side, which prints `Server listening with new address: <address>`, and as `tailcat <address> <port>` in an SSH `ProxyCommand` on the user's side. It is applicable only when `tailcat` is on the user's PATH.

### Reverse SSH <!-- id: reverse-ssh -->

> An opt-in strategy for a remote machine that can reach the user's machine over SSH. It is not raced by default.

When an account sets `reverse_ssh`, the remote side opens an SSH connection to the user's machine and forwards a port back with `ssh -R`. It then takes rank 4 and the provider fallback moves to rank 5.

```toml
[lab_behind_nat]
kind = "shell"
reverse_ssh = { address = "home.example.com", port = 2222, user = "me" }
```

It is opt-in because it needs the user's machine to accept inbound SSH and needs a key on the remote side that can log in to it.

The key is per session. letify generates a new ed25519 key pair for each session and appends the public key to the user's `~/.ssh/authorized_keys` with `restrict,port-forwarding,command="/bin/false"` and the comment `letify-session <alias>`. The private key reaches the remote side through the rendezvous. Closing the link removes the line. Before a new session's key is added, every line carrying the `letify-session` marker for that alias is removed, so a crashed session never leaves a key behind.

### Colab <!-- id: colab-transport -->

> Colab is a `Shell` whose rendezvous is `colab exec`. It has no forward SSH, and its fallback is `colab exec` with the Colab file API.

The Colab CLI runs as `uv tool run --from google-colab-cli colab`, with `jupyter-kernel-client<1` pinned, because release 0.6.0 of the CLI calls an API that jupyter-kernel-client 1.0 removed. `colab new` and `colab stop` manage the session.

Colab limits outbound UDP to roughly 200 packets per second, so rank 3 is expected to lose the probe there. It stays in the list because the ratio rule removes it without a special case.

A `channel = "exec"` entry skips the pipeline and uses the fallback directly. A Colab VM has no SSH server by default, so the rendezvous request asks the remote half to install `openssh-server` and start `sshd` first. The request carries the account's public key, `key` with `.pub` appended, which the remote side adds to `authorized_keys`. SSH over a punched or Tailcat link logs in as `root` unless `user` says otherwise.

The fallback sends calls with `colab exec` and bulk data through the Jupyter contents API that the Colab runtime proxy exposes: uploads are split into parts sent in parallel, each part in chunked `PUT` requests, and downloads read `/files/<path>` in parallel parts. The contents API root is `/` on the VM, not `/content`.

## Configuration

> Accounts live in the home file, project defaults live in the repository file, secrets live in neither.

letify keeps its state in two `.letify` directories. `~/.letify/` belongs to the machine and is never in a repository. `<project>/.letify/` belongs to the repository and is committed.

| Path | Holds |
|---|---|
| `~/.letify/config.toml` | Every account this machine has: kind and connection details, never a secret |
| `~/.letify/accounts/<alias>/` | That account's credentials and provider state, one directory per alias, owner only |
| `<project>/.letify/config.toml` | Project defaults, and the aliases of the accounts the project uses |

The project directory is `.letify/` in the working directory. `Launcher(config=...)` and `--config` name another `.letify` directory, or a `config.toml` directly.

### The two files <!-- id: two-files -->

> The home file is the set of accounts this machine has. The project file chooses from it and may override it. An account the project does not name is not available in that project.

An account in `~/.letify` is available in a project in exactly three cases:

1. **The project file names it.** A table with the same alias, even an empty one, is enough: `[colab_pro]` on its own makes the home account `colab_pro` available with all of its settings.
2. **The home entry is global.** `global = true` in the home entry makes the account available in every project, including one with no `.letify` at all. It is how an account meant for everything, such as a personal Colab, avoids being named in every repository.
3. **It is `local`.** The local machine is always available and never needs a declaration.

Every other home account does not exist as far as that project is concerned: `let.providers.<alias>` raises `UnknownProvider`, it is absent from `let.providers.aliases`, and `let.providers.any` never resolves to it.

When the project file names an account, its fields override the home entry's field by field. Fields the project does not set come from the home entry, including `kind`. A project table that sets `kind` itself is a complete declaration and needs no home entry, which is how an account with no connection details, such as a second `local`, is declared in the repository. A project table with no `kind` whose alias the home file does not have is a configuration error naming `letify login`, because there is nothing to take the kind from.

`global` is read from the home file only. A project cannot make an account global, because a repository deciding what every other repository on the machine can reach is the wrong direction.

An alias must be a Python identifier, because providers are reached by attribute access. `any`, `devices` and `active` are reserved. Declaration order sets the priority for `let.providers.any`.

A credential never appears in either `config.toml`. A field such as `access_token` is resolved in this order: the environment variable named by `access_token_env`, then the file `access_token` in `~/.letify/accounts/<alias>/`, then a literal value, which is accepted only so a home entry can carry a non secret default. A credential file is created with owner only permissions where the platform has them. Provider tools that keep their own login, the Colab CLI and the Modal SDK, keep it in the same account directory rather than in their usual location, which is what lets two accounts of one provider exist on one machine.

The OS keyring is not used. Reading it needs a package in the user's environment, and the provider tools that matter already store their tokens as files, so one mechanism covers every provider.

```toml
[colab_a]
kind = "colab"
account = "someone@example.com"

# Which accelerators this account can get, and how many at once. Colab assigns the device
# itself, so there are no indices to name.
[colab_a.devices]
G4 = { count = 2 }
T4 = { count = 2 }

[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true

# Eight cards in the box, four of them ours. letify takes only those of these four that are
# actually free when a session starts, so a card a colleague is computing on is skipped.
[lab_a100.devices]
A100 = { indices = "0-3" }

[elice_a100]
kind = "elice"
zone_id = "00000000-0000-0000-0000-000000000000"
machine_id = "00000000-0000-0000-0000-000000000000"
access_token_env = "ELICE_ACCESS_TOKEN"
```

The older `gpus = ["A100", "H100"]` list still works and means one of each, with letify choosing no indices. `devices` is what an entry uses once a count or an index range matters.

### Generated provider types

> Loading the configuration writes a type stub that names this project's accounts and their accelerators, so an editor completes `let.providers.colab_pro.G4` and flags a misspelled alias.

Aliases live in `.letify/config.toml`, not in code, so a type checker cannot know them. `Launcher()` therefore writes `letify_providers.pyi` describing the accounts this project can use under the two file rule above. `Launcher.providers` is typed as `letify_providers.ProvidersView`, and letify ships a `letify_providers` module whose `ProvidersView` is the plain `Providers`, so a project with no generated file keeps exactly today's types.

Each alias becomes a class named after it in CamelCase, `colab_pro` as `ColabPro` and `lab_a100` as `LabA100`, subclassing the provider class of its kind. A name that is a Python keyword gets `Provider` appended, and a name two aliases would share gets a number appended in declaration order. The class declares one `Instance` attribute per accelerator the account offers, and `ProvidersView` declares one attribute per alias. Where the accelerators are known, the class declares no fallback attribute lookup, so a misspelled accelerator is a type error; where they are not, attribute access stays typed as `Instance`.

Accelerators are taken from what can be known without a network call: the entry's `devices` table or `gpus` list, or the provider's fixed list for Colab and Modal, or this machine's own cards for `local`. Writing the stub never connects to a machine or an API. An alias whose provider cannot be built is typed as the plain `Provider`.

The file goes in a `typings` directory at the project root, which is Pyright's and Pylance's default stub path. The project root is the nearest directory upward from the working directory that holds a `pyproject.toml`, or the working directory when there is none. `[tool.letify] typings = "<path>"` in that `pyproject.toml` moves it, relative to the root, and the type checker's stub path has to point at the same place. `typings = false` turns generation off, and so does the environment variable `LETIFY_STUBS=0`.

The file is rewritten only when its content would change, so loading the configuration does not touch it on every run. `letify stubs` writes it on demand. It reflects one machine's `~/.letify/`, so it belongs in the project's `.gitignore`.

### Logging in

> One command writes the account to the home file and a reference to it in the project file, so a repository names the accounts it needs without holding any of them.

`letify login <kind> [alias]` declares one account. It writes two entries in two files, because the two files answer different questions.

`~/.letify/config.toml` gets the account: the address, the user, the key path, the zone, whatever that kind of provider needs to connect. It belongs to the machine and is never in a repository, so it is where a connection detail may live. A credential the login collects goes to `~/.letify/accounts/<alias>/`, created with owner only permissions where the platform has them.

The project's `.letify/config.toml` gets the alias as an empty table, `[colab_pro]`. Nothing else, because everything else is either a secret or a detail of one person's machine, and naming the alias is what makes the account available in the project. That table is what makes the repository self describing: a teammate who clones it can run `letify login` for the aliases it names and nothing else has to be explained. A named alias the home file does not declare is a configuration error naming the command that fixes it.

An account that is already in the home file is not asked for again. `letify login lab` in a second repository writes only the reference, which is the common case: the account was set up once and every project since then just needs to name it.

`letify logout <alias>` removes the account from `~/.letify/config.toml` and deletes `~/.letify/accounts/<alias>/` with everything in it. It leaves the project reference alone, because the repository still needs that account; what changed is only that this machine no longer has it.

`letify login colab <alias>` signs in to Colab itself. It runs `colab sessions` through `uv tool run --python 3.13 --from google-colab-cli colab`, with `HOME` set to `~/.letify/accounts/<alias>/`. The Colab CLI keeps its token at a fixed path under its home directory, so the token lands in the account directory and the CLI refreshes it on later calls. Every later Colab command runs with the same `HOME`, which is what lets two Colab accounts live on one machine. uv's cache, Python installs and tools stay pinned to the real home, so a changed `HOME` downloads nothing again. A sign in that exits non zero writes nothing.

Credentials never enter either `config.toml`. A token goes to a file in the account directory. An SSH password is never stored at all, which the next section explains.

### SSH authentication

> Key authentication, because the call path is non-interactive. A password is accepted once, to install the key, and then discarded.

letify opens sessions with `ssh -o BatchMode=yes`. That is not a preference: a session is started by the pool, in the background, possibly long after the call that needed it, so there is nobody present to answer a password prompt. A transport that requires interaction cannot carry a pooled session.

So `letify login shell` sets up key authentication and treats the password as a one-time input:

1. If the configured key does not exist, an ed25519 key is generated at `~/.ssh/id_letify` with no passphrase, because a passphrase would put the prompt back.
2. The public key is appended to the machine's `~/.ssh/authorized_keys`, over one interactive SSH connection that asks for the password in the terminal.
3. The password is used by that one command and then dropped. It is not written to a file or to the environment.
4. The connection is confirmed with `BatchMode=yes`, which proves the key works before the alias is declared rather than at the first call.

Two other approaches were considered and are not the default. Connection multiplexing with `ControlMaster` authenticates once and reuses the socket, but Windows OpenSSH does not implement it and a dropped socket ends a long run. `sshpass` feeds a stored password to each connection, which needs the password kept somewhere and exposes it in the process arguments of every call. `sshpass` is available as `auth = "password"` for a machine whose administrator forbids key authentication, reading the password from `~/.letify/accounts/<alias>/password`, and it refuses on Windows, where the tool does not exist.

### What each kind asks for

> Where a vendor owns the credential, letify records the account and leaves the credential to the vendor.

| Kind | Written to the home file | Credential |
|---|---|---|
| `shell`, `tunnel` | address, user, port, key path | an SSH key, installed by `login`; no password stored |
| `elice` | endpoint, zone, machine | access token in `~/.letify/accounts/<alias>/access_token` |
| `colab` | account email | the `colab` CLI owns it; `login` checks the CLI is present and says which command authenticates it |
| `modal` | workspace | the `modal` CLI owns it, in `~/.modal.toml`; `login` checks it is present |
| `local` | nothing | none; this machine needs no declaration |

For `colab` and `modal`, letify does not touch the vendor's credential store. Wrapping another tool's login would mean owning a token letify has no way to refresh, and the vendor's own command already works.

## letify-core

> The native component behind `host="local"`. A Rust workspace, built in CI and shipped inside platform wheels.

The Python package is pure Python. Standing in for the CUDA driver cannot be done from Python, so that job lives in `letify-core/` as three crates.

| Crate | Holds |
|---|---|
| `letify-wire` | The protocol. Each request declares whether it needs a reply. |
| `letify-driver` | A cdylib that stands in for the driver and forwards its calls. |
| `letify-agent` | Holds the real device and executes what arrives. |

`python letify-core/build.py` builds them and copies the library into `letify/remoting/lib/` under the name of the one it replaces: `nvcuda.dll` on Windows, `libcuda.so.1` on Linux and WSL2, `libletify_driver.dylib` on macOS. Being found before the real driver is the whole mechanism. The agent is copied beside it.

letify looks for the library in `LETIFY_CORE_PATH`, then in `letify/remoting/lib/`. It looks for the agent in `letify/remoting/lib/`, then on `PATH`.

### Batching

> Only a call whose result the host reads waits for an answer.

A launch, a copy to the device and an allocation change device state and return immediately, so they are queued. A copy back to the host, a stream synchronization and an elapsed time query cannot be, and each one is a round trip. That is why the round trip count is the number of host synchronizations rather than the number of calls, which is what makes the efficiency model hold for a step that issues thousands of calls.

### Virtual pointers

> An allocation returns a pointer immediately, and memory accounting stays local so that running out still fails at the call.

The local driver hands out pointers from a range no real device address falls in, records what they stand for, and lets the agent reconcile them in the background. Waiting for the agent would put a round trip in front of every allocation, and a caching allocator makes many.

The cost is honest failure. A caching allocator learns the device is full when the allocation call fails, frees its cache and retries. With a virtual pointer there is nothing to fail yet, so the local driver keeps its own accounting of device memory and refuses once the budget is gone, with a reserve held back for the driver's own context, library workspaces and fragmentation.

### Module identity

> A compiled module is named by its contents, so a fatbin the agent already holds is not sent again.

PyTorch loads the same modules on every process start and they are large. The agent keeps a table keyed by digest and answers with the handle it already has.

### Loading

> On Windows letify does the injection, because it has to happen before the first CUDA library is loaded.

`letify.remoting.inject()` calls `os.add_dll_directory` on the library's directory, which puts it at the front of the loader's search order. It must be called before `import torch`, and it says so when torch is already imported.

On Linux the equivalent is `LD_PRELOAD`, which cannot be set from inside a running process for libraries already resolved. So `inject()` reports the command to run rather than pretending it succeeded, because a silently ineffective injection would look like forwarding while the real driver was being used all along.

### Unimplemented entry points

> A missing entry point names itself and returns `CUDA_ERROR_NOT_SUPPORTED`.

The implemented set is what a PyTorch process touches to start up and run one kernel: initialization, device queries, allocation and copies, module loading, launches, streams and events. Everything else reports its own name, so the way to find out what a real workload needs is to run one and read the list.

Unified memory is the one exception that no amount of implementation removes. Managed memory works by letting the device fault into host pages, which needs one address space, and there is no such thing across a network. A paged optimizer cannot run under forwarding.

## Packaging

> One install, `uv add letify`, with no extras. It installs cloudpickle and blake3 and nothing else.

letify is a dependency inside a research repository, so it adds as little as possible to that repository's environment.

Provider tools run out of process and never in the user's `.venv`. Colab runs through `uv tool run --from google-colab-cli colab`. Modal runs in a separate uv environment that letify manages. Elice uses the standard library HTTP client. The `gcs` blob store uses a standard library client too.

uv must be installed. letify finds it from the `UV` environment variable, then from `PATH`. If neither has it, letify raises an error that says uv is required.

The Python code is pure and links no Python extension. letify-core binaries are built in CI and placed in `letify/remoting/lib/`, so each release has one wheel per desktop platform, tagged `py3-none-<platform>`:

| Platform | Wheel platform tag |
|---|---|
| Linux x86_64 | `manylinux_2_28_x86_64` |
| Linux aarch64 | `manylinux_2_28_aarch64` |
| Windows x86_64 | `win_amd64` |
| Windows arm64 | `win_arm64` |
| macOS arm64 | `macosx_11_0_arm64` |
| macOS x86_64 | `macosx_10_12_x86_64` |

Linux wheels are built inside the `manylinux_2_28` containers, so the binaries need glibc 2.28 or newer. That covers RHEL 8, Debian 10 and Ubuntu 18.10 onward. The sdist carries no binaries, and an install from it has no letify-core.

## Known gaps

> Implemented and unimplemented, stated plainly so nobody builds on a promise.

- **`letify-driver` covers one milestone.** The entry points a PyTorch process needs to start up and run one kernel are forwarded and verified against a real GPU. Kernel argument marshalling reads the pointer list without knowing the kernel's signature, and fatbin size comes from a conservative window rather than the image header. Both need a real workload to shape them.
- **`Modal` and `Elice` are not exercised against the live services.** Their code follows each service's published interface, and the Elice paths come from Elice's own Terraform provider, but neither has been run end to end.
- **The connection pipeline is not exercised against live networks.** `Rendezvous`, `Strategy`, `Link`, `Probe`, `Pipeline`, `LinkCache` and the remote agent are implemented and tested over loopback sockets and faked commands. Installing and starting `sshd` on a Colab VM over `colab exec` is not yet checked against a live runtime.
- **The Elice API runs no command on a machine.** The paths letify uses (virtual machine, allocation, instance type, pricing) create and power machines only, so Elice's remote half runs over forward SSH to the allocated machine, and the punch and Tailcat strategies need that SSH to succeed first.
- **The Colab file API is not part of the fallback link.** The fallback carries calls over `colab exec`; bulk transfer through the Jupyter contents API is not implemented.
- **Orphan reconciliation is not implemented.** A session whose controlling machine was killed outright is released by the lease on the providers where the process is the cost. Where the platform bills for the machine and takes no deadline, nothing ends it: an Elice allocation bills until a delete is issued. The intended answer is that the next letify process asks the provider what is running under this project's name and ends what nothing is watching, with a command to do it on demand. Neither exists yet.
- **Whether the Elice allocation API takes a deadline is unverified.** If it does, that is where the guarantee belongs, because the platform outlives the caller.
- **Persistence detection is not implemented.** Deciding a machine's disk policy by writing a marker file and looking for it in a later runtime is a decision recorded here, not yet code.
