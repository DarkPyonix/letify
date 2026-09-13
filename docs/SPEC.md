# Specification

> The current design of letify. Decisions only. Derivations and measurements live in the experiment pull requests, which each section links to when one exists.
>
> This file is the source of truth. Code follows it, and every test traces to a section here. See "How this project is built" in [CLAUDE.md](../CLAUDE.md).

## Declaration surface

> A declaration places three things and names no mechanism.

The public surface is four names: `Launcher`, `Env`, the `function` decorator it carries, and `session_cache`, which a declared body uses to keep a value for the length of its session. Everything else is reached through a provider object.

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

The decorator takes `device`, `host`, `env`, `timeout` and `retries`. Any other argument is refused by Python as unexpected. `timeout` has no default: a deadline letify invented would end a two hour training run at whatever hour it guessed, which is letify deciding how long the user's own work is allowed to take. It takes no transport, no mode and no width: the three placements below settle where the work runs, and how much can run at once is the provider's inventory rather than a number on the declaration.

### The two placements <!-- id: the-three-placements -->

> `device` says where the accelerator is and `host` says where the host code runs. Both are said in the declaration and nowhere else.

`device` carries the provider, the account, the accelerator and how many of it one session takes, because those are one decision. `colab.G4` is such a value, and so is `lab.A100 * 2` for a run that trains across two cards. Core count and memory are not arguments: they arrive with the shape the provider registered, and a provider that offers several sizes registers them as separate shapes.

`host` is the CUDA word for the CPU side, paired with the device the declaration already placed. `"local"`, the default, keeps Python and the libraries in this process and forwards only CUDA calls. `"remote"` ships the declared function to the machine that holds the device.

`host` takes `letify.local` or `letify.remote`, which are the two members of a string enum defined next to `Instance`. The enum class itself is not part of the public surface: two named values say everything a declaration needs, and a class at the top of the package was one more name to learn for the same two choices. Because the members are strings, `host="remote"` is the same value. An unrecognized value raises at declaration time with both options named.

An instance carries no placement of its own. `colab.G4` says which card, and only the declaration's `host` says where the host code runs, so there is one place to read to know where a function runs. How long a session lives is not a declaration argument either: it is the `keep_alive` block around the calls, described under Lifetime.

### Invocation

> Calling a declared function runs it. Blocking behaviour is declared at the `def` site.

A declared function is called like any other. There is no second verb such as `.remote()`: a decorator that wraps a `def` and then needs another call to run it has moved the declaration out of the declaration.

A plain `def` blocks and returns its value. An `async def` returns a plain coroutine, so the standard library accepts it wherever a coroutine is expected.

A call takes the arguments the `def` declares and returns what the `def` returns. No argument value changes what a call means.

`Function.local()` runs the body in the calling process. It exists for testing a body with no provider; the preferred way to run locally is a `Local` provider, which keeps the production code path.

### Concurrency <!-- id: fan-out -->

> Many configurations run at once by calling the declared function concurrently. The inventory bounds how many run.

Running several configurations is repeated calls, one per configuration, inside `with let.keep_alive():` so the calls reuse warm sessions. Concurrent calls come from the language: `asyncio.gather` over calls to an `async def` declaration, or threads over a plain `def` one. letify adds no map call and no argument type for it.

Concurrent calls run as wide as the provider has devices for, and no wider. Nothing on the declaration bounds it, because a bound there would be a second statement of the same fact: the inventory already says how many cards exist, and a call that cannot reserve one waits for a call that holds one to finish, as described under Pooling.

A declaration taking two cards halves the width on a four card machine, which is arithmetic rather than policy. That is also why a width knob could not work: with `device=lab.A100 * 2`, a number that says how many runtimes may exist says nothing about how many cards they need.

## Provider model

> A provider object is one account on one kind of infrastructure. It registers the instances it offers, owns storage, opens the channel, and starts and stops the session.

```
Provider (abstract)
├── Local                 persistent
├── Modal                 persistent, reached through the Modal adapter
└── Shell                 ephemeral by default, reached through the connection pipeline
    ├── Colab             session created by the Colab CLI, rendezvous over colab exec
    ├── Tunnel            a machine behind NAT, rendezvous through letify client shell connect
    └── Elice             machine allocated through the Elice Cloud API, rendezvous over SSH to it
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

For a `shell` or `tunnel` account, `letify login` writes this table from what the machine reports, as described under Logging in. An entry with no table is still usable: its accelerators are discovered on first access, as described under Instances.

Which registered indices are free is read with `nvidia-smi` at reservation time, not cached, because the answer changes while a run is queued. A card is taken as busy when another process is computing on it. Nothing else on the machine is inspected, and letify never kills anything.

The reading is taken on the machine that owns the cards. `Local` runs the queries itself. A `shell`, `tunnel` or `elice` account runs them over the account's link, one SSH command each: `nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits` maps each uuid to its index, and `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits` lists the compute processes.

A compute process is excluded when it is a worker of a session this client process started on that provider. A runtime records its worker's process id from the worker's `stat` reply when its persistent channel starts, and gives it up at shutdown. The worker keeps that id when it moves to the project interpreter, because the move is an `execv`. A process started by any other client process, letify or not, counts as busy.

A remote query that cannot run, because SSH failed or `nvidia-smi` exited non zero, raises `RuntimeFailure` naming the busy check. Treating the cards as free would put a run on a card someone else is computing on. When no registered index is free, the `InsufficientDevices` message names the indices another process is computing on.

A reserved session sets `CUDA_VISIBLE_DEVICES` to its reserved physical indices and `CUDA_DEVICE_ORDER=PCI_BUS_ID` in its worker before any user code runs, and keeps both when the worker moves to the project interpreter. The training code sees its cards as 0 upward in `nvidia-smi` order and needs to know nothing about which physical indices it was given. A provider that assigns the device itself sets neither.

### Instances

> An `Instance` is one accelerator shape on one provider account.

`colab.G4` is an `Instance`. It holds the provider, the accelerator name, the host placement, how many devices one session takes, and the core count, memory and VRAM the provider reported. `n * instance` returns a copy taking `n` devices. An instance has no method that changes where the host code runs, because that is the declaration's `host`.

The device count is part of the pool key, because a session holding two cards is not interchangeable with one holding one.

Because an instance carries its provider, `device=colab.G4` fixes provider, account and accelerator in one argument. `let.providers.any.G4` defers the provider choice to the first declared provider that registers a matching accelerator, in configuration order.

Instance discovery is lazy and cached. A provider that must connect to enumerate its accelerators does so on first access, never at import time, and a configuration entry may list `gpus` or a `devices` table to skip the connection. `refresh()` asks again. A `shell` or `tunnel` account gets its `devices` table at login, so the connection on first access is the fallback for an entry written without one: a login where `nvidia-smi` did not answer, a password account, or an entry written by hand.

`Local` reads its accelerator names once per process, because asking `nvidia-smi` takes seconds on a laptop whose discrete GPU is asleep and the answer does not change while the process runs.

Accelerator names are normalized so they can be attributes. `NVIDIA RTX PRO 6000 Blackwell` becomes `RTX_PRO_6000`. Colab calls the same card `G4`, which is what its CLI accepts, and accepts `RTX_PRO_6000` as an alias for it.

Every provider that can start a session without an accelerator registers it as `CPU`, as `Local`, `Colab` and `Modal` do, and `cpu` finds it too. `Colab` creates such a session with `colab new` and no `--gpu` or `--tpu`. `Modal` creates such a sandbox with `gpu` null.

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

A **persistent channel** keeps one worker process alive behind a pipe. Requests are framed lines, so the worker process with its session cache, the blob table and anything written to disk all survive between calls.

A **one-shot channel** can only run a command and collect its output. Every call starts a fresh process, so nothing persists. It exists because some transports offer nothing more, and it refuses the operations that need persistence rather than pretending.

Both hand back the user's own stdout separately from the outcome, because they share one stream.

The worker source cannot be sent on standard input as a script, because `python -` reads to end of file before compiling anything and the pipe has to stay open for requests. A small bootstrap stub passed with `-c` reads a length-prefixed base64 blob, executes it, and leaves standard input where it was.

The worker announces itself with one line, `__LETIFY_WORKER_READY__ <major>.<minor>`, naming the version of the interpreter it runs on. A worker asked to move to another interpreter replies, then replaces its process with `os.execv(<interpreter>, [<interpreter>, "-u", "-c", <bootstrap stub>])` on the same pipes, and the channel sends the worker source again and waits for the new ready line.

### Modal adapter <!-- id: modal-adapter -->

> The letify process never imports `modal`. A small adapter runs in its own uv environment and letify talks to it in JSON lines over its standard input and output.

The adapter is `letify/providers/modal_adapter.py`. It imports only the standard library and `modal`, so it runs by file path and needs none of letify's own dependencies. letify starts it with:

```
uv run --no-project --python 3.12 --with "modal>=1.0,<2" python -P <path to modal_adapter.py>
```

`-P` keeps the adapter's directory off `sys.path`, so `import modal` finds the Modal package and not `letify/providers/modal.py` beside it.

The Modal version range is pinned in `tools.MODAL`. The adapter runs with `MODAL_CONFIG_PATH` set to `~/.letify/accounts/<alias>/modal.toml`, and with any inherited `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and `MODAL_PROFILE` removed. The profile Modal uses is the one the sign in activated in that file. So the account file alone decides which Modal account acts, and two accounts coexist on one machine.

The protocol is one JSON object per line. letify sends `{"id": <int>, "op": <name>, ...}` and waits for the line with the same `id`. The adapter answers `{"id": <int>, "ok": true, "value": <any>}`, or `{"id": <int>, "ok": false, "kind": <kind>, "error": <text>}`. `kind` is `unavailable` when `modal` cannot be imported, `not_found` when a volume path does not exist, and `failure` otherwise. Requests are answered one at a time, in order.

| Op | Fields | Value |
|---|---|---|
| `hello` | none | `{"modal": <installed Modal version>}` |
| `create` | `app`, `args`, `packages`, `gpu`, `timeout` | `{"sandbox": <id>}`. Runs `app` as an ephemeral app on first use, builds `debian_slim` with `packages` installed, and starts `args` in a sandbox |
| `write` | `sandbox`, `data` | `null`. Writes the text to the sandbox's standard input and drains it |
| `read_until` | `sandbox`, `prefixes` | `{"lines": [...], "eof": <bool>}`. The sandbox's stdout lines up to and including the first that starts with one of `prefixes`, or every line left when the stream ends |
| `terminate` | `sandbox` | `null` |
| `volume_put` | `volume`, `version`, `path`, `data` | `null`. `data` is base64 |
| `volume_get` | `volume`, `version`, `path` | base64 of the file |
| `volume_list` | `volume`, `version`, `path` | the paths under `path`, recursively |
| `volume_delete` | `volume`, `version`, `path` | `null`. Removes the file or directory at `path`. A missing path is not an error |

Every volume op creates the volume when it is missing, as version `version`.

The persistent channel to a sandbox is that sandbox's standard input and output, carried by `write` and `read_until`. The sandbox runs the bootstrap stub `python3 -u -c BOOTSTRAP`, and the worker source goes out first as the length-prefixed base64 blob described above. A `read_until` that ends at end of stream without a reply raises `ProtocolError`.

A missing uv raises `ProviderUnavailable` naming uv. A reply of kind `unavailable` raises `ProviderUnavailable` for `modal`. An adapter process that exits, or prints a line that is not the reply it was waiting for, raises `RuntimeFailure` carrying the adapter's standard error, because that is an infrastructure failure. A reply of kind `failure` raises `RuntimeFailure` with the adapter's message. One adapter process serves one provider or one backend and exits when its standard input closes.

The adapter never deploys an app. `create` starts `modal.App(app).run()` the first time it sees an app name and holds that context for the adapter's lifetime. When standard input closes, the adapter terminates its remaining sandboxes and then leaves every app context, which stops the ephemeral app. An adapter that dies stops sending Modal's client heartbeat, and Modal stops the ephemeral app for it. So no app named `app` stays on the account after letify stops.

## Call protocol

> A call is a serialized function plus arguments, and the outcome comes back on the same channel.

The local side pickles `(function, args, kwargs)` with cloudpickle and sends it as a framed request. Framing is one base64 line per message, which survives an SSH channel, a WebSocket bridge and a plain pipe without any of them mangling it.

On a one-shot channel the call travels inside a driver script that prints its outcome between `__LETIFY_RESULT_BEGIN__` and `__LETIFY_RESULT_END__`, so it can be found in a stream that also carries the user's prints. Absence of the marker is not a protocol quirk: it means the remote process died, and letify reports that as `ProtocolError` naming the likely causes.

An `async def` body is awaited on the remote side, so it runs to completion there and can use `await` internally.

### Session cache <!-- id: handles -->

> `letify.session_cache(key, factory)` returns the value stored under `key` in the current session, building it with `factory()` on first use. A call always returns its value.

```python
@let.function(device=colab.G4, host=letify.remote)
def generate(prompt):
    model = letify.session_cache("model", load_model)
    return model(prompt)
```

Inside a declared function running in a runtime, `session_cache` looks `key` up in a store held by the runtime's worker process. On first use it calls the zero-argument `factory` and stores the result. The value lives until the session ends: the call ends it, or the enclosing `keep_alive` block ends it.

Each session has its own cache. Concurrent sessions each build their own value, so no result depends on which session the pool picks for a call.

The store lives in the `letify` module, which the worker imports by reference. A dict at module level in the user's script cannot do this job, because cloudpickle ships `__main__` globals by value with every call, so each call sees a fresh copy. letify is importable in the runtime because the user's environment, keyed by `uv.lock`, includes it. A worker that cannot import letify, whether the call refers to letify when it is loaded or the body imports letify while it runs, fails the call with `RemoteError` whose message says that letify is not installed in the runtime's environment and that `uv add letify` fixes it.

Concurrent first use of one key builds once: the other callers wait for that build and receive its value. A factory that raises stores nothing, so the next use tries again.

Outside a runtime, in the local process or in `Function.local()`, `session_cache` is a memo for the life of the process with the same semantics, so a body tested locally behaves the same.

On a one-shot channel every call is a fresh process. `session_cache` works within that call and keeps nothing for the next one.

A returned value always comes back to the caller. There is no declaration argument that returns a reference instead.

### Argument addressing

> Large arguments are named by the hash of their contents, so the same value travels once.

An argument above 64 KB is pickled and hashed, the runtime is asked which digests it already holds, and only the rest is sent. A later call carrying the same value sends a `Blob` reference instead of the bytes.

Hashing is not a bottleneck at any link speed involved: blake3 runs at gigabytes per second where a home uplink runs at megabytes per second. blake2b from the standard library is the fallback.

### Failure and retry

> Infrastructure failure may be retried. User code failure never is. Neither falls back to a slower path.

`RuntimeFailure` and `ProtocolError` mean the session misbehaved, so the runtime is discarded and the call is retried on a fresh one up to `retries` times. `RemoteError` means the shipped function raised, and it propagates with the remote traceback attached.

A `RuntimeFailure` raised for a failed command carries `command` and `stderr`, and its message names the command and the last 40 lines of stderr. The `RuntimeLost` raised after the last retry keeps the last failure's message, `command` and `stderr`.

letify never falls back to local execution or to a slower mode when the declared one is unavailable. A silent downgrade turns a four times slowdown into a mystery.

## Sessions

> A runtime is one live session and the only object that costs money.

Everything above a runtime is declaration. Creating one is when a provider actually powers something on; shutting it down is when the charge stops.

A runtime boots in seven steps:

1. Open the channel. The worker starts on the bootstrap interpreter: the account's `python` option, `python3` by default.
2. Arm the lease.
3. Prepare the workspace root, as Workspace root describes: expand `~` on the runtime, create the directory, make it the worker's working directory, and point `TMPDIR` at `<workspace root>/tmp`.
4. Build the environment: restore the environment archive, or run `uv sync` in the project directory, as Environment describes.
5. Move the worker to `<project directory>/.venv/bin/python`.
6. Check the worker's interpreter version against the local one.
7. Attach volumes.

Step 3 is skipped on the local provider, whose worker keeps the working directory of the process that started it. Steps 4 to 6 are skipped on the local provider, which already runs in the project's environment. Steps 4 and 5 are skipped when the account sets `python`, which means the user manages the interpreter on that machine. Step 6 still runs then.

### Pooling

> Runtimes are pooled by instance and environment, so the second call through a declaration pays nothing for setup.

The pool key is the instance key joined with the environment key. Two declarations that agree on both share runtimes, which is why nothing has to be said for two functions on one device to reuse a session.

How many sessions may exist is the provider's inventory and nothing else. Starting one reserves the devices its instance asks for. A call that cannot reserve them waits only while a session in this process that holds that accelerator is serving a call, because that session gives its devices back when the call finishes. A session this process is still starting counts as serving the call that started it; more concurrent calls than the inventory has cards for rely on exactly this. In every other case the devices cannot be allocated, and the call raises `InsufficientDevices` at once, naming what holds them: an idle session that a `keep_alive` block is keeping, a card another process is computing on, or a request for more devices than the account declares. Waiting there would never end, because nothing letify is running would free a device. `InsufficientDevices` is not retried, since a retry asks for the same devices from the same inventory. There is no ceiling on the launcher: a number there would be a guess about hardware the provider entry already describes, and when the two disagreed the smaller would win silently.

A session is never a value the caller holds. Pooling, reuse and teardown are decided from the declaration and the `keep_alive` block around it, so there is no call that starts a session, none that returns one, and none that takes one. `Runtime` exists, and letify hands it to a provider and to a volume, but it does not appear in anything a user writes.

### Lifetime

> A session ends with the call that needed it. `with let.keep_alive():` keeps sessions for the length of a block. Nothing else decides.

1. **The call.** A session ends when the call that started it finishes. Calls that overlap in time share that span: a session released by one call while another is still running stays up for the next call that matches it, and ends when the last overlapping call finishes.
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

Nothing internal is reported. The pool holds a guard so that a session released by one call is not ended while an overlapping call is still running, and whether that guard is currently open is a fact about the pool's implementation rather than about what is running. A field among counts that looks like a count and is actually a boolean is worse than no field, because it is read as a count.

`status()` describes this process only. A session started by a different process is not in it, since the pool lives in the process that owns it. What a machine itself is doing is a different question, answered by `letify utilization`.

## Storage

> A volume is a content addressed blob store on whichever backend a provider has. Each provider holds one volume per project, filled automatically from what a call uses, so a declaration never names one.

Attaching a volume to an ephemeral provider moves the environment archive and the model cache into storage that outlives the runtime. A twenty gigabyte cache takes about 27 minutes to pull from a lab server over a 100 Mbit/s link, three to five minutes from the Hugging Face hub, and 40 to 60 seconds from a bucket inside the same infrastructure as the runtime. That difference is billed as GPU time.

### Content addressed layout

> Blobs are immutable and named by their hash. Mutable names live in a separate, tiny namespace of refs.

```
blobs/<first two hex characters>/<digest>
refs/<name>
```

Immutability buys two things. Concurrent writers cannot conflict, because different contents get different names, where a two way synchronization loses one writer's changes to the other. And nothing is verified twice, because holding a digest is proof of holding the contents.

Refs carry the mutable part, in the way Git keeps branch names apart from objects. A ref is a few dozen bytes, so a last writer wins race on one is harmless and both blobs survive it. letify reserves `env/<env key>-<platform>` for environment archives and `path/<path key>` for project data; nothing else is written there.

### Blob granularity

> The unit of a blob is a decision. Large files stand alone; trees of small files are packed.

A model shard is already large, so one file is one blob. An environment is tens of thousands of small files, so the whole tree is packed into one archive keyed by the environment key and the runtime's platform. That is where the speedup is: tens of thousands of round trips become one.

A content addressed store does not by itself reduce the bytes of a first transfer. What it improves is the metadata exchange, which becomes a single manifest read instead of one request per file, and every repeat transfer, which is skipped by name. Neither it nor a synchronization tool sends deltas within a changed file.

Extraction checks every member's path against the destination before unpacking, so an archive cannot write outside it. An environment archive is extracted with Python's `tar` filter rather than the `data` filter, because a `.venv` links its interpreter by absolute path and the `data` filter refuses such a link.

### Materializing into a runtime

> The runtime pulls a volume straight from the backend, using a short-lived credential borrowed from the local machine. Bytes never pass through the local process on the way in.

A volume holds a copy of the local project's environment and data, so its purpose is to make a new runtime start as if it were the local machine. That copy is only useful if it arrives fast, and the fast path is the backend's own network: a Colab runtime reading a Cloud Storage bucket is a transfer inside Google's infrastructure. Routing the bytes through the local process would cap every session start at the user's uplink.

The local machine writes to the backend directly as well. It does not relay through a runtime, because that adds a hop that is still bounded by the same uplink and spends billed runtime time.

No credential is stored on the remote side. For each materialization the local process derives a short-lived access token from its own login, scoped to reading the volume's prefix where the backend supports scoping, and sends it over the channel. The worker keeps it in memory only, never on disk or in the environment of user code, and drops it when the pull finishes. A backend that has no network path from the runtime, such as `filesystem` on a machine the local process can reach but the runtime cannot, falls back to writing through the channel.

A backend offers a pull by answering with a URL and the request headers that read one blob. The local process sends them in one `pull` request naming the destination path and whether to unpack it in the volume directory. The worker streams the response to the path, unpacks it when asked, and discards the request, headers included, before it reads the next one. `gcs` offers a pull. `filesystem` and `modal` do not, so they write through the channel. A one-shot channel has no worker to pull, so it writes through the channel as well.

The `gcs` read token is downscoped with a Credential Access Boundary: the local token is exchanged at `https://sts.googleapis.com/v1/token` for one that holds `roles/storage.objectViewer` on the volume's bucket only, with an availability condition restricting it to object names under the volume's prefix. A volume option `sts_endpoint` points the exchange elsewhere. A failed exchange raises `RuntimeFailure` rather than sending the unscoped token.

The environment archive is automatic. The first session that runs `uv sync` for an environment packs its project directory, `.venv` included, into its first volume under `env/<env key>-<platform>`. `<platform>` is the runtime's `sys.platform` and `platform.machine()` joined by a hyphen, for example `linux-x86_64`, and the env key includes the interpreter's major.minor version. Every later session on the provider with the same key and platform pulls that archive and unpacks it at `<workspace root>/project` instead of syncing.

A volume's files on a runtime live in its volume directory, `<workspace root>/volumes/<volume name>`. A blob materialized without a named destination is written to `<volume directory>/blobs/<first two hex characters>/<digest>`. The volume option `mount` names another directory for one volume; nothing else sets it. A restored `.venv/bin/python` that does not start is treated as no archive, and the session syncs. Nothing in the public surface names this step.

Nothing hands a session to the caller. There is no call that returns one, no argument that takes one, and no way to hold the wrong one, because which session serves a call is the pool's answer to work out from the declaration.

### Project data <!-- id: project-data -->

> A call's data is found in the call itself. Every `pathlib.Path` the function reaches travels with it: the local contents go up to the provider's volume, the runtime sees the same path already filled, and what the call changes under it comes back when the call ends.

The declared function is sent with cloudpickle, which serializes its closure variables, the globals it references, its default arguments and the call's arguments. letify's pickler intercepts every `pathlib.Path` among them through `reducer_override`. Nothing is declared: the function's own references are the declaration, which is what makes the decorator behave as a closure over the data it uses.

A path is project data when it exists on the local machine at call time, as a file or a directory. For each one:

1. The local contents are packed into content addressed blobs, large files as their own blobs and trees of small files as one archive, and only the digests the volume is missing are uploaded, directly to the backend.
2. The ref `path/<path key>` records the manifest, where the path key is the digest of the path resolved against the project root.
3. In the pickled call, the path is replaced by the path the runtime materializes it at, under the volume directory, so the function body uses it unchanged.
4. Before the call runs, the runtime pulls the manifest's blobs straight from the backend, as Materializing into a runtime describes.

A path that does not exist locally is an output location. It is created empty in the runtime and replaced the same way.

When the call returns, the runtime compares each replaced path with the manifest it started from. Files that are new or changed are stored as blobs in the volume, and the local process writes them back to the local path. So a checkpoint written under a referenced `Path` is on the local disk when the call returns, and the next session receives it as input with no separate resume step.

Only `pathlib.Path` objects are detected. A string that happens to name a local file is left alone, because no rule can tell a path from any other string, and uploading on a guess would send data the user did not mean to send.

Detection costs one `stat` per path per call, and packing is skipped when the manifest ref already matches the local tree's modification times and sizes.


### Backends

| Backend | Used by | Note |
|---|---|---|
| `filesystem` | `Local`, `Shell`, `Elice` | A directory. The local machine can be the origin others pull from. On Elice it sits on the machine's own disk. |
| `gcs` | `Colab` | A Colab runtime is a Compute Engine virtual machine, so this is an internal transfer. Use a multi-region bucket, because runtime placement is not selectable. The client is the standard library HTTP client against the Cloud Storage JSON API. |
| `modal` | `Modal` | A Modal volume, mounted beside the container. Reached through the Modal adapter, never through a `modal` import in the letify process. |

The `modal` backend uses Modal volume version 2, and sends `version` 2 on every volume op. A version 1 volume accepts a file above Modal's 4 MiB large-file limit and then fails to read it back. `ModalBackend.delete` removes a blob, and a blob already absent is not an error.

The `modal` backend acts as one Modal account. A volume option `account` names its alias, and a volume on a `Modal` provider defaults it to that provider's alias. A volume on another provider that names the `modal` backend without `account` raises `ConfigError`, because there is no account to act as.

Every backend answers "which of these digests are missing" with one listing rather than one request per digest, because object level requests are billed and add latency.

### Google login for gcs <!-- id: gcs-login -->

> The `gcs` backend borrows the user's existing Google login. It stores no credential and adds no dependency.

The client calls the Cloud Storage JSON API at `https://storage.googleapis.com`: a listing with `prefix` and `pageToken` for `missing`, a media upload for `put` and `write_ref`, and a media download for `get` and `read_ref`. A volume option `endpoint` points it elsewhere. A request that answers 404 means the object is absent; any other status that is not 2xx raises `RuntimeFailure` naming the status and the object.

The access token is looked up in this order, the first that answers wins:

1. The `GOOGLE_OAUTH_ACCESS_TOKEN` environment variable, used as it is.
2. Application Default Credentials: the file named by `GOOGLE_APPLICATION_CREDENTIALS`, or `application_default_credentials.json` in the gcloud configuration directory (`~/.config/gcloud/` on Linux and macOS, `%APPDATA%\gcloud\` on Windows). A file of type `authorized_user`, which `gcloud auth application-default login` writes, is exchanged for an access token by posting its refresh token to its `token_uri`, `https://oauth2.googleapis.com/token` by default, with the standard library HTTP client.
3. `gcloud auth print-access-token`, when `gcloud` is on `PATH`.

A token is reused until 60 s before it expires. A service account key file is refused with its reason, because exchanging one needs an RSA signature the standard library cannot make; `gcloud auth activate-service-account` followed by rule 3 covers that case. When no rule answers, the backend raises `ProviderUnavailable` for `gcs` saying that no Google login was found and naming `gcloud auth application-default login` and `GOOGLE_OAUTH_ACCESS_TOKEN`.

## Environment

> An `Env` is a declaration keyed by the project's uv files, not a built image. Every remote runtime builds it with `uv sync` into a project `.venv` and runs the worker with that `.venv`'s Python.

`Env()` reads `uv.lock`. `pip_install`, `run`, `vars` and `ship` refine it and return a new value. The project directory is the directory holding the lock file. The key is a hash of the contents of `uv.lock`, `pyproject.toml` and `.python-version` in that directory, `Env.python`, and the refinements, so two declarations that agree share a pooled runtime and a cached archive.

A uv lock file resolves for every platform uv supports, so one lock file drives a Linux runtime from a Windows or macOS client. A `pip freeze` list does not, because it carries platform specific pins.

### Building the environment on a runtime <!-- id: remote-uv-sync -->

> The runtime receives `pyproject.toml`, `uv.lock` and `.python-version`, runs `uv sync --frozen --no-install-project` in a project directory, and starts the worker with that directory's `.venv/bin/python`. A default `Env()` does the same; no path installs into the system Python or installs nothing.

This applies to every provider except `local`: `shell`, `tunnel`, `colab`, `elice` and `modal`. A Modal sandbox image carries only what starts the bootstrap worker, so letify reaches the sandbox through the sync like every other package.

1. The local side reads `pyproject.toml` and `uv.lock` from the project directory, and `.python-version` when it exists. A missing `pyproject.toml` or `uv.lock` raises `ConfigError` naming the file and `uv lock`, before anything runs on the runtime.
2. The worker writes them into the runtime's project directory, `<workspace root>/project/<env key>`. The workspace root is the provider's `workspace_root`, the one place every path letify writes on the runtime derives from, as Workspace root describes.
3. The worker runs `uv sync --frozen --no-install-project --python <major>.<minor>` in that directory, with the version `Env` records, as Interpreter version describes. uv creates `.venv` there, and downloads the interpreter when the runtime has none that matches.
4. Packages from `Env.pip_install` are installed with `uv pip install --python <project directory>/.venv/bin/python <packages>`. Commands from `Env.run` run through the shell with `<project directory>/.venv/bin` first on `PATH` and `VIRTUAL_ENV` set to the `.venv`. Variables from `Env.vars` are set in the worker's environment before the sync, so they reach every later step and user code.
5. The worker moves to `<project directory>/.venv/bin/python`, as Channels describes. On a one-shot channel every later program runs as `<project directory>/.venv/bin/python -c <program>`, a child of the provider's command, with its standard output passed through.

`--frozen` installs exactly what `uv.lock` says and never re-resolves, so the runtime holds the library versions the local `.venv` holds. `--no-install-project` is used because only the three metadata files travel: the project's own source is not on the runtime, so building the project there would fail. The project's own modules reach the runtime with the call instead, as Module shipping describes. letify itself is installed by the sync, because the project depends on it. A path dependency is installed from its locked path on the runtime, so one that exists only on the local disk fails the sync.

A failed sync raises `EnvironmentFailure` saying `uv sync failed on <runtime>`, with the command and the last lines of uv's standard error. `EnvironmentFailure` is a `RuntimeFailure`, so the call is retried on a fresh runtime.

### Interpreter version <!-- id: interpreter-version -->

> The runtime's `.venv` always runs the same Python major.minor as the local process, and `Env` guarantees it.

`Env.python` records the major.minor of the interpreter in the process that declares the `Env`, `sys.version_info[:2]`, and the runtime always runs `uv sync` with `--python <Env.python>`, whether or not the project has a `.python-version`. Without this a wide `requires-python` such as `>=3.11` lets uv pick a different minor version remotely, and cloudpickle's bytecode for a `__main__` function fails on it.

Two declarations cannot diverge from the local process. Before any session starts, a `.python-version` whose major.minor differs from the local interpreter, or an `Env.python` that differs from it, raises `InterpreterMismatch` naming both versions. A `.python-version` naming a patch release, such as `3.12.3`, is compared by its major.minor.

### uv on the runtime <!-- id: uv-on-runtime -->

> The runtime uses the uv it has, and installs uv under the home directory when it has none.

The worker looks for `uv` on `PATH`, then at `~/.local/bin/uv`. When neither exists it downloads `https://astral.sh/uv/install.sh` over HTTPS with the bootstrap interpreter's `urllib` and runs it with `sh`, with `UV_INSTALL_DIR=~/.local/bin` and `UV_NO_MODIFY_PATH=1`. That is the same as `curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh`, without needing `curl`. It needs no root, because it writes only under the home directory. A failed download or a non-zero exit raises `EnvironmentFailure` saying `uv could not be installed on <runtime>` with the reason.

### Interpreter check <!-- id: interpreter-check -->

> A worker whose Python major.minor differs from the local process fails the session start with both versions named, before any call is sent.

The worker reports `sys.version_info[:2]` in its ready line. After the environment step, the local side compares it with its own `sys.version_info[:2]`. A difference raises `InterpreterMismatch`, naming both versions and saying that cloudpickle's bytecode cannot run across them. It is not retried, because a fresh runtime builds the same interpreter. The check applies to every provider that builds its environment, and to an account that sets `python`. It does not apply to the local provider.

### Module shipping

> Modules in the lock file are installed remotely by name. Modules that are not travel with the call.

A package the lock file names is installed in the runtime and referenced by name. A package it does not name, such as the project's own code or an editable install, has to be sent by value, because the remote side either lacks it or holds an older copy. `Env.ship()` overrides the inference.

## Transport

> A `Shell` reaches its machine through a connection pipeline: several strategies are tried at once, the fastest acceptable one wins, and the winner is cached per account.

`Modal` and `Local` are not part of this. Modal is reached through the Modal adapter, described in [Modal adapter](#modal-adapter), and Local starts its worker as a child process.

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

A strategy whose needs are not met is skipped, not attempted. A `Shell` with no rendezvous has only rank 1. An account with no `address`, such as a Tunnel account that names only `tailcat`, has no rank 1: forward SSH is skipped with the reason `no address`, and no SSH command is built for a guessed or empty address. `letify check` on such an account runs over the strategies that remain.

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

Strategies are raced rather than tried in turn, so a strategy that times out does not delay the others. Once the first strategy that can carry the probe connects, the pipeline waits a grace period of 2 s for lower ranked strategies before choosing.

A strategy that cannot carry the probe, such as the provider fallback, does not start the grace period when it connects. It is held back until every strategy that can carry the probe has failed, or until the race timeout of 60 s passes with none of them connected. A rendezvous that takes seconds, such as `colab exec` before a TCP punch, therefore runs inside its strategy's own attempt time, not inside the grace period.

Every connected strategy is probed: 30 round trips, then 2 s of transfer in each direction. A strategy whose throughput in either direction is below 25% of the fastest connected strategy in that direction is rejected. The lowest ranked strategy that remains is chosen. When only one strategy connects, it is chosen without comparison.

Strategies that lose are closed, including one that connects after the choice is made.

When only one strategy is applicable there is nothing to choose between, so it is used directly: it is not raced, not probed and not cached, and a failure surfaces at its first use. When a race of two or more ends with one connected strategy, that strategy is still probed so the cache has throughput to compare against, but a failed probe does not reject it.

A `Link` that cannot carry the probe, such as the provider fallback, is left out of the throughput comparison. It is chosen only when no probed strategy remains, and it is never written to the cache, so the next connection races again. A probe that fails on a strategy in a race of two or more rejects that strategy. When no strategy connects, the error names every strategy with the reason it failed or was skipped.

Every connection decision is printed, one line per event, on stderr with the `letify: ` prefix the session start line uses, so the user's stdout stays clean. The lines are on by default. `Launcher(announce=False)` silences them together with the session start line. The wording is not fixed, but each line names the account and carries these facts:

| Event | The line carries |
|---|---|
| Race start | the strategies attempted, each skipped strategy with its reason, and that the fallback is held back when it is applicable |
| Lone strategy | the one strategy used without a race |
| Strategy outcome | connected with the seconds since the race started, failed with the exception text, timed out, or connected after the choice and closed |
| Probe | round trip in ms, upload and download in MiB/s, or the probe error |
| Choice | the chosen strategy and why: the lowest rank within 25% of the fastest, the only one connected, or the fallback because no probed strategy connected. Each strategy rejected by the 25% rule is named with its upload and download against the fastest |
| Switch | a strategy that connected first replaced by the chosen one, such as `tcp_punch` replacing `tailcat`, a link to the same account connected again after it was closed, with the old and new strategy, and a fall back to the provider's own path |

### Link cache <!-- id: link-cache -->

> The winning strategy is remembered per account and per network, so the next connection starts with it alone.

The cache lives in `~/.letify/accounts/<alias>/link.json`. It records the strategy, the probe results, and a network fingerprint: the local machine's public IP address and the name of its default route interface.

On the next connection the cached strategy is attempted alone. When it connects and its probe is at least 50% of the cached throughput in both directions, it is used. Otherwise, or when the fingerprint differs, the full race runs and the cache is rewritten. A fingerprint whose public IP address could not be learned matches nothing, so the race runs.

The cache's decisions are printed as the race's are, in [Choosing a link](#choosing-a-link): the cached strategy tried alone with its cached throughput, then accepted, or rejected with the reason (below 50% of the cached throughput with both numbers, failed to connect, probe failed, or the network fingerprint changed), and every rewrite of the cache with the strategy written.

The public IP address is learned from the same STUN servers the punch uses. The default route interface is read from `/proc/net/route` on Linux, `route -n get default` on macOS and `Get-NetRoute` on Windows.

### Rendezvous <!-- id: rendezvous -->

> Hole punching needs a way to start a program on the remote side and swap addresses. For Colab and Elice the provider layer fills that role; a plain machine behind NAT runs `letify client shell connect`.

| Provider | Rendezvous |
|---|---|
| `Colab` | the provider layer: `colab new` creates the runtime and `colab exec` runs the remote half |
| `Elice` | the provider layer: the Elice Cloud API creates and allocates the machine, and the remote half runs over forward SSH to it |
| `Tunnel`, and a plain `Shell` behind NAT | the remote agent started by `letify client shell connect`, reached over Tailcat |

Colab and Elice never need `letify client shell connect`. Their create and open step is what puts letify's remote half on the machine.

`letify client shell connect` is run once on a plain machine by its user. The agent needs letify installed on that machine. Before starting anything it checks two things, and each failure prints what to do and exits 1:

1. `tailcat` is on `PATH`. Otherwise it prints the install command for the detected operating system and CPU architecture, for the Tailcat release pinned in `letify.transport.setup.TAILCAT_VERSION`: on Linux amd64, arm64 and armv7, `mkdir -p ~/.local/bin && curl -L <release tar.gz> | tar xz -C ~/.local/bin tailcat` with a note that `~/.local/bin` must be on `PATH`; on macOS, `brew install tailcat`; on Windows amd64 and arm64, the release zip and where to put `tailcat.exe`. Any other platform gets the releases page.
2. An SSH server answers on `--ssh-port`, default 22: a TCP connection to `127.0.0.1` on that port must send a line starting with `SSH-` within 3 s. Otherwise it prints how to install and start one, for a Debian or Ubuntu container `apt-get install -y openssh-server`, `mkdir -p /run/sshd` and `/usr/sbin/sshd`.

It then starts the remote agent on a port the operating system chooses, starts `tailcat serve <agent port>` in front of it, and prints exactly one command for the user's own machine, `letify login tunnel <alias> --connect <token>`. The alias is `--name`, or this machine's host name with every character that is not a letter, digit or underscore replaced by `_`. The token is the URL-safe base64 encoding, without `=` padding, of the compact JSON object `{"tailcat": <address>, "tailcat_port": <agent port>, "user": <this machine's user name>, "port": <SSH port>}`. After the command it prints that the agent must keep running, how to keep it running with `tmux` or `nohup`, and that a restart prints a new address, so the login is run again with the new token.

A connection to the agent is told apart by its first bytes: `SSH-` is spliced to the machine's SSH server, and `LETIFY-RDV ` is followed by one JSON request line and answered with one JSON line. For such an account the pipeline connects over Tailcat first, runs `tailcat <address> <agent port>` to exchange the TCP punch mapping and start time over that link, and then races as specified: the Tailcat link is the rank 3 candidate, and when TCP punching passes the probe it takes over.

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

The proxy URL and token are the session's `url` and `token` in `.config/colab-cli/sessions.json` under the account directory, where the Colab CLI records them when `colab new` creates the session. A session that file does not name raises `RuntimeFailure` naming the file. Every request carries the token as the `colab-runtime-proxy-token` query parameter and the `X-Colab-Runtime-Proxy-Token` header, with `authuser=0`, as the CLI does.

An upload of a file to `<path>` sends parts of 32 MiB, eight at a time, each to `<path>.letify-part-<n>`. Eight is the most parallel parts [measured](NETWORK.md#colabs-own-paths), and it was the fastest in both directions. A part is sent as base64 chunks of 8 MiB: chunk `1` creates the file, later chunks append, and the last chunk of a part of more than one chunk is numbered `-1`. One `colab exec` program then joins the parts into `<path>` in order, removes them, and unpacks the archive when the request asks for it. A download reads the file's size from the contents model with `content=0`, then reads `/files/<path>` with `Range` requests of 32 MiB, eight at a time. `put_file`, `get_file` and `pack_dir` on the fallback channel use this path; `pack_dir` packs into a temporary file under `<workspace root>/tmp` with `colab exec` first. Any other status than 2xx raises `RuntimeFailure` naming the path and the status.

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
workspace = "/workspace/researcher/letify"   # this server allows writes under /workspace only

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

### Workspace root <!-- id: workspace-root -->

> Everything letify writes on a remote machine lives under one directory per account, the workspace root. An account sets it with `workspace`; otherwise its kind decides.

A machine sets rules on where an account may write. A department GPU server may allow writes only under `/workspace`, a Modal sandbox loses its disk when it ends while a Modal volume persists, and a Colab VM is discarded after the session. So the root is a property of one account on one machine.

`workspace = "<path>"` is set on an account entry in `~/.letify/config.toml`. The path starts with `/` or `~`, and `~` is expanded on the runtime, not on the local machine. Any other value raises `ConfigError` naming the account. A `workspace` field in a project `.letify/config.toml` raises `ConfigError` saying that it belongs in the home file, because a repository cannot know the rules of each machine its users reach.

When `workspace` is not set, the root is:

| Kind | Workspace root |
|---|---|
| `shell`, `tunnel`, `elice` | `~/.letify-runtime` |
| `colab` | `/content/letify` |
| `modal` | `/letify`, where a Modal volume named `<app>-workspace` is mounted in every sandbox, so the root persists across sandboxes |
| `local` | not used. A volume materialized on `local` without `mount` lands under `~/.letify-runtime` on this machine |

A `modal` account with `workspace` set mounts the same volume at that path instead, when the path starts with `/`.

Everything letify writes on the runtime is under the root:

| Path | Holds |
|---|---|
| `<workspace root>/project/<env key>` | the project files `uv sync` reads, and the `.venv` it builds |
| `<workspace root>/project/.<digest>.tar.gz` | an environment archive while it is unpacked, removed once the `.venv` starts |
| `<workspace root>/volumes/<volume name>` | a volume's materialized blobs and project data |
| `<workspace root>/tmp` | temporary files, including the archive `pack_dir` builds on a one-shot channel; `TMPDIR` points here |

The worker's working directory is the root, so a relative path in user code resolves under it. The uv installer is the one exception to the root: uv goes to `~/.local/bin`, as uv on the runtime describes, because it is shared by every account on that home directory.

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

Every kind except `local` takes `--workspace PATH`, which writes `workspace` to the account. `letify login shell` and `letify login tunnel` also ask for it, after the BatchMode key check and before devices are detected, with the kind's default in brackets: `Workspace root on <address> [~/.letify-runtime]: `. A blank answer, or `--no-input` without `--workspace`, takes the default and writes no field. The chosen path is then checked over the same BatchMode SSH options: `mkdir -p` on the path, then a write and removal of the probe file `<path>/.letify-probe`, as the account's own user, so a path that needs root fails. A failure writes nothing to either file and raises `LoginError` naming the path, the machine and the error the command printed. A password account has no BatchMode connection, so its workspace is recorded without the check. `colab`, `modal` and `elice` record `--workspace` without a remote check, because there is no machine to reach at login.

An account that is already in the home file is checked again only when `--workspace` is given. The check then runs with the connection details the home file holds, and the field is written when it passes.

`letify check <alias>` on a `shell`, `tunnel`, `colab` or `elice` account runs the same probe on the account's workspace root in the command that confirms the machine answers, and prints `workspace <path>: writable` or `workspace <path>: not writable: <error>`.

The project's `.letify/config.toml` gets the alias as an empty table, `[colab_pro]`. Nothing else, because everything else is either a secret or a detail of one person's machine, and naming the alias is what makes the account available in the project. That table is what makes the repository self describing: a teammate who clones it can run `letify login` for the aliases it names and nothing else has to be explained. A named alias the home file does not declare is a configuration error naming the command that fixes it.

An account that is already in the home file is not asked for again. `letify login lab` in a second repository writes only the reference, which is the common case: the account was set up once and every project since then just needs to name it.

`letify logout <alias>` removes the account from `~/.letify/config.toml` and deletes `~/.letify/accounts/<alias>/` with everything in it. It leaves the project reference alone, because the repository still needs that account; what changed is only that this machine no longer has it.

`letify login colab <alias>` signs in to Colab itself. It runs `colab sessions` through `uv tool run --python 3.13 --from google-colab-cli colab`, with `HOME` set to `~/.letify/accounts/<alias>/`. The Colab CLI keeps its token at a fixed path under its home directory, so the token lands in the account directory and the CLI refreshes it on later calls. Every later Colab command runs with the same `HOME`, which is what lets two Colab accounts live on one machine. uv's cache, Python installs and tools stay pinned to the real home, so a changed `HOME` downloads nothing again. A sign in that exits non zero writes nothing.

`letify login modal <alias>` signs in to Modal itself. It first asks for an optional Modal profile, which names the Modal workspace to sign in to. It then runs `modal token new` through `uv tool run --python 3.12 --with "modal>=1.0,<2" --from modal modal`, with `MODAL_CONFIG_PATH` set to `~/.letify/accounts/<alias>/modal.toml` and, when a profile was given, `--profile <profile>`. The profile is written as `profile`, because `workspace` is the workspace root. Modal's command prints a link and waits for the browser approval, so `modal` never has to be on `PATH` or in the project's environment. The token lands in the account directory, and the adapter reads it from there. A sign in that exits non zero, or exits zero without writing `modal.toml`, writes nothing to either `config.toml` and removes a `modal.toml` the attempt created.

`letify login tunnel <alias> --connect <token>` declares a machine behind NAT from the command `letify client shell connect` printed on it. Without `--connect`, a terminal is asked `Token printed by 'letify client shell connect': `, and `--no-input` refuses. The steps run in this order, and a failure at any step writes nothing to either file and raises `LoginError` whose message starts with `tunnel login failed at <step>: `:

1. `tailcat`: `tailcat` must be on the local `PATH`. Otherwise the message is the same install instructions `letify client shell connect` prints, for this machine's operating system and architecture.
2. `token`: the token is decoded. A token that is not the base64 JSON described under Rendezvous, or that lacks `tailcat` or `tailcat_port`, is refused.
3. `key install`: the key is generated and installed as SSH authentication describes, over SSH with `-o ProxyCommand=tailcat <address> <agent port>`, logging in as the token's `user` on the token's `port`. `--skip-key-install` and `--key` work as for `shell`.
4. `key confirmation`: the key is confirmed with `BatchMode=yes` over the same `ProxyCommand`.
5. `workspace`: the workspace root is chosen and checked as for `shell`, over the same `ProxyCommand`.
6. `devices`: the GPUs are recorded as Recording devices at login describes, over the same `ProxyCommand`.

The account is written with `kind = "tunnel"`, `tailcat`, `tailcat_port`, `user`, `port` and `key` from the token and the options, and no `address`.

Credentials never enter either `config.toml`. A token goes to a file in the account directory. An SSH password is never stored at all, which the next section explains.

### Recording devices at login <!-- id: login-records-devices -->

> `letify login shell` and `letify login tunnel` ask the machine for its GPUs once, right after the key is confirmed, and write the ones the user chose to `[<alias>.devices]` in `~/.letify/config.toml`.

Detection happens at login and not at first use, for two reasons. The generated provider types read the `devices` table without a network call, so an editor completes the accelerator names only once the table exists. And on a shared machine the table is where the user says which cards are theirs, so asking at login is what makes the first session use those cards and no others.

1. After the key is confirmed with `BatchMode=yes`, the same SSH connection options run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader` once. A password account has no BatchMode check to follow, so it is not detected.
2. Each name is normalized with the accelerator name normalizer, and the cards are grouped by that name with their physical indices kept.
3. With a terminal, each group is shown as one line, `A100: 4 cards, indices 0-3 (80 GB each)`, and the user is asked `Indices letify may use for A100 [0-3]: `. A blank answer takes every card found. An answer takes the forms the inventory takes, `0-3` or `0,1,6`. An answer naming an index that was not found, or that cannot be read, is refused with the reason and asked again.
4. With `--no-input`, `--indices NAME=SPEC` chooses the indices for one name and may be repeated. A name without the option gets every card found. An option naming an accelerator that was not found, or an index that was not found, fails the login with nothing written.
5. The result is written as `[<alias>.devices]` with one `NAME = { indices = "..." }` line per name, through the same textual writer as the account, so every other table and comment in the file stays as it was. A contiguous choice is written as a range, `"0-3"`, and any other choice as a list, `"0,1,6"`.
6. When `nvidia-smi` is missing, exits non zero, or lists no GPU, the login still succeeds. Nothing is recorded, a note says why, and the accelerators are discovered on first access.
7. An account already in the home file is not detected again. `--detect-devices` asks the machine again with the connection details the home file holds, shows what it found, and replaces the table after the user confirms with `y`. With `--no-input` the flag itself is the confirmation. When nothing is found, the existing table is left alone.
8. After a table is written, the provider types are regenerated as `Launcher()` does, so completion picks up the names at once.

`letify logout <alias>` removes `[<alias>.devices]` together with `[<alias>]`, because a devices table without its account would declare an account with no kind.

### SSH authentication

> Key authentication, because the call path is non-interactive. A password is accepted once, to install the key, and then discarded.

letify opens sessions with `ssh -o BatchMode=yes`. That is not a preference: a session is started by the pool, in the background, possibly long after the call that needed it, so there is nobody present to answer a password prompt. A transport that requires interaction cannot carry a pooled session.

So `letify login shell` sets up key authentication and treats the password as a one-time input:

1. If the configured key does not exist, an ed25519 key is generated at `~/.ssh/id_letify` with no passphrase, because a passphrase would put the prompt back.
2. The public key is appended to the machine's `~/.ssh/authorized_keys`, over one interactive SSH connection that asks for the password in the terminal.
3. The password is used by that one command and then dropped. It is not written to a file or to the environment.
4. The connection is confirmed with `BatchMode=yes`, which proves the key works before the alias is declared rather than at the first call.
5. The machine's GPUs are detected over that confirmed connection, as described under Recording devices at login.

Two other approaches were considered and are not the default. Connection multiplexing with `ControlMaster` authenticates once and reuses the socket, but Windows OpenSSH does not implement it and a dropped socket ends a long run. `sshpass` feeds a stored password to each connection, which needs the password kept somewhere and exposes it in the process arguments of every call. `sshpass` is available as `auth = "password"` for a machine whose administrator forbids key authentication, reading the password from `~/.letify/accounts/<alias>/password`, and it refuses on Windows, where the tool does not exist.

### What each kind asks for

> Where a vendor owns the credential format, letify runs the vendor's own sign in and points it at the account directory.

| Kind | Written to the home file | Credential |
|---|---|---|
| `shell` | address, user, port, key path, `workspace` when it is not the default, and the `devices` table the machine reported | an SSH key, installed by `login`; no password stored |
| `tunnel` | `tailcat`, `tailcat_port`, user and port from the token `letify client shell connect` printed, key path, `workspace` when it is not the default, and the `devices` table; no address | an SSH key, installed by `login` over Tailcat; no password stored |
| `elice` | endpoint, zone, machine, `workspace` when given | access token in `~/.letify/accounts/<alias>/access_token` |
| `colab` | account email, `workspace` when given | the Colab CLI's token, written by its own sign in under `~/.letify/accounts/<alias>/` |
| `modal` | `profile` and `workspace`, each when given | Modal's token, written by `modal token new` to `~/.letify/accounts/<alias>/modal.toml` |
| `local` | nothing | none; this machine needs no declaration |

For `colab` and `modal`, letify runs the vendor's sign in through uv and does not parse or refresh the token. The vendor's client reads and refreshes it from the account directory.

### Interpreter override <!-- id: python-option -->

> `python` on an account names the interpreter the worker runs with. Setting it means the user manages that interpreter, so letify does not build the environment there.

Without `python`, a `shell`, `tunnel`, `colab` or `elice` account starts its bootstrap worker with `python3` and then runs the worker from the project `.venv`, as Building the environment on a runtime describes. With `python = "/path/to/python"`, the worker is started with that interpreter and stays on it: no project files are sent, no uv runs and no environment archive is read or written. The interpreter check still applies. On `local`, `python` names the interpreter of the worker subprocess, which defaults to the interpreter running letify.

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

### Framing

> Every message is an 8 byte little-endian length followed by that many bytes of body. The protocol version is 2.

The body is a tag byte and then fixed-width fields. The length is 64 bits because a single copy to the device may exceed 4 GiB, and a 32 bit length would wrap without an error. A reader grows its buffer as bytes arrive rather than trusting the length up front, so a corrupt length ends the connection with an error instead of an allocation failure. The driver and the agent refuse each other when their protocol versions differ.

### Copies to the device

> The bytes of a copy to the device are written from the caller's buffer and read into the agent's staging buffer, with no copy in between.

On the driver, `cuMemcpyHtoD_v2` writes the frame header and the fixed fields, then hands the caller's slice to `write_vectored`. A payload larger than the 8 KiB write buffer goes to the socket without being copied into it. On the agent, a `CopyToDevice` frame is recognised by its tag before its body is read, and the payload is read with `read_exact` into a staging buffer the session keeps and reuses, which is then passed to the real driver. Batching and `TCP_NODELAY` are the same as for every other request. Measured throughput is in [NETWORK.md](NETWORK.md#letify-core-copy-throughput).

### Virtual pointers

> An allocation returns a pointer immediately, and memory accounting stays local so that running out still fails at the call.

The local driver hands out pointers from a range no real device address falls in, records what they stand for, and lets the agent reconcile them in the background. Waiting for the agent would put a round trip in front of every allocation, and a caching allocator makes many.

The cost is honest failure. A caching allocator learns the device is full when the allocation call fails, frees its cache and retries. With a virtual pointer there is nothing to fail yet, so the local driver keeps its own accounting of device memory and refuses once the budget is gone, with a reserve held back for the driver's own context, library workspaces and fragmentation.

### Module identity

> A compiled module is named by its contents, so a fatbin the agent already holds is not sent again.

PyTorch loads the same modules on every process start and they are large. The agent keeps a table keyed by digest and answers with the handle it already has.

`cuModuleLoadData` receives a pointer with no length, so the driver reads the size from the image itself and sends the whole image. A fatbinary is `fat_size` bytes, read from its header after the magic `0xBA55ED50`. An ELF object is `e_shoff + e_shentsize * e_shnum` bytes, or the end of its program headers when that is larger. Anything else is a PTX text image and runs to its terminating NUL, which is included.

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

Provider tools run out of process and never in the user's `.venv`. Colab runs through `uv tool run --from google-colab-cli colab`. Modal's client runs in the Modal adapter, in a uv environment started with `uv run --no-project --with "modal>=1.0,<2"`, and its sign in runs through `uv tool run --from modal modal`. Elice uses the standard library HTTP client. The `gcs` blob store uses a standard library client too.

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
- **`Modal` and `Elice` are not exercised against the live services.** Their code follows each service's published interface, and the Elice paths come from Elice's own Terraform provider, but neither has been run end to end. The Modal adapter's calls were checked against the signatures of Modal 1.5.5, and `letify login modal` has not been run against Modal's sign in.
- **The connection pipeline is not exercised against live networks.** `Rendezvous`, `Strategy`, `Link`, `Probe`, `Pipeline`, `LinkCache` and the remote agent are implemented and tested over loopback sockets and faked commands. Installing and starting `sshd` on a Colab VM over `colab exec` is not yet checked against a live runtime.
- **The Elice API runs no command on a machine.** The paths letify uses (virtual machine, allocation, instance type, pricing) create and power machines only, so Elice's remote half runs over forward SSH to the allocated machine, and the punch and Tailcat strategies need that SSH to succeed first.
- **Orphan reconciliation is not implemented.** A session whose controlling machine was killed outright is released by the lease on the providers where the process is the cost. Where the platform bills for the machine and takes no deadline, nothing ends it: an Elice allocation bills until a delete is issued. The intended answer is that the next letify process asks the provider what is running under this project's name and ends what nothing is watching, with a command to do it on demand. Neither exists yet.
- **Whether the Elice allocation API takes a deadline is unverified.** If it does, that is where the guarantee belongs, because the platform outlives the caller.
- **Persistence detection is not implemented.** Deciding a machine's disk policy by writing a marker file and looking for it in a later runtime is a decision recorded here, not yet code.
