# Components

> What each object is, what it owns, and how the vocabulary fits together.

## The vocabulary

Read this table first. Most confusion about letify is one of these words meaning something else in another tool.

| Word | In letify |
|---|---|
| **Launcher** | The public entry point. Loads configuration, hands out providers, accepts declarations, owns the pool. The variable name for it is `let`. |
| **Provider** | One account on one kind of infrastructure. A class such as `Colab` or `Shell`, instantiated once per configuration entry. |
| **provider object** | What you get from `let.providers.<alias>`. Said this way when the object-oriented sense of "instance" would be ambiguous. |
| **Instance** | One accelerator shape offered by a provider. `colab.G4` is an Instance. Not a virtual machine and not an object instance. |
| **device** | Where the accelerator is. The decorator argument that carries provider, account and accelerator. |
| **host** | Where the host code runs. CUDA's word for the CPU side, paired with the device. |
| **Runtime** | One live session. The only object that costs money. |
| **keep_alive** | The block, `with let.keep_alive():`, that keeps sessions between calls. Outside it a call ends its session. |
| **Channel** | How letify talks to a session: persistent, or one-shot. |
| **Env** | A declaration of the remote environment, keyed by a uv lock file. Not a container image. |
| **Volume** | A named content addressed blob store on a provider's storage. Not a mounted filesystem. |
| **Handle** | A reference to an object that lives in a session. |
| **Blob** | A large argument named by the hash of its contents. |
| **persistence** | Whether a provider's storage outlives a session: `persistent` or `ephemeral`. |
| **letify-core** | The Rust workspace behind `host="local"`. Its crates are `letify-wire`, `letify-driver` and `letify-agent`. |

## The shape of the thing

```
Launcher (let)
├── Config                  .letify/config.toml from home and project, merged
├── Providers               attribute access by alias
│   └── Provider            one account on one infrastructure
│       ├── Instance        accelerator shape
│       ├── Volume          content addressed store
│       │   └── Store ── Backend    filesystem, gcs, modal
│       └── Runtime         one live session
│           ├── Channel     persistent, or one-shot
│           └── Lease       the deadline that outlives nothing
├── RuntimePool             sessions keyed by (Instance, Env)
└── Function                one declaration, created by @let.function
```

Declaration flows down and cost appears at exactly one point. `Launcher`, `Provider`, `Instance`, `Env`, `Volume` and `Function` are all descriptions. `Runtime` is the only thing that powers hardware on.

## Package layout

One directory per concern, so the file you need is the one named after the thing you are changing.

| Package | Owns |
|---|---|
| `config/` | Reading `.letify/config.toml`, the schema, login, resolving credentials |
| `declare/` | `Launcher`'s surface, `Function`, `Instance`, `Env` |
| `protocol/` | Handles, the codec, framing, the remote worker, the one-shot driver |
| `runtime/` | The channel, the session, the pool, the lease, bootstrap source |
| `providers/` | The base class, name normalization, and one module per provider |
| `store/` | The content addressed store, volumes, and `backends/` |
| `remoting/` | The capability probe and the loader for `letify-core` |
| `launcher.py` | `Launcher` itself, which ties the rest together |
| `errors.py` | The exception hierarchy, which everything imports |
| `cli.py` | The command line |
| `stubs.py` | Writing `typings/letify_providers.pyi` for editor completion |

## Launcher

Owns the configuration, the provider cache and the runtime pool.

It exists as an object rather than as module-level state because it holds the pool. It does not exist to hold per-instance configuration: everything that varies between calls, which is the device, the host placement, the environment and the account, varies per declaration.

`let.providers` is a view, not a dictionary of its own. Attribute access returns a `Provider` and builds it on first use, so a provider that has to connect does not connect at import time. Three names are reserved:

- `any` returns a stand-in whose attributes are provider-free accelerator requests, resolved in configuration order.
- `devices` returns the accelerators each provider offers, reporting a reason instead of raising for one that cannot be reached.
- `active` returns the providers that currently hold a session, which is the quickest answer to what is costing money.

There is nothing to tear down. `keep_alive()` is the one scope, and it only keeps sessions. `invocation()` is internal: one call brackets itself with it so calls that overlap in time reuse the sessions they release until the last of them finishes.

## Provider

Abstract base with five responsibilities: say which accelerators the account offers, say whether storage outlives a session, own the volumes, open the channel, and start and stop the session.

Four class attributes drive every default, and none is a user-facing switch.

`persistence` says whether storage outlives a session. `has_fast_path` says whether the machine is close enough for CUDA call forwarding to pay off, which controls whether a warning is issued rather than whether the mode is allowed. `persistent_channel` says whether a worker process can be kept alive. `needs_lease` says whether a session can outlive this process and keep billing.

### The hierarchy

```
Provider
├── Local      persistent, no environment install, no lease
├── Modal      persistent, a sandbox with framed pipes
└── Shell      ephemeral by default, SSH transport
    ├── Colab  Colab CLI for sessions, its SSH bridge or exec for the channel
    ├── Tunnel Tailscale or frp first, then SSH
    └── Elice  Elice Cloud API allocates, then SSH
```

`Shell` is named for the capability rather than the transport. SSH is its default, and its subclasses differ in how the connection is obtained: `Colab` creates the session with a CLI, `Tunnel` builds a network path, `Elice` allocates a machine through an API. A future transport that is not SSH fits without renaming the base.

`Local` overrides more than the others: it installs no environment, needs no lease, and runs the worker in a subprocess of this machine. That makes it the test path for everything else, because the same serialized call goes through the same worker.

## Instance

An accelerator shape, carrying the provider it came from, the accelerator name, the host placement, and the core count, memory and VRAM the provider reported.

It carries the provider so that `device=colab.G4` fixes provider, account and accelerator in one argument. Those three are one decision, and splitting them across three keyword arguments made it possible to write a combination that does not exist.

Core count and memory are reported, not requested. A provider that offers several sizes registers them as separate shapes, so there is nothing for a declaration to choose and no way to ask for a shape the provider does not have.

An instance carries no placement; `host=letify.local` or `letify.remote` on the declaration does. `Instance.key` is what the pool matches on. `AnyInstance` is the deferred form, produced by `let.providers.any.G4` and resolved by the launcher against declaration order.

## Channel

How letify talks to a session, and the reason some providers can do more than others.

`PersistentChannel` keeps one worker process alive behind a pipe. Requests are framed lines, so the object table, the blob table and anything written to disk survive between calls. That is what makes a handle resolvable and a large argument sendable once.

`OneShotChannel` can only run a command and collect its output. Every call starts a fresh process, so nothing persists, and it refuses the operations that need persistence rather than pretending.

The worker cannot be sent as a script on standard input, because `python -` reads to end of file before compiling and the pipe has to stay open. A bootstrap stub passed with `-c` reads a length-prefixed blob, executes it, and leaves standard input where it was.

## Runtime, Lease and RuntimePool

A `Runtime` is one live session. It knows its provider, its instance, its environment, its volumes, and its channel.

Booting one opens the channel, arms the lease, installs the environment and attaches volumes. The lease is the part that protects the bill: the local process renews a deadline inside the session, and the session terminates itself if renewal stops for longer than the grace period. A crashed script therefore cannot leave a GPU billing, while a brief network drop does not kill a training run.

`RuntimePool` keys sessions by instance and environment, which is the whole economic argument for the library. Starting a session per call would pay provider boot, environment installation and the first transfer every time, and all of it is billed as GPU time.

The pool also enforces the release rule. A session ends when it is released, unless a `keep_alive` block holds the pool; idle sessions end when the outermost block exits. An internal hold brackets a single invocation so an overlapping call can reuse a session another call released. A session still starting counts as serving a call, so a call that finds every card reserved by this process waits for one. A call whose devices cannot be allocated raises `InsufficientDevices` instead of waiting.

## Function

What `@let.function` returns. Holds the declaration and the launcher, and decides nothing at call time except which session to use.

`is_async` is read from the wrapped `def`, which is what makes blocking behaviour a property of the declaration. A sync declaration returns its value. An async one returns a plain coroutine, so `await`, `asyncio.gather` and `asyncio.as_completed` accept it, and concurrency comes from those rather than from a type letify adds.

Retry policy lives here. `RuntimeFailure` and `ProtocolError` discard the session and retry, because the session is at fault. `RemoteError` propagates, because the user's code is at fault and a retry reproduces it.

## Env

A declaration keyed by the hash of a uv lock file plus any refinements. The key is what the pool and the blob store both use, so two declarations that agree reuse the same session and the same prebuilt archive.

It is called `Env` rather than `Image` because there is no image: no container is built, and what letify can arrange is an environment installed from a lock file. Naming it for a mechanism it does not use would set the wrong expectation.

## Volume, Store and Backend

Three layers with one job each.

`Volume` is the user-facing name on a provider. It knows the mount point, the reserved refs for environments and checkpoints, and how to materialize a blob into a session: the session pulls it from the backend with a borrowed short-lived token when the backend offers a pull, and otherwise the blob goes through the channel.

`Store` is the content addressed logic: put and get blobs, pack and unpack trees, read and write refs, and plan which of a list of files still needs uploading.

`Backend` is where bytes live. All backends keep the same layout, so a blob written by one is readable by another pointed at the same bucket.

The layering exists so that provider and storage vary independently. Colab reads from Google Cloud Storage because a Colab session is a Compute Engine virtual machine, Elice reads from the machine's own disk, and neither fact is visible in the declaration.

## Handle, Blob and the call protocol

`protocol` owns the wire: serialize a call, frame it, run it, decode the outcome.

A `Handle` is a reference to an object registered in a session's object table. It carries the name of the live session, not the pool key, because two sessions can share a key and an object lives in only one of them. Passing one to a call on a different session raises rather than materializing the object.

A `Blob` is a large argument named by the hash of its contents. The runtime is asked which digests it already holds before anything is sent, so a value passed repeatedly crosses the network once.

Three protocol rules exist for performance rather than tidiness. Large arguments are content addressed. Values may stay remote. And a failure never falls back to a slower path, because a silent downgrade turns a four times difference into a mystery.

## remoting

The Python half of CUDA call forwarding: find the built library, measure the round trip, report what is missing, and get the library loaded before the real driver.

It refuses rather than degrading when the library or the agent is absent. It does not refuse merely because the link is slow, because that judgement belongs to whoever wrote the declaration.

`inject()` is where the platforms differ. On Windows it calls `os.add_dll_directory`, which puts the library at the front of the loader's search order, and it must run before `import torch`. On Linux the equivalent is `LD_PRELOAD`, which cannot be set from inside a running process, so it reports the command instead of pretending it succeeded.

## letify-core

The native component, outside the Python package because standing in for the CUDA driver cannot be done from Python.

| Crate | Owns |
|---|---|
| `letify-wire` | The protocol. A request declares whether it needs a reply, which is where the batching rule lives. |
| `letify-driver` | The cdylib that stands in for the driver. Virtual pointers, local memory accounting, and the entry points themselves. |
| `letify-agent` | Holds the real device. Maps handles onto real pointers and keys modules by content. |

The split matters because the batching rule is a property of the protocol rather than of any one intercepted symbol. A launch, a copy to the device and an allocation are queued; a copy back, a stream synchronization and an elapsed time query are not. That is why the round trip count is the number of host synchronizations rather than the number of calls.
