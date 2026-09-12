# Components

> What each object is, what it owns, and how the vocabulary fits together.

## The vocabulary

Read this table first. Most confusion about letify is one of these words meaning something else in another tool.

| Word | In letify |
|---|---|
| **Launcher** | The public entry point. Loads configuration, hands out providers, accepts declarations, owns the pool. The variable name for it is `let`. |
| **Provider** | One account on one kind of infrastructure. A class such as `Colab` or `Shell`, instantiated once per configuration entry. |
| **provider object** | What you get from `let.providers.<alias>`. Said this way when the object-oriented sense of "instance" would be ambiguous. |
| **Instance** | One accelerator shape offered by a provider, carrying CPU placement. `colab.G4` is an Instance. Not a virtual machine and not an object instance. |
| **Runtime** | One live session. The only object that costs money. |
| **Env** | A declaration of the remote environment, keyed by a uv lock file. Not a container image. |
| **Volume** | A named content addressed blob store on a provider's storage. Not a mounted filesystem. |
| **Sweep** | A declared search space, produced by `grid` or `zip`. |
| **Handle** | A reference to an object that lives in a runtime. |
| **persistence** | Whether a provider's storage outlives a runtime. `persistent` or `ephemeral`. |
| **placement** | Where the Python side of the work runs. `remote` ships the function, `local` forwards CUDA calls. |

## The shape of the thing

```
Launcher (let)
├── Config                  .letify from home and project, merged
├── Providers               attribute access by alias
│   └── Provider            one account on one infrastructure
│       ├── Instance        accelerator shape plus CPU placement
│       ├── Volume          content addressed store
│       │   └── Store ── Backend    filesystem, gcs, s3, modal
│       └── Runtime         one live session
├── RuntimePool             runtimes keyed by (Instance, Env)
├── Function                one declaration, created by @let.function
└── Sweep                   a declared space, passed into a call
```

Declaration flows down and cost appears at exactly one point. `Launcher`, `Provider`, `Instance`, `Env`, `Volume`, `Function` and `Sweep` are all descriptions. `Runtime` is the only thing that powers hardware on.

## Launcher

Owns the configuration, the provider cache, the runtime pool and the run scope.

It exists as an object rather than as module-level state because it is the thing that holds the pool, and because the scope has to have something to hang off. It does not exist to hold per-instance configuration: everything that varies between calls, which is the accelerator, the placement, the environment, the provider and the account, varies per declaration rather than per launcher.

`let.providers` is a view, not a dictionary of its own. Attribute access returns a `Provider` and builds it on first use, so a provider that has to connect does not connect at import time. Three names are reserved:

- `any` returns a stand-in whose attributes are provider-free accelerator requests, resolved in configuration order.
- `gpus` returns the accelerators each provider offers, reporting a reason instead of raising for one that cannot be reached.
- `active` returns the providers that currently hold a runtime, which is the quickest answer to what is costing money.

## Provider

Abstract base with four responsibilities: say which accelerators the account offers, say whether storage outlives a runtime, own the volumes, and start a runtime.

Two class attributes drive every default, and neither is a user-facing switch.

`persistence` says whether storage outlives a runtime. `has_fast_path` says whether the machine is close enough for CUDA call forwarding to pay off. Together they produce `default_cpu_placement`, which is what a declaration inherits when its instance does not state a placement.

### The hierarchy

```
Provider
├── Local      persistent, has_fast_path, no environment install, no lease
├── Modal      persistent, its own API, lazily imported
└── Shell      ephemeral by default, SSH transport, has_fast_path
    ├── Colab  Colab CLI for sessions, no fast path
    ├── Tunnel Tailscale or frp first, then SSH
    └── Elice  Elice Cloud API allocates, then SSH
```

`Shell` is named for the capability rather than the transport. SSH is its default, and its subclasses differ in how the connection is obtained: `Colab` creates the session with a CLI, `Tunnel` builds a network path, `Elice` allocates a machine through an API. A future transport that is not SSH fits without renaming the base.

`Colab` is a `Shell` subclass because the Colab CLI offers `colab ssh --proxy-mode`, an OpenSSH ProxyCommand bridge. Once a session exists it is reached the same way any other machine is.

`Local` overrides more than the others: it installs no environment, needs no lease, and runs code in a subprocess of this machine. That makes it the test path for everything else, because the same serialized call goes through the same driver script.

### What each provider reports

| Provider | Persistence | Fast path | Store backend | Session created by |
|---|---|---|---|---|
| `Local` | persistent | yes | `filesystem` | a subprocess |
| `Modal` | persistent | no | `modal` | the Modal API |
| `Colab` | ephemeral | no | `gcs` | `colab new` |
| `Shell` | ephemeral, overridable | yes | `filesystem` | SSH |
| `Tunnel` | ephemeral, overridable | yes | `filesystem` | tunnel, then SSH |
| `Elice` | persistent | yes | `s3` | an allocation POST |

## Instance

An accelerator shape, carrying the provider it came from, the accelerator name, the CPU placement, the core count, the memory sizes and whether it is preemptible.

The reason it carries the provider is that `gpu=colab.G4` then fixes provider, account, accelerator and placement in one argument. Those four are one decision, and splitting them across four keyword arguments made it possible to write a combination that does not exist.

Calling an instance refines it and returns a new value: `colab.G4(cpu="local", cpus=8)`. The registered instance is never mutated, so two declarations that refine the same shape do not interfere.

`Instance.key` is what the pool matches on. `AnyInstance` is the deferred form, produced by `let.providers.any.G4` and resolved by the launcher against declaration order.

## Runtime and RuntimePool

A `Runtime` is one live session. It knows its provider, its instance, its environment and its volumes, and it holds the lock that makes it exclusive to one call at a time.

Booting a runtime installs the environment, attaches volumes and arms the lease. The lease is the part that protects the bill: the local process renews a deadline inside the session, and the session terminates itself if renewal stops for longer than the grace period. A crashed script therefore cannot leave a GPU billing, while a brief network drop does not kill a training run.

`RuntimePool` keys runtimes by instance and environment, which is the whole economic argument for the library. Starting a session per call would pay provider boot, environment installation and the first transfer every time, and all of it is billed as GPU time.

The pool bounds itself with a semaphore. A call that finds every slot occupied waits for a runtime of its own key rather than starting a session that the provider would refuse.

## Function

What `@let.function` returns. Holds the declaration and the launcher, and decides nothing at call time except which runtime to use.

`is_async` is read from the wrapped `def`, which is what makes blocking behaviour a property of the declaration. A sync declaration returns its value; an async one returns `_AsyncCall`, which is both awaitable and async-iterable so that `await` and `async for` express ordering.

Retry policy lives here. `RuntimeFailure` and `ProtocolError` discard the runtime and retry, because the session is at fault. `RemoteError` propagates, because the user's code is at fault and a retry reproduces it.

## Env

A declaration keyed by the hash of a uv lock file plus any refinements. The key is what the pool and the blob store both use, so two declarations that agree reuse the same runtime and the same prebuilt archive.

It is called `Env` rather than `Image` because there is no image: no container is built, and what letify can arrange is an environment installed from a lock file. Naming it for the mechanism it does not use would set the wrong expectation.

## Volume, Store and Backend

Three layers with one job each.

`Volume` is the user-facing name on a provider. It knows the mount point and the reserved refs for environments and checkpoints.

`Store` is the content addressed logic: put and get blobs, pack and unpack trees, read and write refs, and plan which of a list of files still needs uploading.

`Backend` is where bytes live. All backends keep the same layout, so a blob written by one is readable by another pointed at the same bucket.

The layering exists so that provider and storage vary independently. Colab reads from Google Cloud Storage because a Colab runtime is a Compute Engine virtual machine, Elice reads from an S3 compatible store because Data Hub speaks S3, and neither fact is visible in the declaration.

## Sweep

A finite set of keyword argument combinations. `grid` is the product, `zip` is the pairing, `|` is the union.

It is a value rather than a method on a function because the space is data. The same space can be declared once and passed to several functions, stored in a configuration file, or built by code, none of which works when the iteration is a method call.

## Handle and the call protocol

`wire` owns the protocol: serialize a call, run it under a driver script, find the outcome between markers, decode it.

A `Handle` is a reference to an object registered in a runtime's object table. It carries the runtime key, and passing it to a call on a different runtime raises rather than materializing the object, because a handle is a pointer into one process and one CUDA context.

Three protocol rules exist for performance rather than tidiness. Large arguments are content addressed, so the same weights passed ten times travel once. Values may stay remote, so a model is not copied back and forth. Callbacks to the local process are one way, because an acknowledgement inside a token loop would put a round trip on every token.

## remoting

The capability probe for CUDA call forwarding. It reports whether a layer 3 tunnel is possible, whether the driver shim is present, and what the round trip is, and it refuses when forwarding would not pay off.

It refuses rather than degrading because a silent downgrade is the failure this design is built to avoid. A user who asked for `cpu="local"` and got function shipping instead would see a four times difference in throughput with nothing in the output explaining it.

The forwarding client itself is not implemented. See the known gaps in [SPEC.md](SPEC.md).
