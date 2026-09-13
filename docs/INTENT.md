# Intent

> Why letify exists, what it claims, and what it will not do.

## The problem

A researcher with no GPU of their own, or one whose lab GPU is always queued, has to rent one. The cheapest way to rent a given card is often the least convenient: a notebook session that is evicted, has no persistent disk, and hands out whichever accelerator is free. The convenient way costs several times more per hour for the same silicon.

The gap is large enough to change what research is affordable. The same RTX PRO 6000 costs about 975 KRW per hour through Colab credits and about 4,070 KRW per hour on Modal, a factor of four. For a graduate student paying out of pocket, that factor decides how many experiments get run.

letify exists so that the cheap option can be used like the expensive one. You declare what a function needs, and it runs on whichever infrastructure you have, without the session management, the repeated environment setup and the repeated data transfer that make the cheap option painful.

## Goals

1. **One declaration, several providers.** The same declared function runs on Colab, on Modal, on a lab server over SSH, and on the local machine. Changing provider is changing one value, not rewriting the code.
2. **Setup cost paid once, not per call.** Provider boot, environment installation and data transfer are billed as GPU time, so they must be amortized across calls rather than repeated.
3. **No silent performance cliff.** If the fast path is unavailable, letify says so and stops. It never quietly takes a path that is four times slower.
4. **A bill that cannot run away.** A session cannot outlive the process that started it. A crashed script must not leave a GPU billing.
5. **Short sessions are the normal case.** The design assumes one to two hours of work, frequent restarts, and eviction at any moment.

## Claims

Each claim is one sentence to agree or disagree with. An experiment that tests one lists it in its pull request YAML as `claims: [N1]`.

### N1. Shipping the whole loop keeps efficiency near a local run even over a high-latency link

Sending a training or generation loop to the remote machine and running it there reaches at least 95 percent of the throughput of running on that machine directly, at a 150 ms round trip, because the loop's host synchronizations become local to the remote process.

### N2. Forwarding CUDA calls is only viable at low latency and low synchronization count

Efficiency for call forwarding follows `T / (T + k * RTT)`, where `T` is GPU time per step and `k` is host synchronizations per step. It is therefore unusable for token by token decoding at any useful round trip, and it gets worse as the GPU gets faster.

### N3. Pooling runtimes by declaration removes per-call setup cost

Reusing a session across calls with the same instance and environment reduces the amortized setup cost per call to near zero, where starting a session per call pays provider boot time every time.

### N4. A content addressed store with archive-level granularity beats file-level synchronization

For an environment or a model cache, packing a tree into one hash-named blob transfers faster than synchronizing files individually, and immutable naming removes the write conflicts that a two way synchronization has between concurrent sessions.

### N5. A declaration should state the execution mode rather than have it derived

Two modes that differ by a factor of two in throughput and by a factor of fifty for decoding are too far apart to be chosen implicitly. A reader of the declaration should be able to see which one it uses, and nothing should infer it from storage or link properties.

### N6. Three placements are enough to cover every provider without naming a mechanism

A declaration that says where the accelerator is, where the host code runs and how long the session lives is enough to place any supported workload, and a user never has to name a transport, a channel kind or a storage backend.

### N7. Batching driver calls makes the round trip count the synchronization count

Forwarding is viable only if a step that issues thousands of driver calls pays a handful of round trips. Queueing every call whose result the host does not read achieves that, so the efficiency model is `T / (T + k * RTT)` with `k` counting host synchronizations rather than calls.

## Constraints

- **The Python package is pure Python.** No compiled extension in `letify/`. A wheel that has to be built for each platform is a maintenance cost this project will not carry, and hashing and transfer are not CPU bound at the link speeds involved.
- **One native component, built separately.** Standing in for the CUDA driver cannot be done from Python, so that job lives in `letify-core/` as a Rust workspace. Only whoever uses `host="local"` builds it, and the Python package works without it.
- **The local process stays alive for the duration of a run.** letify does not offer detached execution. A detached run whose remote side is evicted loses its results, so the local process stays the owner and the durable artifacts are checkpoints in the store.
- **Nothing is torn down by hand.** No release call and no shutdown call on the public surface. A call ends its own session, an idle one is reaped, and the lease covers a crash.
- **No credential in a tracked file.** Accounts and keys live in `~/.letify` or in the environment or the OS keyring.
- **Colab accelerators require a paid entitlement.** The remote control features letify uses are permitted on paid plans while the compute unit balance is positive.

## Non-goals

- **Not a scheduler for a shared cluster.** letify targets one researcher's own accounts, not queue management for a group.
- **Not a training framework.** It runs the user's code. It does not own the training loop, the metrics or the checkpoint format.
- **Not a way around a provider's limits.** It does not attempt to bypass session limits, quotas or terms.
- **No bitwise reproducibility.** Kernel selection by measured timing and batch dependent reduction order make that unachievable in general. The project aims for comparable results under fixed seeds, fixed batch size and fixed padding.

## Open decisions

Each of these would change a claim or a default. Answering one is a good first experiment.

1. **How many concurrent sessions does one Colab account allow?** Undocumented, and it moves with tier, credit balance and demand. Until it is measured, a `devices` count in the provider entry is where the answer goes, so a user who has measured their own account is not overruled by a number letify guessed.
2. **What is the real host synchronization count per step, `k`, for the target workload?** Measurable with `torch.cuda.set_sync_debug_mode("warn")`. This sets whether call forwarding is worth implementing at all.
3. **Does `colab ssh --proxy-mode` support port forwarding with `ssh -L`?** The CLI documents the ProxyCommand bridge but not forwarding. This decides whether a data channel separate from `colab exec` is available.
4. **Is NVFP4 reachable in a stock Colab runtime?** Needs the CUDA version, the compute capability and whether the quantization stack installs.
5. **Is the Elice SSH port stable across a restart?** If it is not, the configuration needs a command that resolves the current port.
6. **What does Elice spot pricing cost?** The API exposes a pricing id, which suggests preemptible instances are available. This is a direct cost lever.
7. **Which driver entry points does a real NVFP4 fine-tune actually reach?** `letify-driver` implements what a PyTorch process needs to start up and run one kernel, and names anything else it is asked for. One real run produces the list of what to implement next, which is the only honest way to size the remaining work.
