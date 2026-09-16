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

### N2. Forwarding PyTorch operators is only viable at low latency and low synchronization count

Efficiency for PyTorch forwarding follows `T / (T + n * d + k * RTT)`, where `T` is GPU time per step, `n` is operators per step, `d` is local dispatch cost per operator and `k` is host synchronizations per step. It is therefore unusable for token by token decoding at any useful round trip, and it gets worse as the GPU gets faster.

### N3. Pooling runtimes by declaration removes per-call setup cost

Reusing a session across calls with the same instance and environment reduces the amortized setup cost per call to near zero, where starting a session per call pays provider boot time every time.

### N4. A content addressed store with archive-level granularity beats file-level synchronization

For an environment or a model cache, packing a tree into one hash-named blob transfers faster than synchronizing files individually, and immutable naming removes the write conflicts that a two way synchronization has between concurrent sessions.

### N5. A declaration should state the execution mode rather than have it derived

Two modes that differ by a factor of two in throughput and by a factor of fifty for decoding are too far apart to be chosen implicitly. A reader of the declaration should be able to see which one it uses, and nothing should infer it from storage or link properties.

### N6. Three placements are enough to cover every provider without naming a mechanism

A declaration that says where the accelerator is and where the host code runs, plus a `keep_alive` block for sessions to keep, is enough to place any supported workload, and a user never has to name a transport, a channel kind or a storage backend.

### N7. Batching operators makes the round trip count the synchronization count

Forwarding is viable only if a step that dispatches dozens to thousands of operators pays a handful of round trips. Queueing every operator whose result the host does not read, and computing output shapes locally on meta tensors, achieves that, so the network term of the efficiency model is `k * RTT` with `k` counting host synchronizations rather than operators.

### N8. A remote call waits for the data its first step reads, not for the dataset

The wait before a remote call's first step is proportional to the bytes that step reads, not to the size of the dataset, because letify derives the read order from the pickled call itself, sends only the first wave of that order before the call, and keeps sending the rest in the background while the call runs.

## How an experiment reports efficiency

This standard sits here rather than in `docs/SPEC.md` because it governs how a claim is tested, not what the system does. The spec records the design; this records what a report has to show before a claim above may be called supported. Every experiment pull request follows it.

1. **Report efficiency twice.** Once as whole wall time, from the user's command to the result, including connection, session setup, the wait before the first step, all step time, transfer during the call and write-back. Once as step time only. Every table says which of the two it holds, in its caption or its column name, and no table mixes them.
2. **Measure the direct-run baseline both ways too.** Running on the machine directly also moves data: report it once with the `scp` of the dataset and the copy back of the results included, and once without them. A letify whole wall time compared against a bare step time is not a comparison.
3. **Separate the first run from repeated runs.** A first run on a machine pays a transfer in both directions, while a repeat may pay none, so the two go in different rows and are never averaged together. State the run index. The question the repeated case answers is whether letify adds overhead once the data is already there, and that is the number a reader is looking for.
4. **Name the mode in every table.** `host="remote"` and `host="local"` have different cost models, so a table names the one it measured and never holds both.

A report that omits one of the four is incomplete, and its claim stays open.

## Constraints

- **The Python package is pure Python.** No compiled extension in `letify/`. A wheel that has to be built for each platform is a maintenance cost this project will not carry, and hashing and transfer are not CPU bound at the link speeds involved.
- **No native component on the forwarding path.** `host="local"` forwards PyTorch operators through PyTorch's own extension points from Python, so it needs no build step. PyTorch is the project's own dependency and letify never installs it.
- **The local process stays alive for the duration of a run.** letify does not offer detached execution. A detached run whose remote side is evicted loses its results, so the local process stays the owner and the durable artifacts are checkpoints in the store.
- **Nothing is torn down by hand.** No release call and no shutdown call on the public surface. A call ends its own session, an idle one is reaped, and the lease covers a crash.
- **No credential in a tracked file.** Accounts live in `~/.letify/config.toml`, and credentials live in the environment or in `~/.letify/accounts`.
- **Colab accelerators require a paid entitlement.** The remote control features letify uses are permitted on paid plans while the compute unit balance is positive.

## Non-goals

- **Not a scheduler for a shared cluster.** letify targets one researcher's own accounts, not queue management for a group.
- **Not a training framework.** It runs the user's code. It does not own the training loop, the metrics or the checkpoint format.
- **Not a way around a provider's limits.** It does not attempt to bypass session limits, quotas or terms.
- **No bitwise reproducibility.** Kernel selection by measured timing and batch dependent reduction order make that unachievable in general. The project aims for comparable results under fixed seeds, fixed batch size and fixed padding.

## Open decisions

Each of these would change a claim or a default. Answering one is a good first experiment.

1. **How many concurrent sessions does one Colab account allow?** Undocumented, and it moves with tier, credit balance and demand. Until it is measured, a `devices` count in the provider entry is where the answer goes, so a user who has measured their own account is not overruled by a number letify guessed.
2. **What is the real host synchronization count per step, `k`, for the target workload?** Measurable with `torch.cuda.set_sync_debug_mode("warn")`. This sets whether PyTorch forwarding is worth using for that workload.
3. **How often does TCP hole punching succeed on the networks researchers actually use?** It succeeded between a Colab VM and a university network in Korea, where both NATs preserved the port. Home routers, office networks and mobile tethering are unmeasured. This decides how often the pipeline falls to UDP or to the provider's own path.
4. **Is NVFP4 reachable in a stock Colab runtime?** Needs the CUDA version, the compute capability and whether the quantization stack installs.
5. **Is the Elice SSH port stable across a restart?** If it is not, the configuration needs a command that resolves the current port.
6. **What does Elice spot pricing cost?** The API exposes a pricing id, which suggests preemptible instances are available. This is a direct cost lever.
7. **How often can the read order be derived from the pickled call?** N8 rests on it. A `Dataset` or a list of paths in the arguments carries the exact order, and a sampler with a fixed seed carries the shuffled one, but a body that builds its file list at runtime carries nothing. The share of real calls in each group decides whether the manifest order fallback is the common case or the rare one, and how often the blocking backstop fires.
8. **How many operators does a real NVFP4 fine-tune dispatch per step, and what is `d` on a researcher's laptop?** The benchmark model dispatches 31 operators per step. A transformer step dispatches thousands, where `n * d` may dominate `T`, and that decides whether operator forwarding needs a faster local dispatch path for large models.
