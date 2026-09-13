# 6️⃣ Cost control

> How letify keeps a session from outliving you, and what you still have to do.

[← Sweeps](05-sweeps.md) · [Guides](README.md) · [Next: Troubleshooting →](07-troubleshooting.md)

---

## The three layers

Nothing has to be torn down by hand. Four things end a session, and you only declare one of
them.

**1. The call.** A session ends when the call that needed it finishes. A search space counts
as one call, so a sweep of six points starts one set of sessions and ends them once.

```python
@let.function(device=colab.G4, host="remote")
def train(lr): ...

train(lr=1e-4)        # the session starts here and ends here
```

**2. A `keep_alive` block.** `with let.keep_alive():` keeps sessions past their calls until
the block ends, for a run of separate calls that would otherwise pay session start each time.
On Colab that start is provider boot plus environment installation, which is minutes.

```python
@let.function(device=colab.G4, host="remote")
def train(lr): ...

with let.keep_alive():
    train(lr=1e-4)    # starts a session
    train(lr=3e-4)    # reuses it
# the block ends here, and so does the idle session
```

Two declarations that agree on device and environment share a session either way, because
the pool keys by those rather than by which function asked.

Nothing ends a session on a timer. The block has a visible end, and that end is what closes
the sessions it kept. Blocks nest, and only the outermost exit ends anything.

A kept idle session still holds its card. A call in the block that needs that card with a
different environment raises `letify.InsufficientDevices` instead of waiting. See
[Troubleshooting](07-troubleshooting.md).

**3. The lease.** The session holds a deadline that your process renews every 30 seconds,
and the worker exits on its own if the deadline passes. The grace period is 300 seconds, so a
flaky connection does not kill a training run. This is for the one case nothing else covers:
`kill -9`, the out of memory killer and a power cut all run no code, so a session that ends
only when asked would never be asked.

**What that does and does not buy.** The worker exiting releases the card. It does not switch
off a machine that somebody else's platform is billing you for, because infrastructure cannot
choose to end itself: something that owns it has to.

| Provider | What you pay for | Your machine is killed |
|---|---|---|
| `local` | nothing | the subprocess dies with its parent |
| `shell`, `tunnel` | nothing; the card is occupied | the worker exits, the card frees |
| `modal` | the sandbox | **covered.** letify sets a deadline when the sandbox is created and Modal enforces it |
| `colab` | the runtime | not covered. Colab's own idle policy ends it, which can be 12 or 24 hours |
| `elice` | the allocation | **not covered.** It bills until something issues the delete |

For the last two, nothing letify does today will end them. Reconciliation on the next run is
the intended answer and is not built yet, so an Elice allocation left behind by a killed
machine bills until somebody deletes it. Check the provider's own console after a crash, and
if you know your machine is about to go down, stop the run yourself first.

## Why there is no detached mode

You might want to start a long run and close your terminal. letify deliberately does not offer that.

A detached run whose remote side gets preempted, which on Colab happens routinely, loses its results with nothing watching. So the local process stays the owner, and durability comes from checkpoints in a volume instead of from a session that survives you.

The practical consequence: **a long run needs your machine awake.** A 90 minute experiment is fine. An overnight run means leaving the machine on, and it means checkpointing so a restart is cheap.

## Making a restart cheap

Write checkpoints to a volume on a time interval rather than a step count. Then whatever kills the session, you lose at most one interval.

```python
cache = colab.volume("hf-cache")

@let.function(device=colab.G4, host="remote", env=env, volumes=[cache])
def train(lr, run="run-1"):
    # resume from the newest checkpoint under this name, if there is one
    # save every N minutes, not every N steps
    ...
```

Time based saving bounds your loss in wall clock terms, which is what you actually care about when a session can vanish at any moment.

## Seeing what is running

```python
let.providers.active        # {'colab_a': ['letify-g4-a1b2c3']}
let.status()                # counts against the ceiling, then a row per session
let.pool.live               # the Runtime objects
```

```bash
letify status
```

`let.providers.active` is the quickest answer to the question that matters, which is what is costing money right now.

## Knowing what it costs

Rough hourly rates at 1,343 KRW to the dollar, for planning only. Check current prices before relying on these.

| Accelerator | Colab credits | Modal | Elice |
|---|---|---|---|
| RTX PRO 6000, `G4` | ~975 KRW | ~4,070 KRW | not offered |
| H100 80 GB | ~1,340 to 1,680 KRW | ~5,300 KRW | 5,500 KRW |
| A100 80 GB | ~841 KRW | ~3,360 KRW | 2,500 KRW |
| L4 | ~191 KRW | ~1,070 KRW | not offered |
| T4 | ~133 KRW | ~790 KRW | not offered |

Colab figures assume the 600 credit pack at 49.99 USD, which is about 0.083 USD per credit. Credit consumption rates move with demand, so check the resource panel in a live session.

### Multipliers that are easy to miss

**Modal.** Pinning a region costs 1.15 to 1.75 times the base rate, and non-preemptible execution costs 3 times. A non-preemptible RTX PRO 6000 is about 12,000 KRW per hour, not 4,070.

**Colab.** A session consumes credits whenever a runtime is attached, whether or not code is running. Editing code with a session open is billed.

**Elice.** Block storage keeps billing while a machine is stopped, and it disappears when the machine is deleted. A forgotten machine costs money with no allocation running.

**Cloud storage.** A bucket charges for storage, for requests and, if you get the region wrong, for cross-region egress. 100 GB is a few thousand KRW per month; a badly configured region can cost more than that in transfer. Use a multi-region bucket and set a billing alert.

## Habits that save real money

**Set the idle timeout low.** If your pattern is bursts of work with thinking in between, 300 seconds beats 600.

**Use a volume.** A 20 GB cache pulled from a nearby bucket in 60 seconds instead of 27 minutes from a remote origin, every session, is the largest single saving available.

**Size the accelerator to the work.** L4 is a fifth the price of an A100 80 GB. If the work fits in 22 GB, and for hyperparameter search with a smaller batch it often does, that is a direct multiple off the bill.

**Batch your evaluation.** Evaluating 100 utterances at batch 1 takes 250 seconds; at batch 64 it takes 5 to 10. If you evaluate every 500 steps, that is the difference between doubling your training cost and adding 3%.

**Apply for credits.** Modal offers academic credits up to 10,000 USD. Half a day writing an application has a better expected value than weeks of optimizing around a budget, and if it is approved the entire cost question disappears. Ask your advisor to sponsor the request.

**Keep the queued lab GPU.** A queued GPU is still free. Put the long runs in the queue and use paid credits only for what you need to see now.

## Checking before you spend

```bash
letify probe gpu.lab.example.edu     # round trip, and whether forwarding is viable
letify check lab_a100                # does the machine answer at all?
letify login shell lab_a100          # declare the machine before spending on it
letify usage                         # what is left on each account
letify utilization                   # how busy each declared GPU is right now
letify devices                          # what each provider offers
```

For a new provider, run one trivial function first. A `check` declaration that returns the torch version and device name costs seconds and confirms the whole path before you commit a long run to it.

```python
@let.function(device=colab.G4, host="remote")
def check():
    import torch

    return torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
```

---

[← Sweeps](05-sweeps.md) · [Guides](README.md) · [Next: Troubleshooting →](07-troubleshooting.md)
