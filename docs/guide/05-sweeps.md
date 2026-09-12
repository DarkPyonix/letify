# 5️⃣ Sweeps and concurrency

> Running many configurations, in parallel, without a map call.

[← Environments and data](04-environments-and-data.md) · [Guides](README.md) · [Next: Cost control →](06-cost.md)

---

## Declare the space, not the loop

A sweep is a value. Passing it where a scalar is expected declares that the argument varies.

```python
space = letify.grid(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32])   # 6 points
pairs = letify.zip(lr=[1e-4, 3e-4], bs=[16, 32])          # 2 points
```

| Builder | Does | Use for |
|---|---|---|
| `grid` | Cartesian product of the axes | hyperparameter search |
| `zip` | pairs the axes position by position | a prepared list of configurations |
| `a | b` | union, dropping duplicate points | combining two searches |
| `space.with_fixed(**kw)` | adds arguments constant across every point | epochs, seed, output path |

A scalar axis stays fixed, so `grid(lr=[1e-4, 3e-4], bs=32)` is two points.

`zip` rejects axes of unequal length rather than silently truncating:

```python
letify.zip(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32])
# ValueError: zip axes must have equal length, got lr=3, bs=2
```

## Consuming a space

The two orderings you might want are already in the language.

```python
@let.function(gpu=colab.G4, concurrency=3)
async def train(lr, bs):
    ...
    return {"lr": lr, "bs": bs, "loss": loss}

with let.run():
    results = await train(space)          # list, in input order

    async for result in train(space):     # as each finishes
        print(result)
```

`await` collects in input order, which is what you want for a results table. `async for` yields by completion, which is what you want when a sweep takes an hour and you would rather see the early results than wait.

A sync declaration returns a list in input order:

```python
@let.function(gpu=colab.G4, concurrency=3)
def train(lr, bs): ...

with let.run():
    rows = train(space)     # list
```

## Concurrency belongs to the declaration

```python
@let.function(gpu=colab.G4, concurrency=3)
```

This is how many runtimes the declaration may occupy at once, not a property of any single call. It is part of the declaration because it describes the infrastructure that declaration is allowed to use.

Six points across three runtimes finish in roughly a third of the wall clock time. Credits spent are the same either way, since three GPUs for 30 minutes costs what one GPU costs for 90.

> ⏱️ **Wall clock time is worth paying for in research.** Seeing results three times sooner changes how fast you can design the next experiment, even though the credit total is unchanged.

## How many runtimes can you actually get

Two limits apply and they are different.

`concurrency` on the declaration is what that declaration may use. `max_runtimes` on the launcher is the ceiling for the whole process.

```python
let = letify.Launcher(max_runtimes=4)
```

```toml
[defaults]
max_runtimes = 4
```

The provider has its own limit, which for Colab is undocumented and moves with your tier, credit balance and current demand. The default of 3 is a guess. Measure it:

```python
let = letify.Launcher(max_runtimes=6)

@let.function(gpu=colab.L4, concurrency=6)
async def probe(n):
    return n

with let.run():
    await probe(letify.grid(n=list(range(6))))
    print(len(let.pool.live))        # how many actually came up
```

Use `L4` for this. It is the cheapest accelerator, so finding the limit costs almost nothing.

More accounts is the other way to raise the ceiling, since the limit is per account. See [Providers and accounts](02-providers.md).

## Cheaper accelerators, more of them

For search, how many run at once often matters more than how fast each one is.

| Accelerator | Per hour | Three at once |
|---|---|---|
| A100 80 GB | ~841 KRW | 2,523 KRW |
| L4 | ~191 KRW | 573 KRW |

Three L4s cost less per hour than one A100 and finish three configurations in the time one A100 finishes one. If the work fits in 22 GB, which hyperparameter search often does with a smaller batch, this is the better trade.

Confirm the final candidate on the bigger card.

> ⚠️ This does not apply when you need a specific capability. NVFP4 needs Blackwell tensor cores, which on Colab means `G4` and nothing else. There is no cheaper substitute.

## Practical patterns

**Run a space, collect a table.**

```python
with let.run():
    rows = await train(letify.grid(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32]))

best = min(rows, key=lambda r: r["loss"])
print(best)
```

**Log as results arrive.**

```python
with let.run():
    async for row in train(space):
        wandb.log(row)
```

**Train and evaluate at the same time, on different providers.**

```python
@let.function(gpu=colab.G4, concurrency=1)
async def train(lr): ...

@let.function(gpu=lab.A100, concurrency=1)
async def evaluate(ckpt): ...

with let.run():
    await asyncio.gather(train(lr=1e-4), evaluate(ckpt="run-1"))
```

Because each declaration names its own instance, one runs on Colab and the other on your lab server with no extra machinery.

**Evaluation batching matters more than a second GPU.** For a speech model at 12.5 frames per second, a 10 second utterance is 125 frames. At batch 1 with a 20 ms frame, 100 utterances take 250 seconds. At batch 64, the same evaluation is 5 to 10 seconds. If you evaluate every 500 training steps, that is the difference between 100% overhead and 3%.

## Only one space per call

```python
train(letify.grid(lr=[1e-4]), letify.grid(bs=[16]))
# TypeError: only one search space may be passed per call.
#            Combine them with let.grid(...) or the | operator instead.
```

Combine axes into one space instead. That keeps the number of points visible in one place rather than being the product of two arguments.

---

[← Environments and data](04-environments-and-data.md) · [Guides](README.md) · [Next: Cost control →](06-cost.md)
