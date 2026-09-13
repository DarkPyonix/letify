# 5️⃣ Sweeps and capacity

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
@let.function(device=colab.G4, host="remote")
async def train(lr, bs):
    ...
    return {"lr": lr, "bs": bs, "loss": loss}

results = await train(space)          # list, in input order

async for result in train(space):     # as each finishes
    print(result)
```

`await` collects in input order, which is what you want for a results table. `async for` yields by completion, which is what you want when a sweep takes an hour and you would rather see the early results than wait.

A sync declaration returns a list in input order:

```python
@let.function(device=colab.G4, host="remote")
def train(lr, bs): ...

rows = train(space)     # list
```

## Width comes from the inventory

Nothing on the declaration says how wide a sweep runs. A provider entry declares what the
account has, and that is the answer:

```toml
[colab_pro.devices]
G4 = { count = 2 }            # two concurrent sessions on this account

[lab_a100.devices]
A100 = { indices = "0-3" }    # four cards in a shared box are ours
```

Six points on four cards finish in roughly a quarter of the wall clock time. Credits spent
are the same either way, since four GPUs for 15 minutes costs what one GPU costs for an
hour.

> ⏱️ **Wall clock time is worth paying for in research.** Seeing results four times sooner
> changes how fast you can design the next experiment, even though the credit total is
> unchanged.

There is no second number. A width on the declaration and a ceiling on the launcher were
two statements of one decision, and when they disagreed the smaller won silently: a sweep
was slow and neither number said why.

## A run that takes more than one card

Multiply the instance:

```python
@let.function(device=lab.A100 * 2, host="remote")
def train(lr):
    ...
```

On the four card box above, that is two concurrent sessions rather than four. This is the
plain reason a width knob could not work: a number saying how many sessions may exist says
nothing about how many cards each one needs.

The session sets its own visible devices, so the code inside sees its cards as `cuda:0` and
`cuda:1` and needs to know nothing about which physical indices it was given.

## Sharing a machine with other people

On a department box the usable indices move with whoever else is logged in. Register the
ones that are yours, and letify takes only those that are actually free when a session
starts:

```toml
[lab_a100.devices]
A100 = { indices = "0-3" }
```

A card a colleague is computing on is skipped, not fought over. Compute processes are what
is read, rather than utilization, because a card between steps reads as idle and is not.
letify never kills anything and never touches an index you did not register.

## What actually came up

```python
print(let.status()["devices"])   # inventory against what is reserved
print(let.status()["live"])      # sessions that exist right now
```

```bash
letify status
```

If a sweep is narrower than you expected, the inventory is the only place to look. A point
that cannot reserve its cards waits for one that can to finish, rather than asking the
provider for a machine it would refuse.

More accounts is how to get more cards, since an inventory belongs to one account. See [Providers and accounts](02-providers.md).

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
rows = await train(letify.grid(lr=[1e-4, 3e-4, 1e-3], bs=[16, 32]))

best = min(rows, key=lambda r: r["loss"])
print(best)
```

**Log as results arrive.**

```python
async for row in train(space):
    wandb.log(row)
```

**Train and evaluate at the same time, on different providers.**

```python
@let.function(device=colab.G4, host="remote")
async def train(lr): ...

@let.function(device=lab.A100, host="remote")
async def evaluate(ckpt): ...

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
