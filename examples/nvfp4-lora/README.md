# NVFP4 LoRA on a rented card

> [Korean](README_ko.md)

The scenario this project exists for. You have a laptop with a 6 GiB GPU and a model that
needs 96 GiB. Renting an RTX PRO 6000 Blackwell through Colab compute units costs about
975 KRW per hour against about 4,070 KRW per hour for the same card on Modal, so the cheap
option is worth using, and the only thing making it inconvenient is everything around the
training: starting the session, installing the environment, moving the data, keeping the
checkpoints, and stopping the session before it bills for an hour of nothing.

That is what these four scripts remove. The training code in [recipe.py](recipe.py) never
mentions letify.

## Run it

Start on the local provider, which needs no account and no configuration:

```bash
python 00_smoke.py --provider local
python 01_sweep.py --provider local --steps 20
python 02_resume.py --provider local
python 03_watch.py --provider local
```

Then point the same scripts at a rented card by declaring one account and changing one flag:

```bash
letify login colab colab_pro --account you@example.com
python 00_smoke.py  --provider colab_pro --device G4
python 01_sweep.py  --provider colab_pro --device G4
```

The scripts are numbered in the order they are worth running, not as chapters of a tutorial.

| Script | What it is for | What it proves |
|---|---|---|
| [00_smoke.py](00_smoke.py) | One trivial call before spending anything | The provider answers, the environment installed, this project's code arrived, and the card is the one that was paid for |
| [01_sweep.py](01_sweep.py) | Six configurations across pooled sessions | A declared search space, one warm session per slot, and the best adapter kept in the store |
| [02_resume.py](02_resume.py) | Continue after a preemption | A checkpoint put back inside a fresh session, and immutable blobs behind a moving name |
| [03_watch.py](03_watch.py) | Watch the money while it runs | Remaining account usage, and whether the card is actually busy |

## What each piece is doing

**The declaration is the only place the site appears.** `recipe.train` is a plain function
that imports torch and returns a dictionary. `let.function(device=..., host=...)` decides
where it runs. That is why the same file is what you debug on a laptop and what runs on a
rented card, and why `--provider local` is a real test rather than a mock.

**`ship("recipe")` sends the training code by value.** The machine installs what `uv.lock`
names and has no copy of this project, so a function imported from `recipe.py` has to travel
inside the call. Anything the lock file does install goes by name, because sending torch over
the network on every call would be absurd.

**`with let.keep_alive():` is what makes a sweep affordable.** Session start, environment
install and the first transfer are all billed as GPU time. The block keeps the session between
calls, so the best adapter can still be pulled out after the sweep. Without it each separate
call would pay setup again, and on a short run the setup costs more than the training.

**How wide the sweep runs is the provider's inventory**, not an argument. Declare what the
account has and that is the width:

```toml
[colab_pro.devices]
G4 = { count = 2 }
```

Two cards halves the wall clock of the sweep at the same total GPU cost. A run that needs two
cards at once asks with `device=lab.A100 * 2`, and on a four card box that is two concurrent
sessions rather than four, which is why a number on the declaration could not have said it.

**A volume is a content addressed store, not a mounted disk.** `absorb` pulls a file out of
a session and files it under a name; `resume` puts whatever a name points at back into a
session. Both take the declaration rather than a session, because which session ran the calls
is letify's answer and naming a different one would read from a session holding no files. Blobs are immutable and a name is a few dozen bytes, so two sessions writing at once
cannot lose each other's work: one name wins and both checkpoints remain.

**There is nothing to release.** A call ends its own session, and leaving the `keep_alive`
block ends the sessions it kept. Nothing ends one on a timer. A heartbeat lease covers a process
killed outright, which frees the card; whether it also stops the billing depends on the
provider, and [docs/guide/06-cost.md](../../docs/guide/06-cost.md) says which. Durability is
the checkpoint in the store, not a session that outlives you.

## The two execution modes, and which one to use here

`00_smoke.py` prints the arithmetic that decides it. Forwarding CUDA calls keeps Python and
the data here and sends only driver calls, so its cost is one network round trip at every
point where the host reads a value back. Efficiency against a direct run is
`T / (T + k * RTT)`, where `T` is GPU time per step and `k` is host synchronizations per
step.

On a 150 ms link to a Colab runtime:

| Step time | k = 3 (default loop) | k = 1 (tuned) |
|---|---|---|
| 0.05 s, a decode step | 10% | 25% |
| 0.5 s, an NVFP4 micro step | 53% | 77% |
| 4 s, a large batch | 90% | 96% |

So `host="remote"` is the default for training, and it is what these scripts use.
`host="local"` becomes interesting when the data must not leave this machine, and
`recipe.train` is written for it: it reads the loss back once every ten steps rather than
every step, which is the single change that moves a 0.5 s step from about half to nearly all
of a direct run. Token by token decoding stays bad at any useful latency, because the round
trip sets the ceiling.

## What is synthetic here, and what is not

The data is synthetic, on purpose: a real dataset would make the example depend on a
download, a tokenizer and a licence, and the scenario being shown is the infrastructure.
Swap the two lines in `recipe.train` that build `inputs` and `targets` for a dataloader and
the rest of the function is what you would actually run.

The quantization is a stand-in too. `recipe.train` sends the base weight through a narrow
type and back to the compute dtype, which has the accuracy cost of weight-only quantization
without needing a Blackwell kernel present to demonstrate it. On a card with compute
capability below 10.0, `00_smoke.py` says so rather than pretending otherwise.

Everything else is real: real sessions, a real worker process behind a real framed pipe, real
content addressed storage, and real checkpoint round trips.
