# 4️⃣ Environments and data

> Why session start is slow, and the one change that fixes it.

[← Execution modes](03-execution-modes.md) · [Guides](README.md) · [Next: Concurrency →](05-concurrency.md)

---

## The problem in one table

Every session on an ephemeral provider rebuilds the same things. All of it is billed as GPU time.

| Pulling a 20 GB model cache | Time |
|---|---|
| 🐢 From a lab server over 100 Mbit/s | ~27 min |
| 🚶 From the Hugging Face hub | 3 to 5 min |
| 🚀 From a bucket next to the runtime | **40 to 60 s** |

On a 90 minute session, the first row is 30% of your credits spent waiting. The third is 1%.

## Environments

`Env` declares the remote environment. It is keyed by the hash of your lock file, so two declarations that agree share a pooled runtime and a cached archive.

```python
env = letify.Env()                        # uv.lock
env = letify.Env.from_lock("train.lock")  # a different lock file
env = env.pip_install("flash-attn")       # packages the lock file does not carry
env = env.run("apt-get install -y git")   # commands after installation
env = env.vars(HF_HOME="/opt/cache")      # environment variables in the runtime
```

### Why uv.lock and not pip freeze

A uv lock file resolves for **every** platform uv supports, with markers. One lock file therefore drives a Linux runtime from a Windows or macOS client.

A `pip freeze` list does not. It carries the pins of the machine that produced it, including platform specific packages, so a Windows list cannot install on Linux.

### Which modules travel with your call

Packages named in the lock file are installed in the runtime and referenced by name. Your own code is not in the lock file, so it has to travel with the call: the remote side either does not have it or has an older copy.

letify infers this from the lock file. Override the inference only when you need to:

```python
env = env.ship("mypkg")     # send this module by value
```

This is why editing code and rerunning is cheap. Code is a few hundred kilobytes; the environment and the data are the expensive parts, and they are cached.

## Volumes

A volume is a content addressed blob store on whatever storage your provider has.

```python
cache = colab.volume("hf-cache")

@let.function(device=colab.G4, host="remote", env=env, volumes=[cache])
def train(lr): ...
```

That one line is what makes an ephemeral provider behave like a persistent one.

### Why content addressed and not a file sync

| | 🐌 Two-way sync | ⚡ Content addressed |
|---|---|---|
| Two sessions writing | last writer wins, the other's work is lost | cannot collide, different contents get different names |
| Is it already there? | compare sizes and timestamps | holding the hash **is** the proof |
| 50,000 small files | 50,000 round trips | one packed archive, one transfer |
| Retry after a failure | re-enumerate everything | skip what is already stored, by name |

Mutable names live in a separate namespace of refs, exactly like Git objects and branch names. A ref is a few dozen bytes, so a race on one is harmless and both blobs survive it.

### What granularity buys you

This is the part that actually makes it fast, and it is a decision rather than a detail.

**Large files stand alone.** A model shard is already big, so per-file latency does not matter.

**Trees of small files get packed.** A virtual environment is tens of thousands of files. Packed into one archive keyed by the hash of the lock file, it is one transfer instead of tens of thousands of round trips.

Be honest about what this does not do. A first transfer still moves the same bytes; what improves is the metadata exchange, which becomes one manifest read instead of one request per file, and every repeat transfer, which is skipped by name.

### Using a volume

```python
cache = colab.volume("hf-cache")

cache.cached_env(env)                 # digest of a prebuilt environment, or None
cache.cache_env(env, "/opt/venv")     # pack one and remember it under this env key

cache.put_checkpoint("run-1", "/tmp/ckpt")   # store, and move the ref
cache.fetch_checkpoint("run-1", "/tmp/ckpt") # newest for that name, or None
cache.latest_checkpoint("run-1")             # just the digest
```

Overriding the backend or the location:

```python
cache = colab.volume("hf-cache", bucket="my-bucket", prefix="letify")
cache = lab.volume("scratch", backend="filesystem", root="/mnt/data/letify")
```

## Backends

Chosen by the provider, so your declaration does not mention one.

| Backend | Used by | Why |
|---|---|---|
| `filesystem` | `Local`, `Shell`, `Elice` | A directory. Your machine can be the origin others pull from. On Elice it is the machine's own disk. |
| `gcs` | `Colab` | A Colab runtime is a Compute Engine virtual machine, so this is an internal transfer. |
| `modal` | `Modal` | A Modal volume, mounted beside the container. |

> 🌏 **For Colab, use a multi-region bucket such as `US`.** You cannot choose where a Colab runtime lands, and a multi-region bucket avoids a cross-region charge on every read. Set a billing alert while you are there.

## Why not Google Drive

Drive looks like a disk when mounted, which is the source of the confusion. It is a consumer file service reached one file at a time, and every file access is an HTTP request of tens to hundreds of milliseconds regardless of the file's size.

The arithmetic: 50,000 images at 100 ms each is over an hour of pure input and output per epoch, with the GPU idle at 10 to 20% utilization the whole time. The same data as one archive, copied once and unpacked locally, is under a minute.

Drive is a fine warehouse. It is not a working disk. If you already keep data there, pull one archive at session start rather than reading from it during training.

## Data your call reads

Use `pathlib.Path` for local data, and letify sends it with the call. A `Path` argument, a `Path` default, or a `Path` in a global or closure the function reads is detected while the call is pickled. The body receives a `Path` on the runtime with the same file names and directory layout.

```python
DATA = Path("data/corpus")

@let.function(device=lab.A100, host=letify.remote)
def train(lr):
    files = sorted(DATA.rglob("*.bin"))   # a directory on the runtime
```

What is detected: a file, a directory or a path that does not exist yet, under the project root, the nearest directory with a `pyproject.toml`, or under a directory listed in `[tool.letify] data_roots`. A path outside those roots and the project root itself stay plain paths. `.git`, `.venv` and `__pycache__` inside a directory are skipped.

Each file is hashed once and remembered by size, modification time and inode in `~/.cache/letify/digests.json`, so an unchanged dataset is not read again. Where the bytes come from depends on the account:

| Account | First session | Later sessions |
|---|---|---|
| persistent (`persistent = true`, Modal, local) | missing files over the link | nothing uploaded; the runtime's disk holds them |
| ephemeral with `bucket = "<name>"` | missing files to the bucket, then the runtime downloads them | nothing uploaded; the runtime downloads from the bucket |
| ephemeral without `bucket` | every file over the link | every file over the link again |

A changed file is sent again on its own; the rest is not. Each call that carries data prints one line, for example `letify: data 8 files 1024.0 MiB detected, 7 files 896.0 MiB already on the runtime, uploaded 1 files 128.0 MiB in 1.4 s (91.4 MiB/s)`.

### Files the call writes

A directory and a path that does not exist yet are output locations. When the call returns, every file the body created or changed there is copied to the same relative path on the local disk.

```python
RUN = Path("runs/exp1")

@let.function(device=lab.A100, host=letify.remote)
def train(epochs):
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUN / "model.pt")   # local runs/exp1/model.pt after the call
```

- A file whose contents did not change is not sent. A second call that only changes the logs receives only the logs.
- A call that raised writes nothing back.
- A file the body deleted on the runtime is not deleted locally.
- Calls writing the same local path at the same time are applied one after the other, and for the same file the call that returned last wins. A file is never half from one call and half from another.
- Files inside a directory are placed on the runtime as writable copies, so the body may overwrite an existing file.

Each such call prints one more line, for example `letify: data wrote back 3 files 512.1 MiB in 2.4 s (213.4 MiB/s), 0 files 0.0 MiB already on the client`.

## Data that is too big to move

Three options, in the order worth trying.

**Pack and transfer once.** One archive, parallel download, unpacked to local disk. Simple and predictable, and it is the right answer whenever the data fits and you run more than one epoch.

**Stream in shards.** With data as tar shards read by URL, training starts immediately and the download overlaps with computation. Set a prefetch depth of at least 2 so the round trip is hidden, and keep shards at 200 to 500 MB. The cost is that shuffling becomes approximate and every epoch re-transfers.

**Move the origin closer.** Keep the dataset in the same infrastructure as the runtime. This is what a volume on a cloud backend does for you.

Arithmetic for the streaming case: a 224 by 224 JPEG at 100 KB, an A100 processing 500 images per second, needs 50 MB/s. A 1 Gbit/s link has room. Double the batch and you need 100 MB/s, and the link becomes the bottleneck.

## Checkpoints

Write them to a volume, not to the runtime's disk, because the runtime can be preempted.

```python
@let.function(device=colab.G4, host="remote", env=env, volumes=[cache])
def train(lr, resume="run-1"):
    ...
    # inside the shipped function, checkpoint on a time interval so that
    # whenever the session dies, the loss is bounded by that interval
```

Save on a time interval rather than a step count. Then whatever happens, you lose at most one interval of work.

> 💡 There is no detached mode in letify, so a long run needs your local process alive. A volume checkpoint is what makes a restart cheap. See [Cost control](06-cost.md).

---

[← Execution modes](03-execution-modes.md) · [Guides](README.md) · [Next: Concurrency →](05-concurrency.md)
