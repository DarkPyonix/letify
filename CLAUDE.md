# CLAUDE.md

Guidance for AI agents working in this repository. Read this before writing code, documentation or commits.

## Hard rules

These are not preferences. Breaking one means the change gets reverted.

1. **Never add a co-author trailer.** Do not add `Co-Authored-By` lines for Claude or any other agent. Commits are authored by the repository owner alone.
2. **Never use an em-dash.** Not in code, comments, docstrings, documentation, commit messages or pull request bodies. Use a hyphen, a comma, a colon or a second sentence instead. This applies to the en-dash used as punctuation as well.
3. **Code comments and docstrings are English only.** No exceptions, including in files whose documentation is Korean. Identifiers, log messages and error strings are English too.
4. **Commit subjects use a capitalized type prefix followed by a colon.** For example `Feat: Add the Colab provider`, `Fix: Release the session lock when exec raises`, `Docs: Describe the blob store layout`. Use the imperative mood and name the thing that changed. Allowed prefixes: `Feat`, `Fix`, `Docs`, `Refactor`, `Test`, `Perf`, `Chore`, `Build`.
5. **Never commit secrets.** Access tokens, SSH private keys and account credentials belong in the environment or the OS keyring, never in tracked files. `.letify` at the repository root holds project defaults only; credentials live in `~/.letify`.

## Writing style

Everything written here is read by people first. Documentation, pull request bodies and commit messages follow the same rules.

- Lead with the conclusion, then the evidence.
- One idea per sentence. If a sentence needs a second read, split it.
- Name the concrete thing: the function, the file, the setting, the number. No metaphors and no personification.
- Numbers carry units and a comparison. Not `latency 150` but `round trip 150 ms, measured from Seoul to a Colab runtime in the United States`.
- Define every label the first time it appears.
- Keep it as short as it can be while still complete.

## Documentation layout

| File | Holds |
|---|---|
| `README.md` | What letify is and how to use it, for a first-time reader |
| `docs/locales/README_ko.md` | Korean translation of `README.md`, kept in sync |
| `PROJECT.md` | The feature set and the public API surface |
| `docs/INTENT.md` | Problem, goals, claims (`N1`, `N2`, ...), constraints, non-goals, open decisions |
| `docs/SPEC.md` | The current design. Decisions only, one feature per section, each section opening with a one-line summary |
| `docs/COMPONENT.md` | Class relationships, responsibilities and the vocabulary of the project |
| `docs/NETWORK.md` | Transport paths, measured latency and throughput, tunnel selection |
| `docs/guide/` | Task-oriented guides, English in `docs/guide/`, Korean in `docs/guide/ko/` |

Rules for these documents:

- `docs/SPEC.md` records only the current state. No "previously we used", no "tried and rejected". History lives in Git and in pull request bodies.
- Derivations, measurement tables and comparisons go in the experiment pull request. The spec links to it.
- Section titles name features and stay stable. Before renaming one, give the heading an id comment so its history continues.
- Keep the English and Korean documentation in step. A change to one is incomplete until the other matches.

## ResearchTree conventions

This repository uses [ResearchTree](https://darkpyonix.github.io/researchtree/) for performance work. One branch is one experiment and one pull request is its lab note.

- Root branch: `develop`. Experiment branch prefix: `feat/`. Both are set in `.researchtree`.
- Never open an experiment pull request against `main`.
- One hypothesis per branch. Two ideas are two branches.
- Give each experiment its own `git worktree`. Do not switch branches in a checkout that has a job running.
- The first YAML block at the top of the pull request body is the record. Keep it valid and first.
- Reuse existing metric keys so experiments stay comparable: `efficiency_pct`, `rtt_ms`, `step_s`, `syncs_per_step`, `setup_s`, `transfer_mbps`.
- Do not merge, close, tag or push to `develop` unless the owner asks.

## Code layout

Python only, no compiled extensions. The package lives in `letify/` at the repository root.

```
letify/
  __init__.py       Public surface. Everything a user needs is re-exported here
  errors.py         Exception hierarchy
  launcher.py       Launcher, the public entry point
  config.py         .letify loading and credential resolution
  env.py            Env, the uv.lock based environment declaration
  instance.py       Instance, a GPU plus CPU placement
  sweep.py          Sweep, grid and zip, the declared search space
  function.py       The @let.function decorator and the callable it returns
  runtime.py        Runtime, one live session, and the session pool
  wire.py           Serialization, the call protocol and result decoding
  store/            Content addressed blob store and its backends
  providers/        Provider base class and one module per provider
  remoting/         CUDA API forwarding, used only where a low-latency path exists
```

Conventions:

- Every module has a docstring saying what it owns and what it does not.
- Type hints on every public function. `from __future__ import annotations` at the top of each module.
- No provider-specific import at package import time. A missing optional dependency disables that provider and nothing else.
- Errors distinguish infrastructure failure from user code failure. Infrastructure failure may be retried, user code failure never is.
- Never fall back to local execution silently. Raise instead.

## Dependencies

The base install has no provider dependencies. Providers are extras:

```
uv add letify              # core only
uv add "letify[colab]"     # Google Colab
uv add "letify[modal]"     # Modal
uv add "letify[shell]"     # SSH, tunnel and Elice
uv add "letify[gcs]"       # Google Cloud Storage blob store backend
uv add "letify[s3]"        # S3 compatible blob store backend
uv add "letify[all]"       # everything
```

Keep the core dependency list minimal. A dependency that only one provider needs belongs in that provider's extra.
