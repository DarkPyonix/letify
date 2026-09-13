# CLAUDE.md

Guidance for AI agents working in this repository. Read this before writing code, documentation or commits.

## Hard rules

These are not preferences. Breaking one means the change gets reverted.

1. **Never add a co-author trailer.** Do not add `Co-Authored-By` lines for Claude or any other agent. Commits are authored by the repository owner alone.
2. **Never use an em-dash.** Not in code, comments, docstrings, documentation, commit messages or pull request bodies. Use a hyphen, a comma, a colon or a second sentence instead. This applies to the en-dash used as punctuation as well.
3. **Code comments and docstrings are English only.** No exceptions, including in files whose documentation is Korean. Identifiers, log messages and error strings are English too.
4. **Commit subjects use a capitalized type prefix followed by a colon.** For example `Feat: Add the Colab provider`, `Fix: Release the session lock when exec raises`, `Docs: Describe the blob store layout`. Use the imperative mood and name the thing that changed. Allowed prefixes: `Feat`, `Fix`, `Docs`, `Refactor`, `Test`, `Perf`, `Chore`, `Build`.
5. **Never commit secrets.** Access tokens, SSH private keys and account credentials belong in the environment or in `~/.letify/accounts`, never in tracked files. The project's `.letify/` holds project defaults only.

## How this project is built

Spec driven and test driven, in that order. The sequence for any change that is not a
typo fix is: settle the spec, write the test that would prove it, then write the code.

### Spec first

`docs/SPEC.md` is the single source of truth for what the system does. Code follows it,
not the other way round.

1. **Change the spec before the code.** If a change alters behaviour, edit the section of
   `docs/SPEC.md` that describes it first. If no section describes it, add one. A change
   with no spec entry has nowhere to be checked against.
2. **A spec change is a hypothesis, not a settled requirement.** In research, whether a
   design is right is decided by the experiment. So a design changing experiment edits the
   spec on its own branch, and the verdict on the pull request, adopted or rejected, is
   what decides whether that spec change reaches `develop`.
3. **The spec records decisions only.** What the system does, the formula it uses, the
   value chosen. Derivations, measurement tables and comparisons go in the pull request
   body, and the spec links to it.
4. **`docs/INTENT.md` sits above the spec.** It holds the claims the design is meant to
   prove, with stable ids (`N1`, `N2`, ...). An experiment lists the claim it tests. When
   evidence contradicts a claim, say so and propose the edit rather than leaving it
   standing.

### Test driven

A test is written to state the expected behaviour before the code that satisfies it
exists. The cycle is red, then green, then tidy.

1. **Write the failing test first.** It names the behaviour, so the name reads as a
   sentence: `test_a_handle_from_another_runtime_is_refused`. Run it and watch it fail for
   the reason you expect, because a test that passes before the code is written is testing
   nothing.
2. **Write the smallest code that makes it pass.** Then clean up with the test as the net.
3. **Every test traces to a spec section.** If you cannot say which section a test is
   pinning, either the spec is missing an entry or the test is asserting an accident of the
   implementation. Both are worth fixing.
4. **A bug gets a failing test before a fix.** That is how it is shown to be real, and how
   it stays fixed.
5. **Exercise the real code path.** The `Local` provider starts the same worker behind the
   same framed protocol a remote runtime uses, so anything testable through it should be.
   A fake belongs only where the real thing needs a live account, a network or a GPU, and
   it goes in `tests/conftest.py` rather than being written twice.
6. **Coverage is a smoke detector, not a goal.** A line with no test is a question to ask.
   Chasing a percentage with tests that assert nothing makes the suite worse, and a line
   that genuinely needs a live service is marked `# pragma: no cover` with the reason.

### What this rules out

- Writing the implementation first and the tests afterwards to match whatever it happened
  to do. That blesses accidents as behaviour.
- Changing behaviour without touching the spec. The next reader then has two sources of
  truth that disagree.
- Deleting or loosening a test to make a change pass. If the test was wrong, say why in
  the commit; if the behaviour changed on purpose, the spec changes with it.

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

The Python package is pure Python and links no Python extension. The package lives in `letify/` at the repository root. letify-core, the Rust workspace in `letify-core/`, is built in CI and shipped as plain binaries in `letify/remoting/lib/` inside platform wheels.

```
letify/
  __init__.py       Public surface. Everything a user needs is re-exported here
  errors.py         Exception hierarchy
  launcher.py       Launcher, the public entry point
  cli.py            The letify command line
  stubs.py          Provider type stubs for editor completion
  config/           .letify/config.toml loading, schema, login and credential files
  declare/          Env, Instance and the @let.function callable
  protocol/         Codec, framing, reference types, the remote worker and driver
  runtime/          Channel, session, pool, lease, bootstrap and telemetry
  store/            Content addressed blob store, volumes and backends
  providers/        Provider base class and one module per provider
  remoting/         CUDA API forwarding, used only where a low-latency path exists
  _vendor/          Vendored third party code
```

Conventions:

- Every module has a docstring saying what it owns and what it does not.
- Type hints on every public function. `from __future__ import annotations` at the top of each module.
- No provider-specific import at package import time. A missing optional dependency disables that provider and nothing else.
- Errors distinguish infrastructure failure from user code failure. Infrastructure failure may be retried, user code failure never is.
- Never fall back to local execution silently. Raise instead.

## Dependencies

There is one install and no extras:

```
uv add letify
```

letify installs cloudpickle and blake3 only, because it lives inside researchers' repositories. Provider tools run out of process through uv, never in the user's `.venv`: Colab through `uv tool run --from google-colab-cli colab`, Modal through a separate uv environment. Elice and the GCS blob store use the standard library HTTP client. Do not add a dependency to `pyproject.toml`.
