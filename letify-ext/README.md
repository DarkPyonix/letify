# letify Status

A VS Code extension that shows letify's remaining quota and GPU activity in the status bar. It reads only the letify CLI's JSON output (`docs/SPEC.md`, "Machine-readable output") and makes no network call of its own.

## What it shows

- **Quota item**: the account with the lowest share left, for example `kaggle 10% left (2d 9h)`. It turns yellow below 20 percent left and red below 5 percent.
- **GPU item**: `GPU 2/4 busy 50%`, the cards at 10 percent utilization or more, out of all reported cards, and their mean utilization.
- **Hover**: a summary card with Quota, GPU and Runtimes links, the time of the last reading, Refresh and Settings.
- **letify view** (bottom panel, opened by clicking either item) with three tabs:
  - Quota: one card per account with a gauge of the share used, a marker for the share of the period gone (Kaggle week, Modal month), reset time, pace projection (`At this pace, ~84% by reset`), burn rate and a daily spending chart. Shell and tunnel accounts show `No quota, unmetered`.
  - GPU: one card per instance with utilization and memory gauges per card, temperature, power, and `letify reserved` when `letify status` lists the index. A daily mean utilization chart.
  - Runtimes: live runtimes from `letify status`.

Daily history comes from the extension's own samples, kept 30 days in VS Code's extension storage.

## Install

```bash
cd letify-ext
npm install
npm run package          # writes ../letify-ext.vsix
code --install-extension ../letify-ext.vsix
```

Open a folder whose uv project has letify installed. The extension runs `uv run letify usage --json`, `uv run letify utilization --json` and `uv run letify status --json` in the first workspace folder.

## Settings

| Setting | Default | Means |
|---|---|---|
| `letify.command` | `uv run letify` | The command that runs the CLI, split on spaces |
| `letify.usageIntervalSeconds` | 60 | Seconds between usage reads |
| `letify.utilizationIntervalSeconds` | 10 | Seconds between utilization and status reads while the view is visible |
| `letify.warningPercent` | 20 | Share left below which the quota item turns yellow |
| `letify.errorPercent` | 5 | Share left below which it turns red |
| `letify.busyPercent` | 10 | Utilization at which a card counts as busy |
| `letify.showQuota`, `letify.showGpu` | true | Show each status bar item |

A CLI failure shows its last stderr lines in the hover and the view. Output is never written to a log.

## Develop

```bash
npm run build   # tsc into out/
npm test        # vitest: parsing, formatting, pace, status bar text, history
```
