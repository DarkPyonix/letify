/**
 * HTML for the hover cards and the letify view.
 *
 * Owns the card markup for quota, GPU and runtimes. Every value from the CLI is escaped here
 * before it reaches trusted Markdown or the webview. It does not own the VS Code API.
 */

import { DayBucket, gpuMean } from "./history";
import {
  Device,
  Status,
  UsageRow,
  UtilizationRow,
  escapeHtml as e,
  formatAmount,
  formatDuration,
  holderLabel,
  isReserved,
  paceProjection,
  periodElapsed,
  shareLeft,
} from "./model";

const pct = (share: number) => `${Math.round(share * 100)}%`;

function gauge(share: number, marker: number | null): string {
  const mark = marker === null ? "" : `<span class="mark" style="left:${(marker * 100).toFixed(1)}%"></span>`;
  const level = share >= 0.95 ? "error" : share >= 0.8 ? "warning" : "ok";
  return `<div class="gauge"><span class="fill ${level}" style="width:${(share * 100).toFixed(1)}%"></span>${mark}</div>`;
}

function stamp(seconds: number): string {
  return new Date(seconds * 1000).toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

export function quotaCard(row: UsageRow, now: number): string {
  const title = `<div class="title"><b>${e(row.alias)}</b> <span class="muted">${e(row.kind)}</span></div>`;
  if (row.unavailable) return `<div class="card">${title}<div class="muted">unavailable: ${e(row.unavailable)}</div></div>`;
  if (row.unmetered) return `<div class="card">${title}<div class="muted">No quota, unmetered</div></div>`;
  if (row.remaining === null) {
    return `<div class="card">${title}<div class="muted">Not reported: ${e(row.note ?? row.source)}</div></div>`;
  }
  const lines: string[] = [title];
  const left = shareLeft(row);
  const elapsed = periodElapsed(row, now);
  if (left !== null) {
    lines.push(`<div class="row"><span>${pct(1 - left)} used</span><span>${e(formatAmount(row.remaining, row.unit))} left of ${e(formatAmount(row.limit ?? 0, row.unit))}</span></div>`);
    lines.push(gauge(1 - left, elapsed));
  } else {
    lines.push(`<div class="row"><span>${e(formatAmount(row.remaining, row.unit))} left</span></div>`);
  }
  if (row.resets_at !== null) {
    lines.push(`<div class="muted">Resets in ${formatDuration(row.resets_at - now)}, ${stamp(row.resets_at)}</div>`);
  }
  if (elapsed !== null) lines.push(`<div class="muted">${pct(elapsed)} of the period gone</div>`);
  const pace = paceProjection(row, now);
  if (pace !== null) lines.push(`<div class="muted">At this pace, ~${pct(pace)} by reset</div>`);
  if (row.rate_per_hour !== null) {
    let burn = `Burning ${e(formatAmount(row.rate_per_hour, row.unit))}/hour`;
    if (row.rate_per_hour > 0) burn += `, about ${formatDuration((row.remaining / row.rate_per_hour) * 3600)} left at this rate`;
    lines.push(`<div class="muted">${burn}</div>`);
  }
  if (row.note) lines.push(`<div class="muted">${e(row.note)}</div>`);
  for (const r of row.resources) {
    const amount = r.remaining === null
      ? "not reported"
      : r.limit
        ? `${formatAmount(r.remaining, r.unit)} left of ${formatAmount(r.limit, r.unit)}`
        : `${formatAmount(r.remaining, r.unit)} left`;
    lines.push(`<div class="row resource"><span>${e(r.name)}</span><span>${e(amount)}</span></div>`);
    if (r.remaining !== null && r.limit) lines.push(gauge(Math.max(0, Math.min(1, 1 - r.remaining / r.limit)), null));
    if (r.resets_at !== null) lines.push(`<div class="muted">Resets in ${formatDuration(r.resets_at - now)}, ${stamp(r.resets_at)}</div>`);
  }
  return `<div class="card">${lines.join("")}</div>`;
}

function deviceBlock(alias: string, device: Device, status: Status | null): string {
  const util = device.utilization_percent;
  const mem = device.memory_percent;
  const holder = holderLabel(device, isReserved(status, alias, device.index)) || null;
  const facts = [
    device.temperature_c !== null ? `${Math.round(device.temperature_c)} C` : null,
    device.power_w !== null ? `${Math.round(device.power_w)} W` : null,
    holder,
  ].filter(Boolean).join(" · ");
  const memText = device.memory_total_gb
    ? `${(device.memory_used_gb ?? 0).toFixed(1)}/${device.memory_total_gb.toFixed(1)} GiB`
    : "memory unknown";
  return `<div class="device">
<div class="row"><span>gpu${device.index} ${e(device.name)}</span><span class="muted">${e(facts)}</span></div>
<div class="row small"><span>util ${util === null ? "unknown" : `${Math.round(util)}%`}</span></div>${gauge((util ?? 0) / 100, null)}
<div class="row small"><span>mem ${memText}</span></div>${gauge((mem ?? 0) / 100, null)}
</div>`;
}

export function gpuCard(row: UtilizationRow, status: Status | null): string {
  const title = `<div class="title"><b>${e(row.alias)}</b> <span class="muted">${e(row.accelerator ?? row.kind)}</span></div>`;
  if (row.unavailable) return `<div class="card">${title}<div class="muted">unavailable: ${e(row.unavailable)}</div></div>`;
  if (row.devices.length === 0) {
    return `<div class="card">${title}<div class="muted">${e(row.reason ?? "nothing reported")}</div></div>`;
  }
  return `<div class="card">${title}${row.devices.map((d) => deviceBlock(row.alias, d, status)).join("")}</div>`;
}

export function runtimesCard(status: Status | null): string {
  if (!status) return `<div class="card muted">No status read yet</div>`;
  if (status.runtimes.length === 0) {
    return `<div class="card"><div class="title"><b>Runtimes</b></div><div class="muted">No live runtime in this project's process</div></div>`;
  }
  const rows = status.runtimes
    .map((r) => `<div class="row"><span>${e(r.name)} <span class="muted">${e(r.provider)}.${e(r.accelerator)}</span></span><span>${r.busy ? "busy" : `idle ${formatDuration(r.idle_seconds)}`}</span></div>`)
    .join("");
  return `<div class="card"><div class="title"><b>Runtimes</b> <span class="muted">${status.live} live, ${status.busy} busy</span></div>${rows}</div>`;
}

export function historyChart(history: DayBucket[], kind: "gpu" | "quota"): string {
  if (history.length === 0) return "";
  const values = history.map((b) =>
    kind === "gpu" ? gpuMean(b) ?? 0 : Object.values(b.spent).reduce((a, v) => a + v, 0),
  );
  const top = Math.max(...values, kind === "gpu" ? 100 : 0) || 1;
  const bars = values
    .map((v, i) => `<span class="bar" title="${history[i].day}: ${v.toFixed(1)}" style="height:${Math.max(2, (v / top) * 100).toFixed(0)}%"></span>`)
    .join("");
  const label = kind === "gpu" ? "Daily mean GPU utilization" : "Daily quota spent, all units";
  return `<div class="card"><div class="title muted">${label}, last ${history.length} days</div><div class="chart">${bars}</div></div>`;
}

export const STYLE = `
body{font-family:var(--vscode-font-family);font-size:var(--vscode-font-size);color:var(--vscode-foreground);padding:6px 10px}
.tabs{display:flex;gap:4px;margin-bottom:8px}
.tabs button{background:transparent;color:var(--vscode-foreground);border:0;border-bottom:2px solid transparent;padding:4px 8px;cursor:pointer}
.tabs button.active{border-bottom-color:var(--vscode-focusBorder);font-weight:600}
.card{border:1px solid var(--vscode-widget-border,var(--vscode-panel-border));border-radius:4px;padding:8px 10px;margin-bottom:8px;background:var(--vscode-editorWidget-background)}
.title{margin-bottom:4px}
.muted{color:var(--vscode-descriptionForeground)}
.row{display:flex;justify-content:space-between;gap:8px}
.small{font-size:0.9em}
.gauge{position:relative;height:6px;border-radius:3px;background:var(--vscode-progressBar-background,#888);opacity:0.9;margin:3px 0 5px;overflow:hidden;background:var(--vscode-input-background)}
.fill{position:absolute;left:0;top:0;bottom:0;background:var(--vscode-progressBar-background)}
.fill.warning{background:var(--vscode-editorWarning-foreground)}
.fill.error{background:var(--vscode-editorError-foreground)}
.mark{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--vscode-foreground)}
.device{margin-top:6px}
.resource{margin-top:6px}
.chart{display:flex;align-items:flex-end;gap:2px;height:48px}
.bar{flex:1;background:var(--vscode-charts-blue,var(--vscode-progressBar-background));min-width:3px}
.footer{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.error{color:var(--vscode-errorForeground)}
a{color:var(--vscode-textLink-foreground);cursor:pointer}
`;
