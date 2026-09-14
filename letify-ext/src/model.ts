/**
 * Parsing and formatting of letify's JSON output.
 *
 * Owns the records read from `letify usage --json`, `letify utilization --json` and
 * `letify status --json` (spec "Machine-readable output"), the arithmetic the cards show
 * (share left, period elapsed, pace projection) and the status bar text. It does not own
 * running the CLI or anything that touches the VS Code API, so it is testable in plain Node.
 */

export interface UsageRow {
  alias: string;
  kind: string;
  unit: string;
  source: string;
  remaining: number | null;
  limit: number | null;
  used: number | null;
  rate_per_hour: number | null;
  resets_at: number | null;
  unmetered: boolean;
  as_of: number | null;
  note: string | null;
  resources: Resource[];
  unavailable?: string;
}

/** A further allowance on the same account, such as Kaggle's TPU hours. */
export interface Resource {
  name: string;
  unit: string;
  remaining: number | null;
  used: number | null;
  limit: number | null;
  resets_at: number | null;
}

/** Who holds a card, as `letify utilization --json` reports it; null on a session row. */
export type Holder = "letify" | "others" | "mine" | "free" | "unknown" | null;

const HOLDERS = new Set(["letify", "others", "mine", "free", "unknown"]);

export interface Device {
  index: number;
  name: string;
  utilization_percent: number | null;
  memory_used_gb: number | null;
  memory_total_gb: number | null;
  memory_percent: number | null;
  temperature_c: number | null;
  power_w: number | null;
  holder: Holder;
  users: string[];
  reserved: boolean;
}

export interface UtilizationRow {
  alias: string;
  kind: string;
  scope: string;
  accelerator: string | null;
  devices: Device[];
  reason: string | null;
  unavailable?: string;
}

export interface Runtime {
  name: string;
  provider: string;
  accelerator: string;
  devices: unknown[];
  placement: string;
  busy: boolean;
  persistent_channel: boolean;
  idle_seconds: number;
}

export interface Status {
  name: string;
  live: number;
  busy: number;
  devices: Record<string, Record<string, { count: number; reserved: number; indices: number[] }>>;
  runtimes: Runtime[];
}

export type Severity = "ok" | "warning" | "error";

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/** Parse the JSON text a command printed, naming the command when it is not JSON. */
export function parseJson(text: string, command: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    const head = text.trim().split("\n")[0]?.slice(0, 120) ?? "";
    throw new Error(`letify ${command} did not print JSON: ${head || "empty output"}`);
  }
}

export function parseUsage(data: unknown): UsageRow[] {
  if (!Array.isArray(data)) throw new Error("letify usage --json did not print a list");
  return data.map((raw: Record<string, unknown>) => ({
    alias: str(raw.alias) ?? "?",
    kind: str(raw.kind) ?? "",
    unit: str(raw.unit) ?? "",
    source: str(raw.source) ?? "",
    remaining: num(raw.remaining),
    limit: num(raw.limit),
    used: num(raw.used),
    rate_per_hour: num(raw.rate_per_hour),
    resets_at: num(raw.resets_at),
    unmetered: raw.unmetered === true,
    as_of: num(raw.as_of),
    note: str(raw.note),
    resources: (Array.isArray(raw.resources) ? raw.resources : []).map((r: Record<string, unknown>) => ({
      name: str(r.name) ?? "",
      unit: str(r.unit) ?? "",
      remaining: num(r.remaining),
      used: num(r.used),
      limit: num(r.limit),
      resets_at: num(r.resets_at),
    })),
    ...(typeof raw.unavailable === "string" ? { unavailable: raw.unavailable } : {}),
  }));
}

export function parseUtilization(data: unknown): UtilizationRow[] {
  if (!Array.isArray(data)) throw new Error("letify utilization --json did not print a list");
  return data.map((raw: Record<string, unknown>) => ({
    alias: str(raw.alias) ?? "?",
    kind: str(raw.kind) ?? "",
    scope: str(raw.scope) ?? "",
    accelerator: str(raw.accelerator),
    reason: str(raw.reason),
    devices: (Array.isArray(raw.devices) ? raw.devices : []).map((d: Record<string, unknown>) => ({
      index: num(d.index) ?? 0,
      name: str(d.name) ?? "",
      utilization_percent: num(d.utilization_percent),
      memory_used_gb: num(d.memory_used_gb),
      memory_total_gb: num(d.memory_total_gb),
      memory_percent: num(d.memory_percent),
      temperature_c: num(d.temperature_c),
      power_w: num(d.power_w),
      holder: typeof d.holder === "string" && HOLDERS.has(d.holder) ? (d.holder as Holder) : null,
      users: Array.isArray(d.users) ? d.users.filter((u): u is string => typeof u === "string") : [],
      reserved: d.reserved === true,
    })),
    ...(typeof raw.unavailable === "string" ? { unavailable: raw.unavailable } : {}),
  }));
}

export function parseStatus(data: unknown): Status {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("letify status --json did not print an object");
  }
  const raw = data as Record<string, unknown>;
  return {
    name: str(raw.name) ?? "",
    live: num(raw.live) ?? 0,
    busy: num(raw.busy) ?? 0,
    devices: (raw.devices && typeof raw.devices === "object" ? raw.devices : {}) as Status["devices"],
    runtimes: Array.isArray(raw.runtimes) ? (raw.runtimes as Runtime[]) : [],
  };
}

/** Share of the allowance left, 0 to 1, or null when there is no limit to compare against. */
export function shareLeft(row: UsageRow): number | null {
  if (row.unmetered || row.remaining === null || !row.limit) return null;
  return Math.max(0, Math.min(1, row.remaining / row.limit));
}

/** The account with the lowest share left; accounts with no limit rank after every other. */
export function mostConstrained(rows: UsageRow[]): UsageRow | null {
  const metered = rows.filter((r) => !r.unmetered && !r.unavailable && r.remaining !== null);
  if (metered.length === 0) return null;
  const ranked = [...metered].sort((a, b) => {
    const sa = shareLeft(a);
    const sb = shareLeft(b);
    if (sa === null && sb === null) return 0;
    if (sa === null) return 1;
    if (sb === null) return -1;
    return sa - sb;
  });
  return ranked[0];
}

/** Compact time until a moment: `14m`, `5h 3m`, `2d 9h`, `now` once it has passed. */
export function formatDuration(seconds: number): string {
  if (seconds <= 0) return "now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${Math.max(1, minutes)}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return minutes % 60 ? `${hours}h ${minutes % 60}m` : `${hours}h`;
  const days = Math.floor(hours / 24);
  return hours % 24 ? `${days}d ${hours % 24}h` : `${days}d`;
}

/** An amount in its unit, matching the CLI table: `12,345 KRW`, `$29.50`, `12.5 GPU hours`. */
export function formatAmount(value: number, unit: string): string {
  const fixed = (digits: number) =>
    value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  if (unit === "KRW") return `${fixed(0)} KRW`;
  if (unit === "USD") return `$${fixed(2)}`;
  if (unit === "compute units") return `${fixed(2)} compute units`;
  if (unit === "GPU hours") return `${fixed(1)} GPU hours`;
  return `${value} ${unit}`.trim();
}

/** Length of the allowance period in seconds, where the unit implies one. */
export function periodSeconds(row: UsageRow): number | null {
  if (row.resets_at === null) return null;
  if (row.unit === "GPU hours") return 7 * 86400;
  if (row.unit === "USD") {
    const reset = new Date(row.resets_at * 1000);
    const start = Date.UTC(reset.getUTCFullYear(), reset.getUTCMonth() - 1, 1);
    return row.resets_at - start / 1000;
  }
  return null;
}

/** Share of the current period already gone, 0 to 1, or null when the period is unknown. */
export function periodElapsed(row: UsageRow, now: number): number | null {
  const length = periodSeconds(row);
  if (length === null || row.resets_at === null) return null;
  const elapsed = length - (row.resets_at - now);
  return Math.max(0, Math.min(1, elapsed / length));
}

/** Share used by reset at the current pace, capped at 9.99, or null when it cannot be told. */
export function paceProjection(row: UsageRow, now: number): number | null {
  const elapsed = periodElapsed(row, now);
  const left = shareLeft(row);
  if (elapsed === null || left === null || elapsed <= 0) return null;
  return Math.min(9.99, (1 - left) / elapsed);
}

export function severity(left: number | null, warningPercent: number, errorPercent: number): Severity {
  if (left === null) return "ok";
  if (left * 100 < errorPercent) return "error";
  if (left * 100 < warningPercent) return "warning";
  return "ok";
}

export function quotaStatusText(rows: UsageRow[], now: number): string {
  const row = mostConstrained(rows);
  if (!row) return "letify no quota";
  const left = shareLeft(row);
  const amount = left === null ? formatAmount(row.remaining ?? 0, row.unit) : `${Math.round(left * 100)}%`;
  const reset = row.resets_at !== null ? ` (${formatDuration(row.resets_at - now)})` : "";
  return `${row.alias} ${amount} left${reset}`;
}

export interface GpuSummary {
  total: number;
  busy: number;
  meanPercent: number | null;
  reserved: number;
  /** Devices whose holder is `free`. */
  free: number;
  /** Devices that report any holder at all. */
  withHolder: number;
}

export function gpuSummary(rows: UtilizationRow[], status: Status | null, busyPercent: number): GpuSummary {
  const devices = rows.flatMap((r) => r.devices);
  const readings = devices.map((d) => d.utilization_percent).filter((v): v is number => v !== null);
  const reserved = status
    ? Object.values(status.devices).reduce(
        (sum, accs) => sum + Object.values(accs).reduce((s, e) => s + (e.reserved || 0), 0),
        0,
      )
    : 0;
  return {
    total: devices.length,
    busy: readings.filter((v) => v >= busyPercent).length,
    meanPercent: readings.length ? readings.reduce((a, b) => a + b, 0) / readings.length : null,
    reserved,
    free: devices.filter((d) => d.holder === "free").length,
    withHolder: devices.filter((d) => d.holder !== null).length,
  };
}

export function gpuStatusText(summary: GpuSummary): string {
  if (summary.total === 0) {
    return summary.reserved ? `GPU ${summary.reserved} reserved` : "GPU idle";
  }
  const mean = summary.meanPercent === null ? "" : ` ${Math.round(summary.meanPercent)}%`;
  if (summary.withHolder > 0) return `GPU ${summary.free}/${summary.total} free${mean}`;
  return `GPU ${summary.busy}/${summary.total} busy${mean}`;
}

/** The label a person reads for who holds a card (spec "Editor extension"). */
export function holderLabel(device: Device, reservedByStatus: boolean): string {
  switch (device.holder) {
    case "letify":
      return "letify reserved";
    case "others":
      return device.users.length ? `other users: ${device.users.join(", ")}` : "other users";
    case "mine":
      return "yours";
    case "free":
      return "free";
    case "unknown":
      return "unknown";
    default:
      return reservedByStatus ? "letify reserved" : "";
  }
}

/** Whether letify's own status lists this device index as reserved on the alias. */
export function isReserved(status: Status | null, alias: string, index: number): boolean {
  const accs = status?.devices[alias];
  if (!accs) return false;
  const live = status?.runtimes.filter((r) => r.provider === alias) ?? [];
  return live.some((r) => r.devices.includes(index)) ||
    Object.values(accs).some((e) => e.reserved > 0 && e.indices.includes(index) && live.length > 0);
}

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
