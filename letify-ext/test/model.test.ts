import { describe, expect, it } from "vitest";

import { splitCommand } from "../src/cli";
import { recordSample } from "../src/history";
import {
  UsageRow,
  escapeHtml,
  formatAmount,
  formatDuration,
  gpuStatusText,
  gpuSummary,
  mostConstrained,
  paceProjection,
  parseJson,
  parseStatus,
  parseUsage,
  parseUtilization,
  periodElapsed,
  quotaStatusText,
  severity,
} from "../src/model";
import { quotaCard } from "../src/render";

// Shapes copied from spec "Machine-readable output".
const row = (fields: Partial<UsageRow>): UsageRow =>
  parseUsage([{ alias: "a", kind: "k", unit: "GPU hours", source: "s", unmetered: false, ...fields }])[0];

const NOW = Date.UTC(2026, 8, 14, 12) / 1000;

describe("parsing", () => {
  it("reads every usage key and keeps an unavailable alias", () => {
    const rows = parseUsage([
      { alias: "colab", kind: "colab", unit: "compute units", source: "ccu-info", remaining: 80, limit: 100, used: 20, rate_per_hour: 1.8, resets_at: null, unmetered: false, as_of: NOW, note: null },
      { alias: "odd", unavailable: "unknown kind vastai" },
    ]);
    expect(rows[0].rate_per_hour).toBe(1.8);
    expect(rows[1].unavailable).toBe("unknown kind vastai");
    expect(rows[1].remaining).toBeNull();
  });

  it("reads devices with null readings as gaps", () => {
    const rows = parseUtilization([
      { alias: "lab", accelerator: "A100", reason: null, devices: [{ index: 0, name: "A100", utilization_percent: 87, memory_used_gb: 40, memory_total_gb: 80, memory_percent: 50, temperature_c: null, power_w: null }] },
    ]);
    expect(rows[0].devices[0].temperature_c).toBeNull();
    expect(rows[0].devices[0].utilization_percent).toBe(87);
  });

  it("refuses a status that is not an object", () => {
    expect(() => parseStatus([])).toThrow(/did not print an object/);
  });

  it("names the command when output is not JSON", () => {
    expect(() => parseJson("Traceback (most recent call last)", "usage")).toThrow("letify usage did not print JSON: Traceback");
  });

  it("splits the configured command on spaces", () => {
    expect(splitCommand("  uv run  letify ")).toEqual(["uv", "run", "letify"]);
  });
});

describe("formatting", () => {
  it.each([
    [30, "1m"],
    [14 * 60, "14m"],
    [5 * 3600 + 180, "5h 3m"],
    [2 * 86400 + 9 * 3600, "2d 9h"],
    [-5, "now"],
  ])("formats %d seconds as %s", (seconds, text) => {
    expect(formatDuration(seconds)).toBe(text);
  });

  it("formats amounts like the CLI table", () => {
    expect(formatAmount(12345.4, "KRW")).toBe("12,345 KRW");
    expect(formatAmount(25.5, "USD")).toBe("$25.50");
    expect(formatAmount(99.93343, "compute units")).toBe("99.93 compute units");
    expect(formatAmount(12.54, "GPU hours")).toBe("12.5 GPU hours");
  });

  it("escapes markup from CLI values", () => {
    expect(escapeHtml(`<b a="x">&'`)).toBe("&lt;b a=&quot;x&quot;&gt;&amp;&#39;");
    expect(quotaCard(row({ alias: "<x>", unmetered: true }), NOW)).not.toContain("<x>");
  });
});

describe("pace", () => {
  // Kaggle week: resets in 2 days, so 5 of 7 days are gone.
  const kaggle = row({ remaining: 12, limit: 30, used: 18, resets_at: NOW + 2 * 86400 });

  it("measures the share of the week gone", () => {
    expect(periodElapsed(kaggle, NOW)).toBeCloseTo(5 / 7);
  });

  it("projects the share used by reset", () => {
    expect(paceProjection(kaggle, NOW)).toBeCloseTo(0.6 / (5 / 7));
  });

  it("measures a Modal month from the first of the month", () => {
    const modal = row({ unit: "USD", remaining: 15, limit: 30, resets_at: Date.UTC(2026, 9, 1) / 1000 });
    expect(periodElapsed(modal, NOW)).toBeCloseTo((13.5 * 86400) / (30 * 86400));
  });

  it("has no pace without a period", () => {
    expect(paceProjection(row({ unit: "KRW", remaining: 1000, limit: 5000 }), NOW)).toBeNull();
  });
});

describe("status bar", () => {
  it("shows the account with the lowest share left", () => {
    const rows = [
      row({ alias: "colab", unit: "compute units", remaining: 90, limit: 100 }),
      row({ alias: "kaggle", remaining: 3, limit: 30, resets_at: NOW + 14 * 60 }),
      row({ alias: "elice", unit: "KRW", remaining: 1000 }),
      row({ alias: "lab", unmetered: true }),
    ];
    expect(mostConstrained(rows)?.alias).toBe("kaggle");
    expect(quotaStatusText(rows, NOW)).toBe("kaggle 10% left (14m)");
  });

  it("ranks an account with no limit after every limited one and shows its amount", () => {
    expect(quotaStatusText([row({ alias: "elice", unit: "KRW", remaining: 1000 })], NOW)).toBe("elice 1,000 KRW left");
  });

  it("says so when nothing is metered", () => {
    expect(quotaStatusText([row({ unmetered: true })], NOW)).toBe("letify no quota");
  });

  it("colors by the share left", () => {
    expect(severity(0.5, 20, 5)).toBe("ok");
    expect(severity(0.1, 20, 5)).toBe("warning");
    expect(severity(0.01, 20, 5)).toBe("error");
    expect(severity(null, 20, 5)).toBe("ok");
  });

  it("counts busy cards and the mean load", () => {
    const util = parseUtilization([
      { alias: "lab", accelerator: "A100", devices: [
        { index: 0, name: "A100", utilization_percent: 90 },
        { index: 1, name: "A100", utilization_percent: 0 },
        { index: 2, name: "A100", utilization_percent: 60 },
        { index: 3, name: "A100", utilization_percent: null },
      ] },
    ]);
    const status = parseStatus({ name: "p", live: 1, busy: 1, devices: { lab: { A100: { count: 4, reserved: 1, indices: [0, 1, 2, 3] } } }, runtimes: [] });
    const summary = gpuSummary(util, status, 10);
    expect(summary).toEqual({ total: 4, busy: 2, meanPercent: 50, reserved: 1 });
    expect(gpuStatusText(summary)).toBe("GPU 2/4 busy 50%");
    expect(gpuStatusText({ total: 0, busy: 0, meanPercent: null, reserved: 0 })).toBe("GPU idle");
  });
});

describe("history", () => {
  it("adds drops in remaining as spending and ignores top ups", () => {
    let h = recordSample([], NOW, 40, { k: 20 });
    h = recordSample(h, NOW + 60, 60, { k: 18 });
    h = recordSample(h, NOW + 120, null, { k: 30 });
    expect(h).toHaveLength(1);
    expect(h[0].spent.k).toBe(2);
    expect(h[0].gpuSum / h[0].gpuCount).toBe(50);
  });

  it("keeps 30 days", () => {
    let h = recordSample([], NOW - 40 * 86400, 10, {});
    h = recordSample(h, NOW, 10, {});
    expect(h.map((b) => b.day)).toEqual(["2026-09-14"]);
  });
});
