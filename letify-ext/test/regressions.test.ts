import { describe, expect, it, vi } from "vitest";
import { safeCommand } from "../src/cli";
import { parseSessions, parseStatus, parseUtilization, activityStatusText } from "../src/model";
import { poll } from "../src/poll";
import { gpuCard, runtimesCard, quotaCard, STYLE } from "../src/render";
import { parseUsage } from "../src/model";

// Spec: Editor extension and Provider session discovery.
describe("safe polling", () => {
  it.each([
    ["uv run letify", ["uv", "run", "--frozen", "--no-sync", "letify"]],
    ["/tools/uv run --project /project letify", ["/tools/uv", "run", "--frozen", "--no-sync", "--project", "/project", "letify"]],
    ["C:\\tools\\uv.exe --offline run letify", ["C:\\tools\\uv.exe", "--offline", "run", "--frozen", "--no-sync", "letify"]],
    ["uv run --frozen --no-sync letify", ["uv", "run", "--frozen", "--no-sync", "letify"]],
    ["letify", ["letify"]],
  ])("freezes the configured command %s", (command, expected) => {
    expect(safeCommand(command)).toEqual(expected);
  });
  it("refuses uv commands that could modify the environment", () => {
    expect(() => safeCommand("uv sync")).toThrow(/uv run/);
    expect(() => safeCommand("uv tool run letify")).toThrow(/uv run/);
  });
  it("keeps valid session names separate from process-owned status", () => {
    const sessions = parseSessions([{ alias: "colab", kind: "colab", sessions: ["live-cpu"], reason: null }]);
    const status = parseStatus({ name: "letify", live: 0, busy: 0, devices: {}, runtimes: [] });
    expect(activityStatusText([], status, sessions, {}, 10)).toContain("1 session");
    expect(runtimesCard(status, sessions)).toContain("live-cpu");
    expect(runtimesCard(status, sessions)).toContain("Provider sessions");
    expect(status.live).toBe(0);
    expect(() => parseSessions([{ alias: "colab", sessions: "bad" }])).toThrow(/sessions/);
    expect(() => parseStatus({})).toThrow(/status/);
    expect(() => parseStatus({ name: "p", live: 0, busy: 0, devices: { lab: null }, runtimes: [] })).toThrow(/status/);
    expect(() => parseStatus({ name: "p", live: 1, busy: 0, devices: {}, runtimes: [{}] })).toThrow(/status/);
  });
  it("shows failures even when previous measurements remain", () => {
    const rows = parseUtilization([{ alias: "lab", devices: [{ index: 0, utilization_percent: 42 }] }]);
    expect(activityStatusText(rows, null, [], { status: "bad output" }, 10)).toContain("$(error)");
    expect(activityStatusText(rows, null, [], { sessions: "timeout" }, 10)).toContain("error");
  });
  it("parses each command independently and recovers on the next poll", async () => {
    const state = { status: null, utilization: [], sessions: [], errors: {} };
    const read = vi.fn(async (command: string) => command === "status" ? {} : command === "sessions"
      ? [{ alias: "colab", kind: "colab", sessions: ["live"], reason: null }] : []);
    await Promise.all([poll(state, "status", read, parseStatus), poll(state, "sessions", read, parseSessions)]);
    expect(state.errors).toHaveProperty("status");
    expect(state.sessions).toHaveLength(1);
    await poll(state, "status", async () => ({ name: "letify", live: 0, busy: 0, devices: {}, runtimes: [] }), parseStatus);
    expect(state.errors).not.toHaveProperty("status");
    await poll(state, "sessions", async () => { throw new Error("command failed"); }, parseSessions);
    expect(state.errors).toHaveProperty("sessions", "command failed");
    expect(state.sessions).toHaveLength(1);
  });
});

describe("device monitor", () => {
  it("aligns utilization and memory bars with values and load colors", () => {
    const [row] = parseUtilization([{ alias: "lab", devices: [{ index: 0, name: "A100", utilization_percent: 85,
      memory_used_gb: 76, memory_total_gb: 80, memory_percent: 95, temperature_c: 60, power_w: 200 }] }]);
    const html = gpuCard(row, null);
    expect(html).toContain('class="metric"');
    expect(html).toContain("85%");
    expect(html).toContain("76.0/80.0 GiB");
    expect(html).toContain('fill warning');
    expect(html).toContain('fill error');
    expect(html).toContain("Process details are not provided by the CLI");
    expect(STYLE).toContain("tabular-nums");
    expect(STYLE).toContain("--vscode-editor-background");
    const quota = parseUsage([{ alias: "lab", remaining: 20, limit: 100, unit: "GPU hours" }])[0];
    expect(quotaCard(quota, 0)).toContain('class="metric"');
  });
  it("does not turn missing telemetry into zero or a filled bar", () => {
    const [row] = parseUtilization([{ alias: "lab", devices: [{ index: 0, name: "A100", memory_total_gb: 80 }] }]);
    const html = gpuCard(row, null);
    expect(html).toContain("unknown/80.0 GiB");
    expect(html).not.toContain('class="fill');
    expect(html).not.toContain("0.0/80.0");
  });
  it("escapes session names and preserves unsupported and unavailable providers", () => {
    const sessions = parseSessions([{ alias: "c", kind: "colab", sessions: ["<session>"], reason: null },
      { alias: "local", kind: "local", sessions: [], reason: "session discovery is not supported" },
      { alias: "bad", sessions: [], unavailable: "query failed" }]);
    const html = runtimesCard(null, sessions);
    expect(html).toContain("&lt;session&gt;");
    expect(html).not.toContain("<session>");
    expect(html).toContain("session discovery is not supported");
    expect(html).toContain("query failed");
  });
});
