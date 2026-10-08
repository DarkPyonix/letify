import { describe, expect, it, vi } from "vitest";
import { execFile } from "node:child_process";
import { runLetify, TIMEOUT_MS } from "../src/cli";

vi.mock("node:child_process", () => ({ execFile: vi.fn() }));

// Spec: Editor extension. Exercise the subprocess boundary for each poll.
describe("CLI execution", () => {
  it.each(["usage", "utilization", "status", "sessions"])("freezes the actual %s invocation", async (command) => {
    vi.mocked(execFile).mockImplementationOnce((_program, _args, _options, callback: any) => {
      callback(null, "[]", "");
      return {} as any;
    });
    await expect(runLetify("uv run letify", command, "/workspace")).resolves.toEqual([]);
    expect(execFile).toHaveBeenLastCalledWith("uv", ["run", "--frozen", "--no-sync", "letify", command, "--json"],
      expect.objectContaining({ cwd: "/workspace", timeout: TIMEOUT_MS }), expect.any(Function));
  });
  it("names a missing executable", async () => {
    vi.mocked(execFile).mockImplementationOnce((_program, _args, _options, callback: any) => {
      callback(Object.assign(new Error("missing"), { code: "ENOENT" }), "", "");
      return {} as any;
    });
    await expect(runLetify("uv run letify", "sessions", undefined)).rejects.toThrow("uv was not found");
  });
  it("names invalid JSON and a failed command", async () => {
    vi.mocked(execFile).mockImplementationOnce((_program, _args, _options, callback: any) => {
      callback(null, "not JSON", "");
      return {} as any;
    });
    await expect(runLetify("letify", "status", undefined)).rejects.toThrow("letify status did not print JSON");
    vi.mocked(execFile).mockImplementationOnce((_program, _args, _options, callback: any) => {
      callback(new Error("failure"), "", "query failed");
      return {} as any;
    });
    await expect(runLetify("letify", "sessions", undefined)).rejects.toThrow("letify sessions failed: query failed");
  });
});
