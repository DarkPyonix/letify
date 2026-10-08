/**
 * Running the letify CLI and reading its JSON.
 *
 * Owns spawning `<command> <subcommand> --json` in the workspace folder and turning a failure
 * into a readable message. It does not own parsing the records, which is in model.ts. Output
 * is never logged, because a misconfigured command could print a credential.
 */

import { execFile } from "node:child_process";

import { parseJson } from "./model";

export const TIMEOUT_MS = 90_000;

export function splitCommand(command: string): string[] {
  return command.trim().split(/\s+/).filter(Boolean);
}

/** Freeze uv project execution even when a user retains an older command setting. */
export function safeCommand(command: string): string[] {
  const parts = splitCommand(command);
  const executable = parts[0]?.split(/[\\/]/).pop()?.toLowerCase();
  if (executable !== "uv" && executable !== "uv.exe") return parts;
  const run = parts.indexOf("run");
  if (run < 1 || parts.slice(1, run).some((p) => ["tool", "sync", "pip", "add", "remove"].includes(p))) {
    throw new Error("letify.command must use uv run for read-only polling");
  }
  const flags = ["--frozen", "--no-sync"].filter((flag) => !parts.includes(flag));
  parts.splice(run + 1, 0, ...flags);
  return parts;
}

export function runLetify(command: string, subcommand: string, cwd: string | undefined): Promise<unknown> {
  let parts: string[];
  try {
    parts = safeCommand(command);
  } catch (error) {
    return Promise.reject(error);
  }
  const [program, ...base] = parts;
  if (!program) return Promise.reject(new Error("letify.command is empty"));
  const args = [...base, subcommand, "--json"];
  return new Promise((resolve, reject) => {
    execFile(program, args, { cwd, timeout: TIMEOUT_MS, maxBuffer: 8 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        const detail = (stderr || "").trim().split("\n").slice(-3).join(" ").slice(0, 300);
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "ENOENT") {
          reject(new Error(`${program} was not found; set letify.command`));
        } else {
          reject(new Error(`letify ${subcommand} failed${detail ? `: ${detail}` : ""}`));
        }
        return;
      }
      try {
        resolve(parseJson(stdout, subcommand));
      } catch (parseError) {
        reject(parseError);
      }
    });
  });
}
