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

export function runLetify(command: string, subcommand: string, cwd: string | undefined): Promise<unknown> {
  const [program, ...base] = splitCommand(command);
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
