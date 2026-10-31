import { describe, expect, it } from "vitest";

import { authenticate } from "../src/auth";
import { freshNonce, newNode, sha256Hex, signedHeaders } from "./helpers";

const NOW = 1_800_000_000;
const PATH = "/send/" + "ab".repeat(32);

async function attempt(
  mutate: (headers: Record<string, string>) => void = () => {},
  options: { time?: number; body?: Uint8Array; tamper?: string } = {},
) {
  const node = await newNode();
  const body = options.body ?? new TextEncoder().encode("hello");
  const headers = await signedHeaders(node, {
    method: "POST",
    path: PATH,
    body,
    time: options.time ?? NOW,
    tamper: options.tamper,
  });
  mutate(headers);
  const result = await authenticate({
    method: "POST",
    pathAndQuery: PATH,
    header: (name) => {
      const key = Object.keys(headers).find((k) => k.toLowerCase() === name.toLowerCase());
      return key ? headers[key] : null;
    },
    body,
    now: NOW,
  });
  return { node, result };
}

describe("authenticate", () => {
  it("accepts a request signed over method, path, time, nonce and body hash", async () => {
    const { node, result } = await attempt();
    expect(result).toMatchObject({ ok: true, node: node.id });
  });

  it("accepts a time 60 s old and 60 s ahead, and refuses 61 s either way", async () => {
    expect((await attempt(() => {}, { time: NOW - 60 })).result.ok).toBe(true);
    expect((await attempt(() => {}, { time: NOW + 60 })).result.ok).toBe(true);
    for (const time of [NOW - 61, NOW + 61]) {
      expect((await attempt(() => {}, { time })).result).toMatchObject({ ok: false, status: 401, code: "stale" });
    }
  });

  it("refuses a wrong signature", async () => {
    const { result } = await attempt(() => {}, { tamper: "something else" });
    expect(result).toMatchObject({ ok: false, status: 401, code: "bad_signature" });
  });

  it("refuses a body that is not the one that was signed", async () => {
    const node = await newNode();
    const headers = await signedHeaders(node, {
      method: "POST",
      path: PATH,
      body: new TextEncoder().encode("one"),
      time: NOW,
    });
    const result = await authenticate({
      method: "POST",
      pathAndQuery: PATH,
      header: (name) => headers[Object.keys(headers).find((k) => k.toLowerCase() === name.toLowerCase())!] ?? null,
      body: new TextEncoder().encode("two"),
      now: NOW,
    });
    expect(result).toMatchObject({ ok: false, code: "bad_signature" });
  });

  it("refuses a path other than the one that was signed", async () => {
    const node = await newNode();
    const headers = await signedHeaders(node, { method: "POST", path: PATH, time: NOW });
    const result = await authenticate({
      method: "POST",
      pathAndQuery: "/send/" + "cd".repeat(32),
      header: (name) => headers[Object.keys(headers).find((k) => k.toLowerCase() === name.toLowerCase())!] ?? null,
      body: new Uint8Array(),
      now: NOW,
    });
    expect(result).toMatchObject({ ok: false, code: "bad_signature" });
  });

  it("refuses a missing or malformed header with bad_request", async () => {
    const cases: Array<(h: Record<string, string>) => void> = [
      (h) => delete h["X-Letify-Node"],
      (h) => delete h["X-Letify-Signature"],
      (h) => (h["X-Letify-Time"] = "12x"),
      (h) => (h["X-Letify-Time"] = "-5"),
      (h) => (h["X-Letify-Nonce"] = "abc"),
      (h) => (h["X-Letify-Nonce"] = "G".repeat(32)),
      (h) => (h["X-Letify-Signature"] = "00".repeat(63)),
    ];
    for (const mutate of cases) {
      const { result } = await attempt(mutate);
      expect(result).toMatchObject({ ok: false, status: 400, code: "bad_request" });
    }
  });

  it("refuses a signer that is not a canonical key with bad_key", async () => {
    const identity = "01" + "00".repeat(31);
    const { result } = await attempt((h) => (h["X-Letify-Node"] = identity));
    expect(result).toMatchObject({ ok: false, status: 400, code: "bad_key" });
    const upper = await attempt((h) => (h["X-Letify-Node"] = h["X-Letify-Node"].toUpperCase()));
    expect(upper.result).toMatchObject({ ok: false, code: "bad_key" });
  });

  it("hashes zero bytes for a request with no body", () => {
    expect(sha256Hex(new Uint8Array())).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect(freshNonce()).toHaveLength(32);
  });
});
