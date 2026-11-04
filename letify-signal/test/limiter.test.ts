import { describe, expect, it } from "vitest";

import { LimiterCore } from "../src/limiter";
import { IP_LIMIT, NONCE_MEMORY_S, SEND_LIMIT, SIGNER_LIMIT } from "../src/protocol";
import { MemoryStorage } from "./helpers";

const NOW = 1_800_000_000;
const nonce = (n: number) => n.toString(16).padStart(32, "0");

describe("signer limits", () => {
  it("refuses a nonce used before inside the memory, and takes it again after", async () => {
    const limiter = new LimiterCore(new MemoryStorage());
    expect(await limiter.checkSigned(nonce(1), NOW, false)).toEqual({ ok: true });
    expect(await limiter.checkSigned(nonce(1), NOW + 1, false)).toMatchObject({ ok: false, code: "replay" });
    expect(await limiter.checkSigned(nonce(1), NOW + NONCE_MEMORY_S + 1, false)).toEqual({ ok: true });
  });

  it("keeps the nonces of other signers apart because each signer has its own object", async () => {
    const a = new LimiterCore(new MemoryStorage());
    const b = new LimiterCore(new MemoryStorage());
    expect(await a.checkSigned(nonce(1), NOW, false)).toEqual({ ok: true });
    expect(await b.checkSigned(nonce(1), NOW, false)).toEqual({ ok: true });
  });

  it("allows 120 requests and 30 sends per minute and tells when to retry", async () => {
    const limiter = new LimiterCore(new MemoryStorage());
    for (let i = 0; i < SEND_LIMIT; i++) expect((await limiter.checkSigned(nonce(i), NOW, true)).ok).toBe(true);
    const sixth = await limiter.checkSigned(nonce(1000), NOW + 5, true);
    expect(sixth).toMatchObject({ ok: false, code: "rate_limited" });
    expect((sixth as { retryAfter: number }).retryAfter).toBeGreaterThan(0);
    // Receives still pass until the 120 total.
    for (let i = 0; i < SIGNER_LIMIT - SEND_LIMIT; i++) {
      expect((await limiter.checkSigned(nonce(2000 + i), NOW + 6, false)).ok).toBe(true);
    }
    expect(await limiter.checkSigned(nonce(3000), NOW + 7, false)).toMatchObject({ code: "rate_limited" });
    // The next window starts clean.
    expect((await limiter.checkSigned(nonce(4000), NOW + 61, true)).ok).toBe(true);
  });

  it("does not count a refused replay as a request twice", async () => {
    const limiter = new LimiterCore(new MemoryStorage());
    await limiter.checkSigned(nonce(1), NOW, true);
    for (let i = 0; i < 100; i++) await limiter.checkSigned(nonce(1), NOW, true);
    for (let i = 0; i < SEND_LIMIT - 1; i++) expect((await limiter.checkSigned(nonce(10 + i), NOW, true)).ok).toBe(true);
  });
});

describe("ip limit", () => {
  it("allows 240 requests per minute and no more", async () => {
    const limiter = new LimiterCore(new MemoryStorage());
    for (let i = 0; i < IP_LIMIT; i++) expect((await limiter.hitIp(NOW)).ok).toBe(true);
    expect(await limiter.hitIp(NOW + 1)).toMatchObject({ ok: false, code: "rate_limited" });
    expect((await limiter.hitIp(NOW + 60)).ok).toBe(true);
  });
});
