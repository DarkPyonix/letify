import { describe, expect, it } from "vitest";

import { MailboxCore } from "../src/mailbox";
import { MAX_DEPTH, TTL_S } from "../src/protocol";
import { MemoryStorage } from "./helpers";

const NOW = 1_800_000_000;
const FROM = "ab".repeat(32);

describe("mailbox", () => {
  it("numbers messages from 1, in order, and returns them from a cursor", async () => {
    const box = new MailboxCore(new MemoryStorage());
    expect(await box.put(FROM, "YQ==", NOW)).toEqual({ ok: true, seq: 1 });
    expect(await box.put(FROM, "Yg==", NOW)).toEqual({ ok: true, seq: 2 });
    const all = await box.read(0, NOW);
    expect(all.messages.map((m) => [m.seq, m.from, m.body, m.at])).toEqual([
      [1, FROM, "YQ==", NOW],
      [2, FROM, "Yg==", NOW],
    ]);
    expect(all.cursor).toBe(2);
  });

  it("keeps a message that was read until it is acknowledged", async () => {
    const box = new MailboxCore(new MemoryStorage());
    await box.put(FROM, "YQ==", NOW);
    await box.put(FROM, "Yg==", NOW);
    expect((await box.read(0, NOW)).messages).toHaveLength(2);
    expect((await box.read(0, NOW)).messages).toHaveLength(2);
    expect((await box.read(1, NOW)).messages.map((m) => m.seq)).toEqual([2]);
    expect((await box.read(2, NOW)).messages).toEqual([]);
  });

  it("never reuses a sequence number after the mailbox emptied", async () => {
    const box = new MailboxCore(new MemoryStorage());
    await box.put(FROM, "YQ==", NOW);
    await box.read(1, NOW);
    expect(await box.put(FROM, "Yg==", NOW)).toEqual({ ok: true, seq: 2 });
  });

  it("drops a message once its time to live is over", async () => {
    const box = new MailboxCore(new MemoryStorage());
    await box.put(FROM, "YQ==", NOW);
    expect((await box.read(0, NOW + TTL_S - 1)).messages).toHaveLength(1);
    expect((await box.read(0, NOW + TTL_S)).messages).toEqual([]);
  });

  it("refuses the 33rd message and takes one again when room opens", async () => {
    const box = new MailboxCore(new MemoryStorage());
    for (let i = 0; i < MAX_DEPTH; i++) expect((await box.put(FROM, "YQ==", NOW)).ok).toBe(true);
    expect(await box.put(FROM, "YQ==", NOW)).toEqual({ ok: false, code: "mailbox_full" });
    await box.read(1, NOW);
    expect((await box.put(FROM, "YQ==", NOW)).ok).toBe(true);
  });

  it("frees room by expiry", async () => {
    const box = new MailboxCore(new MemoryStorage());
    for (let i = 0; i < MAX_DEPTH; i++) await box.put(FROM, "YQ==", NOW);
    expect((await box.put(FROM, "YQ==", NOW + TTL_S)).ok).toBe(true);
  });

  it("reports the next expiry so the object can set its alarm", async () => {
    const box = new MailboxCore(new MemoryStorage());
    expect(await box.nextExpiry()).toBeNull();
    await box.put(FROM, "YQ==", NOW);
    await box.put(FROM, "Yg==", NOW + 10);
    expect(await box.nextExpiry()).toBe(NOW + TTL_S);
    await box.expire(NOW + TTL_S);
    expect(await box.nextExpiry()).toBe(NOW + 10 + TTL_S);
  });
});
