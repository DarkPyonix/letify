// The messages waiting for one node: numbering, depth, acknowledgement and expiry. The
// Durable Object that wraps it adds the waiting, the WebSocket and the alarm.

import { MAX_DEPTH, TTL_S } from "./protocol";
import type { KV } from "./storage";

export interface Stored {
  seq: number;
  from: string;
  /** Unix second the service accepted the message. */
  at: number;
  /** The message body, base64. */
  body: string;
}

export type PutResult = { ok: true; seq: number } | { ok: false; code: "mailbox_full" };

export interface ReadResult {
  messages: Stored[];
  cursor: number;
}

const PREFIX = "m:";
const COUNTER = "seq";

const keyOf = (seq: number): string => PREFIX + seq.toString().padStart(12, "0");

export class MailboxCore {
  constructor(private readonly storage: KV) {}

  private async live(now: number): Promise<Stored[]> {
    const all = await this.storage.list<Stored>({ prefix: PREFIX });
    const keep: Stored[] = [];
    const drop: string[] = [];
    for (const [key, message] of all) {
      if (message.at + TTL_S <= now) drop.push(key);
      else keep.push(message);
    }
    if (drop.length > 0) await this.storage.delete(drop);
    return keep;
  }

  async put(from: string, body: string, now: number): Promise<PutResult> {
    const waiting = await this.live(now);
    if (waiting.length >= MAX_DEPTH) return { ok: false, code: "mailbox_full" };
    const seq = ((await this.storage.get<number>(COUNTER)) ?? 0) + 1;
    await this.storage.put(COUNTER, seq);
    await this.storage.put(keyOf(seq), { seq, from, at: now, body } satisfies Stored);
    return { ok: true, seq };
  }

  /** Delete the messages up to `after`, then return the ones above it. */
  async read(after: number, now: number): Promise<ReadResult> {
    const waiting = await this.live(now);
    const acknowledged = waiting.filter((m) => m.seq <= after).map((m) => keyOf(m.seq));
    if (acknowledged.length > 0) await this.storage.delete(acknowledged);
    return {
      messages: waiting.filter((m) => m.seq > after),
      cursor: (await this.storage.get<number>(COUNTER)) ?? 0,
    };
  }

  /** Delete the expired messages. */
  async expire(now: number): Promise<void> {
    await this.live(now);
  }

  /** The unix second the oldest waiting message expires, or null for an empty mailbox. */
  async nextExpiry(): Promise<number | null> {
    const all = await this.storage.list<Stored>({ prefix: PREFIX });
    let earliest: number | null = null;
    for (const message of all.values()) {
      const at = message.at + TTL_S;
      if (earliest === null || at < earliest) earliest = at;
    }
    return earliest;
  }
}
