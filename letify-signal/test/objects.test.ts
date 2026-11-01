// The Mailbox shell registers a Hibernation auto-response, so the client keepalive is answered
// by the runtime without waking the object. Spec "Signalling service" item 1.

import { describe, expect, it, vi } from "vitest";

vi.mock("cloudflare:workers", () => ({
  DurableObject: class {
    ctx: unknown;
    constructor(ctx: unknown) {
      this.ctx = ctx;
    }
  },
}));

class FakePair {
  constructor(
    readonly request: string,
    readonly response: string,
  ) {}
}
(globalThis as Record<string, unknown>).WebSocketRequestResponsePair = FakePair;

describe("Mailbox keepalive", () => {
  it("registers ping answered by pong as the WebSocket auto-response", async () => {
    const { Mailbox } = await import("../src/objects");
    const calls: unknown[] = [];
    const ctx = {
      storage: new Map(),
      setWebSocketAutoResponse: (pair: unknown) => calls.push(pair),
    };
    new Mailbox(ctx as never, {});
    expect(calls).toHaveLength(1);
    const pair = calls[0] as FakePair;
    expect([pair.request, pair.response]).toEqual(["ping", "pong"]);
  });
});
