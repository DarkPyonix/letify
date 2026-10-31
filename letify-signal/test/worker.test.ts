import { beforeEach, describe, expect, it } from "vitest";

import { type Deps, type MailboxApi, handle } from "../src/handler";
import { LimiterCore } from "../src/limiter";
import { MailboxCore } from "../src/mailbox";
import { MAX_BODY, bytesToHex } from "../src/protocol";
import { MemoryStorage, type TestNode, freshNonce, newNode, signedHeaders } from "./helpers";

const NOW = 1_800_000_000;
const BASE = "https://letify.test";

class World {
  limiters = new Map<string, LimiterCore>();
  boxes = new Map<string, MailboxCore>();
  now = NOW;
  logs: string[] = [];

  deps: Deps = {
    now: () => this.now,
    limiter: (name) => {
      if (!this.limiters.has(name)) this.limiters.set(name, new LimiterCore(new MemoryStorage()));
      return this.limiters.get(name)!;
    },
    mailbox: (node): MailboxApi => {
      if (!this.boxes.has(node)) this.boxes.set(node, new MailboxCore(new MemoryStorage()));
      const core = this.boxes.get(node)!;
      return {
        put: (from, body, now) => core.put(from, body, now),
        read: (after, _wait, now) => core.read(after, now),
        connect: async () => new Response(null, { status: 501 }),
      };
    },
    log: (line) => this.logs.push(line),
  };
}

async function send(world: World, from: TestNode, to: string, body: Uint8Array, over: { time?: number; nonce?: string; ip?: string } = {}) {
  const path = `/send/${to}`;
  const headers = await signedHeaders(from, { method: "POST", path, body, time: over.time ?? world.now, nonce: over.nonce });
  return handle(
    new Request(BASE + path, { method: "POST", body, headers: { ...headers, "CF-Connecting-IP": over.ip ?? "203.0.113.5" } }),
    world.deps,
  );
}

async function recv(world: World, node: TestNode, path: string, signer: TestNode = node, over: { time?: number; nonce?: string } = {}) {
  const headers = await signedHeaders(signer, { method: "GET", path, time: over.time ?? world.now, nonce: over.nonce });
  return handle(new Request(BASE + path, { method: "GET", headers: { ...headers, "CF-Connecting-IP": "203.0.113.6" } }), world.deps);
}

let world: World;
let alice: TestNode;
let bob: TestNode;
beforeEach(async () => {
  world = new World();
  alice = await newNode();
  bob = await newNode();
});

const text = (s: string) => new TextEncoder().encode(s);

describe("send and recv", () => {
  it("delivers a message from alice to bob with alice's id stamped by the service", async () => {
    const sent = await send(world, alice, bob.id, text("hello"));
    expect(sent.status).toBe(202);
    expect(await sent.json()).toEqual({ ok: true, seq: 1 });
    const got = await recv(world, bob, `/recv/${bob.id}?wait=0`);
    expect(got.status).toBe(200);
    const body = (await got.json()) as { messages: Array<{ seq: number; from: string; body: string }>; cursor: number };
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0].from).toBe(alice.id);
    expect(atob(body.messages[0].body)).toBe("hello");
    expect(body.cursor).toBe(1);
  });

  it("deletes what the after cursor acknowledges", async () => {
    await send(world, alice, bob.id, text("one"));
    await send(world, alice, bob.id, text("two"));
    const after = await recv(world, bob, `/recv/${bob.id}?after=1&wait=0`);
    const body = (await after.json()) as { messages: Array<{ seq: number }> };
    expect(body.messages.map((m) => m.seq)).toEqual([2]);
  });

  it("refuses to let alice read bob's mailbox even with a valid signature of her own", async () => {
    await send(world, alice, bob.id, text("secret"));
    const stolen = await recv(world, bob, `/recv/${bob.id}?wait=0`, alice);
    expect(stolen.status).toBe(403);
    expect(await stolen.json()).toEqual({ ok: false, error: "forbidden" });
  });

  it("refuses an unsigned send and a send with a wrong signature", async () => {
    const unsigned = await handle(new Request(`${BASE}/send/${bob.id}`, { method: "POST", body: text("x") }), world.deps);
    expect(unsigned.status).toBe(400);
    const path = `/send/${bob.id}`;
    const headers = await signedHeaders(alice, { method: "POST", path, body: text("x"), time: NOW, tamper: "no" });
    const wrong = await handle(new Request(BASE + path, { method: "POST", body: text("x"), headers }), world.deps);
    expect(wrong.status).toBe(401);
    expect(await wrong.json()).toEqual({ ok: false, error: "bad_signature" });
    expect((await recv(world, bob, `/recv/${bob.id}?wait=0`).then((r) => r.json())) as { messages: unknown[] }).toMatchObject({ messages: [] });
  });

  it("refuses a stale request", async () => {
    const response = await send(world, alice, bob.id, text("x"), { time: NOW - 61 });
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ ok: false, error: "stale" });
  });

  it("refuses a replayed nonce and delivers the message once", async () => {
    const nonce = freshNonce();
    expect((await send(world, alice, bob.id, text("x"), { nonce })).status).toBe(202);
    const again = await send(world, alice, bob.id, text("x"), { nonce });
    expect(again.status).toBe(401);
    expect(await again.json()).toEqual({ ok: false, error: "replay" });
    const got = (await (await recv(world, bob, `/recv/${bob.id}?wait=0`)).json()) as { messages: unknown[] };
    expect(got.messages).toHaveLength(1);
  });

  it("refuses a body over 4096 bytes with 413 and takes exactly 4096", async () => {
    const big = await send(world, alice, bob.id, new Uint8Array(MAX_BODY + 1));
    expect(big.status).toBe(413);
    expect(await big.json()).toEqual({ ok: false, error: "too_large" });
    expect((await send(world, alice, bob.id, new Uint8Array(MAX_BODY))).status).toBe(202);
  });

  it("refuses an empty body, a recipient that is not a canonical key and a malformed id", async () => {
    expect((await send(world, alice, bob.id, new Uint8Array())).status).toBe(400);
    const identity = "01" + "00".repeat(31);
    const bad = await send(world, alice, identity, text("x"));
    expect(bad.status).toBe(400);
    expect(await bad.json()).toEqual({ ok: false, error: "bad_key" });
    const short = await send(world, alice, "abc", text("x"));
    expect(short.status).toBe(400);
  });

  it("answers 429 when the mailbox is full", async () => {
    // Senders differ, so no signer limit applies on the way to a full mailbox.
    for (let i = 0; i < 32; i++) expect((await send(world, await newNode(), bob.id, text("x"))).status).toBe(202);
    const full = await send(world, alice, bob.id, text("x"));
    expect(full.status).toBe(429);
    expect(await full.json()).toEqual({ ok: false, error: "mailbox_full" });
  });

  it("limits a signer to 30 sends per minute with Retry-After", async () => {
    for (let i = 0; i < 30; i++) {
      const to = (await newNode()).id;
      expect((await send(world, alice, to, text("x"))).status).toBe(202);
    }
    const limited = await send(world, alice, bob.id, text("x"));
    expect(limited.status).toBe(429);
    expect(Number(limited.headers.get("Retry-After"))).toBeGreaterThan(0);
    expect(await limited.json()).toEqual({ ok: false, error: "rate_limited" });
  });

  it("limits a client address to 240 requests per minute, before any signature is checked", async () => {
    let last = 0;
    for (let i = 0; i < 241; i++) {
      const r = await handle(new Request(`${BASE}/send/${bob.id}`, { method: "POST", body: text("x"), headers: { "CF-Connecting-IP": "192.0.2.1" } }), world.deps);
      last = r.status;
    }
    expect(last).toBe(429);
  });
});

describe("routing", () => {
  it("answers the health check without a signature", async () => {
    const r = await handle(new Request(`${BASE}/health`), world.deps);
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual({ ok: true });
  });

  it("answers 404 for the old versioned paths", async () => {
    for (const path of ["/v1/health", `/v1/send/${bob.id}`, `/v1/recv/${bob.id}`, `/v1/ws/${bob.id}`]) {
      expect((await handle(new Request(`${BASE}${path}`), world.deps)).status, path).toBe(404);
    }
  });

  it("answers 404 for an unknown path and 405 for a wrong method", async () => {
    expect((await handle(new Request(`${BASE}/v2/send/${bob.id}`), world.deps)).status).toBe(404);
    expect((await handle(new Request(`${BASE}/send/${bob.id}`, { method: "GET" }), world.deps)).status).toBe(405);
  });

  it("refuses a bad wait or after value", async () => {
    for (const query of ["wait=26", "wait=-1", "wait=x", "after=-1", "after=1.5"]) {
      const r = await recv(world, bob, `/recv/${bob.id}?${query}`);
      expect(r.status, query).toBe(400);
    }
  });

  it("logs the method, the code and the length and nothing else", async () => {
    await send(world, alice, bob.id, text("private words"));
    expect(world.logs.length).toBeGreaterThan(0);
    const joined = world.logs.join("\n");
    expect(joined).toContain("POST");
    expect(joined).toContain("202");
    expect(joined).not.toContain("private");
    expect(joined).not.toContain(alice.id);
    expect(joined).not.toContain(bob.id);
    expect(joined).not.toContain(bytesToHex(new Uint8Array([1])));
  });
});
