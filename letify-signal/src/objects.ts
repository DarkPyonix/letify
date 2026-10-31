// The Durable Object classes. Each is a thin shell: the logic is in MailboxCore and
// LimiterCore, which the unit tests cover, and the shell adds only what needs the platform:
// storage, alarms, long-poll waiting and WebSocket Hibernation.

import { DurableObject } from "cloudflare:workers";

/** The bindings of the Worker. The objects use none of them. */
type Env = Record<string, unknown>;

import { LimiterCore, type Verdict } from "./limiter";
import { MailboxCore, type PutResult, type ReadResult, type Stored } from "./mailbox";
import { MAX_WAIT_S } from "./protocol";
import type { KV } from "./storage";

export class Limiter extends DurableObject {
  private readonly core: LimiterCore;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.core = new LimiterCore(ctx.storage as unknown as KV);
  }

  hitIp(now: number): Promise<Verdict> {
    return this.core.hitIp(now);
  }

  checkSigned(nonce: string, now: number, send: boolean): Promise<Verdict> {
    return this.core.checkSigned(nonce, now, send);
  }
}

export class Mailbox extends DurableObject {
  private readonly core: MailboxCore;
  /** Long polls waiting for a message. They keep the object awake for at most MAX_WAIT_S. */
  private readonly waiters = new Set<() => void>();

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.core = new MailboxCore(ctx.storage as unknown as KV);
  }

  async put(from: string, body: string, now: number): Promise<PutResult> {
    const result = await this.core.put(from, body, now);
    if (!result.ok) return result;
    await this.armAlarm();
    const message: Stored = { seq: result.seq, from, at: now, body };
    for (const socket of this.ctx.getWebSockets()) {
      try {
        socket.send(JSON.stringify(message));
      } catch {
        // A socket that cannot be written to is closed by the platform; the message waits.
      }
    }
    for (const wake of this.waiters) wake();
    return result;
  }

  async read(after: number, waitS: number, now: number): Promise<ReadResult> {
    let result = await this.core.read(after, now);
    if (result.messages.length > 0 || waitS <= 0) return result;
    await new Promise<void>((resolve) => {
      const done = () => {
        clearTimeout(timer);
        this.waiters.delete(done);
        resolve();
      };
      const timer = setTimeout(done, Math.min(waitS, MAX_WAIT_S) * 1000);
      this.waiters.add(done);
    });
    result = await this.core.read(after, Math.floor(Date.now() / 1000));
    return result;
  }

  /** Accept a WebSocket with Hibernation and send what is already waiting. */
  async fetch(_request: Request): Promise<Response> {
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    this.ctx.acceptWebSocket(server);
    const backlog = await this.core.read(0, Math.floor(Date.now() / 1000));
    for (const message of backlog.messages) server.send(JSON.stringify(message));
    return new Response(null, { status: 101, webSocket: client });
  }

  async webSocketMessage(socket: WebSocket, data: string | ArrayBuffer): Promise<void> {
    if (typeof data !== "string" || data.length > 64) return;
    let ack: unknown;
    try {
      ack = (JSON.parse(data) as { ack?: unknown }).ack;
    } catch {
      return;
    }
    if (typeof ack === "number" && Number.isSafeInteger(ack) && ack >= 0) {
      await this.core.read(ack, Math.floor(Date.now() / 1000));
    } else {
      socket.close(1008, "bad frame");
    }
  }

  async webSocketClose(socket: WebSocket, code: number): Promise<void> {
    try {
      socket.close(code === 1005 ? 1000 : code);
    } catch {
      // Already closed.
    }
  }

  async alarm(): Promise<void> {
    await this.core.expire(Math.floor(Date.now() / 1000));
    await this.armAlarm();
  }

  private async armAlarm(): Promise<void> {
    const next = await this.core.nextExpiry();
    if (next !== null) await this.ctx.storage.setAlarm(next * 1000);
  }
}
