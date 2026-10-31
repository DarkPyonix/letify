// Replay and rate limits for one signer or one client address. One Limiter Durable Object
// exists per name, so the counters and the nonces are shared by every replica of the Worker.

import {
  IP_LIMIT,
  NONCE_MEMORY_S,
  RATE_WINDOW_S,
  SEND_LIMIT,
  SIGNER_LIMIT,
} from "./protocol";
import type { KV } from "./storage";

export type Verdict =
  | { ok: true }
  | { ok: false; code: "replay" | "rate_limited"; retryAfter?: number };

interface State {
  window: number;
  total: number;
  sends: number;
  /** Nonce to the second it was used. */
  nonces: Record<string, number>;
}

const KEY = "state";

export class LimiterCore {
  constructor(private readonly storage: KV) {}

  private async load(now: number): Promise<State> {
    const window = Math.floor(now / RATE_WINDOW_S);
    const stored = await this.storage.get<State>(KEY);
    const state: State = stored ?? { window, total: 0, sends: 0, nonces: {} };
    if (state.window !== window) {
      state.window = window;
      state.total = 0;
      state.sends = 0;
    }
    for (const [nonce, used] of Object.entries(state.nonces)) {
      if (now - used > NONCE_MEMORY_S) delete state.nonces[nonce];
    }
    return state;
  }

  private retryAfter(now: number): number {
    return (Math.floor(now / RATE_WINDOW_S) + 1) * RATE_WINDOW_S - now;
  }

  /** Count one request of a signer and remember its nonce. `send` marks a send. */
  async checkSigned(nonce: string, now: number, send: boolean): Promise<Verdict> {
    const state = await this.load(now);
    if (nonce in state.nonces) {
      await this.storage.put(KEY, state);
      return { ok: false, code: "replay" };
    }
    // A refused request still burns its nonce, so a refusal cannot be retried for free.
    state.nonces[nonce] = now;
    if (state.total >= SIGNER_LIMIT || (send && state.sends >= SEND_LIMIT)) {
      await this.storage.put(KEY, state);
      return { ok: false, code: "rate_limited", retryAfter: this.retryAfter(now) };
    }
    state.total += 1;
    if (send) state.sends += 1;
    await this.storage.put(KEY, state);
    return { ok: true };
  }

  /** Count one request of a client address. */
  async hitIp(now: number): Promise<Verdict> {
    const state = await this.load(now);
    if (state.total >= IP_LIMIT) {
      await this.storage.put(KEY, state);
      return { ok: false, code: "rate_limited", retryAfter: this.retryAfter(now) };
    }
    state.total += 1;
    await this.storage.put(KEY, state);
    return { ok: true };
  }
}
