// The routes of the service. It owns the order of the checks and the reply codes, and takes
// the clock, the limiter objects and the mailbox objects as dependencies so tests can run it
// without Cloudflare.

import { authenticate } from "./auth";
import { checkNodeKey } from "./ed25519";
import type { Verdict } from "./limiter";
import type { PutResult, ReadResult } from "./mailbox";
import {
  MAX_BODY,
  MAX_WAIT_S,
  STATUS,
  bytesToBase64,
  hexToBytes,
  isNodeIdText,
} from "./protocol";

export interface LimiterApi {
  hitIp(now: number): Promise<Verdict>;
  checkSigned(nonce: string, now: number, send: boolean): Promise<Verdict>;
}

export interface MailboxApi {
  put(from: string, bodyBase64: string, now: number): Promise<PutResult>;
  /** Acknowledge up to `after`, then wait up to `waitS` seconds for a message above it. */
  read(after: number, waitS: number, now: number): Promise<ReadResult>;
  /** Hand the WebSocket upgrade request to the mailbox. */
  connect(request: Request): Promise<Response>;
}

export interface Deps {
  now(): number;
  limiter(name: string): LimiterApi;
  mailbox(node: string): MailboxApi;
  /** One log line. It carries no node id, address, header or body. */
  log(line: string): void;
}

function reply(code: string, extra: Record<string, unknown> = {}, headers: HeadersInit = {}): Response {
  return Response.json({ ok: false, error: code, ...extra }, { status: STATUS[code], headers });
}

function limited(verdict: Extract<Verdict, { ok: false }>): Response {
  const headers: Record<string, string> = {};
  if (verdict.retryAfter !== undefined) headers["Retry-After"] = String(verdict.retryAfter);
  return reply(verdict.code, {}, headers);
}

/** Read the body but stop past `cap` bytes. Returns null for a body over the cap. */
async function readCapped(request: Request, cap: number): Promise<Uint8Array | null> {
  const declared = request.headers.get("Content-Length");
  if (declared !== null && Number(declared) > cap) return null;
  if (request.body === null) return new Uint8Array();
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > cap) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const out = new Uint8Array(size);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.byteLength;
  }
  return out;
}

function integer(text: string | null, fallback: number, max: number): number | null {
  if (text === null) return fallback;
  if (!/^[0-9]{1,15}$/.test(text)) return null;
  const value = Number(text);
  return value <= max ? value : null;
}

export async function handle(request: Request, deps: Deps): Promise<Response> {
  let length = 0;
  const response = await route(request, deps, (n) => (length = n));
  // The hour, not the second: a log line cannot be matched to one request.
  deps.log(`${Math.floor(deps.now() / 3600) * 3600} ${request.method} ${response.status} ${length}`);
  return response;
}

async function route(request: Request, deps: Deps, sawBody: (n: number) => void): Promise<Response> {
  const url = new URL(request.url);
  if (url.pathname === "/v1/health") {
    return Response.json({ ok: true, v: 1 });
  }
  const match = /^\/v1\/(send|recv|ws)\/([^/]+)$/.exec(url.pathname);
  if (match === null) return reply("not_found");
  const [, kind, target] = match;
  const method = kind === "send" ? "POST" : "GET";
  if (request.method !== method) return reply("method_not_allowed");

  const now = deps.now();
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const counted = await deps.limiter(`ip:${ip}`).hitIp(now);
  if (!counted.ok) return limited(counted);

  const body = kind === "send" ? await readCapped(request, MAX_BODY) : new Uint8Array();
  if (body === null) return reply("too_large");
  sawBody(body.byteLength);

  const auth = await authenticate({
    method: request.method,
    pathAndQuery: url.pathname + url.search,
    header: (name) => request.headers.get(name),
    body,
    now,
  });
  if (!auth.ok) return reply(auth.code);

  if (!isNodeIdText(target)) return reply("bad_request");
  if (kind !== "send" && target !== auth.node) return reply("forbidden");
  if (kind === "send" && (body.byteLength === 0 || checkNodeKey(hexToBytes(target)) !== "ok")) {
    return reply(body.byteLength === 0 ? "bad_request" : "bad_key");
  }

  const checked = await deps.limiter(`node:${auth.node}`).checkSigned(auth.nonce, now, kind === "send");
  if (!checked.ok) return limited(checked);

  if (kind === "send") {
    const put = await deps.mailbox(target).put(auth.node, bytesToBase64(body), now);
    if (!put.ok) return reply(put.code);
    return Response.json({ ok: true, seq: put.seq }, { status: 202 });
  }
  if (kind === "recv") {
    const after = integer(url.searchParams.get("after"), 0, Number.MAX_SAFE_INTEGER);
    const wait = integer(url.searchParams.get("wait"), MAX_WAIT_S, MAX_WAIT_S);
    if (after === null || wait === null) return reply("bad_request");
    const result = await deps.mailbox(target).read(after, wait, now);
    return Response.json(result);
  }
  if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") return reply("upgrade_required");
  return deps.mailbox(target).connect(request);
}
