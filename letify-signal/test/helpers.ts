import { createHash, webcrypto } from "node:crypto";

import { VERSION_DOMAIN, bytesToHex } from "../src/protocol";

export interface TestNode {
  id: string;
  sign(message: Uint8Array): Promise<Uint8Array>;
}

/** A node with a fresh ed25519 key from WebCrypto, whose public key is the node id. */
export async function newNode(): Promise<TestNode> {
  const pair = (await webcrypto.subtle.generateKey("Ed25519", true, ["sign", "verify"])) as webcrypto.CryptoKeyPair;
  const raw = new Uint8Array(await webcrypto.subtle.exportKey("raw", pair.publicKey));
  return {
    id: bytesToHex(raw),
    async sign(message) {
      return new Uint8Array(await webcrypto.subtle.sign("Ed25519", pair.privateKey, message));
    },
  };
}

export function sha256Hex(body: Uint8Array): string {
  return createHash("sha256").update(body).digest("hex");
}

let counter = 0;
export function freshNonce(): string {
  counter += 1;
  return counter.toString(16).padStart(32, "0");
}

export interface SignOptions {
  method: string;
  path: string;
  body?: Uint8Array;
  time: number;
  nonce?: string;
  /** Sign this string instead of the canonical one, to make a wrong signature. */
  tamper?: string;
}

/** The four headers of a signed request. */
export async function signedHeaders(node: TestNode, o: SignOptions): Promise<Record<string, string>> {
  const nonce = o.nonce ?? freshNonce();
  const text =
    o.tamper ??
    `${VERSION_DOMAIN}\n${o.method}\n${o.path}\n${o.time}\n${nonce}\n${sha256Hex(o.body ?? new Uint8Array())}`;
  const signature = await node.sign(new TextEncoder().encode(text));
  return {
    "X-Letify-Node": node.id,
    "X-Letify-Time": String(o.time),
    "X-Letify-Nonce": nonce,
    "X-Letify-Signature": bytesToHex(signature),
  };
}

import type { KV } from "../src/storage";

/** An in-memory stand-in for the storage of a Durable Object. */
export class MemoryStorage implements KV {
  readonly data = new Map<string, unknown>();

  async get<T>(key: string): Promise<T | undefined> {
    return this.data.get(key) as T | undefined;
  }

  async put<T>(key: string, value: T): Promise<void> {
    this.data.set(key, structuredClone(value));
  }

  async delete(keys: string | string[]): Promise<void> {
    for (const key of Array.isArray(keys) ? keys : [keys]) this.data.delete(key);
  }

  async list<T>(options: { prefix: string }): Promise<Map<string, T>> {
    const out = new Map<string, T>();
    for (const key of [...this.data.keys()].sort()) {
      if (key.startsWith(options.prefix)) out.set(key, this.data.get(key) as T);
    }
    return out;
  }
}
