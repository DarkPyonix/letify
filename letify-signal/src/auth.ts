// Authentication of one signed request: headers in, a verdict out. It does not remember
// nonces (the signer's Limiter object does) and does not know any route.

import { verifySignature, checkNodeKey } from "./ed25519";
import {
  STATUS,
  WINDOW_S,
  hexToBytes,
  isNodeIdText,
  isNonceText,
  isSignatureText,
  sha256Hex,
  signedText,
} from "./protocol";

export interface AuthInput {
  method: string;
  /** The path and query exactly as sent. */
  pathAndQuery: string;
  header(name: string): string | null;
  body: Uint8Array;
  /** The service clock in unix seconds. */
  now: number;
}

export type AuthResult =
  | { ok: true; node: string; nonce: string; time: number }
  | { ok: false; status: number; code: string };

const refuse = (code: string): AuthResult => ({ ok: false, status: STATUS[code], code });

export async function authenticate(input: AuthInput): Promise<AuthResult> {
  const node = input.header("X-Letify-Node");
  const timeText = input.header("X-Letify-Time");
  const nonce = input.header("X-Letify-Nonce");
  const signature = input.header("X-Letify-Signature");
  if (node === null || timeText === null || nonce === null || signature === null) {
    return refuse("bad_request");
  }
  if (!/^[0-9]{1,12}$/.test(timeText) || !isNonceText(nonce) || !isSignatureText(signature)) {
    return refuse("bad_request");
  }
  // Upper case or short hex is not a node id, and neither is a key of small order.
  if (!isNodeIdText(node) || checkNodeKey(hexToBytes(node)) !== "ok") return refuse("bad_key");
  const time = Number(timeText);
  if (Math.abs(time - input.now) > WINDOW_S) return refuse("stale");
  const text = signedText(input.method, input.pathAndQuery, timeText, nonce, await sha256Hex(input.body));
  if (!(await verifySignature(node, new TextEncoder().encode(text), signature))) {
    return refuse("bad_signature");
  }
  return { ok: true, node, nonce, time };
}
