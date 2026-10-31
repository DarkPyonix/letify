// Constants and byte helpers shared by every module. No service state lives here.

/** The domain prefix of every request signature: the first line of the signed text. */
export const VERSION_DOMAIN = "letify-signal v1";

/** The largest message body in bytes. */
export const MAX_BODY = 4096;
/** Seconds a message waits in a mailbox before it is deleted. */
export const TTL_S = 60;
/** The most messages a mailbox holds. */
export const MAX_DEPTH = 32;
/** Seconds the request time may differ from the service clock, in either direction. */
export const WINDOW_S = 60;
/** Seconds a nonce is remembered: twice the window, so no replayable request outlives it. */
export const NONCE_MEMORY_S = 120;
/** The longest long poll in seconds. */
export const MAX_WAIT_S = 25;
/** Requests per signer per rate window. */
export const SIGNER_LIMIT = 120;
/** Sends per signer per rate window. */
export const SEND_LIMIT = 30;
/** Requests per client address per rate window. */
export const IP_LIMIT = 240;
/** Seconds in a rate window. */
export const RATE_WINDOW_S = 60;

const NODE_ID = /^[0-9a-f]{64}$/;
const NONCE = /^[0-9a-f]{32}$/;
const SIGNATURE = /^[0-9a-f]{128}$/;

export function isNodeIdText(text: string): boolean {
  return NODE_ID.test(text);
}

export function isNonceText(text: string): boolean {
  return NONCE.test(text);
}

export function isSignatureText(text: string): boolean {
  return SIGNATURE.test(text);
}

export function hexToBytes(text: string): Uint8Array {
  if (text.length % 2 !== 0 || !/^[0-9a-f]*$/.test(text)) throw new Error("not lowercase hex");
  const out = new Uint8Array(text.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(text.slice(2 * i, 2 * i + 2), 16);
  return out;
}

export function bytesToHex(bytes: Uint8Array): string {
  let out = "";
  for (const byte of bytes) out += byte.toString(16).padStart(2, "0");
  return out;
}

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return bytesToHex(new Uint8Array(digest));
}

/** The text a request signature covers. The body hash is lowercase hex. */
export function signedText(
  method: string,
  pathAndQuery: string,
  time: string,
  nonce: string,
  bodyHash: string,
): string {
  return `${VERSION_DOMAIN}\n${method}\n${pathAndQuery}\n${time}\n${nonce}\n${bodyHash}`;
}

/** A reply code, its HTTP status and nothing else: the service never echoes request data. */
export const STATUS: Record<string, number> = {
  bad_request: 400,
  bad_key: 400,
  bad_signature: 401,
  stale: 401,
  replay: 401,
  forbidden: 403,
  not_found: 404,
  method_not_allowed: 405,
  too_large: 413,
  rate_limited: 429,
  mailbox_full: 429,
  upgrade_required: 426,
};
