// Ed25519 node keys and signatures. WebCrypto verifies the signature; this module adds the
// checks that WebCrypto does not make and that letify-link makes in identity.rs: a node id must
// be the canonical encoding of a point that has no small order and lies in the prime order
// subgroup. The arithmetic is BigInt and runs only on a signer's key, once per request.

const P = (1n << 255n) - 19n;
const L = (1n << 252n) + 27742317777372353535851937790883648493n;
const D = mod(-121665n * invert(121666n));
const SQRT_M1 = pow(2n, (P - 1n) / 4n);

function mod(a: bigint): bigint {
  const r = a % P;
  return r >= 0n ? r : r + P;
}

function pow(base: bigint, exponent: bigint): bigint {
  let result = 1n;
  let b = mod(base);
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % P;
    b = (b * b) % P;
    e >>= 1n;
  }
  return result;
}

function invert(a: bigint): bigint {
  return pow(mod(a), P - 2n);
}

/** A point in extended coordinates (X, Y, Z, T) with x = X/Z, y = Y/Z, xy = T/Z. */
export interface Point {
  x: bigint;
  y: bigint;
  z: bigint;
  t: bigint;
}

const IDENTITY: Point = { x: 0n, y: 1n, z: 1n, t: 0n };

export function addPoints(a: Point, b: Point): Point {
  const A = mod((a.y - a.x) * (b.y - b.x));
  const B = mod((a.y + a.x) * (b.y + b.x));
  const C = mod(a.t * 2n * D * b.t);
  const E2 = mod(a.z * 2n * b.z);
  const E = mod(B - A);
  const F = mod(E2 - C);
  const G = mod(E2 + C);
  const H = mod(B + A);
  return { x: mod(E * F), y: mod(G * H), t: mod(E * H), z: mod(F * G) };
}

function multiply(point: Point, scalar: bigint): Point {
  let result = IDENTITY;
  let addend = point;
  let k = scalar;
  while (k > 0n) {
    if (k & 1n) result = addPoints(result, addend);
    addend = addPoints(addend, addend);
    k >>= 1n;
  }
  return result;
}

function isIdentity(point: Point): boolean {
  return mod(point.x) === 0n && mod(point.y - point.z) === 0n;
}

function littleEndian(bytes: Uint8Array): bigint {
  let n = 0n;
  for (let i = bytes.length - 1; i >= 0; i--) n = (n << 8n) | BigInt(bytes[i]);
  return n;
}

/** Decode a point, or null when the bytes are not the canonical encoding of a curve point. */
export function decodePoint(bytes: Uint8Array): Point | null {
  if (bytes.length !== 32) return null;
  const sign = bytes[31] >> 7;
  const masked = bytes.slice();
  masked[31] &= 0x7f;
  const y = littleEndian(masked);
  if (y >= P) return null;
  const y2 = mod(y * y);
  const u = mod(y2 - 1n);
  const v = mod(D * y2 + 1n);
  let x = mod(u * pow(v, 3n) * pow(u * pow(v, 7n), (P - 5n) / 8n));
  const check = mod(v * x * x);
  if (check === u) {
    // x is a root.
  } else if (check === mod(-u)) {
    x = mod(x * SQRT_M1);
  } else {
    return null;
  }
  if (x === 0n && sign === 1) return null;
  if (Number(x & 1n) !== sign) x = mod(-x);
  return { x, y, z: 1n, t: mod(x * y) };
}

export function encodePoint(point: Point): Uint8Array {
  const zinv = invert(point.z);
  const x = mod(point.x * zinv);
  const y = mod(point.y * zinv);
  const out = new Uint8Array(32);
  for (let i = 0; i < 32; i++) out[i] = Number((y >> BigInt(8 * i)) & 0xffn);
  out[31] |= Number(x & 1n) << 7;
  return out;
}

/** The base point B of the group. */
export const BASE: Point = (() => {
  const y = mod(4n * invert(5n));
  const bytes = new Uint8Array(32);
  for (let i = 0; i < 32; i++) bytes[i] = Number((y >> BigInt(8 * i)) & 0xffn);
  return decodePoint(bytes)!;
})();

function hasSmallOrder(point: Point): boolean {
  return isIdentity(multiply(point, 8n));
}

/** Why `bytes` is not a node id, or "ok". The reasons are the ones identity.rs gives. */
export function checkNodeKey(bytes: Uint8Array): string {
  if (bytes.length !== 32) return "not 32 bytes";
  const point = decodePoint(bytes);
  if (point === null) {
    // Tell a point off the curve from an encoding that is not the canonical one.
    return looksReducible(bytes) ? "not the canonical encoding" : "not a curve point";
  }
  if (hasSmallOrder(point)) return "small order";
  if (!isIdentity(multiply(point, L))) return "not in the prime order subgroup";
  return "ok";
}

function looksReducible(bytes: Uint8Array): boolean {
  const masked = bytes.slice();
  masked[31] &= 0x7f;
  if (littleEndian(masked) >= P) return true;
  return false;
}

/** Verify `signature` (128 lowercase hex) by node `nodeHex` over `message`. */
export async function verifySignature(
  nodeHex: string,
  message: Uint8Array,
  signatureHex: string,
): Promise<boolean> {
  if (!/^[0-9a-f]{64}$/.test(nodeHex) || !/^[0-9a-f]{128}$/.test(signatureHex)) return false;
  const key = fromHex(nodeHex);
  if (checkNodeKey(key) !== "ok") return false;
  const signature = fromHex(signatureHex);
  // S must be below the group order, or S + L would be a second valid signature.
  if (littleEndian(signature.slice(32)) >= L) return false;
  // R must be a canonical point of more than small order, as verify_strict requires.
  const r = decodePoint(signature.slice(0, 32));
  if (r === null || hasSmallOrder(r)) return false;
  try {
    const imported = await crypto.subtle.importKey("raw", key, { name: "Ed25519" }, false, ["verify"]);
    return await crypto.subtle.verify({ name: "Ed25519" }, imported, signature, message);
  } catch {
    return false;
  }
}

function fromHex(text: string): Uint8Array {
  const out = new Uint8Array(text.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(text.slice(2 * i, 2 * i + 2), 16);
  return out;
}
