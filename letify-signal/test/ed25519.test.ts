import { describe, expect, it } from "vitest";

import { BASE, addPoints, checkNodeKey, decodePoint, encodePoint, verifySignature } from "../src/ed25519";
import { bytesToHex, hexToBytes } from "../src/protocol";
import { newNode } from "./helpers";

// RFC 8032 section 7.1, test 1.
const RFC_PUBLIC = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
const RFC_SIGNATURE =
  "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b";

describe("node keys", () => {
  it("accepts the key of a real node", async () => {
    const node = await newNode();
    expect(checkNodeKey(hexToBytes(node.id))).toBe("ok");
    expect(checkNodeKey(hexToBytes(RFC_PUBLIC))).toBe("ok");
  });

  it("refuses the identity and every other small order point", () => {
    const p = (1n << 255n) - 19n;
    const encode = (y: bigint, sign: number): Uint8Array => {
      const out = new Uint8Array(32);
      for (let i = 0; i < 32; i++) out[i] = Number((y >> BigInt(8 * i)) & 0xffn);
      out[31] |= sign << 7;
      return out;
    };
    const order8 = 0x7a03ac9277fdc74ec6cc392cfa53202a0f67100d760b3cba4fd84d3d706a17c7n;
    const smallOrder = [
      encode(1n, 0), // the identity
      encode(p - 1n, 0), // order 2
      encode(0n, 0), // order 4
      encode(0n, 1),
      encode(order8, 0), // order 8
      encode(order8, 1),
      encode(p - order8, 0),
      encode(p - order8, 1),
    ];
    for (const key of smallOrder) expect(checkNodeKey(key)).toBe("small order");
  });

  it("refuses a y coordinate that is not reduced below 2^255 - 19", () => {
    // y = 1 + p encodes the same point as y = 1 and is not the canonical encoding.
    const p = (1n << 255n) - 19n;
    const y = 1n + p;
    const bytes = new Uint8Array(32);
    for (let i = 0; i < 32; i++) bytes[i] = Number((y >> BigInt(8 * i)) & 0xffn);
    expect(checkNodeKey(bytes)).not.toBe("ok");
  });

  it("refuses a point that is not on the curve", () => {
    // y = 2 has no x on the curve.
    const bytes = new Uint8Array(32);
    bytes[0] = 2;
    expect(checkNodeKey(bytes)).toBe("not a curve point");
  });

  it("refuses minus zero, the sign bit set on x = 0", () => {
    const bytes = new Uint8Array(32);
    bytes[0] = 1;
    bytes[31] = 0x80;
    expect(checkNodeKey(bytes)).not.toBe("ok");
  });

  it("refuses a point outside the prime order subgroup", () => {
    // The base point plus a point of order 8: on the curve, canonical, not of small order.
    const order8 = hexToBytes(
      "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
    );
    const torsion = decodePoint(order8);
    expect(torsion).not.toBeNull();
    const mixed = encodePoint(addPoints(BASE, torsion!));
    expect(checkNodeKey(mixed)).toBe("not in the prime order subgroup");
    expect(checkNodeKey(encodePoint(BASE))).toBe("ok");
  });

  it("refuses a length other than 32 bytes", () => {
    expect(checkNodeKey(new Uint8Array(31))).toBe("not 32 bytes");
  });
});

describe("signatures", () => {
  it("verifies the RFC 8032 vector over the empty message", async () => {
    expect(await verifySignature(RFC_PUBLIC, new Uint8Array(), RFC_SIGNATURE)).toBe(true);
  });

  it("verifies a signature of a generated node and refuses another message", async () => {
    const node = await newNode();
    const message = new TextEncoder().encode("hello");
    const signature = bytesToHex(await node.sign(message));
    expect(await verifySignature(node.id, message, signature)).toBe(true);
    expect(await verifySignature(node.id, new TextEncoder().encode("hellO"), signature)).toBe(false);
  });

  it("refuses an S at or above the group order, though S + L is the same scalar", async () => {
    const L = (1n << 252n) + 27742317777372353535851937790883648493n;
    const raw = hexToBytes(RFC_SIGNATURE);
    let s = 0n;
    for (let i = 31; i >= 0; i--) s = (s << 8n) | BigInt(raw[32 + i]);
    const malleable = s + L;
    const out = raw.slice();
    for (let i = 0; i < 32; i++) out[32 + i] = Number((malleable >> BigInt(8 * i)) & 0xffn);
    expect(await verifySignature(RFC_PUBLIC, new Uint8Array(), bytesToHex(out))).toBe(false);
  });

  it("refuses a signature whose R is a small order point", async () => {
    const raw = hexToBytes(RFC_SIGNATURE);
    const out = raw.slice();
    out.fill(0, 0, 32);
    out[0] = 1; // R is the identity
    expect(await verifySignature(RFC_PUBLIC, new Uint8Array(), bytesToHex(out))).toBe(false);
  });

  it("refuses a signature of the wrong length or case", async () => {
    expect(await verifySignature(RFC_PUBLIC, new Uint8Array(), RFC_SIGNATURE.slice(2))).toBe(false);
    expect(await verifySignature(RFC_PUBLIC, new Uint8Array(), RFC_SIGNATURE.toUpperCase())).toBe(false);
  });

  it("refuses a signature by a node whose key is not canonical", async () => {
    const identity = "01" + "00".repeat(31);
    expect(await verifySignature(identity, new Uint8Array(), "00".repeat(64))).toBe(false);
  });
});
