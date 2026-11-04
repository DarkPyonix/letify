import { describe, expect, it } from "vitest";

import { authenticate } from "../src/auth";

// A request signed by letify-link's sidecar (ed25519-dalek) from the key file "ab" x 32, made
// with `LinkDaemon.sign("letify-signal v1", ...)` of letify/transport/linkd.py.
const vector = {
  node: "248acbdbaf9e050196de704bea2d68770e519150d103b587dae2d9cad53dd930",
  path: "/v1/send/248acbdbaf9e050196de704bea2d68770e519150d103b587dae2d9cad53dd930",
  time: 1800000000,
  nonce: "0123456789abcdef0123456789abcdef",
  body: '{"hello":"signal"}',
  sig: "c19469202f50b8c9fe10576954d079a89fa096f81fe93a2909a311b73dac445c8e2307fc6515a5f572010b8609d02c0dc150ddfaa1d0759bcc673e73d1d8d202",
};

function input(over: Partial<{ body: string; now: number }> = {}) {
  const headers: Record<string, string> = {
    "X-Letify-Node": vector.node,
    "X-Letify-Time": String(vector.time),
    "X-Letify-Nonce": vector.nonce,
    "X-Letify-Signature": vector.sig,
  };
  return {
    method: "POST",
    pathAndQuery: vector.path,
    header: (name: string) => headers[name] ?? null,
    body: new TextEncoder().encode(over.body ?? vector.body),
    now: over.now ?? vector.time,
  };
}

describe("a request signed by the Rust sidecar", () => {
  it("is accepted by the Worker", async () => {
    expect(await authenticate(input())).toMatchObject({ ok: true, node: vector.node });
  });

  it("is refused when the body differs by one byte", async () => {
    expect(await authenticate(input({ body: '{"hello":"signaL"}' }))).toMatchObject({
      ok: false,
      code: "bad_signature",
    });
  });
});
