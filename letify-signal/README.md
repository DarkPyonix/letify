# letify-signal

The letify signalling service: a Cloudflare Worker with one Durable Object mailbox per node id. It carries signed bootstrap lines to a machine that has no address, so letify reaches it with no ssh and no Tailcat. The protocol, the limits and the trust model are in `docs/SPEC.md`, section "Signalling service". The Python side is `letify/transport/linksignal.py`.

The service is untrusted for content. Every line is signed by the sending node and verified by the receiving node, and the receiving agent never runs a command that a line supplies.

## Layout

| File | Holds |
|---|---|
| `src/index.ts` | The Worker entry point. Binds the routes to the Durable Objects |
| `src/handler.ts` | The routes, the order of the checks and the reply codes |
| `src/auth.ts` | Authentication of one signed request |
| `src/ed25519.ts` | Canonical key and small-order checks, and signature verification with WebCrypto |
| `src/mailbox.ts` | The messages of one node: numbering, depth, acknowledgement, expiry |
| `src/limiter.ts` | Nonce replay refusal and the per-signer and per-IP rate limits |
| `src/objects.ts` | The `Mailbox` and `Limiter` Durable Object classes |
| `test/` | vitest tests of every module above |

## Test

```
cd letify-signal
npm install
npm test
npm run typecheck
```

The tests run under Node 18 or later and need no Cloudflare account. They cover the logic modules with an in-memory storage. The Durable Object classes in `src/objects.ts` (storage, alarm, long-poll wait, WebSocket Hibernation) are thin shells over those modules and are checked only by `wrangler dev` and by a deployed health check, because `@cloudflare/vitest-pool-workers` needs Node 22 for wrangler.

## Deploy (the owner does this)

1. Use Node 22 or later (wrangler 4 needs it) and log in: `npx wrangler login`.
2. Edit `wrangler.toml`: set `account_id` to the account that owns `darkpyonix.dev`. The route `letify.darkpyonix.dev` is a custom domain, which needs `darkpyonix.dev` to be a zone of that account. For a route on an existing zone, use the `routes` line in the comment of that file.
3. Deploy: `npm run deploy`. The first deploy applies migration `v1`, which creates the two SQLite-backed Durable Object classes (they work on the free plan).
4. Check: `curl https://letify.darkpyonix.dev/health` must print `{"ok":true}`.
5. Check a signed round trip with `tests/test_link_signal.py` pointed at the service. Set `LETIFY_SIGNAL_URL=https://letify.darkpyonix.dev` and run `pytest -q tests/test_link_signal.py -k live` (the live tests are skipped without the variable).

No secret, token or API key is needed: the service authenticates nodes by their own ed25519 keys. Do not add one.

## Operating notes

- Cost. A connect sends about 20 messages. A long poll holds a Durable Object awake for up to 25 s, so the duration charge grows with open polls, not with messages. An idle `letify agent serve` therefore waits on `/ws/<node>`, where the object hibernates, because one agent long polling all day would use about 86,400 s of the roughly 104,000 s per day (13,000 GB-s at 128 MB) that the free plan allows. A connect's short waits use `/recv`.
- Limits are constants in `src/protocol.ts` (body 4096 bytes, time to live 60 s, depth 32, 120 requests and 30 sends per signer per minute, 240 requests per IP per minute). Changing one does not change the wire protocol.
- The service logs the hour, the method, the reply code and the body length. `observability` is off in `wrangler.toml`. Do not turn on request logging for the zone, because it would record node ids in paths.
- Rolling back is `npx wrangler rollback`. Stored messages live at most 60 s, so nothing needs migrating.
