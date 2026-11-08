# letify-signal

The letify signalling service: a Cloudflare Worker with one Durable Object mailbox per node id. It carries signed bootstrap lines to a machine that has no address, so letify reaches it with no ssh. The protocol, the limits and the trust model are in `docs/SPEC.md`, section "Signalling service". The Python side is `letify/transport/linksignal.py`.

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

The tests run under Node 18 or later and need no Cloudflare account. They cover the logic modules with an in-memory storage. The Durable Object classes in `src/objects.ts` (storage, alarm, long-poll wait, WebSocket Hibernation) are thin shells over those modules and are checked only by `cf dev` and by a deployed health check, because `@cloudflare/vitest-pool-workers` needs Node 22 for wrangler.

## Deploy (the owner does this)

The Worker is built and deployed with `cf`, the Cloudflare CLI (beta), from `cloudflare.config.ts`. `cf build` still bundles through wrangler, which stays a dev dependency.

1. Use Node 22.18 or later and run `npm install`.
2. Give `cf` credentials, either `cf auth login` or the variables `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from a file that is not tracked, such as the project's `.letify/cf.env`, which `.gitignore` covers. A token needs Workers Scripts Edit on the account and Workers Routes Edit on the `darkpyonix.dev` zone. `cloudflare.config.ts` names no account.
3. Check: `npx cf deploy --dry-run` lists the bindings `MAILBOX` and `LIMITER` without uploading.
4. Deploy: `npm run deploy`. The `exports` block declares both Durable Object classes with SQLite storage, which the first deploy creates (they work on the free plan). The custom domain `letify.darkpyonix.dev` needs `darkpyonix.dev` to be a zone of the same account.
5. Check: `curl https://letify.darkpyonix.dev/health` must print `{"ok":true}`.
6. Check a signed round trip with `tests/test_link_signal.py` pointed at the service. Set `LETIFY_SIGNAL_URL=https://letify.darkpyonix.dev` and run `pytest -q tests/test_link_signal.py -k live` (the live tests are skipped without the variable).

Clients need no secret, token or API key: the service authenticates nodes by their own ed25519 keys. Do not add one. The Cloudflare token above is for deploying only.

## Operating notes

- Cost. A connect sends about 20 messages. A long poll holds a Durable Object awake for up to 25 s, so the duration charge grows with open polls, not with messages. An idle `letify agent serve` therefore waits on `/ws/<node>`, where the object hibernates, because one agent long polling all day would use about 86,400 s of the roughly 104,000 s per day (13,000 GB-s at 128 MB) that the free plan allows. A connect's short waits use `/recv`. The keepalive is the text `ping` with an auto-response `pong`, so it does not wake the object either.
- Limits are constants in `src/protocol.ts` (body 4096 bytes, time to live 60 s, depth 32, 120 requests and 30 sends per signer per minute, 240 requests per IP per minute). Changing one does not change the wire protocol.
- The service logs the hour, the method, the reply code and the body length. `observability` is off in `cloudflare.config.ts`. Do not turn on request logging for the zone, because it would record node ids in paths.
- Rolling back is deploying an earlier version from the Worker's Deployments page in the Cloudflare dashboard. Stored messages live at most 60 s, so nothing needs migrating.
