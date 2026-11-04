// The Worker entry point: it binds the routes of handler.ts to the Durable Objects.

import { type Deps, handle } from "./handler";
import type { Limiter, Mailbox } from "./objects";

export { Limiter, Mailbox } from "./objects";

interface Env {
  MAILBOX: DurableObjectNamespace<Mailbox>;
  LIMITER: DurableObjectNamespace<Limiter>;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const deps: Deps = {
      now: () => Math.floor(Date.now() / 1000),
      limiter: (name) => env.LIMITER.get(env.LIMITER.idFromName(name)),
      mailbox: (node) => {
        const stub = env.MAILBOX.get(env.MAILBOX.idFromName(node));
        return {
          put: (from, body, now) => stub.put(from, body, now),
          read: (after, wait, now) => stub.read(after, wait, now),
          connect: (upgrade) => stub.fetch(upgrade),
        };
      },
      log: (line) => console.log(line),
    };
    return handle(request, deps);
  },
};
