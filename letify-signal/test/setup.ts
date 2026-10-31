import { webcrypto } from "node:crypto";

// Node 18 has no global crypto. Workers and Node 20 and later do.
if (!(globalThis as { crypto?: unknown }).crypto) {
  Object.defineProperty(globalThis, "crypto", { value: webcrypto });
}
