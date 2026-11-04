import { defineConfig } from "vitest/config";

// The tests cover the pure logic modules under Node. The Durable Object classes are thin
// wrappers around cores that take a storage interface, so the tests give them a Map.
export default defineConfig({
  test: { include: ["test/**/*.test.ts"], environment: "node", setupFiles: ["test/setup.ts"] },
});
