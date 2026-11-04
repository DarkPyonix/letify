// The letify signalling service, built and deployed with the Cloudflare CLI (cf).
// The account comes from CLOUDFLARE_ACCOUNT_ID or `cf auth login`, so no id is written here.
import { bindings, defineConfig, exports } from "cf/config";

export default defineConfig({
	worker: {
		name: "letify-signal",
		compatibilityDate: "2025-09-01",
		entrypoint: "src/index.ts",
		workersDev: false,
		// The service logs no message content and no node id, and Workers logs stay off.
		observability: {
			enabled: false,
		},
		domains: ["letify.darkpyonix.dev"],
		env: {
			MAILBOX: bindings.durableObject({
				worker: "letify-signal",
				exportName: "Mailbox",
			}),
			LIMITER: bindings.durableObject({
				worker: "letify-signal",
				exportName: "Limiter",
			}),
		},
		// Both classes keep their state in SQLite-backed storage, which works on the free plan.
		exports: {
			Mailbox: exports.durableObject({ storage: "sqlite" }),
			Limiter: exports.durableObject({ storage: "sqlite" }),
		},
	},
});
