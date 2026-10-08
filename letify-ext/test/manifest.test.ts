import { describe, expect, it } from "vitest";
import manifest from "../package.json";

// Spec: Editor extension. The publisher has to match the account a release is pushed from,
// or vsce refuses the upload with "Publisher ID ... should match the publisher ID ...".
describe("the extension manifest", () => {
  it("names the publisher the repository belongs to", () => {
    expect(manifest.publisher).toBe("darkpyonix");
  });

  it("carries what the Marketplace listing needs", () => {
    expect(manifest.description).toBeTruthy();
    expect(manifest.license).toBeTruthy();
    expect(manifest.repository.url).toContain("darkpyonix/letify");
  });
});
