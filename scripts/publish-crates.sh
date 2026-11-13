#!/usr/bin/env bash
# Publish the crates letify-link and letify to crates.io, in that order, at the workspace version.
#
# Owns the choice of which crates still need an upload. It does not decide whether a release is
# allowed: the gates are in .github/workflows/publish-crates.yml. See docs/SPEC.md, Versioning
# and releases.
#
#   scripts/publish-crates.sh --dry-run   package and build what would be published
#   scripts/publish-crates.sh             upload it (needs CARGO_REGISTRY_TOKEN)
#
# A crate whose version is already on crates.io is left out, so a rerun after a partial release
# publishes only what is missing and a finished release publishes nothing. One `cargo publish`
# carries the whole list, so cargo uploads letify-link before letify and waits for the index.
set -euo pipefail

dry=()
case "${1:-}" in
  --dry-run) dry=(--dry-run) ;;
  "") ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../letify-core"
version="$(cargo metadata --no-deps --format-version 1 --locked |
  python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"]=="letify"))')"

todo=()
for crate in letify-link letify; do
  status="$(curl -s -o /dev/null -w '%{http_code}' -A 'letify-release (https://github.com/DarkPyonix/letify)' "https://crates.io/api/v1/crates/${crate}/${version}")"
  case "$status" in
    200) echo "${crate} ${version} is already on crates.io, skipping it." ;;
    404) todo+=(-p "$crate") ;;
    *) echo "crates.io answered ${status} for ${crate} ${version}" >&2; exit 1 ;;
  esac
done

if [ "${#todo[@]}" -eq 0 ]; then
  echo "Nothing to publish."
  exit 0
fi
if [ "${#dry[@]}" -eq 0 ] && [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "CARGO_REGISTRY_TOKEN is empty." >&2
  echo "Add a crates.io API token as the secret CARGO_REGISTRY_TOKEN in the crates-io environment." >&2
  exit 1
fi
cargo publish "${todo[@]}" --locked "${dry[@]}"
