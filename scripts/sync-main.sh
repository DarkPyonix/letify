#!/usr/bin/env bash
# Publish develop onto main, without the working documents.
#
# main is what users see and what gets released, so it carries the code, the root README.md
# and the task guides, and nothing that only matters while the project is being built. A
# markdown file survives on main only if it is:
#
#   README.md        at the repository root
#   docs/<dir>/...   anywhere below a subdirectory of docs, such as docs/guide or docs/locales
#
# Every other markdown file is dropped: CLAUDE.md, PROJECT.md, docs/SPEC.md and the other files
# directly inside docs, and the READMEs under examples.
#
# The new main commit is built from develop's tree with git plumbing rather than by merging.
# A merge would conflict on every run, because main deletes files that develop keeps editing.
# The commit has both main and develop as parents, so the history still shows where it came
# from. Nothing in the working tree is touched and no branch is checked out.
#
# Usage:
#   scripts/sync-main.sh             build the commit and move main to it
#   scripts/sync-main.sh --dry-run   list what would be dropped and change nothing
#   scripts/sync-main.sh --push      also push main to origin
set -euo pipefail

SOURCE=develop
TARGET=main
REMOTE=origin
DRY_RUN=0
PUSH=0

for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        --push) PUSH=1 ;;
        -h | --help)
            sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"

fail() {
    echo "sync-main: $*" >&2
    exit 1
}

git rev-parse --verify --quiet "refs/heads/$SOURCE" >/dev/null || fail "no local $SOURCE branch"
git rev-parse --verify --quiet "refs/heads/$TARGET" >/dev/null || fail "no local $TARGET branch"

# Moving main under a worktree that has it checked out would leave that worktree holding the old
# files, so this is refused before anything is built.
if [ "$DRY_RUN" -eq 0 ] && [ "$(git symbolic-ref --quiet --short HEAD || true)" = "$TARGET" ]; then
    fail "$TARGET is checked out in this worktree. Switch to $SOURCE and run again."
fi

# Refuse to publish something that differs from what was pushed, because main is built from the
# branch and a local commit nobody reviewed would go out with it.
if git rev-parse --verify --quiet "refs/remotes/$REMOTE/$SOURCE" >/dev/null; then
    git fetch --quiet "$REMOTE" "$SOURCE" "$TARGET"
    if [ "$(git rev-parse "$SOURCE")" != "$(git rev-parse "$REMOTE/$SOURCE")" ]; then
        fail "$SOURCE differs from $REMOTE/$SOURCE. Push or pull it first."
    fi
    if [ "$(git rev-parse "$TARGET")" != "$(git rev-parse "$REMOTE/$TARGET")" ]; then
        fail "$TARGET differs from $REMOTE/$TARGET. Pull it first, so its history is not lost."
    fi
fi

# True for a markdown path that main keeps.
keeps() {
    case "$1" in
        README.md) return 0 ;;
        docs/*/*) return 0 ;;
        *) return 1 ;;
    esac
}

dropped=()
while IFS= read -r -d '' path; do
    case "${path,,}" in
        *.md) keeps "$path" || dropped+=("$path") ;;
    esac
done < <(git ls-tree -r -z --name-only "$SOURCE")

echo "Dropping ${#dropped[@]} markdown file(s) from $SOURCE:"
for path in "${dropped[@]}"; do
    echo "  $path"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run: nothing changed."
    exit 0
fi

index="$(mktemp)"
trap 'rm -f "$index"' EXIT

export GIT_INDEX_FILE="$index"
git read-tree "$SOURCE"
if [ "${#dropped[@]}" -gt 0 ]; then
    printf '%s\0' "${dropped[@]}" | git update-index --force-remove -z --stdin
fi
tree="$(git write-tree)"
unset GIT_INDEX_FILE

if [ "$tree" = "$(git rev-parse "$TARGET^{tree}")" ]; then
    echo "$TARGET already matches $SOURCE. Nothing to do."
    exit 0
fi

source_commit="$(git rev-parse --short "$SOURCE")"
commit="$(
    git commit-tree "$tree" -p "$TARGET" -p "$SOURCE" \
        -m "Chore: Sync main with develop at $source_commit"
)"

git update-ref "refs/heads/$TARGET" "$commit" "$(git rev-parse "$TARGET")"
echo "$TARGET is now $(git rev-parse --short "$commit"), built from $SOURCE at $source_commit."

if [ "$PUSH" -eq 1 ]; then
    git push "$REMOTE" "$TARGET"
else
    echo "Not pushed. Run: git push $REMOTE $TARGET"
fi
