# Releasing

A release is one version number pushed as the tag `v<version>`. GitHub Actions builds everything and attaches it to a GitHub Release.

## The version rule

`pyproject.toml` `[project].version` is the source of truth. `letify/__init__.py`, `letify-core/Cargo.toml` and `Cargo.lock`, and `letify-ext/package.json` and `package-lock.json` carry the same value. Never edit them by hand:

```
python scripts/version.py show            # the version in each file
python scripts/version.py set 1.2.0       # write 1.2.0 into every file
python scripts/version.py check           # exit 1 when a file disagrees
python scripts/version.py check --tag v1.2.0
```

The `ci` workflow runs `check` on every push and pull request.

## Steps

1. On `develop`, with the tests passing, set the version and commit it:

   ```
   python scripts/version.py set 1.2.0
   git commit -am "Chore: Set the version to 1.2.0"
   ```

2. Bring the commit to `main` as usual, then tag that commit and push the tag:

   ```
   git tag v1.2.0
   git push origin v1.2.0
   ```

3. The `publish` workflow runs. It fails first if the tag and the files disagree. Otherwise it builds:
   - the sdist,
   - six platform wheels carrying the letify-core binaries,
   - `letify-ext-1.2.0.vsix`, after the extension's unit tests pass.

4. It creates the GitHub Release `v1.2.0` with all of those files attached, and uploads the sdist and wheels to PyPI through trusted publishing.

A failed run publishes nothing to the Release. Fix the cause, delete the tag with `git push origin :v1.2.0`, and tag again.

## Installing the extension from a release

Download the `.vsix` from the Release page, then run `code --install-extension letify-ext-1.2.0.vsix`. The extension is not on the VS Code Marketplace. Publishing there would need a `vsce publish` step and a Marketplace token stored as a repository secret.
