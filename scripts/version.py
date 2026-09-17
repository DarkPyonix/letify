"""Read, check and set the one release version across every file that carries it.

Owns the version rule in docs/SPEC.md, Packaging, Versioning and releases: `pyproject.toml`
`[project].version` is the source of truth, and `letify/__init__.py`, the letify-core workspace
(`Cargo.toml` and the crate entries in `Cargo.lock`), and the VS Code extension (`package.json`
and `package-lock.json`, when `letify-ext/` exists) must equal it, as must a `v<version>` tag.
It does not build, tag or publish anything. Standard library only.

    python scripts/version.py show
    python scripts/version.py check [--tag v1.2.0]
    python scripts/version.py set 1.2.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?$")
CRATES = ("letify-wire", "letify-driver", "letify-agent")
INIT_RE = re.compile(r'^__version__ = "([^"]*)"$', re.MULTILINE)
PYPROJECT_RE = re.compile(r'(\[project\][^\[]*?^version = ")([^"]*)(")', re.MULTILINE | re.DOTALL)
CARGO_RE = re.compile(
    r'(\[workspace\.package\][^\[]*?^version = ")([^"]*)(")', re.MULTILINE | re.DOTALL
)


def _lock_re(crate: str) -> re.Pattern[str]:
    return re.compile(rf'(\[\[package\]\]\nname = "{crate}"\nversion = ")([^"]*)(")')


def read_versions(root: Path) -> dict[str, str]:
    """Return the version each file carries, keyed by a label naming the file."""
    found = {
        "pyproject.toml": tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "version"
        ],
    }
    match = INIT_RE.search((root / "letify" / "__init__.py").read_text())
    found["letify/__init__.py"] = match.group(1) if match else "<missing>"
    cargo = tomllib.loads((root / "letify-core" / "Cargo.toml").read_text())
    found["letify-core/Cargo.toml"] = cargo["workspace"]["package"]["version"]
    lock = (root / "letify-core" / "Cargo.lock").read_text()
    locked = {m.group(2) if (m := _lock_re(c).search(lock)) else "<missing>" for c in CRATES}
    found["letify-core/Cargo.lock"] = (
        locked.pop() if len(locked) == 1 else " / ".join(sorted(locked))
    )
    ext = root / "letify-ext"
    if (ext / "package.json").exists():
        found["letify-ext/package.json"] = json.loads((ext / "package.json").read_text())["version"]
    if (ext / "package-lock.json").exists():
        lock_json = json.loads((ext / "package-lock.json").read_text())
        found["letify-ext/package-lock.json"] = lock_json["version"]
        found['letify-ext/package-lock.json packages[""]'] = lock_json["packages"][""]["version"]
    return found


def check(root: Path, tag: str | None = None) -> list[str]:
    """Return one error per file, or tag, that disagrees with pyproject.toml. Empty means agreed."""
    found = read_versions(root)
    expected = found["pyproject.toml"]
    errors = [
        f"{label} has {value}, pyproject.toml has {expected}"
        for label, value in found.items()
        if value != expected
    ]
    if tag is not None and tag != f"v{expected}":
        errors.append(
            f"tag {tag} does not match pyproject.toml version {expected}, expected v{expected}"
        )
    return errors


def _sub(path: Path, pattern: re.Pattern[str], value: str) -> None:
    text, count = pattern.subn(lambda m: m.group(1) + value + m.group(3), path.read_text(), count=1)
    if count != 1:
        raise RuntimeError(f"no version field found in {path}")
    path.write_text(text)


def set_version(root: Path, value: str) -> None:
    """Write `value` into every file that carries the version."""
    if not SEMVER.match(value):
        raise ValueError(f"{value!r} is not a semantic version such as 1.2.0")
    _sub(root / "pyproject.toml", PYPROJECT_RE, value)
    init = root / "letify" / "__init__.py"
    init.write_text(INIT_RE.sub(f'__version__ = "{value}"', init.read_text(), count=1))
    _sub(root / "letify-core" / "Cargo.toml", CARGO_RE, value)
    for crate in CRATES:
        _sub(root / "letify-core" / "Cargo.lock", _lock_re(crate), value)
    ext = root / "letify-ext"
    for name in ("package.json", "package-lock.json"):
        path = ext / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        data["version"] = value
        if name == "package-lock.json":
            data["packages"][""]["version"] = value
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="print the version each file carries")
    check_cmd = commands.add_parser("check", help="fail when any file or the tag disagrees")
    check_cmd.add_argument("--tag", help="a tag such as v1.2.0 that must match")
    set_cmd = commands.add_parser("set", help="write one version into every file")
    set_cmd.add_argument("version")
    args = parser.parse_args(argv)

    if args.command == "show":
        for label, value in read_versions(args.root).items():
            print(f"{label}: {value}")
        return 0
    if args.command == "set":
        set_version(args.root, args.version)
        print(f"set {args.version}")
        return 0
    errors = check(args.root, args.tag)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print(f"ok: every file carries {read_versions(args.root)['pyproject.toml']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
