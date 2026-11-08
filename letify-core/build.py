"""Build letify-link and put it where letify looks.

The wheels carry the letify-link sidecar in ``letify/remoting/lib/``. The spec's "Sidecar
interface" describes what it does, built with the Cargo feature ``relay``, and
``letify/transport/linkd.py`` finds it there. This script builds the workspace and copies
the binary into place.

Run it from the repository root or from here:

    python letify-core/build.py
    python letify-core/build.py --debug
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: The letify-link sidecar each platform's wheel ships.
SIDECARS = {
    "Windows": "letify-link.exe",
    "Linux": "letify-link",
    "Darwin": "letify-link",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="build without optimizations")
    parser.add_argument(
        "--target-dir", default=None, help="where letify should find letify-link"
    )
    args = parser.parse_args(argv)

    system = platform.system()
    if system not in SIDECARS:
        print(f"letify: {system} is not a supported platform for letify-link", file=sys.stderr)
        return 1

    if shutil.which("cargo") is None:
        print(
            "letify: cargo is not on PATH. Install Rust from https://rustup.rs to build "
            "letify-link.",
            file=sys.stderr,
        )
        return 1

    profile = "debug" if args.debug else "release"
    command = ["cargo", "build", "--workspace", "--features", "letify-link/relay"] + (
        [] if args.debug else ["--release"]
    )
    result = subprocess.run(command, cwd=HERE)
    if result.returncode != 0:
        return result.returncode

    built = HERE / "target" / profile
    destination = Path(args.target_dir) if args.target_dir else ROOT / "letify" / "remoting" / "lib"
    destination.mkdir(parents=True, exist_ok=True)

    sidecar = built / SIDECARS[system]
    if not sidecar.is_file():
        print(f"letify: cargo did not produce {sidecar}", file=sys.stderr)
        return 1
    shutil.copy2(sidecar, destination / sidecar.name)
    print(f"letify: letify-link installed at {destination / sidecar.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
