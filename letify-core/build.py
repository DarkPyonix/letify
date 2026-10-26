"""Build letify-quic and put it where letify looks.

The wheels carry the QUIC carrier binary in ``letify/remoting/lib/``. The spec's "QUIC over a
punched UDP pair" describes what it does, and ``letify/transport/quic.py`` finds it there.
This script builds the workspace and copies the binary into place.

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

#: The QUIC carrier each platform's wheel ships. Spec "QUIC over a punched UDP pair": the
#: remote fetches the same wheel to get its own copy.
CARRIERS = {
    "Windows": "letify-quic.exe",
    "Linux": "letify-quic",
    "Darwin": "letify-quic",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="build without optimizations")
    parser.add_argument(
        "--target-dir", default=None, help="where letify should find letify-quic"
    )
    args = parser.parse_args(argv)

    system = platform.system()
    if system not in CARRIERS:
        print(f"letify: {system} is not a supported platform for letify-quic", file=sys.stderr)
        return 1

    if shutil.which("cargo") is None:
        print(
            "letify: cargo is not on PATH. Install Rust from https://rustup.rs to build "
            "letify-quic, or skip it: the QUIC strategy is then left out.",
            file=sys.stderr,
        )
        return 1

    profile = "debug" if args.debug else "release"
    command = ["cargo", "build"] + ([] if args.debug else ["--release"])
    result = subprocess.run(command, cwd=HERE)
    if result.returncode != 0:
        return result.returncode

    built = HERE / "target" / profile
    destination = Path(args.target_dir) if args.target_dir else ROOT / "letify" / "remoting" / "lib"
    destination.mkdir(parents=True, exist_ok=True)

    carrier = built / CARRIERS[system]
    if not carrier.is_file():
        print(f"letify: cargo did not produce {carrier}", file=sys.stderr)
        return 1
    shutil.copy2(carrier, destination / carrier.name)
    print(f"letify: QUIC carrier installed at {destination / carrier.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
