"""Build letify-core and the agent, and put them where letify looks.

The library has to be installed under the name of the one it replaces, because being found before the real
driver is the entire mechanism. Cargo cannot emit those names directly, so this copies the
artifact into place under the right one.

    Linux and WSL2   libcuda.so.1   reached with LD_PRELOAD
    Windows          nvcuda.dll     reached by putting its directory first in the
                                    loader's search order

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

#: What the library must be called on each platform, and what cargo produces.
ARTIFACTS = {
    "Windows": ("letify_driver.dll", "nvcuda.dll"),
    "Linux": ("libletify_driver.so", "libcuda.so.1"),
    "Darwin": ("libletify_driver.dylib", "libletify_driver.dylib"),
}

AGENTS = {
    "Windows": "letify-agent.exe",
    "Linux": "letify-agent",
    "Darwin": "letify-agent",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="build without optimizations")
    parser.add_argument(
        "--target-dir", default=None, help="where letify should find letify-core"
    )
    args = parser.parse_args(argv)

    system = platform.system()
    if system not in ARTIFACTS:
        print(f"letify: {system} is not a supported platform for the shim", file=sys.stderr)
        return 1

    if shutil.which("cargo") is None:
        print(
            "letify: cargo is not on PATH. Install Rust from https://rustup.rs to build "
            "letify-core, or use host='remote' and skip it entirely.",
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

    source_name, install_name = ARTIFACTS[system]
    source = built / source_name
    if not source.is_file():
        print(f"letify: cargo did not produce {source}", file=sys.stderr)
        return 1
    shutil.copy2(source, destination / install_name)
    print(f"letify: letify-core installed at {destination / install_name}")

    agent = built / AGENTS[system]
    if agent.is_file():
        shutil.copy2(agent, destination / agent.name)
        print(f"letify: agent installed at {destination / agent.name}")

    print()
    print("The agent runs on the machine with the GPU:")
    print(f"    {agent.name} 0.0.0.0:7654")
    print()
    print("letify starts it for you when a declaration asks for host='local'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
