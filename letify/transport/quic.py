"""Finding the QUIC carrier letify ships, and the commands that drive it.

Owns where ``letify-quic`` is and how it is invoked. It does not own the punch that gives
it its endpoints; the rendezvous does, exactly as it does for a TCP punch. Spec "QUIC over
a punched UDP pair".
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

#: Where the wheel puts letify-core's binaries, beside the CUDA shim.
LIB_DIR = Path(__file__).resolve().parent.parent / "remoting" / "lib"
#: What cargo names the carrier on each platform.
CARRIER = "letify-quic.exe" if sys.platform == "win32" else "letify-quic"
#: Said when the wheel for this platform carries no binary, so the reason is one string.
MISSING = "letify-quic is not in this wheel"


def carrier_path() -> Path | None:
    """The carrier bundled in the wheel, then one on PATH, then None.

    None is not an error. A source install has no build product, and the strategy that
    needs it is skipped with ``MISSING`` rather than failing the connection.
    """
    bundled = LIB_DIR / CARRIER
    if bundled.is_file():
        return bundled
    found = shutil.which("letify-quic")
    return Path(found) if found else None


def serve_command(
    binary: str, *, bind: int, peer: tuple[str, int], token: str, forward: int
) -> list[str]:
    """The remote side: accept the connection and splice it to the SSH server."""
    return [
        binary,
        "serve",
        "--bind",
        str(bind),
        "--peer",
        f"{peer[0]}:{peer[1]}",
        "--token",
        token,
        "--forward",
        str(forward),
    ]


def connect_command(binary: str, *, bind: int, peer: tuple[str, int], token: str) -> str:
    """The user's side, as an SSH ``ProxyCommand``: the stream on standard input and output."""
    return (
        f"{binary} connect --bind {bind} --peer {peer[0]}:{peer[1]} --token {token}"
    )


__all__ = ["CARRIER", "LIB_DIR", "MISSING", "carrier_path", "connect_command", "serve_command"]
