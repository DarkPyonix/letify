"""Turning device names into attribute-friendly labels.

Providers report accelerators in their own spelling, and letify exposes them as
attributes, so the names have to be normalized in one place. ``colab.G4`` and
``lab.RTX_PRO_6000`` should be reachable without the user knowing which vendor
string produced them.
"""

from __future__ import annotations

#: Suffixes that describe a package or a board rather than the chip.
_TRIM = (
    "-SXM",
    "-PCIE",
    " SXM",
    " PCIe",
    " Blackwell",
    " Laptop",
    " Ada Generation",
)


def normalize_gpu(raw: str) -> str:
    """Turn a vendor product name into a short label usable as an attribute.

    ``NVIDIA RTX PRO 6000 Blackwell`` becomes ``RTX_PRO_6000`` and
    ``NVIDIA A100-SXM4-80GB`` becomes ``A100``.
    """
    text = raw.strip()
    for vendor in ("NVIDIA", "Nvidia", "AMD", "Intel"):
        text = text.removeprefix(vendor).strip()
    for marker in _TRIM:
        index = text.find(marker)
        if index > 0:
            text = text[:index]
    return text.strip().replace(" ", "_").replace("-", "_")


def gib_from_mib(raw: str) -> int | None:
    """Read a memory size reported in mebibytes and return whole gibibytes."""
    digits = "".join(character for character in raw if character.isdigit())
    if not digits:
        return None
    return round(int(digits) / 1024)


__all__ = ["gib_from_mib", "normalize_gpu"]
