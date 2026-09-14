"""Version guards for PyTorch forwarding.

This module owns the checks spec "Version guards" and "The device worker" name: whether
the local PyTorch can forward at all, and whether a worker's PyTorch matches. It imports
nothing from PyTorch at module import, so a provider can ask before torch is known to exist.
"""

from __future__ import annotations

import re

from ...errors import UnsupportedMode

#: The oldest PyTorch whose extension points forwarding uses.
MINIMUM = (2, 1)


def _major_minor(version: str) -> tuple[int, int]:
    found = re.match(r"(\d+)\.(\d+)", version)
    if found is None:
        raise UnsupportedMode(f"cannot read a PyTorch version from {version!r}")
    return int(found.group(1)), int(found.group(2))


def check_torch_version(version: str) -> None:
    """Refuse a PyTorch older than ``MINIMUM``, naming both versions."""
    if _major_minor(version) < MINIMUM:
        raise UnsupportedMode(
            f"this process has PyTorch {version}, and host='local' forwards PyTorch "
            f"operators through extension points that need {MINIMUM[0]}.{MINIMUM[1]} or newer"
        )


def check_worker_version(*, local: str, remote: str) -> None:
    """Refuse a device worker whose PyTorch major.minor differs from this process."""
    if _major_minor(local) != _major_minor(remote):
        mine, theirs = _major_minor(local), _major_minor(remote)
        raise UnsupportedMode(
            f"the runtime has PyTorch {theirs[0]}.{theirs[1]} ({remote}) and this process has "
            f"{mine[0]}.{mine[1]} ({local}). ATen operator schemas change between minor "
            f"versions, so pin the same torch version in the project"
        )


def require_torch() -> None:
    """Refuse host='local' when PyTorch does not import here or is too old."""
    try:
        import torch
    except ImportError as exc:
        raise UnsupportedMode(
            f"host='local' forwards PyTorch operators, and PyTorch does not import in this "
            f"process: {exc}. Add torch to the project, or use host='remote'"
        ) from exc
    check_torch_version(torch.__version__)
