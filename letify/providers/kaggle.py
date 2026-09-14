"""Kaggle, one Kaggle account reached through the official Kaggle CLI run by uv.

This module owns the account's accelerator list and its remaining weekly quota, read from
``kaggle quota --format json``. It does not own the login, which is in
``letify.config.login``, and it opens no tunnel or port forward of any kind, because the
Kaggle Acceptable Use Policy forbids circumvention tools.

The session channel over a Kaggle Jupyter Server is not part of this module yet, so
starting a session raises ``UnsupportedMode`` rather than falling back to anything.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .. import tools
from ..config.secrets import account_directory
from ..declare.instance import Host, Instance
from ..errors import ProviderUnavailable, RuntimeFailure, UnsupportedMode
from .base import Provider
from .usage import Usage

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: Accelerators a Kaggle session can be started with, and the memory of one card.
GPUS = {"P100": {"vram_gb": 16}, "T4": {"vram_gb": 16}}
TPUS = ("TPU_V3_8",)

#: The read-only call that answers the weekly quota.
QUOTA = ("quota", "--format", "json")


def hours(value: Any) -> float | None:
    """Read a figure such as ``3.25h`` as hours."""
    text = str(value or "").strip().removesuffix("h").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_quota(output: str) -> dict[str, dict[str, Any]]:
    """The quota rows keyed by resource, from output that may carry warnings before the JSON."""
    start = output.find("[")
    if start < 0:
        return {}
    try:
        rows = json.loads(output[start:])
    except ValueError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("resource")).upper(): row
        for row in rows
        if isinstance(row, dict) and row.get("resource")
    }


class Kaggle(Provider):
    """One Kaggle account."""

    kind = "kaggle"
    default_persistence = "ephemeral"
    has_fast_path = False
    persistent_channel = False
    needs_lease = False

    #: No device stream can reach a Kaggle session without a tunnel, which Kaggle forbids.
    serves_host_local = False

    usage_unit = "GPU hours"
    usage_source = "kaggle quota, the weekly accelerator quota endpoint"

    default_workspace = "/kaggle/working/letify"

    def available(self) -> bool:
        return tools.find_uv() is not None

    def discover(self) -> Mapping[str, Instance]:
        """The fixed list of accelerators a Kaggle session offers. No call is made."""
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None)}
        table.update(
            {name: Instance(self, gpu=name, vram_gb=spec["vram_gb"]) for name, spec in GPUS.items()}
        )
        table.update({name: Instance(self, tpu=name) for name in TPUS})
        return table

    def store_backend(self) -> str:
        return "filesystem"

    def check_mode(self, instance: Instance) -> None:
        """Refuse ``host="local"``, which reaches here only through ``let.providers.any``."""
        if instance.placement is Host.local:
            raise UnsupportedMode(
                f"{self.alias} cannot serve host='local': Kaggle forbids tunnels and port "
                f"forwarding, so no device stream reaches the session. Use host='remote'."
            )

    def open_channel(self, runtime: Runtime) -> Channel:
        raise UnsupportedMode(
            f"{self.alias}: running on a Kaggle Jupyter Server session is not available in "
            f"this version of letify, only login and usage are"
        )

    def _secrets(self) -> list[str]:
        """Values in the account's credential files, to hide from error output."""
        directory = account_directory(self.alias)
        found: list[str] = []
        token = directory / "access_token"
        if token.is_file():
            found.append(token.read_text(encoding="utf-8").strip())
        legacy = directory / "kaggle.json"
        if legacy.is_file():
            try:
                found.append(str(json.loads(legacy.read_text(encoding="utf-8")).get("key", "")))
            except (ValueError, AttributeError):
                pass
        return [value for value in found if value]

    def report_usage(self) -> Usage:
        uv = tools.find_uv()
        if uv is None:
            raise ProviderUnavailable(self.kind, tools.missing_uv_message())
        command = [*tools.command(tools.KAGGLE, uv), *QUOTA]
        shown = " ".join([tools.KAGGLE.executable, *QUOTA])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            env=tools.kaggle_environment(self.alias),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            for secret in self._secrets():
                detail = detail.replace(secret, "***")
            raise RuntimeFailure(
                f"{self.alias}: `{shown}` exited {result.returncode}", command=shown, stderr=detail
            )
        rows = parse_quota(result.stdout)
        gpu = rows.get("GPU")
        if gpu is None:
            raise RuntimeFailure(f"{self.alias}: `{shown}` reported no GPU quota", command=shown)
        notes = []
        refresh = gpu.get("refreshAt") or gpu.get("refresh_at")
        if refresh:
            notes.append(f"resets {refresh}")
        tpu = rows.get("TPU")
        if tpu is not None:
            notes.append(
                f"TPU {hours(tpu.get('used')):g} h used, {hours(tpu.get('remaining')):g} h left "
                f"of {hours(tpu.get('total')):g}"
            )
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            remaining=hours(gpu.get("remaining")),
            limit=hours(gpu.get("total")),
            used=hours(gpu.get("used")),
            note="; ".join(notes) or None,
        )


__all__ = ["GPUS", "QUOTA", "TPUS", "Kaggle", "hours", "parse_quota"]
