"""Modal, reached through its own Python API.

The Modal client is an optional dependency and is imported lazily. If it is not
installed, this provider reports itself unavailable and every other provider keeps
working. Nothing in the rest of letify imports ``modal``.

Storage outlives a container because a Modal volume is mounted from outside it, so
this provider is persistent and the loop is shipped. Forwarding CUDA calls is not
offered: Modal's model is a function call into a container, which is the same
thing letify already does natively.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..errors import ProviderUnavailable, UnsupportedMode
from ..instance import Instance
from .base import Provider

if TYPE_CHECKING:
    from ..env import Env
    from ..runtime import Runtime

#: GPU names Modal accepts, with the memory each one carries.
GPUS = {
    "T4": dict(vram_gb=16),
    "L4": dict(vram_gb=24),
    "A10": dict(vram_gb=24),
    "A100": dict(vram_gb=40),
    "A100_80GB": dict(vram_gb=80),
    "L40S": dict(vram_gb=48),
    "H100": dict(vram_gb=80),
    "H200": dict(vram_gb=141),
    "B200": dict(vram_gb=180),
    "RTX_PRO_6000": dict(vram_gb=96),
}

#: Modal spells some names with a hyphen or a suffix.
WIRE_NAMES = {
    "A100_80GB": "A100-80GB",
    "RTX_PRO_6000": "RTX-PRO-6000",
}


class Modal(Provider):
    """One Modal workspace."""

    kind = "modal"
    extra = "modal"
    default_persistence = "persistent"
    has_fast_path = False

    def client(self) -> Any:
        """Import and return the Modal module, or report the provider unavailable."""
        try:
            import modal
        except ImportError as exc:
            raise ProviderUnavailable(
                self.kind, "the modal package is not installed", self.extra
            ) from exc
        return modal

    def discover(self) -> Mapping[str, Instance]:
        """Return Modal's published GPU list.

        The list is static, so no call to Modal is made here. Asking for a GPU
        that the workspace cannot get fails when a runtime starts, not now.
        """
        table = {
            name: Instance(self, gpu=name, vram_gb=spec.get("vram_gb"))
            for name, spec in GPUS.items()
        }
        return table

    def store_backend(self) -> str:
        return "modal"

    def wire_name(self, instance: Instance) -> str:
        """Translate an instance into the string Modal's API expects."""
        gpu = instance.gpu or ""
        return WIRE_NAMES.get(gpu, gpu)

    def start(self, instance: Instance, env: Env, *, name: str) -> Runtime:
        from ..runtime import Runtime

        if instance.placement == "local":
            raise UnsupportedMode(
                "Modal does not support cpu='local'. Modal exposes function calls "
                "into a container, not a device to forward CUDA calls to."
            )
        self.client()
        runtime = Runtime(name=name, provider=self, instance=instance, env=env)
        runtime.boot()
        return runtime
