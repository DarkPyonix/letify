"""Modal, reached through its own Python API.

The Modal client is an optional dependency and is imported lazily, so a missing
install disables this provider and nothing else. Nothing outside this module
imports ``modal``.

Storage outlives a container because a Modal volume is mounted from outside it, so
this provider is persistent and the loop is shipped. CUDA call forwarding is not
offered: Modal exposes a function call into a container, not a device to forward
calls at.

A sandbox is used rather than a function call, because letify needs a process that
stays alive. Without one there is no object table for a handle to point at and no
blob table to keep a large argument from travelling twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..config import ProviderConfig
from ..declare.instance import Host, Instance
from ..errors import ProtocolError, ProviderUnavailable, UnsupportedMode
from .base import Provider

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: GPU names Modal accepts, with the memory each one carries.
GPUS = {
    "T4": {"vram_gb": 16},
    "L4": {"vram_gb": 24},
    "A10": {"vram_gb": 24},
    "A100": {"vram_gb": 40},
    "A100_80GB": {"vram_gb": 80},
    "L40S": {"vram_gb": 48},
    "H100": {"vram_gb": 80},
    "H200": {"vram_gb": 141},
    "B200": {"vram_gb": 180},
    "RTX_PRO_6000": {"vram_gb": 96},
}

#: Modal spells some of these with a hyphen or a suffix.
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

    #: Modal bills in dollars and exposes no workspace balance through its SDK, so the
    #: figure has to come from a configured command or from the dashboard.
    usage_unit = "USD"
    usage_source = "the Modal SDK exposes no workspace balance"

    #: A sandbox keeps a process alive, so handles and blob reuse work.
    persistent_channel = True

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._sandboxes: dict[str, Any] = {}

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

        The list is static, so no call to Modal is made here. Asking for a GPU the
        workspace cannot get fails when a runtime starts, not now.
        """
        return {
            name: Instance(self, gpu=name, vram_gb=spec.get("vram_gb"))
            for name, spec in GPUS.items()
        }

    def store_backend(self) -> str:
        return "modal"

    def wire_name(self, instance: Instance) -> str:
        """Translate an instance into the string Modal's API expects."""
        gpu = instance.gpu or ""
        return WIRE_NAMES.get(gpu, gpu)

    def check_mode(self, instance: Instance) -> None:
        if instance.placement is Host.local:
            raise UnsupportedMode(
                "Modal cannot serve host='local'. It exposes function calls into a "
                "container, so there is no device to forward CUDA calls to. This is a "
                "limit of the service, not a speed judgement. Use host='remote'."
            )

    def open_channel(self, runtime: Runtime) -> Channel:
        """A Modal sandbox running one Python that reads framed requests."""
        modal = self.client()
        app_name = str(self.config.option("app", "letify"))
        app = modal.App.lookup(app_name, create_if_missing=True)
        # The worker carries the vendored cloudpickle, so the image needs nothing added.
        image = modal.Image.debian_slim()
        sandbox = modal.Sandbox.create(
            "python3",
            "-u",
            "-",
            app=app,
            image=image,
            gpu=self.wire_name(runtime.instance) or None,
            timeout=int(self.config.option("timeout", 3600)),
        )
        self._sandboxes[runtime.name] = sandbox
        return SandboxChannel(sandbox, name=runtime.name)

    def stop(self, runtime: Runtime) -> None:
        sandbox = self._sandboxes.pop(runtime.name, None)
        if sandbox is None:
            return
        try:
            sandbox.terminate()
        except Exception:
            # Terminating is best effort. A sandbox that is already gone is fine.
            pass


class SandboxChannel:
    """Adapts a Modal sandbox's streams to the channel interface.

    The framed protocol is the same one that runs over SSH. Only the plumbing to
    reach the pipes differs, because Modal exposes them through its own objects.
    """

    persistent = True

    def __init__(self, sandbox: Any, *, name: str):
        self.sandbox = sandbox
        self.name = name
        self._started = False

    def start(self) -> None:
        from ..protocol.worker import SOURCE

        if self._started:
            return
        self._write(SOURCE + "\n")
        self._started = True

    def close(self) -> None:
        from .. import protocol

        try:
            self._write(protocol.SHUTDOWN + "\n")
        except Exception:
            pass

    def _write(self, text: str) -> None:
        self.sandbox.stdin.write(text.encode())
        self.sandbox.stdin.drain()

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, str]:
        from .. import protocol

        self.start()
        self._write(protocol.encode_request(payload) + "\n")
        logs: list[str] = []
        for raw in self.sandbox.stdout:
            line = raw if isinstance(raw, str) else raw.decode()
            if protocol.is_ready(line):
                continue
            if protocol.is_reply(line):
                outcome = protocol.decode_reply(line)
                return protocol.unwrap(outcome, runtime_key=self.name), "".join(logs)
            logs.append(line)
        raise ProtocolError(
            f"{self.name}: the sandbox stopped without replying, so the process died "
            f"before it finished.\n--- last remote output ---\n"
            f"{''.join(logs)[-2000:]}"
        )

    def call(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        *,
        keep_remote: bool = False,
        timeout: float | None = None,
    ) -> tuple[Any, str]:
        import base64

        from .. import protocol

        return self.request(
            {
                "op": "call",
                "payload": base64.b64encode(protocol.dumps_call(fn, args, kwargs)).decode(),
                "keep_remote": keep_remote,
            },
            timeout=timeout,
        )


__all__ = ["GPUS", "WIRE_NAMES", "Modal", "SandboxChannel"]
