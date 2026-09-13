"""Elice, a machine on Elice Cloud Infrastructure.

Elice separates the definition of a machine from the fact that it is running. A
virtual machine is the declared machine and an allocation is the machine actually
powered on, so starting is a POST to the allocation collection and stopping is a
DELETE. letify maps that onto its own split: the virtual machine is the provider's
instance and the allocation is the runtime.

Elice has two GPU product lines and only one can be automated. Elice Cloud
Infrastructure has a REST API, a Terraform provider and a CLI, all published by
Elice, and the paths below come from that Terraform provider's source. Run Box, the
container product, is driven from the web console only; a Run Box machine can still
be used by declaring it as a plain Shell with its tunnel address and port.

Storage is a separate resource from the machine, so it survives a stop and a
restart, which makes this provider persistent. Compute bills by the second while
allocated, and block storage keeps billing while the machine is stopped, so a
forgotten machine still costs money with no allocation running.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..declare.instance import Instance
from ..errors import ProviderUnavailable, RuntimeFailure
from .naming import normalize_gpu
from .shell import Shell

if TYPE_CHECKING:
    from ..runtime.session import Runtime

DEFAULT_ENDPOINT = "https://portal.elice.cloud/api"

#: Paths taken from Elice's published Terraform provider.
VM_PATH = "/user/resource/compute/virtual_machine"
ALLOCATION_PATH = "/user/resource/compute/virtual_machine_allocation"
INSTANCE_TYPE_PATH = "/user/infra/instance_type"
PRICING_PATH = "/user/pricing"


class Elice(Shell):
    """One Elice Cloud Infrastructure zone under one access token."""

    kind = "elice"
    extra = "shell"
    default_persistence = "persistent"
    has_fast_path = True

    # -- configuration -------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return str(self.config.option("endpoint", DEFAULT_ENDPOINT)).rstrip("/")

    @property
    def zone_id(self) -> str:
        value = self.config.option("zone_id")
        if not isinstance(value, str):
            raise ProviderUnavailable(self.kind, f"{self.alias} has no 'zone_id' field")
        return value

    @property
    def organization_id(self) -> str | None:
        value = self.config.option("organization_id")
        return value if isinstance(value, str) else None

    @property
    def machine_id(self) -> str:
        value = self.config.option("machine_id")
        if not isinstance(value, str):
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} has no 'machine_id' field. Declare the virtual machine "
                f"once in the Elice console or with Terraform, then put its id here. "
                f"letify allocates and releases it, but does not create it",
            )
        return value

    def _token(self) -> str:
        token = self.config.secret("access_token")
        if not token:
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} needs an access token. Set access_token_env or "
                f"access_token_keyring so the token stays out of tracked files",
            )
        return token

    # -- the API -------------------------------------------------------------

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            import httpx
        except ImportError as exc:
            raise ProviderUnavailable(
                self.kind, "the httpx package is not installed", self.extra
            ) from exc

        with httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=60.0,
        ) as client:
            response = client.request(method, path, **kwargs)

        # This API answers 200 for every success, so anything else is a failure.
        if response.status_code != 200:
            try:
                body = response.json()
                detail = body.get("message") or str(body)
            except Exception:
                detail = response.text[:500]
            raise RuntimeFailure(f"Elice {method} {path} returned {response.status_code}: {detail}")
        return response.json()

    @staticmethod
    def _items(body: Any) -> list[dict[str, Any]]:
        return body if isinstance(body, list) else body.get("items", [])

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """List the instance types this zone offers.

        A configuration entry may name them instead, which avoids an API call during
        import.
        """
        declared = self.config.option("gpus")
        if isinstance(declared, list) and declared:
            return {str(name): Instance(self, gpu=str(name)) for name in declared}

        body = self._call("GET", INSTANCE_TYPE_PATH, params={"zone_id": self.zone_id})
        table: dict[str, Instance] = {}
        for item in self._items(body):
            raw = item.get("gpu_model") or item.get("name") or ""
            if not raw:
                continue
            label = normalize_gpu(str(raw))
            table[label] = Instance(
                self,
                gpu=label,
                cpus=item.get("cpu_count"),
                memory_gb=item.get("memory_gb"),
                vram_gb=item.get("gpu_memory_gb"),
            )
        return table

    def store_backend(self) -> str:
        """Data Hub, which speaks the S3 API."""
        backend = self.config.option("store")
        return str(backend) if isinstance(backend, str) else "s3"

    def pricing(self) -> list[dict[str, Any]]:
        """The zone's price list, including any preemptible option."""
        return self._items(self._call("GET", PRICING_PATH))

    def machines(self) -> list[dict[str, Any]]:
        return self._items(self._call("GET", VM_PATH, params={"zone_id": self.zone_id}))

    # -- allocations, which are runtimes -------------------------------------

    def allocations(self, machine_id: str | None = None) -> list[dict[str, Any]]:
        params = {"filter_machine_id": machine_id} if machine_id else None
        return self._items(self._call("GET", ALLOCATION_PATH, params=params))

    def allocate(self, machine_id: str) -> str:
        """Power a declared machine on and return the allocation id."""
        payload: dict[str, Any] = {"zone_id": self.zone_id, "machine_id": machine_id}
        if self.organization_id:
            payload["organization_id"] = self.organization_id
        body = self._call("POST", ALLOCATION_PATH, json=payload)
        allocation_id = body.get("id") or body.get("allocation_id")
        if not allocation_id:
            raise RuntimeFailure(f"Elice did not return an allocation id: {body!r}")
        return str(allocation_id)

    def release(self, allocation_id: str) -> None:
        """Power off, which stops compute billing. Storage keeps being billed."""
        try:
            self._call("DELETE", f"{ALLOCATION_PATH}/{allocation_id}")
        except RuntimeFailure:
            # Releasing is best effort. An allocation that is already gone is fine.
            pass

    # -- sessions ------------------------------------------------------------

    def create_session(self, instance: Instance, name: str) -> None:
        self._pending_allocation = self.allocate(self.machine_id)

    def start(self, instance: Instance, env: Any, *, name: str, volumes: Any = ()) -> Runtime:
        runtime = super().start(instance, env, name=name, volumes=volumes)
        runtime.external_id = getattr(self, "_pending_allocation", None)
        return runtime

    def stop(self, runtime: Runtime) -> None:
        if runtime.external_id:
            self.release(runtime.external_id)


__all__ = [
    "ALLOCATION_PATH",
    "DEFAULT_ENDPOINT",
    "INSTANCE_TYPE_PATH",
    "PRICING_PATH",
    "VM_PATH",
    "Elice",
]
