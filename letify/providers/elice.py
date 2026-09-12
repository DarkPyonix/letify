"""Elice, a machine on Elice Cloud Infrastructure.

Elice separates the definition of a machine from the fact that it is running. A
``virtual_machine`` is the declared machine and an allocation is the machine
actually powered on, so starting is a POST to the allocation collection and
stopping is a DELETE. letify maps that onto its own split: the virtual machine is
the provider's instance and the allocation is the runtime.

Elice has two GPU product lines and only one of them can be automated. Elice
Cloud Infrastructure has a REST API, a Terraform provider and a CLI, all
published by Elice. Run Box, the container product, is driven from the web
console only. This provider targets Elice Cloud Infrastructure.

Storage is a separate resource from the machine, so it survives a stop and a
restart, which makes this provider persistent. Compute is billed by the second
while allocated, and block storage keeps being billed while the machine is
stopped, so a forgotten machine still costs money even with no allocation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..errors import ProviderUnavailable, RuntimeFailure
from ..instance import Instance
from .shell import Shell

if TYPE_CHECKING:
    from ..env import Env
    from ..runtime import Runtime

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

    @property
    def endpoint(self) -> str:
        value = self.config.option("endpoint", DEFAULT_ENDPOINT)
        return str(value).rstrip("/")

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

    def _token(self) -> str:
        token = self.config.secret("access_token")
        if not token:
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} needs an access token. Set access_token_env or "
                f"access_token_keyring so the token stays out of tracked files",
            )
        return token

    def _http(self) -> Any:
        try:
            import httpx
        except ImportError as exc:
            raise ProviderUnavailable(
                self.kind, "the httpx package is not installed", self.extra
            ) from exc
        return httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=60.0,
        )

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        with self._http() as client:
            response = client.request(method, path, **kwargs)
        # This API answers 200 for every success, so anything else is a failure.
        if response.status_code != 200:
            detail = ""
            try:
                body = response.json()
                detail = body.get("message") or str(body)
            except Exception:
                detail = response.text[:500]
            raise RuntimeFailure(f"Elice {method} {path} returned {response.status_code}: {detail}")
        return response.json()

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """List the instance types this zone offers.

        The configuration may name them instead, which avoids an API call during
        import.
        """
        declared = self.config.option("gpus")
        if isinstance(declared, list) and declared:
            return {str(name): Instance(self, gpu=str(name)) for name in declared}

        body = self._call("GET", INSTANCE_TYPE_PATH, params={"zone_id": self.zone_id})
        items = body if isinstance(body, list) else body.get("items", [])
        table: dict[str, Instance] = {}
        for item in items:
            gpu_name = item.get("gpu_model") or item.get("name") or ""
            if not gpu_name:
                continue
            from .shell import _normalize_gpu_name

            label = _normalize_gpu_name(str(gpu_name))
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

    # -- allocations, which are runtimes -------------------------------------

    def allocate(self, machine_id: str) -> str:
        """Power on a declared machine and return the allocation id."""
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
            pass

    def allocations(self, machine_id: str | None = None) -> list[dict[str, Any]]:
        params = {"filter_machine_id": machine_id} if machine_id else None
        body = self._call("GET", ALLOCATION_PATH, params=params)
        return body if isinstance(body, list) else body.get("items", [])

    def machines(self) -> list[dict[str, Any]]:
        body = self._call("GET", VM_PATH, params={"zone_id": self.zone_id})
        return body if isinstance(body, list) else body.get("items", [])

    # -- runtimes ------------------------------------------------------------

    def start(self, instance: Instance, env: Env, *, name: str) -> Runtime:
        from ..runtime import Runtime

        machine_id = self.config.option("machine_id")
        if not isinstance(machine_id, str):
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} has no 'machine_id' field. Declare the virtual machine "
                f"once in the Elice console or with Terraform, then put its id here. "
                f"letify allocates and releases it, but does not create it",
            )
        allocation_id = self.allocate(machine_id)
        self.connect()
        runtime = Runtime(name=name, provider=self, instance=instance, env=env)
        runtime.external_id = allocation_id
        runtime.boot()
        return runtime
