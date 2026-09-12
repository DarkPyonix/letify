"""Instance, one machine shape offered by a provider.

An Instance carries everything the scheduler needs to place a call: which
provider and account it belongs to, which accelerator to ask for, where the
Python side of the work runs, and how many cores to request. Because it is one
value, ``@let.function(gpu=colab.G4)`` fixes provider, account, accelerator and
CPU placement in a single argument.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .providers.base import Provider

#: Where the Python process that drives the work runs.
#:
#: ``"remote"`` ships the declared function to the remote machine, so the whole
#: loop runs there. ``"local"`` keeps Python in this process and forwards only
#: CUDA calls, which costs one network round trip per host synchronization.
CpuPlacement = Literal["local", "remote"]


@dataclass(frozen=True, slots=True)
class Instance:
    """One accelerator shape on one provider account."""

    provider: Provider
    gpu: str | None = None
    tpu: str | None = None
    cpu: CpuPlacement | None = None
    cpus: int | None = None
    memory_gb: int | None = None
    vram_gb: int | None = None
    spot: bool = False

    def __call__(
        self,
        *,
        cpu: CpuPlacement | None = None,
        cpus: int | None = None,
        memory_gb: int | None = None,
        spot: bool | None = None,
    ) -> Instance:
        """Return a copy with placement or size overridden.

        ``colab.G4(cpu="local", cpus=8)`` reads as a refinement of the registered
        shape rather than a new declaration.
        """
        return replace(
            self,
            cpu=cpu if cpu is not None else self.cpu,
            cpus=cpus if cpus is not None else self.cpus,
            memory_gb=memory_gb if memory_gb is not None else self.memory_gb,
            spot=spot if spot is not None else self.spot,
        )

    @property
    def accelerator(self) -> str:
        return self.gpu or self.tpu or "cpu"

    @property
    def placement(self) -> CpuPlacement:
        """Resolved CPU placement, falling back to the provider's default."""
        return self.cpu or self.provider.default_cpu_placement

    @property
    def key(self) -> str:
        """Identity used to pool runtimes. Instances with equal keys share one."""
        parts = [
            self.provider.alias,
            self.accelerator,
            self.placement,
            str(self.cpus or "auto"),
            "spot" if self.spot else "ondemand",
        ]
        return ":".join(parts)

    def __repr__(self) -> str:
        return f"<Instance {self.provider.alias}:{self.accelerator} cpu={self.placement}>"


@dataclass(frozen=True, slots=True)
class AnyInstance:
    """A request for an accelerator without naming the provider.

    Produced by ``let.providers.any.G4``. Resolution order is the declaration
    order in the configuration file, and only providers that registered a
    matching accelerator are considered.
    """

    accelerator: str
    cpu: CpuPlacement | None = None
    cpus: int | None = None

    def __call__(self, *, cpu: CpuPlacement | None = None, cpus: int | None = None) -> AnyInstance:
        return replace(
            self,
            cpu=cpu if cpu is not None else self.cpu,
            cpus=cpus if cpus is not None else self.cpus,
        )

    def __repr__(self) -> str:
        return f"<AnyInstance {self.accelerator}>"
