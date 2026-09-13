"""Instance, one machine shape offered by a provider, and where its host runs.

A declaration places two things. ``device=colab.G4`` says where the device is, carrying
the provider and the account with it. ``host=Host.local`` says where the host code
runs, which is the other half of the same picture.

The words come from CUDA, where the host is the CPU side and the device is the GPU.
Naming this axis ``host`` rather than ``cpu`` keeps it about placement: a core count is
a resource, and it arrives with the shape the provider registered rather than being
asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.base import Provider


class Lifetime(StrEnum):
    """How long the runtime a declaration uses stays alive.

    ``call`` is the default. The session ends when the call that needed it finishes, and a
    search space counts as one call, so a sweep starts its sessions once and ends them
    once. Nothing keeps billing after the work is done.

    ``process`` keeps the session past the call, because starting one costs provider boot
    plus environment installation, which is minutes on Colab and worth avoiding across a
    run of separate calls. It then ends when the process exits. Nothing ends it sooner,
    because a timer would overrule the declaration that asked to keep it.

    A string enum, so ``lifetime="process"`` works wherever ``lifetime=Lifetime.process``
    does.
    """

    call = "call"
    process = "process"


class Host(StrEnum):
    """Where the host code runs, relative to this process.

    ``local`` is the default. Python and the libraries stay here and only CUDA calls
    cross the network, so the code and the data stay where they already are. The cost is
    one round trip at every point where the host reads a value back from the device.

    ``remote`` ships the declared function, so the host code runs on the machine that
    holds the device and those reads never cross the network. That is what a heavy loop
    opts into, and it needs the environment and the data on the far side.

    A string enum, so ``host="local"`` works wherever ``host=Host.local`` does.
    """

    local = "local"
    remote = "remote"


@dataclass(frozen=True, slots=True)
class Instance:
    """One accelerator shape on one provider account."""

    provider: Provider
    gpu: str | None = None
    tpu: str | None = None
    host: Host | None = None

    #: Reported by the provider when it registered this shape, not requested. A
    #: provider that offers several sizes registers them as separate shapes.
    cpus: int | None = None
    memory_gb: int | None = None
    vram_gb: int | None = None
    spot: bool = False

    #: How many of this accelerator one session takes. A run that trains across two cards
    #: asks for two, which is a property of the shape rather than a separate argument, so it
    #: travels with the value the declaration already carries.
    devices: int = 1

    def on_host(self, host: Host | str | None) -> Instance:
        """Return a copy whose host code runs in the given place."""
        if host is None:
            return self
        return replace(self, host=Host(host))

    def __mul__(self, count: int) -> Instance:
        """Return a copy taking ``count`` devices, as in ``lab.A100 * 2``.

        A value rather than a mutation, so the single card shape stays usable next to it.
        """
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"a device count must be a whole number, not {type(count).__name__}")
        if count < 1:
            raise ValueError(f"a session takes at least one device, not {count}")
        return replace(self, devices=count)

    def __rmul__(self, count: int) -> Instance:
        return self.__mul__(count)

    @property
    def accelerator(self) -> str:
        return self.gpu or self.tpu or "cpu"

    @property
    def placement(self) -> Host:
        """Where the host code runs, defaulting to this process."""
        return self.host or Host.local

    @property
    def key(self) -> str:
        """Identity used to pool runtimes. Instances with equal keys share one."""
        return ":".join(
            [
                self.provider.alias,
                self.accelerator,
                str(self.placement),
                # A session holding two cards is not interchangeable with one holding one.
                f"x{self.devices}",
                "spot" if self.spot else "ondemand",
            ]
        )

    def __repr__(self) -> str:
        count = f"x{self.devices}" if self.devices > 1 else ""
        return f"<Instance {self.provider.alias}:{self.accelerator}{count} host={self.placement}>"


@dataclass(frozen=True, slots=True)
class AnyInstance:
    """A request for an accelerator without naming the provider.

    Produced by ``let.providers.any.G4``. Resolution order is the order the entries
    appear in the configuration file, and only providers that registered a matching
    accelerator are considered.
    """

    accelerator: str
    host: Host | None = None

    def on_host(self, host: Host | str | None) -> AnyInstance:
        if host is None:
            return self
        return replace(self, host=Host(host))

    def __repr__(self) -> str:
        return f"<AnyInstance {self.accelerator}>"


__all__ = ["AnyInstance", "Host", "Instance", "Lifetime"]
