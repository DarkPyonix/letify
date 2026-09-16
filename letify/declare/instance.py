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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..providers.base import Provider

#: The price types an instance may name. Elice's reserved pricing is not offered.
PRICE_TYPES = ("ondemand", "spot")


class Host(StrEnum):
    """Where the host code runs, relative to this process.

    ``local`` is the default. Python and the libraries stay here and only PyTorch operators
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

    #: ``"ondemand"`` or ``"spot"``, or None for the account's own ``price_type``. Set with
    #: ``priced``. Only providers with spot pricing accept ``"spot"``.
    price_type: str | None = None

    #: How many of this accelerator one session takes. A run that trains across two cards
    #: asks for two, which is a property of the shape rather than a separate argument, so it
    #: travels with the value the declaration already carries.
    devices: int = 1

    def _placed(self, host: Host | str | None) -> Instance:
        """Return a copy whose host code runs in the given place.

        Internal. A declaration folds its host into the instance it runs on, and the pool keys
        sessions by the result. It is not public because where the host code runs is said in
        the declaration and nowhere else.
        """
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

    def priced(self, price_type: Literal["ondemand", "spot"]) -> Instance:
        """Return a copy that runs at this price type, as in ``elice.A100.priced("spot")``.

        A value rather than a mutation, like ``n * instance``. A spot machine may be taken
        back by the provider at any time.
        """
        if price_type not in PRICE_TYPES:
            raise ValueError(f"a price type is 'ondemand' or 'spot', not {price_type!r}")
        return replace(self, price_type=price_type)

    @property
    def spot(self) -> bool:
        return self.price_type == "spot"

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
                # None follows the account, so it is not the same session as a named type.
                self.price_type or "default",
            ]
        )

    def __repr__(self) -> str:
        count = f"x{self.devices}" if self.devices > 1 else ""
        return f"<Instance {self.provider.alias}:{self.accelerator}{count} host={self.placement}>"


if TYPE_CHECKING:

    class RemoteOnlyInstance:
        """An accelerator of a provider that cannot serve ``host="local"``, for a type checker.

        Deliberately not a subtype of ``Instance`` for a type checker, so ``Launcher.function``
        accepts it only in the overload whose ``host`` is ``remote``. At run time it is
        ``Instance``, as spec "Placements a provider cannot serve" describes.
        """

        provider: Provider
        gpu: str | None
        tpu: str | None
        host: Host | None
        cpus: int | None
        memory_gb: int | None
        vram_gb: int | None
        spot: bool
        devices: int

        def __init__(self, provider: Provider, gpu: str | None = None) -> None: ...
        def __mul__(self, count: int) -> RemoteOnlyInstance: ...
        def __rmul__(self, count: int) -> RemoteOnlyInstance: ...
        @property
        def accelerator(self) -> str: ...
        @property
        def placement(self) -> Host: ...
        @property
        def key(self) -> str: ...

else:
    RemoteOnlyInstance = Instance


@dataclass(frozen=True, slots=True)
class AnyInstance:
    """A request for an accelerator without naming the provider.

    Produced by ``let.providers.any.G4``. Resolution order is the order the entries
    appear in the configuration file, and only providers that registered a matching
    accelerator are considered.
    """

    accelerator: str
    host: Host | None = None

    def _placed(self, host: Host | str | None) -> AnyInstance:
        """Internal, for the same reason as ``Instance._placed``."""
        if host is None:
            return self
        return replace(self, host=Host(host))

    def __repr__(self) -> str:
        return f"<AnyInstance {self.accelerator}>"


__all__ = ["AnyInstance", "Host", "Instance", "RemoteOnlyInstance"]
