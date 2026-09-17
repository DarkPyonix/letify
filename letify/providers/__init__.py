"""The provider registry.

Every provider class is imported here, but no provider's optional dependency is.
A provider reports itself unavailable when its dependency is missing, and the rest
of letify keeps working.
"""

from __future__ import annotations

from ..config import ProviderConfig
from ..errors import ConfigError
from .base import Persistence, Provider
from .colab import Colab
from .elice import Elice
from .kaggle import Kaggle
from .local import Local
from .modal import Modal
from .shell import Shell
from .tunnel import Tunnel

#: Value of a configuration entry's ``kind`` field to the class it selects.
KINDS: dict[str, type[Provider]] = {
    "local": Local,
    "colab": Colab,
    "modal": Modal,
    "shell": Shell,
    "ssh": Shell,
    "tunnel": Tunnel,
    "elice": Elice,
    "kaggle": Kaggle,
}


def build(config: ProviderConfig) -> Provider:
    """Instantiate the provider a configuration entry describes."""
    kind = config.kind.lower()
    cls = KINDS.get(kind)
    if cls is None:
        known = ", ".join(sorted(KINDS))
        raise ConfigError(
            f"provider {config.alias!r} has kind {config.kind!r}, which is not known. "
            f"Use one of: {known}"
        )
    return cls(config)


__all__ = [
    "KINDS",
    "Colab",
    "Elice",
    "Kaggle",
    "Local",
    "Modal",
    "Persistence",
    "Provider",
    "Shell",
    "Tunnel",
    "build",
]
