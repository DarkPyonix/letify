"""What a configuration file describes.

One entry is one provider account. The options are whatever that kind of provider
needs, kept as a plain mapping so a provider owns the meaning of its own fields
rather than the loader knowing them all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .secrets import resolve_secret

#: Aliases are reached with attribute access, as in ``let.providers.colab_a``, so
#: these names are taken.
RESERVED_ALIASES = frozenset({"any", "devices", "active", "aliases", "kinds"})


@dataclass(slots=True)
class ProviderConfig:
    """One provider declaration, as written in the configuration file."""

    alias: str
    kind: str
    options: dict[str, Any] = field(default_factory=dict)
    order: int = 0

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)

    def secret(self, name: str, default: str | None = None) -> str | None:
        """Resolve a credential without it ever living in a tracked file."""
        return resolve_secret(self.options, name, default)


@dataclass(slots=True)
class Config:
    """The merged configuration."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)

    @property
    def order(self) -> list[str]:
        """Aliases in declaration order, which sets ``any`` resolution priority."""
        return [entry.alias for entry in sorted(self.providers.values(), key=lambda c: c.order)]


__all__ = ["RESERVED_ALIASES", "Config", "ProviderConfig"]
