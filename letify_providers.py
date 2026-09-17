"""The type of ``Launcher.providers`` when a project has not generated its own.

letify writes ``letify_providers.pyi`` into a project to name that project's accounts, and a type
checker reads it in place of this module. With no generated stub, ``ProvidersView`` is the plain
``Providers``, so the types are exactly what they would be without generation.
"""

from __future__ import annotations

from letify.launcher import Providers as ProvidersView

__all__ = ["ProvidersView"]
