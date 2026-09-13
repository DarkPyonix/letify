"""The vendored cloudpickle, packed into source that runs where letify is not installed.

A call payload pickled by ``letify._vendor.cloudpickle`` names that module path, so the
process that unpickles it has to be able to import the same name. A remote worker has
nothing installed, so the worker source and the one-shot driver both start with a prelude
that registers the vendored modules under their real names from source carried inside it.
A machine that can already import them, such as the local provider's worker, uses its own.
"""

from __future__ import annotations

import base64
import zlib
from functools import cache
from pathlib import Path

#: Where the vendored package lives in this installation.
VENDORED = Path(__file__).resolve().parents[1] / "_vendor" / "cloudpickle"

#: Module name to file, in the order they are executed. The submodule comes first because
#: the package's ``__init__`` imports from it.
MODULES = (
    ("letify._vendor.cloudpickle.cloudpickle", "cloudpickle.py"),
    ("letify._vendor.cloudpickle", "__init__.py"),
)

_PRELUDE = """
def _letify_vendored_cloudpickle():
    import base64, sys, types, zlib
    try:
        from letify._vendor import cloudpickle as found
        return found
    except ImportError:
        pass
    for name in ("letify", "letify._vendor", "letify._vendor.cloudpickle"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            module.__package__ = name
            sys.modules[name] = module
    for name, packed in {packed!r}:
        if name in sys.modules and not name.endswith(".cloudpickle.cloudpickle"):
            module = sys.modules[name]
        else:
            module = types.ModuleType(name)
            module.__package__ = name.rpartition(".")[0]
            sys.modules[name] = module
        module.__file__ = "<letify vendored " + name + ">"
        source = zlib.decompress(base64.b64decode(packed)).decode("utf-8")
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        parent, _, leaf = name.rpartition(".")
        setattr(sys.modules[parent], leaf, module)
    return sys.modules["letify._vendor.cloudpickle"]


cloudpickle = _letify_vendored_cloudpickle()
"""


@cache
def prelude() -> str:
    """Source that makes ``letify._vendor.cloudpickle`` importable and binds ``cloudpickle``."""
    packed = tuple(
        (name, base64.b64encode(zlib.compress((VENDORED / file).read_bytes(), 9)).decode())
        for name, file in MODULES
    )
    return _PRELUDE.format(packed=packed)


__all__ = ["MODULES", "VENDORED", "prelude"]
