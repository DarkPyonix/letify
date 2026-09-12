"""Local, this machine.

Storage is this machine's disk, so it is persistent by definition. There is no
transport, no shipping and no forwarding: a declared function runs in a
subprocess with the declared environment, or in this process when no environment
is declared.

Two uses make this worth having as a real provider rather than a special case.
Someone with a GPU in their own machine can use letify without any remote account.
Everyone else can run the same declarations locally in tests, which exercises the
production code path instead of a mock.

Because the disk is local, this provider can also serve as the origin of the blob
store. Data that is already here does not need to be uploaded anywhere before a
remote runtime can pull it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..instance import Instance
from .base import Provider

if TYPE_CHECKING:
    from ..env import Env
    from ..runtime import Runtime


class Local(Provider):
    """The machine letify is running on."""

    kind = "local"
    extra = None
    default_persistence = "persistent"

    #: Nothing crosses a network, so there is no round trip to pay.
    has_fast_path = True

    #: This machine already runs in its environment.
    prepares_env = False

    #: A local subprocess ends with the call and costs nothing.
    needs_lease = False

    def discover(self) -> Mapping[str, Instance]:
        """List the GPUs in this machine, plus a plain CPU instance."""
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None, cpu="local")}
        if not shutil.which("nvidia-smi"):
            return table
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return table
        if out.returncode != 0:
            return table

        from .shell import _normalize_gpu_name, _parse_mib

        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            name, _, memory = line.partition(",")
            label = _normalize_gpu_name(name)
            table[label] = Instance(self, gpu=label, cpu="local", vram_gb=_parse_mib(memory))
        return table

    @property
    def default_cpu_placement(self) -> str:  # type: ignore[override]
        """Always local. There is nowhere else for the Python side to be."""
        return "local"

    def store_backend(self) -> str:
        return "filesystem"

    # -- execution -----------------------------------------------------------

    def exec(self, name: str, code: str, *, timeout: float | None = None) -> str:
        """Run the code in a subprocess of this machine.

        A subprocess rather than this process, so a crash in shipped code does not
        take the caller down and the declared environment can differ from the one
        letify is running in.
        """
        import sys

        from ..errors import RuntimeFailure

        result = subprocess.run(
            [sys.executable, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeFailure(
                f"{name}: local python exited {result.returncode}",
                command=f"{sys.executable} -",
                stderr=result.stderr.strip(),
            )
        return result.stdout

    def stop_session(self, name: str) -> None:
        """Nothing to stop. A local subprocess ends with the call."""
        return None

    def start(self, instance: Instance, env: Env, *, name: str) -> Runtime:
        from ..runtime import Runtime

        runtime = Runtime(name=name, provider=self, instance=instance, env=env)
        runtime.boot()
        return runtime
