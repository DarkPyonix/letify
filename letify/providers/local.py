"""Local, this machine.

Storage is this machine's disk, so it is persistent by definition. Nothing crosses
a network, so there is no transport, no shipping and no forwarding: a declared
function runs in a subprocess of this machine.

Two things make it worth being a real provider rather than a special case. Someone
with their own GPU can use letify without any remote account. And everyone else's
tests exercise the production code path, because the same serialized call goes
through the same worker.

It is also the origin of the blob store when a remote runtime has to pull from
somewhere, since data that is already here needs no upload first.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..declare.instance import Instance
from .base import Provider
from .naming import gib_from_mib, normalize_gpu

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime


class Local(Provider):
    """The machine letify is running on."""

    kind = "local"
    extra = None
    default_persistence = "persistent"

    #: Nothing crosses a network, so there is no round trip to pay.
    has_fast_path = True

    #: A subprocess with pipes, so the object and blob tables persist.
    persistent_channel = True

    #: This machine already runs in its environment.
    prepares_env = False

    #: A local subprocess ends with this process and costs nothing.
    needs_lease = False

    def discover(self) -> Mapping[str, Instance]:
        """List the GPUs in this machine, plus a plain CPU instance."""
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None)}
        if not shutil.which("nvidia-smi"):
            return table
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return table
        if result.returncode != 0:
            return table

        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            name, _, memory = line.partition(",")
            label = normalize_gpu(name)
            table[label] = Instance(self, gpu=label, vram_gb=gib_from_mib(memory))
        return table

    def store_backend(self) -> str:
        return "filesystem"

    def open_channel(self, runtime: Runtime) -> Channel:
        """A Python subprocess of this machine, with pipes for framed requests.

        A subprocess rather than this process, so a crash in shipped code does not
        take the caller down and the declared environment can differ from the one
        letify itself is running in.
        """
        from ..runtime.channel import PersistentChannel

        interpreter = self.config.option("python") or sys.executable
        from ..protocol.worker import BOOTSTRAP

        return PersistentChannel([str(interpreter), "-u", "-c", BOOTSTRAP], name=runtime.name)


__all__ = ["Local"]
