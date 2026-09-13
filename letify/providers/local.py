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
from functools import cache
from typing import TYPE_CHECKING

from ..declare.instance import Instance
from .base import Provider
from .naming import gib_from_mib, normalize_gpu
from .usage import Usage

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

    #: The device is in this machine, so there is no second machine to install on.
    needs_remote_agent = False

    #: A subprocess with pipes, so the object and blob tables persist.
    persistent_channel = True

    #: This machine already runs in its environment.
    prepares_env = False

    #: A local subprocess ends with this process and costs nothing.
    needs_lease = False

    #: The worker keeps the working directory of the process that started it.
    prepares_workspace = False

    #: Nothing to run out of, which is a different answer from an unknown balance.
    usage_unit = "hours"
    usage_source = "nothing to ask; this machine bills nobody"

    def report_usage(self) -> Usage:
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            unmetered=True,
        )

    def discover(self) -> Mapping[str, Instance]:
        """List the GPUs in this machine, plus a plain CPU instance.

        The names are read once per process, because asking nvidia-smi takes seconds on a
        laptop whose discrete GPU is asleep and the answer does not change while the
        process runs. ``refresh()`` asks again.
        """
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None)}
        for label, vram_gb in _device_names():
            table[label] = Instance(self, gpu=label, vram_gb=vram_gb)
        return table

    def refresh(self) -> Mapping[str, Instance]:
        _device_names.cache_clear()
        return super().refresh()

    def busy(self) -> tuple[int, ...]:
        """Ask this machine which cards another process is computing on.

        Excluding this process, because a session asking for a second card must not see its
        own first one as taken.
        """
        import os

        from ..runtime import telemetry

        return telemetry.busy_indices(exclude_pids={os.getpid()})

    def store_backend(self) -> str:
        return "filesystem"

    @property
    def workspace_root(self) -> str:
        """The default root, where a volume without ``mount`` lands. ``workspace`` is not used."""
        from ..runtime import bootstrap

        return bootstrap.DEFAULT_WORKSPACE_ROOT

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


@cache
def _device_names() -> tuple[tuple[str, int | None], ...]:
    """Read the machine's GPU names and memory sizes once."""
    if not shutil.which("nvidia-smi"):
        return ()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()

    found = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, _, memory = line.partition(",")
        found.append((normalize_gpu(name), gib_from_mib(memory)))
    return tuple(found)
