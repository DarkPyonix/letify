"""Runtime, one live session, and the pool that reuses them.

A Runtime is the only object in letify that costs money. Everything above it is
declaration: the provider, the instance and the environment describe what to
build, and creating a runtime is when Colab, Modal or a cloud machine actually
powers on. Destroying it is when the charge stops.

The pool is the reason this library pays for itself. Starting a session per call
would put provider boot time, environment installation and the first data transfer
in front of every invocation, and all of that is billed. Runtimes are pooled by
instance and environment, so the second call through the same declaration pays
nothing for setup.

Lifetime has three layers, and the middle one is the one that protects the bill.

The scope is explicit: leaving ``with let.run():`` tears runtimes down. An idle
timeout catches a scope that stays open with nothing happening in it. And a
heartbeat lease makes the remote side shut itself down when this process stops
renewing, with a grace period so a brief network drop does not kill a training
run. Without the lease, a crashed local process leaves a GPU running and billing.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import wire
from .env import Env
from .errors import RuntimeFailure
from .instance import Instance

if TYPE_CHECKING:
    from .providers.base import Provider
    from .store.volume import Volume

#: How long a runtime may sit unused inside an open scope before it is torn down.
DEFAULT_IDLE_TIMEOUT = 600.0

#: How often the local process renews the lease.
HEARTBEAT_INTERVAL = 30.0

#: How long the remote side waits for a renewal before shutting itself down. The
#: gap between this and the interval is what survives a short network drop.
LEASE_GRACE = 300.0


@dataclass
class Runtime:
    """One live session on one provider."""

    name: str
    provider: Provider
    instance: Instance
    env: Env
    volumes: tuple[Volume, ...] = ()

    #: Provider side identifier, such as an Elice allocation id.
    external_id: str | None = None

    started: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    ready: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _heartbeat: threading.Thread | None = field(default=None, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)

    # -- identity ------------------------------------------------------------

    @property
    def key(self) -> str:
        """Pool key. Runtimes with equal keys are interchangeable."""
        return f"{self.instance.key}|{self.env.key}"

    @property
    def idle_for(self) -> float:
        return time.monotonic() - self.last_used

    # -- lifecycle -----------------------------------------------------------

    def boot(self) -> None:
        """Install the environment, attach volumes and start the lease."""
        self.install_env()
        for volume in self.volumes:
            self.attach(volume)
        self.start_heartbeat()
        self.ready = True

    def install_env(self) -> None:
        """Bring the declared environment up inside the session.

        A volume that already holds an archive for this environment key is used,
        because unpacking one archive is a single transfer while installing from
        the lock file is thousands of small ones.
        """
        if not self.provider.prepares_env:
            return
        for volume in self.volumes:
            digest = volume.cached_env(self.env)
            if digest:
                self.exec(
                    _fetch_env_script(volume.mount, digest),
                    timeout=1800,
                )
                return
        self.exec(_install_env_script(self.env), timeout=3600)

    def attach(self, volume: Volume) -> None:
        """Make a volume's mount point available inside the session."""
        self.exec(
            f"import pathlib; pathlib.Path({volume.mount!r}).mkdir(parents=True, exist_ok=True)",
            timeout=120,
        )

    def start_heartbeat(self) -> None:
        """Renew the lease until this runtime is shut down.

        The remote side terminates itself if the renewal stops for longer than
        the grace period, so a crashed or killed local process cannot leave a
        paid session running.
        """
        if self._heartbeat is not None or not self.provider.needs_lease:
            return

        def loop() -> None:
            while not self._stop.wait(HEARTBEAT_INTERVAL):
                try:
                    self.exec(_lease_script(LEASE_GRACE), timeout=60)
                except RuntimeFailure:
                    # The session is gone. The pool notices on the next call.
                    return

        self._heartbeat = threading.Thread(
            target=loop, name=f"letify-lease-{self.name}", daemon=True
        )
        self._heartbeat.start()

    def shutdown(self) -> None:
        """Stop the session and everything that bills for it."""
        self._stop.set()
        self.ready = False
        provider = self.provider
        if hasattr(provider, "stop_session"):
            provider.stop_session(self.name)  # type: ignore[attr-defined]
        if self.external_id is not None and hasattr(provider, "release"):
            provider.release(self.external_id)  # type: ignore[attr-defined]

    # -- execution -----------------------------------------------------------

    def exec(self, code: str, *, timeout: float | None = None) -> str:
        """Run Python source inside the session and return its stdout."""
        self.last_used = time.monotonic()
        provider = self.provider
        if hasattr(provider, "exec"):
            return provider.exec(self.name, code, timeout=timeout)  # type: ignore[attr-defined]
        if hasattr(provider, "ssh_command"):
            import subprocess

            command = provider.ssh_command("python3 -")  # type: ignore[attr-defined]
            result = subprocess.run(
                command, input=code, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                raise RuntimeFailure(
                    f"{self.name}: remote python exited {result.returncode}",
                    command=" ".join(command),
                    stderr=result.stderr.strip(),
                )
            return result.stdout
        raise RuntimeFailure(f"{type(provider).__name__} cannot execute code")

    def call(
        self,
        fn: Any,
        args: tuple,
        kwargs: dict,
        *,
        keep_remote: bool = False,
        timeout: float | None = None,
    ) -> tuple[Any, str]:
        """Ship one call and return ``(value, logs)``.

        Handles that belong to another runtime are rejected before anything is
        sent, because resolving them would mean copying the object across the
        network without the caller asking for it.
        """
        wire.check_handles(self.key, args, kwargs)
        script = wire.encode_call(fn, args, kwargs, keep_remote=keep_remote)
        stdout = self.exec(script, timeout=timeout)
        logs, value = wire.parse(stdout, runtime_key=self.key)
        return value, logs

    def __repr__(self) -> str:
        state = "ready" if self.ready else "starting"
        return f"<Runtime {self.name} {self.instance.accelerator} {state}>"


class RuntimePool:
    """Keeps runtimes warm and hands them out by declaration.

    ``max_runtimes`` is a guess until it is measured. Google does not document how
    many Colab sessions one account may hold at once, and the answer moves with
    the account's tier, credit balance and current demand. Start sessions until
    one is refused to find the real number.
    """

    def __init__(self, *, max_runtimes: int = 3, idle_timeout: float = DEFAULT_IDLE_TIMEOUT):
        self.max_runtimes = max_runtimes
        self.idle_timeout = idle_timeout
        self._runtimes: dict[str, list[Runtime]] = {}
        self._guard = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_runtimes)
        self._count = 0

    # -- acquisition ---------------------------------------------------------

    def acquire(
        self,
        instance: Instance,
        env: Env,
        volumes: Sequence[Volume] = (),
    ) -> Runtime:
        """Return an idle runtime for this declaration, starting one if allowed."""
        key = f"{instance.key}|{env.key}"
        while True:
            with self._guard:
                for runtime in self._runtimes.get(key, ()):
                    if runtime.lock.acquire(blocking=False):
                        runtime.last_used = time.monotonic()
                        return runtime
                may_start = self._count < self.max_runtimes

            if may_start and self._slots.acquire(blocking=False):
                try:
                    runtime = self._start(instance, env, volumes, key)
                except BaseException:
                    self._slots.release()
                    raise
                runtime.lock.acquire()
                return runtime

            # Every slot is taken. Wait for one of this declaration's runtimes.
            with self._guard:
                candidates = list(self._runtimes.get(key, ()))
            if not candidates:
                self._slots.acquire()
                self._slots.release()
                continue
            candidates[0].lock.acquire()
            candidates[0].last_used = time.monotonic()
            return candidates[0]

    def release(self, runtime: Runtime) -> None:
        runtime.last_used = time.monotonic()
        try:
            runtime.lock.release()
        except RuntimeError:
            pass

    def discard(self, runtime: Runtime) -> None:
        """Drop a runtime that can no longer be trusted, freeing its slot."""
        with self._guard:
            bucket = self._runtimes.get(runtime.key, [])
            if runtime in bucket:
                bucket.remove(runtime)
                self._count -= 1
                self._slots.release()
        try:
            runtime.shutdown()
        except RuntimeFailure:
            pass
        self.release(runtime)

    def _start(
        self,
        instance: Instance,
        env: Env,
        volumes: Sequence[Volume],
        key: str,
    ) -> Runtime:
        name = f"letify-{instance.accelerator.lower()}-{uuid.uuid4().hex[:6]}"
        runtime = instance.provider.start(instance, env, name=name)
        runtime.volumes = tuple(volumes)
        with self._guard:
            self._runtimes.setdefault(key, []).append(runtime)
            self._count += 1
        return runtime

    # -- upkeep --------------------------------------------------------------

    def reap_idle(self) -> list[str]:
        """Shut down runtimes that have been unused past the idle timeout."""
        stopped: list[str] = []
        with self._guard:
            candidates = [
                runtime
                for bucket in self._runtimes.values()
                for runtime in bucket
                if runtime.idle_for > self.idle_timeout
            ]
        for runtime in candidates:
            if runtime.lock.acquire(blocking=False):
                self.discard(runtime)
                stopped.append(runtime.name)
        return stopped

    def shutdown(self) -> None:
        with self._guard:
            everything = [r for bucket in self._runtimes.values() for r in bucket]
            self._runtimes.clear()
            self._count = 0
        for runtime in everything:
            try:
                runtime.shutdown()
            except RuntimeFailure:
                pass

    @property
    def live(self) -> list[Runtime]:
        with self._guard:
            return [r for bucket in self._runtimes.values() for r in bucket]


def _install_env_script(env: Env) -> str:
    """Source that installs the declared environment inside a session."""
    lines = [
        "import os, subprocess, shutil, sys",
    ]
    for name, value in env.variables:
        lines.append(f"os.environ[{name!r}] = {value!r}")
    lines += [
        "if shutil.which('uv') is None:",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)",
    ]
    if env.packages:
        packages = ", ".join(repr(p) for p in env.packages)
        lines.append(
            f"subprocess.run(['uv', 'pip', 'install', '--system', '-q', {packages}], check=True)"
        )
    for command in env.commands:
        lines.append(f"subprocess.run({command!r}, shell=True, check=True)")
    lines.append("print('letify: environment ready')")
    return "\n".join(lines)


def _fetch_env_script(mount: str, digest: str) -> str:
    """Source that unpacks a prebuilt environment archive from a volume."""
    return (
        "import pathlib, tarfile\n"
        f"root = pathlib.Path({mount!r})\n"
        f"archive = root / 'blobs' / {digest[:2]!r} / {digest!r}\n"
        "if archive.is_file():\n"
        "    with tarfile.open(archive, 'r:gz') as tar:\n"
        "        tar.extractall(root)\n"
        "    print('letify: environment restored from cache')\n"
        "else:\n"
        "    raise SystemExit('letify: cached environment archive is missing')\n"
    )


def _lease_script(grace: float) -> str:
    """Source that arms the self termination timer inside the session.

    The timer is reset on every renewal. If this process stops renewing, the
    session shuts itself down after the grace period rather than running on and
    billing.
    """
    return (
        "import os, threading, time\n"
        "_state = globals().setdefault('__letify_lease__', {})\n"
        f"_state['deadline'] = time.time() + {grace!r}\n"
        "def _watch():\n"
        "    while True:\n"
        "        time.sleep(15)\n"
        "        if time.time() > _state.get('deadline', 0):\n"
        "            os._exit(0)\n"
        "if not _state.get('armed'):\n"
        "    _state['armed'] = True\n"
        "    threading.Thread(target=_watch, daemon=True).start()\n"
    )
