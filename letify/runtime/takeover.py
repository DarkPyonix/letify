"""Taking over a running call: this process becomes the owner of a call another one started.

Owns the sequence of spec "Taking over a running call": read the run's record, ask the old
owner to let go, reach the same runtime through the record's provider, adopt the call on the
worker's second door, renew the lease, receive the output and the result, write the outputs
back under the record's project and end the session. It does not own the registry or the
control socket, which ``letify.runtime.control`` owns, the door itself, which the worker
source owns, or how a provider reaches its runtime, which ``Provider.attach_channel`` does.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from ..errors import LetifyError, RemoteError, RunNotFound, UnsupportedMode
from . import control
from .lease import Lease
from .session import Runtime

if TYPE_CHECKING:
    from ..launcher import Launcher


def take(run: str, *, launcher: Launcher | None = None) -> Any:
    """Make this process the owner of ``run`` and return the call's value.

    Raises ``RunNotFound`` when no record names ``run``, ``UnsupportedMode`` when the run's
    provider cannot be reached by a second process or this process already owns the run, and
    ``RemoteError`` carrying the traceback when the body raised.
    """
    record = control.find(run)
    if record.get("pid") == os.getpid():
        raise UnsupportedMode(f"run {run} is owned by this process, which cannot take it over")
    if not record.get("door"):
        raise UnsupportedMode(
            f"run {run} has no second door in its record, so no process can take it over"
        )
    accounts = launcher or _launcher_for(record)
    provider = accounts.provider(str(record["provider"]))
    name = str(record["runtime"])
    # Built before the old owner is asked to let go, so a provider that refuses does so
    # without having disturbed a run it cannot take.
    channel = provider.attach_channel(name, record)
    _release(record)
    runtime = Runtime(
        name=name,
        provider=provider,
        instance=SimpleNamespace(key="takeover", accelerator="", devices=0),  # type: ignore[arg-type]
        env=SimpleNamespace(key=run),  # type: ignore[arg-type]
        channel=channel,
    )
    runtime._pool_key = f"takeover|{run}"
    owned = control.begin(runtime, str(record.get("function", "")), details=_details(record),
                          run_id=run)
    owned.started_at = float(record.get("started_at", owned.started_at))
    owned.rewrite()
    return _adopt_and_finish(runtime, record, owned)


def _details(record: dict[str, Any]) -> dict[str, Any]:
    """The fields of a record that describe the call rather than its owner."""
    keep = ("project", "config", "call_dir", "blobs", "outputs", "door", "external_id", "link")
    return {key: record[key] for key in keep if key in record}


def _launcher_for(record: dict[str, Any]) -> Launcher:
    """A launcher over the configuration the run was started with, without touching stubs."""
    from .. import stubs
    from ..launcher import Launcher

    config = record.get("config")
    source = config if isinstance(config, str) and Path(config).is_file() else None
    previous = os.environ.get(stubs.ENVIRONMENT)
    os.environ[stubs.ENVIRONMENT] = "0"
    try:
        return Launcher(source, announce=False)
    finally:
        if previous is None:
            del os.environ[stubs.ENVIRONMENT]
        else:
            os.environ[stubs.ENVIRONMENT] = previous


def _release(record: dict[str, Any]) -> None:
    """Ask the old owner, when it is alive, to let go. One that does not answer is gone."""
    pid = record.get("pid")
    if not isinstance(pid, int) or not control._alive(pid):
        return
    answer = control.ask(
        str(record["run"]), {"cmd": "release", "owner": os.getpid()}, wait=control.RELEASE_WAIT_S
    )
    if answer.get("ok"):
        return
    if answer.get("type") == "UnsupportedMode":
        raise UnsupportedMode(str(answer.get("error")))
    print(
        f"letify: the owner of {record['run']} did not let go ({answer.get('error')}); "
        f"treating it as gone",
        file=sys.stderr,
        flush=True,
    )


def _adopt_and_finish(runtime: Runtime, record: dict[str, Any], owned: control.Run) -> Any:
    """Connect, adopt, wait for the call, write its outputs back and end the session."""
    run = str(record["run"])
    assert runtime.channel is not None
    failure: RemoteError | None = None
    value: Any = None
    pieces = None
    try:
        runtime.channel.start()
        pieces = runtime.stream({"op": "adopt", "run": run, "owner": os.getpid()})
        try:
            next(pieces)
        except RemoteError as exc:
            raise LetifyError(f"run {run} cannot be taken over: {exc}") from exc
        if runtime.provider.needs_lease:
            runtime.lease = Lease(runtime)
            runtime.lease.arm()
        try:
            value = next(pieces)
        except RemoteError as exc:
            failure = exc
    except BaseException:
        # Not adopted, or interrupted while waiting: the run is without an owner again, and
        # the record says so by naming the process that had it before. The worker keeps the
        # call until the lease deadline, and its door takes the next owner.
        owned.forget()
        with contextlib.suppress(OSError):
            control.write_record(record)
        if runtime.lease is not None:
            runtime.lease.release()
        with contextlib.suppress(Exception):
            runtime.channel.detach()
        raise
    finally:
        if pieces is not None:
            pieces.close()
    # From here the call has ended at this process, so a failure below ends the session too.
    try:
        _write_back(runtime, record)
    finally:
        owned.end()
        runtime.shutdown(reason="takeover_done")
    if failure is not None:
        raise failure
    return value


def _write_back(runtime: Runtime, record: dict[str, Any]) -> None:
    """Copy the outputs the call made to the project the record names, not the working directory."""
    outputs = record.get("outputs") or []
    if not outputs or not record.get("call_dir"):
        return
    from ..store import pathdata

    project = Path(str(record.get("project") or "."))
    collector = pathdata.Collector(str(record["call_dir"]))
    for output in outputs:
        local = (project / str(output["local"])).resolve()
        collector.placed[str(local)] = pathdata.Placed(
            local, str(output["runtime"]), bool(output.get("directory")), output=True
        )
    pathdata.write_back(runtime, collector, str(record["blobs"]))


__all__ = ["RunNotFound", "take"]
