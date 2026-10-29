"""The letify workers on a shell host: listing them, finding orphans and stopping one.

Spec section pinned here: "Provider session discovery", "Workers on a shell host" (id
shell-workers). The worker is the real one behind the real framed protocol, so the record it
writes is the real record. The remote program that lists and stops workers runs on this
machine through the same interpreter; only the SSH hop to a host is replaced.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import provider_of

import letify
from letify.cli import main
from letify.declare.env import Env
from letify.declare.instance import Instance
from letify.protocol.worker import BOOTSTRAP
from letify.providers.colab import Colab
from letify.providers.local import Local
from letify.providers.shell import Shell
from letify.runtime import workers
from letify.runtime.channel import PersistentChannel

HOST = socket.gethostname()


def wait_for(condition, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


class LocalHop:
    """Stands in for the SSH hop: runs the remote program on this machine."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, provider: Shell, source: str, *args: str, timeout: float) -> str:
        self.calls.append(args)
        result = subprocess.run(
            [sys.executable, "-c", source, *args], capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise letify.RuntimeFailure("program failed", stderr=result.stderr)
        return result.stdout


def install_hop(monkeypatch) -> LocalHop:
    hop = LocalHop()
    monkeypatch.setattr(
        Shell,
        "_run_program",
        lambda provider, source, *args, timeout=120: hop(provider, source, *args, timeout=timeout),
    )
    return hop


@pytest.fixture
def hop(monkeypatch) -> LocalHop:
    return install_hop(monkeypatch)


@pytest.fixture
def machine(tmp_path: Path, hop: LocalHop) -> Shell:
    return provider_of(Shell, "lab", address="gpu.example.edu", workspace=str(tmp_path / "ws"))


def records_dir(provider: Shell) -> Path:
    path = Path(os.path.expanduser(provider.workspace_root)) / workers.RECORD_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def write_record(provider: Shell, process: subprocess.Popen | None, **fields: Any) -> dict:
    """A worker record for ``process``, with the fields a test overrides."""
    pid = fields.pop("pid", None) or (process.pid if process is not None else 2**22 + 7)
    record = {
        "pid": pid,
        "proc_start": workers.proc_start(pid) if process is not None else None,
        "started": "2026-10-29T01:02:03Z",
        "runtime": "letify-a100-1",
        "key": "A100|env",
        "devices": "0,1",
        "owner": {"host": HOST, "pid": os.getpid()},
        "run": None,
        "lease_deadline": time.time() + 300,
        **fields,
    }
    (records_dir(provider) / f"{pid}.json").write_text(json.dumps(record))
    return record


@pytest.fixture
def children():
    started: list[subprocess.Popen] = []
    yield started
    for process in started:
        process.kill()
        process.wait()


def dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


# -- the worker records itself -------------------------------------------------------------


def register(channel: PersistentChannel, root: Path) -> Any:
    value, _logs = channel.request(
        {
            "op": "register",
            "workspace": str(root),
            "runtime": "letify-a100-1",
            "key": "A100|env",
            "owner": {"host": HOST, "pid": os.getpid()},
        },
        timeout=60,
    )
    return value


def test_a_worker_writes_its_record_when_registered_and_removes_it_at_exit(tmp_path) -> None:
    channel = PersistentChannel([sys.executable, "-u", "-c", BOOTSTRAP], name="t")
    channel.start()
    try:
        register(channel, tmp_path)
        pid = int(channel.request({"op": "stat"}, timeout=60)[0]["pid"])
        path = tmp_path / workers.RECORD_DIR / f"{pid}.json"
        record = json.loads(path.read_text())
        assert record["pid"] == pid
        assert record["runtime"] == "letify-a100-1"
        assert record["key"] == "A100|env"
        assert record["owner"] == {"host": HOST, "pid": os.getpid()}
        assert record["proc_start"] == workers.proc_start(pid)
        assert record["started"].endswith("Z")
        assert record["run"] is None
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    finally:
        channel.close()
    assert wait_for(lambda: not path.exists())


def test_a_lease_renewal_rewrites_the_deadline_in_the_record(tmp_path) -> None:
    channel = PersistentChannel([sys.executable, "-u", "-c", BOOTSTRAP], name="t")
    channel.start()
    try:
        register(channel, tmp_path)
        pid = int(channel.request({"op": "stat"}, timeout=60)[0]["pid"])
        path = tmp_path / workers.RECORD_DIR / f"{pid}.json"
        assert json.loads(path.read_text())["lease_deadline"] is None
        before = time.time()
        channel.request({"op": "lease", "grace": 300}, timeout=60)
        deadline = json.loads(path.read_text())["lease_deadline"]
        assert before + 299 < deadline < before + 310
    finally:
        channel.close()


def test_a_worker_that_is_terminated_removes_its_record(tmp_path) -> None:
    channel = PersistentChannel([sys.executable, "-u", "-c", BOOTSTRAP], name="t")
    channel.start()
    register(channel, tmp_path)
    pid = int(channel.request({"op": "stat"}, timeout=60)[0]["pid"])
    path = tmp_path / workers.RECORD_DIR / f"{pid}.json"
    assert path.exists()
    os.kill(pid, signal.SIGTERM)
    assert wait_for(lambda: not path.exists())
    channel.close()


class RecordingLocal(Local):
    """A local provider that boots like a shell host: a workspace root and a worker record."""

    prepares_workspace = True
    records_workers = True
    needs_lease = True

    @property
    def workspace_root(self) -> str:
        return str(self.config.option("workspace"))


def test_a_booted_session_registers_its_worker_with_its_runtime_name(tmp_path) -> None:
    root = tmp_path / "ws"
    provider = provider_of(RecordingLocal, "lab", python=sys.executable, workspace=str(root))
    instance = Instance(provider, gpu=None)._placed("remote")
    runtime = provider.start(instance, Env(lock=str(tmp_path / "absent.lock")), name="lab-7")
    try:
        (path,) = (root / workers.RECORD_DIR).glob("*.json")
        record = json.loads(path.read_text())
        assert record["pid"] == runtime.worker_pid
        assert record["runtime"] == "lab-7"
        assert record["key"] == runtime.key
        assert record["owner"] == {"host": HOST, "pid": os.getpid()}
        assert record["lease_deadline"] is not None
    finally:
        runtime.shutdown()
    assert wait_for(lambda: not path.exists())


# -- listing ---------------------------------------------------------------------------------


def test_a_shell_host_lists_its_running_workers(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, run="abc123def456")
    assert machine.discovers_sessions is True
    (line,) = machine.sessions()
    assert f"pid {process.pid}" in line
    assert "gpu 0,1" in line
    assert "started 2026-10-29T01:02:03Z" in line
    assert "runtime letify-a100-1" in line
    assert "key A100|env" in line
    assert "run abc123def456" in line
    assert f"owner {HOST}:{os.getpid()} alive" in line


def test_a_dead_worker_record_is_dropped_and_deleted(machine, children) -> None:
    live = sleeper()
    children.append(live)
    write_record(machine, live)
    gone = write_record(machine, None, pid=dead_pid())
    stale = records_dir(machine) / f"{gone['pid']}.json"
    assert stale.exists()
    lines = machine.sessions()
    assert len(lines) == 1
    assert f"pid {live.pid}" in lines[0]
    assert not stale.exists()


def test_a_reused_pid_is_not_a_worker(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, proc_start=workers.proc_start(process.pid) + 99999)
    assert machine.sessions() == []


def test_a_host_with_no_workers_lists_nothing(machine) -> None:
    assert machine.sessions() == []
    assert machine.orphans() == []


def test_a_worker_with_no_devices_reads_gpu_none(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, devices=None)
    assert "gpu none" in machine.sessions()[0]


def test_an_owner_on_another_host_is_remote(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, owner={"host": HOST + "-other", "pid": 1})
    assert machine.sessions()[0].endswith("remote")


# -- orphans ---------------------------------------------------------------------------------


def test_a_worker_whose_owner_died_is_an_orphan(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, owner={"host": HOST, "pid": dead_pid()})
    assert machine.orphans() == [str(process.pid)]
    assert machine.sessions()[0].endswith("gone")


def test_a_worker_whose_owner_lives_is_never_an_orphan(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process)
    assert machine.orphans() == []


def test_a_worker_whose_owner_is_on_another_host_is_not_judged(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, owner={"host": HOST + "-other", "pid": dead_pid()})
    assert machine.orphans() == []


def test_a_worker_whose_lease_passed_is_an_orphan_whoever_owns_it(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, lease_deadline=time.time() - 5)
    assert machine.orphans() == [str(process.pid)]


def test_a_dead_owner_with_a_running_call_and_a_live_lease_can_still_be_taken_over(
    machine, children
) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, owner={"host": HOST, "pid": dead_pid()}, run="abc123def456")
    assert machine.orphans() == []


# -- stopping --------------------------------------------------------------------------------


def test_stopping_a_worker_sends_sigterm_and_reports_it_gone(machine, children) -> None:
    process = sleeper()
    children.append(process)
    write_record(machine, process, owner={"host": HOST, "pid": dead_pid()})
    assert machine.stop_orphan(str(process.pid)) is True
    assert process.wait(timeout=20) == -signal.SIGTERM


STUBBORN = (
    "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "print('ready', flush=True); time.sleep(120)"
)


def test_stopping_an_orphan_sends_sigterm_then_sigkill_after_the_grace(
    machine, children, monkeypatch
) -> None:
    monkeypatch.setattr(workers, "STOP_GRACE_S", 1.0)
    process = subprocess.Popen([sys.executable, "-c", STUBBORN], stdout=subprocess.PIPE)
    children.append(process)
    assert process.stdout.readline().strip() == b"ready"
    write_record(machine, process, owner={"host": HOST, "pid": dead_pid()})
    started = time.monotonic()
    assert machine.stop_orphan(str(process.pid)) is True
    assert time.monotonic() - started >= 1.0
    assert process.wait(timeout=20) == -signal.SIGKILL


def test_a_pid_that_is_not_a_worker_record_is_refused(machine, children) -> None:
    process = sleeper()
    children.append(process)
    with pytest.raises(letify.LetifyError, match="not a letify worker"):
        machine.stop_orphan(str(process.pid))
    assert process.poll() is None


def test_a_non_numeric_ref_is_refused(machine) -> None:
    with pytest.raises(letify.LetifyError, match="not a letify worker"):
        machine.stop_orphan("1; rm -rf /")


# -- the command line ------------------------------------------------------------------------


def test_stop_worker_without_an_alias_is_a_usage_error(capsys) -> None:
    assert main(["stop", "--worker", "4242"]) == 2
    assert "--alias" in capsys.readouterr().err


def test_the_launcher_stops_one_named_worker(tmp_path, monkeypatch, children) -> None:
    project = tmp_path / "proj" / ".letify"
    project.mkdir(parents=True)
    (project / "config.toml").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    install_hop(monkeypatch)
    let = letify.Launcher(project, home=False, announce=False)
    process = sleeper()
    children.append(process)
    write_record(let.provider("lab"), process)
    row = let.stop_worker("lab", str(process.pid))
    assert row == {"alias": "lab", "ref": str(process.pid), "stopped": True}
    assert process.wait(timeout=20) == -signal.SIGTERM


def test_the_launcher_lists_a_shell_host_instead_of_refusing(tmp_path, monkeypatch, children):
    project = tmp_path / "proj" / ".letify"
    project.mkdir(parents=True)
    (project / "config.toml").write_text(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    install_hop(monkeypatch)
    let = letify.Launcher(project, home=False, announce=False)
    process = sleeper()
    children.append(process)
    write_record(let.provider("lab"), process)
    (row,) = [row for row in let.sessions() if row["alias"] == "lab"]
    assert row["reason"] is None
    assert f"pid {process.pid}" in row["sessions"][0]


# -- Colab ---------------------------------------------------------------------------------


def test_colab_session_discovery_is_unchanged() -> None:
    assert Colab.sessions is not Shell.sessions
    assert Colab.orphans is not Shell.orphans
    assert Colab.stop_orphan is not Shell.stop_orphan
    assert Colab.discovers_sessions is True
    assert Colab.records_workers is False
    assert Shell.records_workers is True


def test_colab_refuses_to_stop_a_worker_by_pid() -> None:
    provider = provider_of(Colab, "c")
    with pytest.raises(letify.UnsupportedMode):
        provider.stop_worker("4242")


def test_hosts_reached_only_through_a_connection_pipeline_do_not_discover_sessions() -> None:
    # Spec "Workers on a shell host": listing must not punch, start Tailcat or allocate a
    # machine, so a host with no direct address answers that it cannot be asked.
    from letify.providers.elice import Elice
    from letify.providers.tunnel import Tunnel

    assert Tunnel.discovers_sessions is False
    assert Elice.discovers_sessions is False
