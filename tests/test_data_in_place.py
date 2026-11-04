"""A provider that shares this machine's file system leaves a call's data in place.

Spec section pinned here: "Data in place" of "Project data".

Everything runs through the Local provider and the real framed worker, whose runtime is this
machine. The staging that other providers perform is pinned in test_data.py and
test_data_streaming.py, which run it through the staging_local fixture.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import letify


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = tmp_path / "study"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'study'\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def call_directories(home: Path) -> list[Path]:
    calls = home / ".letify" / "runtime" / "data" / "calls"
    return sorted(calls.iterdir()) if calls.is_dir() else []


def wait_for(path: Path, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def test_a_local_call_writes_its_output_straight_into_the_project_path(
    let, cpu, project
) -> None:
    out = project / "runs" / "exp1"
    release = project / "release"

    @let.function(device=cpu, host=letify.remote)
    def train(target: Path, gate: str) -> str:
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.pt").write_bytes(b"checkpoint")
        import os
        import time

        while not os.path.exists(gate):
            time.sleep(0.02)
        return str(target)

    result: list[str] = []
    caller = threading.Thread(target=lambda: result.append(train(out, str(release))))
    caller.start()
    try:
        # The call has not returned, so no final write-back could have put this here.
        wait_for(out / "model.pt")
        assert (out / "model.pt").read_bytes() == b"checkpoint"
        assert caller.is_alive()
    finally:
        release.write_text("go", encoding="utf-8")
        caller.join(30)
    assert result == [str(out)]


def test_a_local_body_sees_the_path_the_caller_passed(let, cpu, project) -> None:
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def where(path: Path) -> tuple[str, str]:
        return str(path), path.read_text(encoding="utf-8")

    assert where(data) == (str(data), "hello")


def test_a_local_call_creates_no_call_directory_and_sends_no_data(
    let, cpu, project, capsys, tmp_path
) -> None:
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")
    out = project / "out"

    @let.function(device=cpu, host=letify.remote)
    def work(src: Path, dst: Path) -> None:
        dst.mkdir()
        (dst / "copy.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    work(data, out)
    assert (out / "copy.txt").read_text(encoding="utf-8") == "hello"
    assert call_directories(tmp_path / "home") == []
    assert "letify: data" not in capsys.readouterr().err


def test_a_local_call_that_raises_keeps_what_it_wrote(let, cpu, project) -> None:
    out = project / "partial.txt"

    @let.function(device=cpu, host=letify.remote, retries=0)
    def fail(target: Path) -> None:
        target.write_text("half done", encoding="utf-8")
        raise RuntimeError("boom")

    with pytest.raises(Exception, match="boom"):
        fail(out)
    assert out.read_text(encoding="utf-8") == "half done"


def test_a_local_call_stopped_by_a_signal_keeps_what_it_wrote(project, tmp_path) -> None:
    out = project / "ckpt.txt"
    ready = project / "ready"
    script = tmp_path / "driver.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path
            import letify

            let = letify.Launcher(Path(sys.argv[1]) / ".letify", home=False, announce=False)
            cpu = let.providers.local.CPU

            @let.function(device=cpu, host=letify.remote)
            def train(target: Path, ready: Path) -> None:
                import time
                target.write_text("saved", encoding="utf-8")
                ready.write_text("go", encoding="utf-8")
                time.sleep(120)

            train(Path(sys.argv[2]), Path(sys.argv[3]))
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, HOME=os.environ["HOME"])
    proc = subprocess.Popen(
        [sys.executable, "-I", str(script), str(project), str(out), str(ready)],
        cwd=project, env=env,
    )
    try:
        wait_for(ready)
        proc.send_signal(signal.SIGTERM)
        proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert out.read_text(encoding="utf-8") == "saved"


def test_a_local_call_does_not_apply_the_data_root_refusal(let, cpu, project, tmp_path) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("still readable", encoding="utf-8")

    @let.function(device=cpu, host=letify.remote)
    def read(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    assert read(outside) == "still readable"
