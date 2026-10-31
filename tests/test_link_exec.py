"""Helper commands over a letify-link session as ``exec`` streams.

Spec "Sidecar interface" item 7, the exec stream. The machine is this one: the shim, both
sidecars and the session are real, as in tests/test_link_channel.py. The tests need the built
sidecar and are skipped, with the reason, without cargo.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from typing import Any

import pytest
from test_link_channel import (  # noqa: F401  (fixtures)
    FakeSshLink,
    LocalShell,
    linked,
    load_shim,
    local_sidecar,
    runtime,
    ssh,
)

from letify.errors import RuntimeFailure
from letify.transport import linkchannel


def remote_of(provider: Any) -> Any:
    return provider.link_remote(runtime())


def test_an_exec_stream_returns_the_status_and_both_outputs(linked) -> None:
    code = "import sys; print('out'); sys.stderr.write('err'); sys.exit(3)"
    result = remote_of(linked).exec([sys.executable, "-c", code], 20)
    assert (result.status, result.stdout, result.stderr) == (3, "out\n", "err")
    assert not result.timed_out and not result.truncated


def test_an_exec_argument_is_never_split_or_expanded_by_a_shell(linked) -> None:
    tricky = "a b; echo $HOME `id` *"
    result = remote_of(linked).exec([sys.executable, "-c", "import sys; print(sys.argv[1])", tricky], 20)
    assert result.stdout == tricky + "\n"


def test_an_exec_argv_larger_than_the_header_limit_arrives_whole(linked) -> None:
    big = "x" * 200_000
    code = "import sys; print(len(sys.argv[1]))"
    assert remote_of(linked).exec([sys.executable, "-c", code, big], 20).stdout == "200000\n"


def test_an_exec_that_outlives_its_timeout_is_killed_and_says_so(linked) -> None:
    result = remote_of(linked).exec([sys.executable, "-c", "import time; time.sleep(60)"], 1)
    assert result.timed_out and result.status != 0


def test_an_exec_whose_output_passes_the_bound_is_killed_and_truncated(linked) -> None:
    code = "import sys\nwhile True:\n    sys.stdout.write('y' * 4096)\n    sys.stdout.flush()\n"
    result = remote_of(linked).exec([sys.executable, "-c", code], 20, max_output=10_000)
    assert result.truncated and len(result.stdout) <= 10_000


def test_an_exec_of_a_missing_program_reports_status_minus_one(linked) -> None:
    result = remote_of(linked).exec(["/no/such/program-letify"], 20)
    assert result.status == -1 and "No such file" in result.stderr


def test_an_exec_argv_that_is_not_a_list_of_strings_is_closed_with_no_frame() -> None:
    shim = load_shim().Shim({"resume_s": 10, "wait_s": 60, "python": sys.executable})
    for argv in (b'"ls"', b"[]", b"[1, 2]", b"not json"):
        near, far = socket.socketpair()
        far.settimeout(10)
        thread = threading.Thread(target=shim.serve_socket, args=(near,), daemon=True)
        thread.start()
        far.sendall(
            linkchannel.stream_header(
                "exec", "ab", argv_bytes=len(argv), timeout_s=5, max_output=100
            )
            + argv
        )
        assert far.recv(10) == b""
        thread.join(10)
        far.close()


def test_a_link_account_runs_its_helper_commands_over_exec_not_ssh(linked, ssh, monkeypatch) -> None:
    remote_of(linked)
    ssh.commands.clear()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a helper command used a subprocess: {args}")

    monkeypatch.setattr(subprocess, "run", refuse)
    code = "import sys; print(sys.argv[1])"
    out = linked._run_program(code, "hello")
    assert out == "hello\n"
    run = linked._remote_runner("failed")
    assert run((sys.executable, "-c", "print(5)")) == "5\n"
    assert ssh.commands == []


def test_a_failing_helper_over_exec_raises_with_its_standard_error(linked) -> None:
    run = linked._remote_runner("the query failed")
    remote_of(linked)
    with pytest.raises(RuntimeFailure) as raised:
        run((sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(2)"))
    assert "the query failed" in str(raised.value) and "boom" in raised.value.stderr


def test_without_a_live_session_a_link_account_keeps_the_ssh_route(linked, ssh) -> None:
    out = linked._run_program("print('via ssh')")
    assert out == "via ssh\n"
    assert ssh.commands, "the bootstrap route was not used"


def test_a_plain_ssh_account_never_asks_for_a_session(isolated_home, tmp_path, ssh) -> None:
    from test_link_channel import machine

    provider = machine(tmp_path, ssh)
    assert provider.link_exec_remote() is None
    assert json  # keep the import used
