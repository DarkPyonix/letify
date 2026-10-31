"""The development upload of letify-link to a runtime.

Spec "Sidecar interface" item 7 (Development upload). The runtime is this machine: the command
channel is the fake of tests/test_link_command_channel.py and the ssh hop is ``sh -c``.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest
from test_link_channel import FakeSsh, machine, runtime, worker_pid
from test_link_command_channel import FakeCommands, colab, worker_facts

from letify.errors import ProviderUnavailable
from letify.transport import linkd, linkupload

MIB = 1 << 20


@pytest.fixture(autouse=True)
def local_sidecar(sidecar_binary, monkeypatch, udp_stun):
    monkeypatch.setenv(linkd.ENV_BINARY, str(sidecar_binary))
    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def elf(path: Path, machine: int, size: int = 1000) -> Path:
    head = b"\x7fELF\x02\x01\x01" + bytes(9) + (2).to_bytes(2, "little")
    head += machine.to_bytes(2, "little")
    path.write_bytes(head + os.urandom(size))
    return path


def chunk_calls(commands: FakeCommands) -> int:
    return sum(1 for s in commands.sources if "base64.b64decode(DATA)" in s)


def test_an_uploaded_binary_is_sent_once_and_reused_by_checksum(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    commands = FakeCommands(0.05, sidecar_binary)
    provider = colab(tmp_path, commands, monkeypatch, link_binary_upload=str(sidecar_binary))
    channel = provider.open_channel(runtime())
    try:
        assert channel.call(worker_facts(), (), {})[0]["pid"] != os.getpid()
        placed = tmp_path / "ws" / ".letify" / "bin" / f"letify-link-{sha16(sidecar_binary)}"
        assert placed.read_bytes() == sidecar_binary.read_bytes()
        assert placed.stat().st_mode & 0o777 == 0o700
        assert commands.extractions == 0
        first = chunk_calls(commands)
        assert first >= 1
    finally:
        channel.close()
        provider.close_link(runtime())
    # A second session of the same runtime finds the file by its checksum.
    again = provider.open_channel(runtime())
    try:
        assert again.call(worker_facts(), (), {})[0]["pid"] != os.getpid()
        assert chunk_calls(commands) == first
    finally:
        again.close()
        provider.close_link(runtime())


def test_the_environment_names_the_binary_when_the_account_does_not(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(linkupload.ENV_UPLOAD, str(sidecar_binary))
    commands = FakeCommands(0.05, sidecar_binary)
    provider = colab(tmp_path, commands, monkeypatch)
    channel = provider.open_channel(runtime())
    try:
        assert chunk_calls(commands) >= 1
        assert commands.extractions == 0
    finally:
        channel.close()
        provider.close_link(runtime())


def test_a_corrupted_upload_fails_naming_the_checksum(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    commands = FakeCommands(0.05, sidecar_binary)
    original = commands.run

    def flip(source: str, timeout: float | None = None) -> str:
        if "base64.b64decode(DATA)" in source:
            head, rest = source.split("DATA = '", 1)
            rest = ("B" if rest[0] != "B" else "C") + rest[1:]
            source = head + "DATA = '" + rest
        return original(source, timeout)

    commands.run = flip  # type: ignore[method-assign]
    provider = colab(tmp_path, commands, monkeypatch, link_binary_upload=str(sidecar_binary))
    with pytest.raises(ProviderUnavailable) as raised:
        provider.open_channel(runtime())
    message = str(raised.value)
    assert "checksum mismatch" in message
    assert hashlib.sha256(sidecar_binary.read_bytes()).hexdigest() in message
    # The bad part file is gone and nothing is installed under the final name.
    directory = tmp_path / "ws" / ".letify" / "bin"
    assert not list(directory.glob("letify-link-*"))
    provider.close_link(runtime())


def test_a_binary_built_for_another_architecture_is_refused_naming_both(
    sidecar_binary, isolated_home, tmp_path, monkeypatch
) -> None:
    import platform

    other = 183 if platform.machine() == "x86_64" else 62
    wrong = elf(tmp_path / "letify-link", other)
    commands = FakeCommands(0.05, sidecar_binary)
    provider = colab(tmp_path, commands, monkeypatch, link_binary_upload=str(wrong))
    with pytest.raises(ProviderUnavailable) as raised:
        provider.open_channel(runtime())
    message = str(raised.value)
    assert "aarch64" in message and "x86_64" in message
    assert f"Linux {platform.machine()}" in message
    assert chunk_calls(commands) == 0
    provider.close_link(runtime())


def test_a_file_that_is_not_elf_is_refused(tmp_path) -> None:
    text = tmp_path / "letify-link"
    text.write_text("#!/bin/sh\n")
    with pytest.raises(ProviderUnavailable) as raised:
        linkupload.ensure("a", text, workspace=str(tmp_path), run=lambda s, t: "")
    assert "is not an ELF file" in str(raised.value)


def test_ten_mib_over_a_one_second_channel_takes_a_few_calls(tmp_path) -> None:
    binary = elf(tmp_path / "letify-link", 62 if os.uname().machine == "x86_64" else 183)
    with open(binary, "ab") as handle:
        handle.write(os.urandom(10 * MIB))
    commands = FakeCommands(1.0, binary)
    started = time.monotonic()
    path = linkupload.ensure("a", binary, workspace=str(tmp_path / "ws"), run=commands.run)
    elapsed = time.monotonic() - started
    assert Path(path).read_bytes() == binary.read_bytes()
    # One check, 3 chunks of 4 MiB, one finish: 5 calls of 1 s.
    assert len(commands.sources) == 5
    assert 5.0 <= elapsed < 8.0, elapsed


def test_a_shell_account_uploads_over_the_ssh_standard_input(
    sidecar_binary, isolated_home, tmp_path
) -> None:
    ssh = FakeSsh()
    provider = machine(tmp_path, ssh, link="letify-link", link_binary_upload=str(sidecar_binary))
    channel = provider.open_channel(runtime())
    try:
        assert channel.call(worker_pid(), (), {})[0] != os.getpid()
        placed = tmp_path / "ws" / ".letify" / "bin" / f"letify-link-{sha16(sidecar_binary)}"
        assert placed.read_bytes() == sidecar_binary.read_bytes()
        assert any(c.startswith("cat > ") for c in ssh.commands)
        count = len(ssh.commands)
    finally:
        channel.close()
        provider.close_link()
    again = provider.open_channel(runtime())
    try:
        # Only the check and the shim: no second push.
        assert sum(1 for c in ssh.commands if c.startswith("cat > ")) == 1
        assert len(ssh.commands) == count + 2
    finally:
        again.close()
        provider.close_link()
