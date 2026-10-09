"""QUIC over a punched UDP pair.

Spec "QUIC over a punched UDP pair". The binary is a build product, so the tests that
need it are skipped where it is absent, and everything else is exercised against fakes.
"""

from __future__ import annotations

import pytest
from conftest import CannedRendezvous

from letify.transport import nat, quic
from letify.transport.strategies import QuicUDP, Target

# -- Spec: QUIC over a punched UDP pair, finding the binary ------------------------


def test_the_carrier_is_found_in_the_wheel_before_the_path(monkeypatch, tmp_path) -> None:
    bundled = tmp_path / "letify-quic"
    bundled.write_text("#!/bin/true\n", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    assert quic.carrier_path() == bundled


def test_no_carrier_anywhere_is_not_an_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    monkeypatch.setattr(quic.shutil, "which", lambda name: None)
    assert quic.carrier_path() is None


# -- Spec: QUIC over a punched UDP pair, the strategy ------------------------------


def test_quic_needs_the_carrier_and_a_rendezvous(monkeypatch, tmp_path) -> None:
    assert QuicUDP.rank == 5
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    monkeypatch.setattr(quic.shutil, "which", lambda name: None)
    target = Target(alias="lab", rendezvous=CannedRendezvous())
    assert "letify-quic is not in this wheel" in QuicUDP().needs(target)

    bundled = tmp_path / "letify-quic"
    bundled.write_text("", encoding="utf-8")
    bundled.chmod(0o755)
    assert QuicUDP().needs(Target(alias="lab")) == "no rendezvous"
    assert QuicUDP().needs(target) is None


def test_the_strategy_exchanges_both_endpoints_and_proxies_ssh(
    monkeypatch, tmp_path, patch_run
) -> None:
    # Spec "QUIC over a punched UDP pair": the rendezvous carries both punched endpoints
    # and the token, and the client side becomes an SSH ProxyCommand.
    bundled = tmp_path / "letify-quic"
    bundled.write_text("", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(quic, "LIB_DIR", tmp_path)
    monkeypatch.setattr(
        nat, "stun_mapping", lambda port, server=None, **kw: ("198.51.100.1", port)
    )
    recorder = patch_run(__import__("letify.transport.strategies", fromlist=["x"]))
    rendezvous = CannedRendezvous({"mapping": ["203.0.113.9", 44444]})
    target = Target(alias="lab", rendezvous=rendezvous, user="root")

    link = QuicUDP().attempt(target)

    request = rendezvous.requests[0]
    assert request["kind"] == "quic"
    assert request["mapping"][0] == "198.51.100.1"
    assert len(bytes.fromhex(request["token"])) == nat.TOKEN_BYTES
    command = link.ssh_command("uname")
    proxy = next(part for part in command if part.startswith("ProxyCommand="))
    assert "letify-quic" in proxy and "connect" in proxy
    assert "--peer 203.0.113.9:44444" in proxy
    assert f"--token {request['token']}" in proxy
    assert recorder.command[-1] == "exit 0"


# -- Spec: QUIC over a punched UDP pair, the remote half --------------------------


def test_the_remote_half_reports_its_mapping_and_starts_the_carrier(
    patch_popen, monkeypatch
) -> None:
    monkeypatch.setattr(
        nat, "stun_mapping", lambda port, server=None, **kw: ("203.0.113.9", port)
    )
    started = patch_popen(nat, [])
    answer, _ = nat.begin(
        {
            "kind": "quic",
            "mapping": ["198.51.100.1", 41000],
            "token": "ab" * 16,
            "ssh_port": 22,
            "binary": "/tmp/letify-quic",
            "stun": ["stun.example", 443],
        }
    )
    assert "mapping" in answer
    command = started[0].command
    assert command[0] == "/tmp/letify-quic"
    assert command[1] == "serve"
    assert "--peer" in command and "198.51.100.1:41000" in command
    assert "--forward" in command and "22" in command


def test_the_remote_source_fetches_the_wheel_for_its_own_platform() -> None:
    # Spec "QUIC over a punched UDP pair": the remote gets the binary from the letify
    # wheel at this client's version, which PyPI verifies, rather than a pinned hash.
    from letify import install

    source = install.remote_quic_source("9.9.9")
    # The version reaches the remote as the call's argument; the requirement itself is
    # built there, so the source carries the pattern rather than the finished string.
    assert "pip" in source and 'f"letify=={version}"' in source
    assert "_remote_quic('9.9.9')" in source
    assert "letify/remoting/lib/letify-quic" in source
    assert "LETIFY-QUIC " in source
    compile(source, "<remote>", "exec")


# -- Spec: QUIC over a punched UDP pair, the binary itself ------------------------


@pytest.mark.skipif(quic.carrier_path() is None, reason="letify-quic is a build product")
def test_the_carrier_refuses_a_call_with_no_token() -> None:
    import subprocess

    done = subprocess.run(
        [str(quic.carrier_path()), "serve", "--bind", "0", "--peer", "127.0.0.1:1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 2
    assert "--token is required" in done.stderr


@pytest.mark.skipif(quic.carrier_path() is None, reason="letify-quic is a build product")
def test_the_carrier_carries_a_stream_between_two_of_itself(tmp_path) -> None:
    # Spec "QUIC over a punched UDP pair": a real QUIC v1 handshake, then the stream is
    # spliced to the forwarded port on one side and to stdio on the other.
    import secrets
    import socket
    import subprocess
    import threading

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    forward = listener.getsockname()[1]
    message = b"over a QUIC stream" * 64

    def answer() -> None:
        conn, _ = listener.accept()
        conn.sendall(message)
        conn.close()

    threading.Thread(target=answer, daemon=True).start()

    def free() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    token = secrets.token_bytes(16).hex()
    server_port, client_port = free(), free()
    carrier = str(quic.carrier_path())
    server = subprocess.Popen(
        [carrier, "serve", "--bind", str(server_port), "--peer",
         f"127.0.0.1:{client_port}", "--token", token, "--forward", str(forward)],
        stderr=subprocess.PIPE,
    )
    try:
        client = subprocess.run(
            [carrier, "connect", "--bind", str(client_port), "--peer",
             f"127.0.0.1:{server_port}", "--token", token],
            capture_output=True,
            timeout=60,
        )
        assert client.stdout.startswith(message[:64]), client.stderr[-400:]
    finally:
        server.kill()
        listener.close()
