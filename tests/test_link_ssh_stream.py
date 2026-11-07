"""The ssh stream of a letify-link session.

Spec "Sidecar interface" item 11 and "Agent and pairing" item 12. A local TCP echo server stands
in for sshd. The shim and the forward are tested directly, and one test goes the whole way
through a real agent, both sidecars and a real session, which needs the built sidecar and is
skipped, with the reason, without cargo.
"""

from __future__ import annotations

# ruff: noqa: F811
import json
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from test_link_agent import (  # noqa: F401  (fixtures and helpers)
    agent_account,
    daemons,
    local_sidecar,
    start_service,
    user_pair,
)
from test_link_channel import load_shim

from letify.transport import linkagent, linkchannel, linkd, linksignal


class Echo:
    """A TCP server on loopback that sends back every byte, standing in for sshd."""

    def __init__(self, banner: bytes = b"") -> None:
        self.banner = banner
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.accepted = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            self.accepted += 1
            threading.Thread(target=self._echo, args=(conn,), daemon=True).start()

    def _echo(self, conn: socket.socket) -> None:
        with conn:
            if self.banner:
                conn.sendall(self.banner)
            while True:
                data = conn.recv(65536)
                if not data:
                    return
                conn.sendall(data)

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"

    def close(self) -> None:
        self.listener.close()


@pytest.fixture
def echo():
    server = Echo(b"SSH-2.0-echo\r\n")
    yield server
    server.close()


def read_until(sock: socket.socket, size: int, timeout: float = 10.0) -> bytes:
    sock.settimeout(timeout)
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def shim_serving(sock: socket.socket, **config: Any) -> threading.Thread:
    shim = load_shim().Shim({"resume_s": 10, "wait_s": 60, **config})
    thread = threading.Thread(target=shim.serve_socket, args=(sock,), daemon=True)
    thread.start()
    return thread


# -- the shim side (Sidecar interface item 11, the stream and the target) ------------------


def test_an_ssh_stream_copies_bytes_to_the_configured_target_and_back(echo) -> None:
    near, far = socket.socketpair()
    shim_serving(near, ssh_target=echo.target)
    far.sendall(linkchannel.stream_header("ssh", "ab01"))
    assert read_until(far, 14) == b"SSH-2.0-echo\r\n"
    far.sendall(b"hello")
    assert read_until(far, 5) == b"hello"
    far.close()


def test_an_ssh_stream_ends_its_direction_when_the_stream_half_closes(echo) -> None:
    near, far = socket.socketpair()
    shim_serving(near, ssh_target=echo.target)
    far.sendall(linkchannel.stream_header("ssh", "ab01") + b"last")
    far.shutdown(socket.SHUT_WR)
    assert read_until(far, 14 + 4) == b"SSH-2.0-echo\r\nlast"
    assert far.recv(10) == b""
    far.close()


def test_an_ssh_stream_never_takes_its_target_from_its_header(echo) -> None:
    other = Echo(b"SSH-2.0-other\r\n")
    try:
        near, far = socket.socketpair()
        shim_serving(near, ssh_target=echo.target)
        header = linkchannel.stream_header(
            "ssh", "ab01", host="127.0.0.1", port=other.port, target=other.target
        )
        far.sendall(header)
        assert read_until(far, 14) == b"SSH-2.0-echo\r\n"
        assert other.accepted == 0
        far.close()
    finally:
        other.close()


def test_an_ssh_stream_to_a_refused_target_is_closed_with_no_byte() -> None:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))  # bound and never listening: a connection is refused
    try:
        near, far = socket.socketpair()
        thread = shim_serving(near, ssh_target=f"127.0.0.1:{holder.getsockname()[1]}")
        far.sendall(linkchannel.stream_header("ssh", "ab01"))
        far.settimeout(10)
        assert far.recv(10) == b""
        thread.join(10)
        far.close()
    finally:
        holder.close()


def test_the_ssh_target_defaults_to_loopback_port_22() -> None:
    shim = load_shim().Shim({"resume_s": 10, "wait_s": 60})
    assert shim.ssh_address() == ("127.0.0.1", 22)
    configured = load_shim().Shim({"resume_s": 10, "wait_s": 60, "ssh_target": "10.0.0.5:2222"})
    assert configured.ssh_address() == ("10.0.0.5", 2222)


# -- the agent side (Agent and pairing item 12) ---------------------------------------------


def test_a_good_ssh_target_is_host_and_port_and_a_bad_one_is_refused_by_name() -> None:
    assert linkagent.parse_ssh_target("127.0.0.1:2222") == "127.0.0.1:2222"
    for bad in ("", "localhost", "host:", ":22", "host:0", "host:65536", "host:ssh", "a b:22"):
        with pytest.raises(linkagent.AgentError, match="ssh target"):
            linkagent.parse_ssh_target(bad)


def test_the_ssh_target_comes_from_the_flag_then_the_file_then_the_default(tmp_path) -> None:
    home = linkagent.AgentHome(tmp_path / "agent")
    assert linkagent.ssh_target(home) == "127.0.0.1:22"
    (home.path / "config.toml").write_text('ssh_target = "127.0.0.1:2200"\n', encoding="utf-8")
    assert linkagent.ssh_target(home) == "127.0.0.1:2200"
    assert linkagent.ssh_target(home, "127.0.0.1:2300") == "127.0.0.1:2300"


def test_the_agent_passes_its_ssh_target_into_every_shim_configuration() -> None:
    agent = linksignal.SignalAgent(
        SimpleNamespace(),  # type: ignore[arg-type]
        paired=["n"],
        binary="b",
        python="p",
        bootstrap="x",
        relay="y",
        ssh_target="127.0.0.1:2200",
    )
    config = agent.shim_config("n", {"ssh_target": "evil:1", "wait_s": 5})
    assert config["ssh_target"] == "127.0.0.1:2200"


# -- the local side (Sidecar interface item 11, the forward) --------------------------------


class StreamSession:
    """A session whose streams are socket pairs, recording the priority of each open."""

    def __init__(self, fail: bool = False) -> None:
        self.opened: list[str] = []
        self.far: list[socket.socket] = []
        self.fail = fail
        self.closed = threading.Event()

    def open_stream(self, priority: str = "interactive") -> socket.socket:
        if self.fail:
            raise linkd.LinkError("session gone")
        self.opened.append(priority)
        near, far = socket.socketpair()
        self.far.append(far)
        return near

    def close(self) -> None:
        self.closed.set()


def remote_over(session: Any) -> linkchannel.LinkRemote:
    daemon = SimpleNamespace(node_id="a" * 64, close=lambda: None)
    return linkchannel.LinkRemote(daemon, session, "box")  # type: ignore[arg-type]


def test_the_ssh_forward_listens_on_loopback_only() -> None:
    remote = remote_over(StreamSession())
    forward = remote.forward_ssh()
    try:
        assert forward.port > 0
        assert forward.listener.getsockname()[0] == "127.0.0.1"
        socket.create_connection(("127.0.0.1", forward.port), timeout=5).close()
    finally:
        forward.close()


def test_a_connection_to_the_forward_opens_one_interactive_stream_with_an_ssh_header() -> None:
    session = StreamSession()
    forward = remote_over(session).forward_ssh()
    try:
        client = socket.create_connection(("127.0.0.1", forward.port), timeout=5)
        deadline = time.monotonic() + 5
        while not session.far and time.monotonic() < deadline:
            time.sleep(0.01)
        far = session.far[0]
        assert session.opened == ["interactive"]
        far.settimeout(5)
        line = b""
        while not line.endswith(b"\n"):
            line += far.recv(1)
        header = json.loads(line)
        assert header["kind"] == "ssh" and header["token"]
        client.sendall(b"ping")
        assert read_until(far, 4) == b"ping"
        far.sendall(b"pong")
        assert read_until(client, 4) == b"pong"
        client.close()
    finally:
        forward.close()


def test_a_forward_whose_stream_cannot_be_opened_closes_the_connection_and_keeps_listening() -> (
    None
):
    forward = remote_over(StreamSession(fail=True)).forward_ssh()
    try:
        client = socket.create_connection(("127.0.0.1", forward.port), timeout=5)
        client.settimeout(5)
        assert client.recv(10) == b""
        client.close()
        socket.create_connection(("127.0.0.1", forward.port), timeout=5).close()
    finally:
        forward.close()


def test_closing_the_remote_closes_its_forwards() -> None:
    remote = remote_over(StreamSession())
    port = remote.forward_ssh().port
    remote.close()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2)


# -- the whole path --------------------------------------------------------------------------


def test_an_ssh_stream_reaches_the_agents_sshd(
    isolated_home, start_service, signal_service, sidecar_binary, udp_stun, monkeypatch, echo
) -> None:
    from letify.config.secrets import account_directory

    monkeypatch.setenv("LETIFY_LINK_STUN", udp_stun.env)
    agent = start_service(ssh_target=echo.target)
    provider = agent_account(isolated_home, agent, signal_service)
    directory = account_directory("box")
    directory.mkdir(mode=0o700, parents=True)
    user = linkd.LinkDaemon(directory / "link.key", binary=sidecar_binary)
    try:
        user_pair(signal_service, user, agent.service.new_token(signal_service.url))
    finally:
        user.close()
    forward = provider.ssh_forward()
    try:
        client = socket.create_connection(("127.0.0.1", forward.port), timeout=10)
        assert read_until(client, 14, 30) == b"SSH-2.0-echo\r\n"
        client.sendall(b"through the session")
        assert read_until(client, 19, 30) == b"through the session"
        client.close()
        assert echo.accepted == 1
    finally:
        forward.close()
        provider.close_link()
