"""The remote half of a letify-link worker channel. Runs on the machine, standard library only.

Owns starting the remote ``letify-link`` sidecar, listening for the one local node, relaying
rendezvous messages as JSON lines on standard input and output until the hand-over, and
serving each stream the local side opens: a new worker process, the relay to a running
worker's second door, or the tail of a worker's standard error, as the first line of the
stream says. It does not own the local side (``linkchannel.py``), the worker program and the
relay program (both are passed in) or any path decision. Spec "Sidecar interface" item 7.
This file is sent to the machine as the argument of ``python -c``, so it imports nothing from
letify and avoids syntax newer than Python 3.6.
"""

# ruff: noqa: SIM115, UP031
# The file is sent to old interpreters as source: no context-manager rewrites, no f-strings.
import json
import os
import queue
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time


def say(**fields):
    """One JSON line on the bootstrap's standard output."""
    sys.stdout.write(json.dumps(fields) + "\n")
    sys.stdout.flush()


def fail(error):
    say(event="error", error=error)
    sys.exit(1)


def connect(address, timeout=10.0, secret=None):
    kind, _, rest = address.partition(":")
    if kind == "unix":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(rest)
    else:
        host, _, port = rest.rpartition(":")
        sock = socket.create_connection((host, int(port)), timeout=timeout)
        if secret:
            sock.sendall(bytes.fromhex(secret))
    sock.settimeout(None)
    return sock


class Control:
    """The sidecar's framed JSON control channel."""

    def __init__(self, address, on_event):
        self.sock = connect(address)
        self.on_event = on_event
        self.ids = 0
        self.lock = threading.Lock()
        self.waiting = {}
        threading.Thread(target=self.read_loop, daemon=True).start()

    def read_exact(self, n):
        chunks = []
        while n:
            data = self.sock.recv(min(n, 65536))
            if not data:
                raise EOFError
            chunks.append(data)
            n -= len(data)
        return b"".join(chunks)

    def read_loop(self):
        try:
            while True:
                (length,) = struct.unpack(">I", self.read_exact(4))
                message = json.loads(self.read_exact(length))
                if "event" in message:
                    self.on_event(message)
                else:
                    with self.lock:
                        waiter = self.waiting.pop(message.get("id"), None)
                    if waiter is not None:
                        waiter.put(message)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            with self.lock:
                waiters = list(self.waiting.values())
            for waiter in waiters:
                waiter.put({"ok": False, "error": "the sidecar is gone"})
            self.on_event({"event": "control_closed"})

    def request(self, message):
        with self.lock:
            self.ids += 1
            number = self.ids
            waiter = queue.Queue(1)
            self.waiting[number] = waiter
        body = json.dumps(dict(message, id=number)).encode()
        self.sock.sendall(struct.pack(">I", len(body)) + body)
        reply = waiter.get()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "request failed"))
        return reply


LINES = 40
HEADER_LIMIT = 4096
HEADER_WAIT_S = 10.0
TOKEN = re.compile(r"^[0-9a-f]{1,32}$")


class Shim:
    def __init__(self, config):
        self.config = config
        #: Live paths by session number. A session is a key from its first event on.
        self.sessions = {}
        self.last_path = time.monotonic()
        self.handed_over = False
        self.events = queue.Queue()
        self.out_lock = threading.Lock()
        self.directory = None

    def start_sidecar(self):
        binary = self.config["binary"]
        if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            fail("letify-link is not at %s" % binary)
        self.directory = tempfile.mkdtemp(prefix="letify-linkshim-")
        os.chmod(self.directory, 0o700)
        control = os.path.join(self.directory, "c")
        stderr = open(os.path.join(self.directory, "stderr"), "wb")
        self.proc = subprocess.Popen(
            [binary, "serve", "--control", control, "--key", os.path.join(self.directory, "k")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        line = self.proc.stdout.readline().decode("utf-8", "replace")
        try:
            ready = json.loads(line)
            assert ready["event"] == "ready"
        except (ValueError, KeyError, AssertionError):
            detail = open(os.path.join(self.directory, "stderr"), errors="replace").read()
            fail("letify-link did not start: %s" % (detail.strip()[-400:] or line.strip()))
        threading.Thread(target=self.proc.stdout.read, daemon=True).start()
        self.node_id = ready["node_id"]
        self.control = Control(ready["control"], self.events.put)

    def emit(self, **fields):
        with self.out_lock:
            if not self.handed_over:
                say(**fields)

    def listen_request(self):
        """The ``listen`` request: only the local node may start a session, for the lease."""
        return {
            "op": "listen",
            "allow": [self.config["allow"]],
            "bind": self.config["bind"],
            "config": {"resume_window_s": int(self.config["resume_s"])},
        }

    def on_event(self, event):
        """Track the sessions and their paths. Returns True when the shim has nothing to serve."""
        name = event["event"]
        session = event.get("session")
        if name == "session_incoming":
            self.sessions[session] = set()
            self.last_path = time.monotonic()
        elif name == "path_added" and session in self.sessions:
            self.sessions[session].add(event["name"])
        elif name == "path_lost" and session in self.sessions:
            self.sessions[session].discard(event["name"])
            self.last_path = time.monotonic()
        elif name == "stream_incoming" and session in self.sessions:
            threading.Thread(target=self.serve_stream, args=(event["socket"], event.get("secret")), daemon=True).start()
        elif name == "session_closed" and session in self.sessions:
            del self.sessions[session]
            return not self.sessions
        elif name == "control_closed":
            return True
        return False

    def pump(self):
        """Dispatch sidecar events until the shim has nothing left to serve."""
        while True:
            try:
                event = self.events.get(timeout=1.0)
            except queue.Empty:
                self.check_idle()
                continue
            if event["event"] == "rendezvous_out":
                self.emit(event="rendezvous", to=event["to"], message=event["message"])
            elif self.on_event(event):
                return

    def check_idle(self):
        """Leave when no session came, or no session has a path for the resume window."""
        limit = self.config["resume_s"]
        if not self.sessions:
            if self.handed_over and time.monotonic() - self.started > self.config["wait_s"]:
                sys.exit(0)
        elif not any(self.sessions.values()) and time.monotonic() - self.last_path > limit:
            sys.exit(0)

    def serve_stream(self, address, secret=None):
        """Serve one stream: read its first line, then start what the line asks for."""
        self.serve_socket(connect(address, secret=secret))

    def read_header(self, sock):
        """The first line of the stream as an object, or None. Reads no byte past the newline."""
        sock.settimeout(HEADER_WAIT_S)
        data = b""
        try:
            while not data.endswith(b"\n") and len(data) < HEADER_LIMIT:
                byte = sock.recv(1)
                if not byte:
                    return None
                data += byte
            header = json.loads(data.decode("utf-8"))
        except (OSError, ValueError):
            return None
        sock.settimeout(None)
        if not isinstance(header, dict) or not TOKEN.match(str(header.get("token", ""))):
            return None
        return header

    def log_path(self, token):
        return os.path.join(self.directory, "stream-%s.log" % token)

    def serve_socket(self, sock):
        header = self.read_header(sock)
        kind = header.get("kind") if header else None
        if kind == "worker":
            command = [self.config["python"], "-u", "-c", self.config["bootstrap"]]
        elif kind == "door" and isinstance(header.get("path"), str):
            command = [self.config["python"], "-c", self.config["relay"], header["path"]]
        elif kind == "stderr":
            self.send_tail(sock, header["token"])
            return
        else:
            sock.close()
            return
        log = open(self.log_path(header["token"]), "wb")
        proc = subprocess.Popen(command, stdin=sock, stdout=sock, stderr=log)
        sock.close()
        log.close()
        threading.Thread(target=proc.wait, daemon=True).start()

    def send_tail(self, sock, token):
        """Send the last lines of the standard error kept for ``token`` and close."""
        try:
            with open(self.log_path(token), "rb") as handle:
                text = b"".join(handle.read().splitlines(True)[-LINES:])
            sock.sendall(text)
        except OSError:
            pass
        finally:
            sock.close()

    def stdin_loop(self):
        """Commands from the local side: rendezvous messages, punch, hand-over."""
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                continue
            op = command.get("op")
            if op == "rendezvous":
                self.control.request(
                    {
                        "op": "rendezvous_message",
                        "from": command["from"],
                        "message": command["message"],
                    }
                )
            elif op == "punch":
                threading.Thread(target=self.punch, args=(command,), daemon=True).start()
            elif op == "handover":
                self.hand_over()
                return
        if not self.handed_over:
            # The local side went away before the session started: nothing to serve.
            os._exit(0)

    def punch(self, command):
        try:
            self.control.request(
                {
                    "op": "punch",
                    "peer": command["peer"],
                    "role": "responder",
                    "sockets": command.get("sockets", 4),
                    "public_ip": command.get("public_ip", "127.0.0.1"),
                    "listener": self.listener,
                }
            )
            self.emit(event="punched")
        except RuntimeError as error:
            self.emit(event="punch_failed", error=str(error))

    def hand_over(self):
        with self.out_lock:
            say(event="handed_over")
            self.handed_over = True
            self.started = time.monotonic()
        # The local side ends the ssh command now. Nothing of ssh's session may reach this
        # process: no hang-up, no process group of the command, no standard stream.
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            os.setsid()
        except OSError:
            pass
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(devnull, 1)

    def run(self):
        self.started = time.monotonic()
        self.start_sidecar()
        reply = self.control.request(self.listen_request())
        self.listener = reply["listener"]
        port = int(reply["addr"].rpartition(":")[2])
        say(event="ready", node_id=self.node_id, port=port)
        threading.Thread(target=self.stdin_loop, daemon=True).start()
        try:
            self.pump()
        finally:
            self.proc.kill()


def main():
    config = json.loads(sys.stdin.readline())
    Shim(config).run()


if __name__ == "__main__":
    main()
