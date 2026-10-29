"""The remote half of a letify-link worker channel. Runs on the machine, standard library only.

Owns starting the remote ``letify-link`` sidecar, listening for the one local node, relaying
rendezvous messages as JSON lines on standard input and output until the hand-over, and
starting one worker process per stream the local side opens. It does not own the local side
(``linkchannel.py``), the worker program (it is passed in) or any path decision. Spec
"Sidecar interface" item 7. This file is sent to the machine as the argument of ``python -c``,
so it imports nothing from letify and avoids syntax newer than Python 3.6.
"""

# ruff: noqa: SIM115, UP031
# The file is sent to old interpreters as source: no context-manager rewrites, no f-strings.
import json
import os
import queue
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


def connect(address, timeout=10.0):
    kind, _, rest = address.partition(":")
    if kind == "unix":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(rest)
    else:
        host, _, port = rest.rpartition(":")
        sock = socket.create_connection((host, int(port)), timeout=timeout)
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


class Shim:
    def __init__(self, config):
        self.config = config
        self.session = None
        self.paths = set()
        self.last_path = time.monotonic()
        self.handed_over = False
        self.events = queue.Queue()
        self.out_lock = threading.Lock()
        self.workers = 0

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

    def pump(self):
        """Dispatch sidecar events until the shim has nothing left to serve."""
        while True:
            try:
                event = self.events.get(timeout=1.0)
            except queue.Empty:
                self.check_idle()
                continue
            name = event["event"]
            if name == "rendezvous_out":
                self.emit(event="rendezvous", to=event["to"], message=event["message"])
            elif name == "session_incoming":
                self.session = event["session"]
            elif name == "path_added" and event.get("session") == self.session:
                self.paths.add(event["name"])
            elif name == "path_lost" and event.get("session") == self.session:
                self.paths.discard(event["name"])
                self.last_path = time.monotonic()
            elif name == "stream_incoming" and event.get("session") == self.session:
                self.serve_stream(event["socket"])
            elif name == "control_closed" or (
                name == "session_closed" and event.get("session") == self.session
            ):
                return

    def check_idle(self):
        """Leave when no session came, or every path stayed lost past the resume window."""
        limit = self.config["resume_s"]
        if self.session is None:
            if self.handed_over and time.monotonic() - self.started > self.config["wait_s"]:
                sys.exit(0)
        elif not self.paths and time.monotonic() - self.last_path > limit:
            sys.exit(0)

    def serve_stream(self, address):
        """Start the worker program with the stream socket as its standard input and output."""
        sock = connect(address)
        self.workers += 1
        log = open(os.path.join(self.directory, "worker%d.log" % self.workers), "wb")
        proc = subprocess.Popen(
            [self.config["python"], "-u", "-c", self.config["bootstrap"]],
            stdin=sock,
            stdout=sock,
            stderr=log,
        )
        sock.close()
        threading.Thread(target=proc.wait, daemon=True).start()

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
        reply = self.control.request(
            {"op": "listen", "allow": [self.config["allow"]], "bind": self.config["bind"]}
        )
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
