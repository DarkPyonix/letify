"""The shim's standard streams carried over a provider's command channel.

Owns starting ``linkpump.py`` on a runtime through a function that runs one Python program there
(``colab exec``, the Kaggle bridge's ``eval``), looping on that function to move lines both ways,
and measuring the channel's round trip. It does not own the shim, the punch or the lead that is
sized from the round trip (``linkchannel.py``). Spec "Sidecar interface" item 8.
"""

from __future__ import annotations

import collections
import contextlib
import json
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import linkpump

#: Seconds a call waits on the runtime for the outbox to grow before it returns empty.
POLL_WAIT_S = 0.5
#: Seconds a call may take on top of its wait before it counts as failed.
CALL_TIMEOUT_S = 120.0
#: Calls that wait for nothing, made once the pump is up, to measure the round trip.
PINGS = 2
#: Samples the round trip is the largest of.
SAMPLES = 3
#: Seconds close waits for a call in flight to come back.
JOIN_S = 30.0
#: Seconds the close call may take before close gives up on it.
CLOSE_TIMEOUT_S = 30.0


def pump_source() -> str:
    """The text of ``linkpump.py``, which the runtime runs with its own Python."""
    return Path(linkpump.__file__).read_text(encoding="utf-8")


def remote_program(request: dict[str, object]) -> str:
    """The pump module plus one call that answers ``request`` and prints its answer."""
    source = pump_source()
    return source + f"\nrun({json.dumps(request)}, {source!r})\n"


class MailboxBootstrap:
    """A ``Bootstrap`` whose lines travel as calls of a command channel.

    ``run(source, timeout)`` runs a Python program on the runtime and returns what it printed.
    One thread makes calls back to back, each carrying the lines sent since the last one and
    returning the lines the shim printed, so a slow call delays a message by at most one call.
    """

    def __init__(
        self,
        run: Callable[[str, float], str],
        *,
        python: str,
        shim: str | None = None,
    ) -> None:
        self._run = run
        self._python = python
        self._shim = shim
        self._directory = ""
        self._offset = 0
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._inbound: queue.Queue[bytes | None] = queue.Queue()
        self._samples: collections.deque[float] = collections.deque(maxlen=SAMPLES)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stderr = ""

    @property
    def rtt_s(self) -> float | None:
        """The channel's round trip in seconds: the largest of the last few calls, or None."""
        return max(self._samples) if self._samples else None

    def _call(self, request: dict[str, object], timeout: float) -> dict[str, object]:
        output = self._run(remote_program(request), timeout)
        for line in output.splitlines():
            if line.startswith(linkpump.MARKER):
                answer = json.loads(line[len(linkpump.MARKER) :])
                if not answer.get("ok"):
                    raise OSError(str(answer.get("error")))
                return answer
        raise OSError(f"the runtime gave no mailbox answer: {output.strip()[-400:]}")

    def _exchange(self, send: list[str], wait_s: float) -> dict[str, object]:
        began = time.monotonic()
        answer = self._call(
            {
                "op": "exchange",
                "dir": self._directory,
                "send": send,
                "offset": self._offset,
                "wait_s": wait_s,
            },
            wait_s + CALL_TIMEOUT_S,
        )
        elapsed = time.monotonic() - began
        # A call that came back empty spent ``wait_s`` waiting on the runtime, not travelling.
        self._samples.append(max(0.0, elapsed - (wait_s if not answer["lines"] else 0.0)))
        return answer

    def start(self) -> None:
        """Start the pump on the runtime, measure the round trip, and begin moving lines."""
        from . import linkchannel

        shim = self._shim if self._shim is not None else linkchannel.shim_source()
        answer = self._call(
            {"op": "start", "shim": shim, "python": self._python}, linkchannel.PREPARE_TIMEOUT
        )
        self._directory = str(answer["dir"])
        for _ in range(PINGS):
            self._take(self._exchange([], 0.0))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _take(self, answer: dict[str, object]) -> bool:
        """Queue the lines of an answer. Returns False once the shim has ended."""
        self._offset = int(answer["offset"])  # type: ignore[call-overload]
        for line in answer["lines"]:  # type: ignore[attr-defined]
            if line.startswith('{"event": "eof"'):
                with contextlib.suppress(ValueError):
                    self._stderr = str(json.loads(line).get("stderr", ""))
                self._inbound.put(None)
                return False
            self._inbound.put(line.encode("utf-8"))
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                send, self._pending = self._pending, []
            try:
                more = self._take(self._exchange(send, POLL_WAIT_S))
            except Exception as error:
                if not self._stop.is_set():
                    self._stderr = f"the command channel failed: {error}"
                    self._inbound.put(None)
                return
            if not more:
                return

    def send_line(self, line: bytes | str) -> None:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        with self._lock:
            self._pending.append(text.rstrip("\n"))

    def recv_line(self) -> bytes | None:
        """The next line the shim printed, blocking, or None once it has ended."""
        return self._inbound.get()

    def stderr_text(self) -> str:
        return self._stderr

    def close(self) -> None:
        """Stop moving lines and tell the pump to end its runtime side.

        The pump stops a shim that has not handed over and removes its directory. A pump the
        call cannot reach ends the same way by itself after ``linkpump.IDLE_S``.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=JOIN_S)
        if self._directory:
            with contextlib.suppress(Exception):
                self._call({"op": "close", "dir": self._directory}, CLOSE_TIMEOUT_S)
        self._inbound.put(None)


__all__ = ["CALL_TIMEOUT_S", "POLL_WAIT_S", "MailboxBootstrap", "pump_source", "remote_program"]
