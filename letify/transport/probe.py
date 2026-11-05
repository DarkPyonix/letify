"""Probe, a short measurement of a Link.

Owns the client side of the probe: up to 30 round trips, then a discarded warm-up and a
timed transfer in each direction against the responder in ``nat.serve_probe``, with the
rules that end a stage early. It does not own what the numbers decide; the pipeline does.
"""

from __future__ import annotations

import statistics
import struct
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from . import nat


class Stream(Protocol):
    def recv(self, size: int) -> bytes: ...
    def sendall(self, data: bytes) -> None: ...


@dataclass(frozen=True)
class ProbeResult:
    """Round trip median in milliseconds and throughput in bytes per second."""

    rtt_ms: float
    upload_bps: float
    download_bps: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeResult:
        return cls(float(data["rtt_ms"]), float(data["upload_bps"]), float(data["download_bps"]))


#: Round trips of warm-up before the measured transfer, and the bounds on it in seconds.
WARMUP_ROUND_TRIPS = 8
WARMUP_MIN_S = 0.5
WARMUP_MAX_S = 2.0

#: Round trips after which the round trip counts as settled, when the medians of the two
#: halves of the samples differ by at most this share of the median of all of them.
SETTLED_SHARE = 0.10


def settled(rtts: list[float]) -> bool:
    """Whether the median of the first half of ``rtts`` matches that of the second half."""
    middle = len(rtts) // 2
    if middle == 0:
        return False
    overall = statistics.median(rtts)
    gap = abs(statistics.median(rtts[:middle]) - statistics.median(rtts[middle:]))
    return gap <= SETTLED_SHARE * overall


class Probe:
    """Up to 30 round trips, a discarded warm-up, then ``seconds`` of transfer each way.

    By default every stage runs its full length. ``min_round_trips`` lets the round trips
    stop once ``settled``. ``early_bps`` ends a direction after ``early_window`` seconds
    when it already measures at least that rate, and otherwise measures the rest of
    ``seconds``, so only a rate well above the caller's floor is cut short.

    The warm-up exists because TCP's ramp is fixed in round trips while the measurement
    window is fixed in seconds, so without it a far link is charged for its own slow
    start and reports the ramp average. Spec "Choosing a link". Durations are injectable.
    """

    def __init__(
        self,
        round_trips: int = 30,
        seconds: float = 2.0,
        *,
        min_round_trips: int | None = None,
        early_bps: float | None = None,
        early_window: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.round_trips = round_trips
        self.seconds = seconds
        self.min_round_trips = min_round_trips
        self.early_bps = early_bps
        self.early_window = early_window
        self.clock = clock

    def warmup_for(self, rtt_ms: float) -> float:
        """Seconds of transfer to discard before measuring, from the round trip."""
        wanted = WARMUP_ROUND_TRIPS * rtt_ms / 1000.0
        return min(WARMUP_MAX_S, max(WARMUP_MIN_S, wanted))

    def measure(self, stream: Stream) -> ProbeResult:
        rtts = []
        for index in range(self.round_trips):
            payload = struct.pack("!Q", index)
            began = self.clock()
            stream.sendall(b"P" + payload)
            if nat.recv_exact(stream, 9) != b"P" + payload:
                raise OSError("the probe responder answered out of order")
            rtts.append((self.clock() - began) * 1000.0)
            if self.min_round_trips and len(rtts) >= self.min_round_trips and settled(rtts):
                break
        rtt_ms = statistics.median(rtts) if rtts else 0.0

        # The warm-up runs on the same connection, so the window it opens carries over.
        warmup = self.warmup_for(rtt_ms)
        self._upload(stream, warmup)
        upload = self._window(self._send, stream)
        self._download(stream, warmup)
        download = self._window(self._receive, stream)
        return ProbeResult(rtt_ms, upload, download)

    def _window(
        self, segment: Callable[[Stream, float], tuple[int, float]], stream: Stream
    ) -> float:
        """The measured rate of one direction, in bytes per second.

        With ``early_bps`` set the first ``early_window`` seconds decide: at or above that
        rate the direction is done, below it the rest of ``seconds`` is measured and both
        segments make the rate.
        """
        if self.early_bps is None or self.early_window >= self.seconds:
            total, took = segment(stream, self.seconds)
            return total / max(took, 1e-9)
        total, took = segment(stream, self.early_window)
        if total / max(took, 1e-9) < self.early_bps:
            more, longer = segment(stream, self.seconds - self.early_window)
            total, took = total + more, took + longer
        return total / max(took, 1e-9)

    def _upload(self, stream: Stream, seconds: float) -> float:
        total, took = self._send(stream, seconds)
        return total / max(took, 1e-9)

    def _download(self, stream: Stream, seconds: float) -> float:
        total, took = self._receive(stream, seconds)
        return total / max(took, 1e-9)

    def _send(self, stream: Stream, seconds: float) -> tuple[int, float]:
        block = b"\0" * nat.CHUNK
        stream.sendall(b"U")
        began = self.clock()
        while self.clock() - began < seconds:
            stream.sendall(struct.pack("!I", len(block)) + block)
        stream.sendall(struct.pack("!I", 0))
        received = struct.unpack("!Q", nat.recv_exact(stream, 8))[0]
        return received, self.clock() - began

    def _receive(self, stream: Stream, seconds: float) -> tuple[int, float]:
        began = self.clock()
        stream.sendall(b"D" + struct.pack("!d", seconds))
        total = 0
        while True:
            size = struct.unpack("!I", nat.recv_exact(stream, 4))[0]
            if size == 0:
                break
            total += len(nat.recv_exact(stream, size))
        return total, self.clock() - began


__all__ = ["Probe", "ProbeResult", "Stream", "settled"]
