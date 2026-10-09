"""Probe, a short measurement of a Link.

Owns the client side of the probe: 30 round trips, then a discarded warm-up and a timed
transfer in each direction against the responder in ``nat.serve_probe``. It does not own
what the numbers decide; the pipeline does.
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


class Probe:
    """30 round trips, a discarded warm-up, then ``seconds`` of transfer each way.

    The warm-up exists because TCP's ramp is fixed in round trips while the measurement
    window is fixed in seconds, so without it a far link is charged for its own slow
    start and reports the ramp average. Spec "Choosing a link". Durations are injectable.
    """

    def __init__(
        self,
        round_trips: int = 30,
        seconds: float = 2.0,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.round_trips = round_trips
        self.seconds = seconds
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
        rtt_ms = statistics.median(rtts) if rtts else 0.0

        # The warm-up runs on the same connection, so the window it opens carries over.
        warmup = self.warmup_for(rtt_ms)
        self._upload(stream, warmup)
        upload = self._upload(stream, self.seconds)
        self._download(stream, warmup)
        download = self._download(stream, self.seconds)
        return ProbeResult(rtt_ms, upload, download)

    def _upload(self, stream: Stream, seconds: float) -> float:
        block = b"\0" * nat.CHUNK
        stream.sendall(b"U")
        began = self.clock()
        while self.clock() - began < seconds:
            stream.sendall(struct.pack("!I", len(block)) + block)
        stream.sendall(struct.pack("!I", 0))
        received = struct.unpack("!Q", nat.recv_exact(stream, 8))[0]
        return received / max(self.clock() - began, 1e-9)

    def _download(self, stream: Stream, seconds: float) -> float:
        began = self.clock()
        stream.sendall(b"D" + struct.pack("!d", seconds))
        total = 0
        while True:
            size = struct.unpack("!I", nat.recv_exact(stream, 4))[0]
            if size == 0:
                break
            total += len(nat.recv_exact(stream, size))
        return total / max(self.clock() - began, 1e-9)


__all__ = ["Probe", "ProbeResult", "Stream"]
