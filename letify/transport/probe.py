"""Probe, a short measurement of a Link.

Owns the client side of the probe: 30 round trips, then a timed transfer in each
direction against the responder in ``nat.serve_probe``. It does not own what the numbers
decide; the pipeline does.
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


class Probe:
    """30 round trips, then ``seconds`` of transfer each way. Durations are injectable."""

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

    def measure(self, stream: Stream) -> ProbeResult:
        rtts = []
        for index in range(self.round_trips):
            payload = struct.pack("!Q", index)
            began = self.clock()
            stream.sendall(b"P" + payload)
            if nat.recv_exact(stream, 9) != b"P" + payload:
                raise OSError("the probe responder answered out of order")
            rtts.append((self.clock() - began) * 1000.0)

        block = b"\0" * nat.CHUNK
        stream.sendall(b"U")
        began = self.clock()
        while self.clock() - began < self.seconds:
            stream.sendall(struct.pack("!I", len(block)) + block)
        stream.sendall(struct.pack("!I", 0))
        received = struct.unpack("!Q", nat.recv_exact(stream, 8))[0]
        upload = received / max(self.clock() - began, 1e-9)

        began = self.clock()
        stream.sendall(b"D" + struct.pack("!d", self.seconds))
        total = 0
        while True:
            size = struct.unpack("!I", nat.recv_exact(stream, 4))[0]
            if size == 0:
                break
            total += len(nat.recv_exact(stream, size))
        download = total / max(self.clock() - began, 1e-9)
        return ProbeResult(statistics.median(rtts) if rtts else 0.0, upload, download)


__all__ = ["Probe", "ProbeResult", "Stream"]
