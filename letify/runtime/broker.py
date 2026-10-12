"""The client-side broker for a distributed round.

Owns matching replicas' joins to the peer table they get back. It does not own the
reduction itself, the socket work or the wire format of the payload the replicas
exchange: ``letify.declare.reduce`` owns that, and the client here only ever sees the
small join and peer-table messages, never the array bytes a round reduces, which is
claim N10.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

#: Guards ``_TOTAL_BYTES``.
_COUNT_LOCK = threading.Lock()

#: Total size of the payloads the broker has passed between replicas, for
#: ``brokered_bytes``.
_TOTAL_BYTES = 0

#: Seconds between polls of a replica's joins, and between sweeps for an expired round.
_POLL_S = 0.02


def _account(payload: dict) -> None:
    """Add the size of one brokered payload to the running total."""
    global _TOTAL_BYTES
    size = len(json.dumps(payload, default=str).encode("utf-8"))
    with _COUNT_LOCK:
        _TOTAL_BYTES += size


def brokered_bytes() -> int:
    """Total size of the payloads the broker has passed, across every call this process made.

    Spec "Reducing between replicas", claim N10: the client is the signalling broker and
    carries nothing else, so this stays far below the size of what a round reduces.
    """
    with _COUNT_LOCK:
        return _TOTAL_BYTES


class _Round:
    """One round's joins, how many it needs, and when it gives up."""

    __slots__ = ("deadline", "joins", "needed", "resolved")

    def __init__(self, needed: int, deadline: float) -> None:
        self.needed = needed
        self.deadline = deadline
        self.joins: dict[int, dict] = {}
        self.resolved = False


class Broker:
    """Matches replicas' joins to the peer table they get back, for one distributed call.

    A replica thread calls ``watch`` with its channel once the runtime is up. A
    background thread per replica polls that channel's joins and feeds them here, the
    same shape the data sender polls ``take_wants`` in. A replica whose call has
    already returned never joins, so a round resolves, successfully or not, from the
    joins it actually got, bounded by the deadline the first join to arrive named.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rounds: dict[int, _Round] = {}
        self._channels: dict[int, Any] = {}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def watch(self, rank: int, channel: Any) -> None:
        """Poll ``channel`` for joins on a thread of its own, until the call ends."""
        self._channels[rank] = channel
        thread = threading.Thread(
            target=self._poll, args=(rank, channel), name="letify-broker", daemon=True
        )
        self._threads.append(thread)
        thread.start()

    def close(self) -> None:
        """Stop polling. Called once every replica's call has returned."""
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)

    # -- polling -----------------------------------------------------------------

    def _poll(self, rank: int, channel: Any) -> None:
        while not self._stop.is_set():
            connection = getattr(channel, "connection", None)
            if connection is not None:
                try:
                    joins = connection.take_joins()
                except Exception:
                    joins = []
                for join in joins:
                    self._join(rank, join)
            self._sweep()
            time.sleep(_POLL_S)

    # -- resolving a round ---------------------------------------------------------

    def _join(self, rank: int, message: dict) -> None:
        _account(message)
        round_id = int(message["round"])
        with self._lock:
            state = self._rounds.get(round_id)
            if state is None:
                deadline = time.monotonic() + float(message["timeout"])
                state = _Round(int(message["min_peers"]), deadline)
                self._rounds[round_id] = state
            state.joins[rank] = message
            ready = not state.resolved and len(state.joins) >= state.needed
            if ready:
                state.resolved = True
                joins = dict(state.joins)
        if ready:
            self._answer(round_id, joins)

    def _sweep(self) -> None:
        """Fail every round whose deadline passed without enough joins."""
        expired: list[tuple[int, dict]] = []
        now = time.monotonic()
        with self._lock:
            for round_id, state in self._rounds.items():
                if not state.resolved and now > state.deadline:
                    state.resolved = True
                    expired.append((round_id, dict(state.joins)))
        for round_id, joins in expired:
            self._fail(round_id, joins)

    def _answer(self, round_id: int, joins: dict[int, dict]) -> None:
        """Send every joined replica the peer table, naming the lowest rank leader."""
        leader = min(joins)
        peers = {str(r): list(join["address"]) for r, join in joins.items()}
        payload = {"round": round_id, "peers": peers, "leader": leader, "count": len(joins)}
        self._send_to(joins, payload)

    def _fail(self, round_id: int, joins: dict[int, dict]) -> None:
        """Tell every replica that did join that the round never reached its count."""
        if not joins:
            return
        needed = next(iter(joins.values())).get("min_peers")
        reason = f"only {len(joins)} of {needed} needed replicas joined before the timeout"
        self._send_to(joins, {"round": round_id, "failed": reason})

    def _send_to(self, joins: dict[int, dict], payload: dict) -> None:
        _account(payload)
        timeout = max(next(iter(joins.values()))["timeout"], 1.0)
        for rank in joins:
            channel = self._channels.get(rank)
            if channel is None:
                continue
            try:
                channel.request({"op": "reduce_peers", **payload}, timeout=timeout)
            except Exception:
                # The replica is gone, or its own call already failed for another
                # reason; either way there is no one left here to tell.
                pass


__all__ = ["Broker", "brokered_bytes"]
