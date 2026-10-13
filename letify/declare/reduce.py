"""All-reduce between replicas, over a direct socket link brokered only at rendezvous.

Owns ``all_reduce``, ``mean`` and ``sum``: the user-facing reduction a distributed body
calls, and the TCP transport it opens between replicas for one round. It does not own
rank or world_size, which ``letify.declare.replicas`` owns, and it does not own matching
joins to peer tables, which ``letify.runtime.broker`` owns on the client side. This
module runs on the runtime, inside the worker process, never on the client.

Spec "Reducing between replicas", claim N10.
"""

from __future__ import annotations

import copy
import json
import socket
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from ..errors import ReductionFailed, UnsupportedMode
from .replicas import rank, world_size

#: Default bound on one round, in seconds.
DEFAULT_TIMEOUT = 60.0

#: Counts rounds within this process, so two rounds in the same call never collide.
_ROUND_LOCK = threading.Lock()
_ROUND = 0

#: Header length and buffer length each travel as this many big-endian bytes.
_LEN = 8


def mean(arrays: list) -> Any:
    """Average the arrays elementwise. Spec "Reducing between replicas": one of the two
    provided reductions. Imports numpy on the runtime, and says so if it is absent."""
    numpy = _require_numpy()
    stacked = numpy.stack([numpy.asarray(array) for array in arrays], axis=0)
    return numpy.mean(stacked, axis=0)


def sum(arrays: list) -> Any:
    """Add the arrays elementwise. Spec "Reducing between replicas": the other provided
    reduction. Imports numpy on the runtime, and says so if it is absent."""
    numpy = _require_numpy()
    stacked = numpy.stack([numpy.asarray(array) for array in arrays], axis=0)
    return numpy.sum(stacked, axis=0)


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise UnsupportedMode(
            "letify.mean and letify.sum need numpy on this runtime, and it is not "
            "installed here. Add it to the project's lock, or pass a callable of your "
            "own as op instead."
        ) from exc
    return numpy


def all_reduce(
    tensor: Any,
    *,
    op: Callable[[list], Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    min_peers: int | None = None,
    _watch: dict | None = None,
) -> int:
    """Replace ``tensor``'s contents with the reduction of every replica's, in place.

    Spec "Reducing between replicas". Returns how many replicas were reduced. A CUDA
    tensor is refused before any rendezvous, naming ``.cpu()``. Outside a distributed
    call there is nothing to reduce with, and ``rank()`` says so.
    """
    if op is None:
        op = mean
    is_cuda = getattr(tensor, "is_cuda", False)
    device_type = getattr(getattr(tensor, "device", None), "type", None)
    if is_cuda or device_type == "cuda":
        raise UnsupportedMode(
            "all_reduce takes a CPU tensor. Move it with .cpu() first: the copy to "
            "host is needed for any transfer, and letify never does it silently."
        )
    my_rank = rank()
    world = world_size()
    needed = world if min_peers is None else int(min_peers)

    try:
        view = memoryview(tensor)
    except TypeError as exc:
        raise UnsupportedMode(
            f"all_reduce cannot send a {type(tensor).__name__!r}: it exposes no buffer, "
            "so there is nothing to send."
        ) from exc
    raw = view.cast("B").tobytes()

    main = sys.modules.get("__main__")
    if main is None or not hasattr(main, "_reduce_join"):
        raise UnsupportedMode(
            "all_reduce needs the worker's signalling channel to the client, which is "
            "not available here. It works inside a call declared with "
            "host=letify.distributed, run through a persistent channel."
        )

    global _ROUND
    with _ROUND_LOCK:
        _ROUND += 1
        round_id = _ROUND

    deadline = time.monotonic() + timeout
    listener, port, mapped = _listener(world)
    host = _local_address()
    try:
        main._reduce_join(
            {
                "op": "reduce_join",
                "round": round_id,
                "rank": my_rank,
                "address": [host, port],
                "mapped": list(mapped) if mapped is not None else None,
                "timeout": timeout,
                "min_peers": needed,
            }
        )

        table = _await_table(main, round_id, deadline)
        if table.get("failed"):
            raise ReductionFailed(
                f"all_reduce round {round_id} could not reach {needed} replicas within "
                f"{timeout}s: {table['failed']}"
            )

        leader = int(table["leader"])
        count = int(table["count"])
        token = str(table.get("token", ""))
        start_at = float(table.get("start_at", time.time()))
        peers = {
            int(r): {
                "address": tuple(entry["address"]),
                "mapped": tuple(entry["mapped"]) if entry.get("mapped") else None,
            }
            for r, entry in table["peers"].items()
        }
        header = {"rank": my_rank, "round": round_id, "nbytes": len(raw)}

        if _watch is not None:
            _watch["peers"] = table.get("peers")
            _watch["token"] = token
            _watch["start_at"] = start_at

        if my_rank == leader:
            result = _lead(
                listener, deadline, my_rank, tensor, header, raw, count, op, peers, token,
                start_at, port, round_id,
            )
        else:
            result = _follow(leader, peers, deadline, header, raw, token, start_at, port, count)
    except OSError as exc:
        raise ReductionFailed(f"all_reduce round {round_id} failed: {exc}") from exc

    _write_back(tensor, result)
    return count


#: The replica's listener, its port and its STUN mapping, opened once and kept.
#: Spec "Reducing between replicas": a loop that reduces every H steps would otherwise
#: pay a STUN round trip and a fresh punch on every round.
_LISTENER: list = []
_LISTENER_LOCK = threading.Lock()


def _listener(world: int) -> tuple[Any, int, tuple[str, int] | None]:
    """This replica's listening socket, its port and its mapped address, made once."""
    with _LISTENER_LOCK:
        if not _LISTENER:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("0.0.0.0", 0))
            sock.listen(max(1, world))
            port = sock.getsockname()[1]
            _LISTENER.append((sock, port, _mapped_address(port)))
        return _LISTENER[0]


def _local_address() -> str:
    """The host peers can reach this one on. No configuration knob: it is the machine's
    own address, as spec "Reducing between replicas" scopes the local transport."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _mapped_address(port: int) -> tuple[str, int] | None:
    """The STUN mapping of the listener's own ``port``, or None when it cannot be had.

    Spec "Reducing between replicas": a replica joins a round with two addresses, its
    own and this one. Never raises: a replica with no mapping still joins, with
    ``mapped`` absent, and is still reachable on its own address.
    """
    try:
        from ..transport import nat
    except ImportError:
        return None
    try:
        return nat.stun_mapping(port)
    except Exception:
        return None


def _pair_for_punching(*, count: int, ranks: list[int]) -> tuple[int, int]:
    """The two ranks a punch is between, or a refusal naming every rank in the round.

    Spec "Reducing between replicas": one punched connection takes one local port on
    each side, so punching a round of more than two replicas is refused rather than
    attempted as a mesh letify has not measured.
    """
    ordered = sorted(ranks)
    if count == 2 and len(ordered) == 2:
        return ordered[0], ordered[1]
    named = ", ".join(str(r) for r in ordered)
    raise ReductionFailed(
        f"a round of {count} replicas over ranks {named} cannot punch to reach each "
        "other directly: punching is between a pair, not a mesh of more than two"
    )


def _await_table(main: Any, round_id: int, deadline: float) -> dict:
    """Block until the client answers this round's join, or the deadline passes."""
    cond = main._REDUCE_LOCK
    tables = main._REDUCE_TABLES
    with cond:
        while round_id not in tables:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReductionFailed(
                    f"all_reduce round {round_id} got no answer from the client within the timeout"
                )
            cond.wait(min(remaining, 1.0))
        return tables.pop(round_id)


def _lead(
    listener: socket.socket,
    deadline: float,
    my_rank: int,
    tensor: Any,
    header: dict,
    raw: bytes,
    count: int,
    op: Callable[[list], Any],
    peers: dict[int, dict],
    token: str,
    start_at: float,
    port: int,
    round_id: int,
) -> Any:
    """Accept every other joined peer, punching for one that never connects, then reduce
    and send the result back to each.

    Spec "Reducing between replicas": the leader accepts on its own listener first,
    because two replicas on one machine or one network reach each other without a
    punch, and punches with ``initiator=False`` only for a peer that never arrives.
    """
    collected: dict[int, Any] = {my_rank: tensor}
    conns: list[socket.socket] = []
    try:
        window_deadline = min(deadline, start_at + _window())
        while len(collected) < count:
            remaining_window = window_deadline - time.monotonic()
            if remaining_window <= 0:
                break
            listener.settimeout(max(0.01, min(remaining_window, _remaining(deadline))))
            try:
                conn, _addr = listener.accept()
            except TimeoutError:
                break
            conn.settimeout(_remaining(deadline))
            peer_header, payload = _recv_message(conn)
            if int(peer_header.get("round", round_id)) != round_id:
                # The listener outlives a round, so a connection a previous one left
                # behind can be accepted here. Spec "Reducing between replicas".
                conn.close()
                continue
            collected[int(peer_header["rank"])] = _rebuild(tensor, payload)
            conns.append(conn)

        missing = sorted(r for r in peers if r not in collected)
        if missing:
            _pair_for_punching(count=count, ranks=sorted(peers))
            from ..transport import nat

            for peer_rank in missing:
                mapped = peers[peer_rank]["mapped"]
                if mapped is None:
                    raise ReductionFailed(
                        f"replica {peer_rank} never connected directly and reported no "
                        "mapped address to punch to"
                    )
                conn = nat.punch(
                    port, mapped, bytes.fromhex(token), initiator=False, start_at=start_at
                )
                conn.settimeout(_remaining(deadline))
                peer_header, payload = _recv_message(conn)
                collected[int(peer_header["rank"])] = _rebuild(tensor, payload)
                conns.append(conn)

        arrays = [collected[r] for r in sorted(collected)]
        result = op(arrays)
        result_header, result_raw = _encode(result)
        for conn in conns:
            conn.settimeout(_remaining(deadline))
            _send_message(conn, result_header, result_raw)
        return result
    finally:
        for conn in conns:
            conn.close()


def _follow(
    leader_rank: int,
    peers: dict[int, dict],
    deadline: float,
    header: dict,
    raw: bytes,
    token: str,
    start_at: float,
    port: int,
    count: int,
) -> bytes:
    """Send this replica's array to the leader, and wait for the reduced result's bytes.

    Spec "Reducing between replicas": the address the leader reported for itself is
    tried first, and the mapped address is punched to with ``initiator=True`` only when
    that direct connection fails.
    """
    peer = peers[leader_rank]
    try:
        conn = socket.create_connection(peer["address"], timeout=_remaining(deadline))
    except OSError:
        _pair_for_punching(count=count, ranks=sorted(peers))
        mapped = peer["mapped"]
        if mapped is None:
            raise
        from ..transport import nat

        conn = nat.punch(port, mapped, bytes.fromhex(token), initiator=True, start_at=start_at)
    try:
        conn.settimeout(_remaining(deadline))
        _send_message(conn, header, raw)
        conn.settimeout(_remaining(deadline))
        _result_header, payload = _recv_message(conn)
        return payload
    finally:
        conn.close()


def _window() -> float:
    """The punch window, from ``letify.transport.nat`` when it can be imported."""
    try:
        from ..transport import nat
    except ImportError:
        return 45.0
    return nat.DEFAULT_WINDOW


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise ReductionFailed("all_reduce ran out of its timeout while reducing")
    return left


def _send_message(conn: socket.socket, header: dict, payload: bytes) -> None:
    body = json.dumps(header).encode("utf-8")
    conn.sendall(len(body).to_bytes(_LEN, "big"))
    conn.sendall(body)
    conn.sendall(payload)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    pieces = []
    left = size
    while left:
        chunk = conn.recv(left)
        if not chunk:
            raise ReductionFailed("the other replica closed the link before sending enough")
        pieces.append(chunk)
        left -= len(chunk)
    return b"".join(pieces)


def _recv_message(conn: socket.socket) -> tuple[dict, bytes]:
    length = int.from_bytes(_recv_exact(conn, _LEN), "big")
    header = json.loads(_recv_exact(conn, length).decode("utf-8"))
    payload = _recv_exact(conn, int(header["nbytes"]))
    return header, payload


def _encode(array: Any) -> tuple[dict, bytes]:
    """The header and raw bytes of an array this process already holds."""
    try:
        raw = memoryview(array).cast("B").tobytes()
    except TypeError as exc:
        raise UnsupportedMode(
            f"op returned a {type(array).__name__!r}, which exposes no buffer. A "
            "reduction returns an array of the same kind it was given."
        ) from exc
    return {"rank": -1, "nbytes": len(raw)}, raw


def _rebuild(local: Any, payload: bytes) -> Any:
    """A peer's array, as a copy of the local one holding the peer's bytes.

    Spec "Reducing between replicas": nothing is imported and no dtype crosses the wire,
    because the local array already says what kind of thing this is. A peer whose buffer
    is a different length is refused naming both lengths.
    """
    rebuilt = copy.deepcopy(local)
    view = memoryview(rebuilt).cast("B")
    if len(view) != len(payload):
        raise ReductionFailed(
            f"a peer sent {len(payload)} bytes where this replica holds {len(view)}, so "
            f"the replicas disagree on the shape and are not reducing the same thing"
        )
    view[:] = payload
    return rebuilt


def _write_back(tensor: Any, result: Any) -> None:
    """Write the reduced value into ``tensor`` in place.

    ``result`` is whatever ``op`` returned on the leader, and the peer's own bytes
    everywhere else, so both an array and a buffer of the right length land correctly.
    """
    if isinstance(result, (bytes, bytearray, memoryview)):
        memoryview(tensor).cast("B")[:] = result
        return
    try:
        tensor[...] = result
    except Exception:
        memoryview(tensor).cast("B")[:] = memoryview(result).cast("B")


__all__ = ["all_reduce", "mean", "sum"]
