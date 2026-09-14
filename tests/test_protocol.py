"""The wire between this process and a runtime.

Spec sections pinned here: "Channels", "Call protocol", "Session cache" and
"Argument addressing". The one-shot driver is exercised by really running the script it
builds in a fresh interpreter, because that is what a provider such as ``colab exec``
does with it, and the markers and the decoder are then the real ones.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import cloudpickle
import pytest

import letify
from letify import protocol
from letify.declare.env import Env
from letify.protocol import codec, driver, wire
from letify.protocol.handle import Blob, RemoteFile
from letify.protocol.worker import BOOTSTRAP


def run_script(source: str) -> str:
    """Run a one-shot driver script the way a one-shot provider would."""
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=120
    ).stdout


# -- Spec: Call protocol -------------------------------------------------------


def test_a_call_is_a_pickled_function_with_its_arguments() -> None:
    # cloudpickle rather than pickle, because the declared function is usually defined
    # in the caller's own module and has to travel by value.
    def double(x: int) -> int:
        return x * 2

    fn, args, kwargs = cloudpickle.loads(codec.dumps_call(double, (21,), {"scale": 2}))
    assert fn(21) == 42
    assert args == (21,)
    assert kwargs == {"scale": 2}


def test_a_payload_is_addressed_by_the_hash_of_its_contents() -> None:
    first = codec.digest_of(b"same contents")
    assert first == codec.digest_of(b"same contents")
    assert first != codec.digest_of(b"other contents")
    assert len(first) == 32


def test_content_addressing_falls_back_to_the_standard_library(no_module) -> None:
    # blake3 is preferred for speed, but it is a wheel, so blake2b has to carry the same
    # guarantee when it is absent.
    no_module("blake3")
    digest = codec.digest_of(b"same contents")
    assert len(digest) == 32
    assert digest == codec.digest_of(b"same contents")


def test_an_outcome_carrying_a_value_returns_it() -> None:
    assert codec.unwrap({"ok": True, "value": 42}, runtime_key="r") == 42
    # A body that returns nothing still produces an outcome.
    assert codec.unwrap({"ok": True}, runtime_key="r") is None


def test_a_failed_outcome_raises_what_the_remote_side_raised() -> None:
    with pytest.raises(letify.RemoteError) as caught:
        codec.unwrap(
            {"ok": False, "error": "ValueError: intentional", "traceback": "Traceback ..."},
            runtime_key="r",
        )
    assert "intentional" in str(caught.value)
    # The traceback travels with it, because the frames are on the other machine.
    assert caught.value.remote_traceback == "Traceback ..."
    assert "remote traceback" in str(caught.value)


def test_a_failed_outcome_with_no_message_still_says_the_call_failed() -> None:
    with pytest.raises(letify.RemoteError, match="the remote call failed"):
        codec.unwrap({"ok": False}, runtime_key="r")


@pytest.mark.parametrize("payload", [{"value": 1}, "not a dict", None])
def test_an_unrecognized_payload_is_a_protocol_error(payload: object) -> None:
    # Anything that is not an outcome means the far side is not the worker letify sent.
    with pytest.raises(letify.ProtocolError, match="unexpected remote payload"):
        codec.unwrap(payload, runtime_key="r")  # type: ignore[arg-type]


# -- Spec: Session cache -------------------------------------------------------


def test_a_session_cache_works_within_one_one_shot_call() -> None:
    # A one-shot process keeps nothing for the next call, but inside the call the cache holds.
    def work() -> int:
        builds = []

        def load() -> int:
            builds.append(1)
            return 3

        first = letify.session_cache("value", load)
        second = letify.session_cache("value", load)
        return first + second + len(builds)

    stdout = run_script(driver.build(work, (), {}))
    assert codec.parse(stdout, runtime_key="one-shot")[1] == 7


def test_a_runtime_that_cannot_import_letify_names_the_reason(tmp_path: Path) -> None:
    blocker = tmp_path / "letify"
    blocker.mkdir()
    (blocker / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'letify'\", name='letify')\n",
        encoding="utf-8",
    )

    def work() -> int:
        return letify.session_cache("value", lambda: 1)

    import os

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    stdout = subprocess.run(
        [sys.executable, "-c", driver.build(work, (), {})],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    ).stdout
    with pytest.raises(letify.RemoteError, match=r"letify is not installed.*uv add letify"):
        codec.parse(stdout, runtime_key="one-shot")


def test_a_body_that_imports_letify_while_running_names_the_reason(tmp_path: Path) -> None:
    # The call loads without letify, and only the body's own import needs it.
    blocker = tmp_path / "letify"
    blocker.mkdir()
    (blocker / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'letify'\", name='letify')\n",
        encoding="utf-8",
    )

    def work() -> int:
        import letify as inside

        return inside.session_cache("value", lambda: 1)

    import os

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    stdout = subprocess.run(
        [sys.executable, "-c", driver.build(work, (), {})],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    ).stdout
    with pytest.raises(letify.RemoteError, match=r"letify is not installed.*uv add letify"):
        codec.parse(stdout, runtime_key="one-shot")


def test_a_worker_call_whose_body_imports_letify_while_running_names_the_reason(
    tmp_path: Path,
) -> None:
    # The persistent worker's call op, loaded by path in a process where letify cannot
    # be imported, as in a runtime whose environment lacks it.
    blocker = tmp_path / "letify"
    blocker.mkdir()
    (blocker / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'letify'\", name='letify')\n",
        encoding="utf-8",
    )

    def work() -> int:
        import letify as inside

        return inside.session_cache("value", lambda: 1)

    from letify.protocol.worker import SOURCE

    source = SOURCE.encode()
    stdin = bytearray(b"%d\n" % len(source) + source)
    sender = wire.Sender(lambda view: stdin.extend(view) or view.nbytes)
    head, buffers = codec.dumps_call_parts(work, (), {})
    sender.message(wire.REQUEST, 1, {"op": "call", "payload": head, "buffers": buffers})
    sender.frame(wire.SHUTDOWN, 0)
    import os

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    stdout = subprocess.run(
        [sys.executable, "-u", "-c", BOOTSTRAP],
        input=bytes(stdin),
        capture_output=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    ).stdout
    receiver = wire.Receiver(_reader(stdout))
    replies = []
    while True:
        try:
            event = receiver.next_event()
        except EOFError:
            break
        if event is not None and event[0] == wire.REPLY:
            replies.append(wire.loads(*event[2]))
    [reply] = replies
    assert reply["ok"] is False
    assert "letify is not installed" in reply["error"] and "uv add letify" in reply["error"]


def test_a_reference_names_what_it_points_at() -> None:
    assert repr(Blob("0123456789abcdef", 2048)) == "<Blob 01234567 2048 bytes>"
    assert repr(RemoteFile("/opt/letify/x.bin", "abc", 10)) == (
        "<RemoteFile /opt/letify/x.bin 10 bytes>"
    )


def test_the_worker_recognizes_a_blob_by_marker() -> None:
    # The worker source reads a marker by attribute rather than doing an isinstance check.
    assert Blob.__letify_kind__ == "blob"


# -- Spec: Argument addressing -------------------------------------------------


def test_the_inline_limit_is_sixty_four_kilobytes() -> None:
    # Above this an argument is content addressed instead of travelling with the call.
    assert codec.INLINE_LIMIT == 64 * 1024


# -- Spec: Channels, frames ---------------------------------------------------


def test_a_request_travels_as_frames_with_no_base64() -> None:
    written = bytearray()
    sender = wire.Sender(lambda view: written.extend(view) or view.nbytes)
    sender.message(wire.REQUEST, 1, {"op": "stat"})
    magic, kind, _flags, stream, length = wire.HEADER.unpack_from(written)
    assert (magic, kind, stream) == (b"LF", wire.REQUEST, 1)
    assert len(written) == wire.HEADER.size + length
    receiver = wire.Receiver(_reader(bytes(written)))
    kind, stream, (head, buffers) = receiver.next_event()
    assert wire.loads(head, buffers) == {"op": "stat"}


def test_output_frames_are_told_apart_from_replies_by_their_type() -> None:
    # letify's replies and the user's prints travel on one pipe, so the frame type is what
    # separates them.
    written = bytearray()
    sender = wire.Sender(lambda view: written.extend(view) or view.nbytes)
    sender.frame(wire.STDOUT, 0, b"a line the user printed\n")
    sender.message(wire.REPLY, 3, {"ok": True, "value": 7})
    receiver = wire.Receiver(_reader(bytes(written)))
    assert receiver.next_event() == (wire.STDOUT, 0, b"a line the user printed\n")
    kind, stream, (head, buffers) = receiver.next_event()
    assert (kind, stream) == (wire.REPLY, 3)
    assert wire.loads(head, buffers) == {"ok": True, "value": 7}


def test_the_hello_frame_names_the_python_version_the_worker_runs_on() -> None:
    # Spec "Frames" and "Interpreter check".
    written = bytearray()
    wire.Sender(lambda view: written.extend(view) or view.nbytes).frame(wire.HELLO, 0, b"3.12")
    assert wire.Receiver(_reader(bytes(written))).next_event() == (wire.HELLO, 0, "3.12")


def test_a_frame_split_across_reads_is_reassembled_in_order() -> None:
    payload = bytes(range(256)) * 40000
    written = bytearray()
    sender = wire.Sender(lambda view: written.extend(view) or view.nbytes)
    sender.message(wire.REPLY, 5, {"ok": True, "value": bytearray(payload)})
    receiver = wire.Receiver(_reader(bytes(written), step=7777))
    event = None
    while event is None:
        event = receiver.next_event()
    assert wire.loads(event[2][0], event[2][1])["value"] == payload


def test_text_frames_carry_the_same_frames_as_base64_lines() -> None:
    # Spec "Modal adapter": a transport whose output is text gets one line per frame.
    import base64

    written = bytearray()
    wire.TextSender(lambda view: written.extend(view) or view.nbytes).message(
        wire.REPLY, 1, {"ok": True, "value": b"x" * (2 << 20)}
    )
    lines = bytes(written).splitlines()
    assert all(b"\n" not in line for line in lines)
    batches = iter([[base64.b64decode(line)] for line in lines] + [[]])
    receiver = wire.Receiver(wire.chunks_readinto(lambda: next(batches)))
    event = None
    while event is None:
        event = receiver.next_event()
    assert wire.loads(event[2][0], event[2][1])["value"] == b"x" * (2 << 20)


def _reader(data: bytes, step: int = 1 << 30):
    view = memoryview(data)
    position = 0

    def readinto(target: memoryview) -> int:
        nonlocal position
        count = min(target.nbytes, step, len(view) - position)
        target[:count] = view[position : position + count]
        position += count
        return count

    return readinto


# -- Spec: Call protocol, the one-shot path ------------------------------------


def test_an_outcome_is_found_in_a_stream_that_also_carries_the_user_prints() -> None:
    stdout = f"training\n{codec.BEGIN}cGF5bG9hZA=={codec.END}\ndone\n"
    logs, encoded = codec.split_output(stdout)
    assert encoded == "cGF5bG9hZA=="
    assert logs == "training\n\ndone\n"


def test_output_with_no_marker_is_all_user_output() -> None:
    assert codec.split_output("just prints\n") == ("just prints\n", None)


def test_a_begin_marker_with_no_end_marker_is_not_an_outcome() -> None:
    # A process killed while writing its result leaves exactly this.
    assert codec.split_output(f"logs\n{codec.BEGIN}half") == (f"logs\n{codec.BEGIN}half", None)


def test_the_last_marker_wins_when_the_stream_holds_more_than_one() -> None:
    stdout = f"{codec.BEGIN}old{codec.END}{codec.BEGIN}bmV3{codec.END}"
    assert codec.split_output(stdout)[1] == "bmV3"


def test_a_missing_result_marker_means_the_process_died() -> None:
    # Not a protocol quirk. The message names the causes worth checking first, and
    # carries the tail of the remote output.
    with pytest.raises(letify.ProtocolError) as caught:
        codec.parse("Killed\n", runtime_key="r")
    message = str(caught.value)
    assert "out of memory" in message
    assert "preempted" in message
    assert "Killed" in message


def test_an_outcome_that_cannot_be_decoded_says_so() -> None:
    stdout = f"{codec.BEGIN}not-base64!!{codec.END}"
    with pytest.raises(letify.ProtocolError, match="could not be decoded"):
        codec.parse(stdout, runtime_key="r")


def test_the_one_shot_driver_runs_the_call_and_prints_its_outcome() -> None:
    # A real interpreter, the real script, the real markers.
    def add(a: int, b: int = 0) -> int:
        print("working")
        return a + b

    stdout = run_script(driver.build(add, (40,), {"b": 2}))
    logs, value = codec.parse(stdout, runtime_key="one-shot")
    assert value == 42
    assert "working" in logs


def test_the_one_shot_driver_returns_what_the_body_raised() -> None:
    def boom() -> None:
        raise ValueError("intentional")

    stdout = run_script(driver.build(boom, (), {}))
    with pytest.raises(letify.RemoteError) as caught:
        codec.parse(stdout, runtime_key="one-shot")
    assert "intentional" in str(caught.value)
    assert "ValueError" in caught.value.remote_traceback


def test_an_async_body_is_awaited_on_the_far_side() -> None:
    # The body runs to completion there, which is what lets it use await internally.
    async def work() -> int:
        import asyncio

        await asyncio.sleep(0)
        return 7

    stdout = run_script(driver.build(work, (), {}))
    assert codec.parse(stdout, runtime_key="one-shot")[1] == 7


def test_a_return_value_that_cannot_be_serialized_reports_that_rather_than_hanging() -> None:
    def build() -> object:
        import threading

        return threading.Lock()

    stdout = run_script(driver.build(build, (), {}))
    with pytest.raises(letify.RemoteError, match="could not be serialized"):
        codec.parse(stdout, runtime_key="one-shot")


def test_the_protocol_version_is_carried_in_the_driver_script() -> None:
    # Bumped when the request or reply shape changes in a breaking way.
    assert f"_VERSION = {protocol.PROTOCOL_VERSION}" in driver.build(len, ((),), {})


# -- Spec: Module shipping -----------------------------------------------------


@contextmanager
def a_module_only_this_process_has(directory: Path, name: str):
    """Import a module that exists nowhere else, then take every trace of it away.

    Taking it away is the point: it puts this process in the position the machine on the
    other end is in, which is the only way to tell a payload that carries the code from
    one that carries a reference to it.
    """
    source = directory / f"{name}.py"
    source.write_text("def score(x):\n    return x * 3\n", encoding="utf-8")
    sys.path.insert(0, str(directory))
    try:
        yield import_module(name)
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop(name, None)
        source.unlink(missing_ok=True)


def test_a_shipped_module_travels_with_the_call(tmp_path) -> None:
    # The machine on the other end has no copy of the project's own code, so a function
    # imported from it has to be serialized by value.
    with a_module_only_this_process_has(tmp_path, "shipped_recipe") as module:
        codec.ship_by_value(("shipped_recipe",))
        payload = codec.dumps_call(module.score, (2,), {})

    function, args, kwargs = pickle.loads(payload)
    assert function(*args, **kwargs) == 6


def test_a_module_that_is_not_shipped_travels_by_name(tmp_path) -> None:
    # The default, and the right one for anything the lock file installs: sending numpy by
    # value would mean sending numpy over the network on every call.
    with a_module_only_this_process_has(tmp_path, "named_recipe") as module:
        payload = codec.dumps_call(module.score, (2,), {})

    assert b"named_recipe" in payload
    with pytest.raises((ImportError, ModuleNotFoundError)):
        pickle.loads(payload)


def test_shipping_a_module_that_is_not_there_says_which_one() -> None:
    # A typo in ship() is found at the declaration rather than on the machine.
    with pytest.raises(letify.ConfigError, match="no_such_recipe"):
        codec.ship_by_value(("no_such_recipe",))


def test_shipping_the_same_module_twice_is_not_an_error(tmp_path) -> None:
    # Every call asks again, because the registration lives in cloudpickle rather than in
    # the declaration.
    with a_module_only_this_process_has(tmp_path, "twice_recipe"):
        codec.ship_by_value(("twice_recipe",))
        codec.ship_by_value(("twice_recipe",))


def test_a_declaration_ships_what_its_environment_asked_for(let, cpu, tmp_path) -> None:
    # Through the real path: a declared environment, a pooled local runtime, and a worker
    # subprocess that never sees the module on disk because sys.path there does not have
    # the directory it was written to.
    with a_module_only_this_process_has(tmp_path, "declared_recipe") as module:

        @let.function(device=cpu, host="remote", env=Env().ship("declared_recipe"))
        def scored(x: int) -> int:
            return module.score(x) + 1

        assert scored(x=5) == 16


def test_a_large_bytes_value_is_read_into_the_object_it_becomes() -> None:
    # Spec "Frames": a buffer marked as bytes is filled in place, so no copy follows the read.
    payload = os.urandom(2 << 20)
    written = bytearray()
    wire.Sender(lambda view: written.extend(view) or view.nbytes).message(
        wire.REPLY, 1, {"ok": True, "value": payload}
    )
    receiver = wire.Receiver(_reader(bytes(written), step=1 << 20))
    event = None
    while event is None:
        event = receiver.next_event()
    head, buffers = event[2]
    assert [type(buffer) for buffer in buffers] == [bytes]
    value = wire.loads(head, buffers)["value"]
    assert value is buffers[0]
    assert value == payload
    assert hash(value) == hash(payload)


def test_a_bytearray_value_still_arrives_as_a_bytearray() -> None:
    # Only buffers marked as bytes are filled in place; a bytearray keeps its own type.
    written = bytearray()
    wire.Sender(lambda view: written.extend(view) or view.nbytes).message(
        wire.REPLY, 1, {"ok": True, "value": bytearray(b"y" * (2 << 20))}
    )
    receiver = wire.Receiver(_reader(bytes(written)))
    event = None
    while event is None:
        event = receiver.next_event()
    value = wire.loads(*event[2])["value"]
    assert type(value) is bytearray and value == bytearray(b"y" * (2 << 20))


# -- striped byte stream: spec "Parallel data streams" ------------------------------------


class _CountingLane:
    """One end of a socket pair whose writes are recorded by size."""

    def __init__(self, sock):
        self.sock = sock
        self.segments: list[int] = []

    def send(self, view) -> int:
        count = self.sock.send(view)
        self.segments.append(count)
        return count


def _striped_pair(lanes: int):
    import socket

    pairs = [socket.socketpair() for _ in range(lanes)]
    near = [_CountingLane(a) for a, _ in pairs]
    far = [b for _, b in pairs]
    sender = wire.Striped([lane.send for lane in near], [lane.sock.recv_into for lane in near])
    receiver = wire.Striped([sock.send for sock in far], [sock.recv_into for sock in far])
    return sender, receiver, near, far


def _close_all(near, far) -> None:
    for sock in [lane.sock for lane in near] + list(far):
        sock.close()


def _read_exactly(stream, size: int) -> bytes:
    out = bytearray()
    buffer = bytearray(1 << 20)
    while len(out) < size:
        count = stream.recv_into(memoryview(buffer))
        if not count:
            break
        out.extend(buffer[:count])
    return bytes(out)


def _send_all(stream, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[stream.send(view) :]


def test_a_striped_stream_delivers_interleaved_writes_in_order() -> None:
    import threading

    sender, receiver, near, far = _striped_pair(4)
    writes = [os.urandom(size) for size in (10, 3 << 20, 1, (1 << 20) - 1, 9 << 20 | 7, 100)]
    expected = b"".join(writes)
    got: list[bytes] = []
    reader = threading.Thread(target=lambda: got.append(_read_exactly(receiver, len(expected))))
    reader.start()
    for data in writes:
        _send_all(sender, data)
    reader.join(60)
    assert got == [expected]
    _close_all(near, far)


def test_a_write_of_one_mib_or_more_uses_every_lane_and_a_smaller_one_only_lane_zero() -> None:
    import threading

    sender, receiver, near, far = _striped_pair(4)
    size = 100 + wire.STRIPE_MIN * 4
    got: list[bytes] = []
    reader = threading.Thread(target=lambda: got.append(_read_exactly(receiver, size)))
    reader.start()
    _send_all(sender, b"x" * 100)
    assert [bool(lane.segments) for lane in near] == [True, False, False, False]
    _send_all(sender, os.urandom(wire.STRIPE_MIN * 4))
    assert all(lane.segments for lane in near)
    reader.join(60)
    assert len(got[0]) == size
    _close_all(near, far)


def test_a_segment_that_repeats_received_bytes_ends_the_stream() -> None:
    import socket

    a, b = socket.socketpair()
    c, d = socket.socketpair()
    receiver = wire.Striped([b.send, d.send], [b.recv_into, d.recv_into])
    first = wire.HEADER.pack(wire.MAGIC, wire.SEGMENT, 0, 0, 4) + (0).to_bytes(8, "little")
    a.sendall(first + b"abcd")
    assert _read_exactly(receiver, 4) == b"abcd"
    again = wire.HEADER.pack(wire.MAGIC, wire.SEGMENT, 0, 1, 4) + (2).to_bytes(8, "little")
    c.sendall(again + b"cdef")
    assert receiver.recv_into(memoryview(bytearray(16))) == 0
    for sock in (a, b, c, d):
        sock.close()


def test_a_lane_that_closes_ends_the_stream_after_the_bytes_before_it() -> None:
    import socket

    sender, receiver, near, far = _striped_pair(2)
    _send_all(sender, b"hello")
    assert _read_exactly(receiver, 5) == b"hello"
    # Shut down first: close alone leaves the descriptor open under this end's blocked reader.
    near[1].sock.shutdown(socket.SHUT_RDWR)
    near[1].sock.close()
    assert receiver.recv_into(memoryview(bytearray(8))) == 0
    near[0].sock.close()
    for sock in far:
        sock.close()
