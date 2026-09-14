"""The binary wire of a persistent channel, exercised through the real Local worker.

Spec sections pinned here: "Frames", "Worker output", "Waiting for a reply" and "Argument
addressing". Every call goes through ``Local``, which starts the same worker behind the
same frames an SSH runtime uses, so sizes, speeds and memory measured here are the
protocol's own.
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import letify
from letify.protocol import codec, wire

MiB = 1 << 20


# -- Spec: Frames --------------------------------------------------------------


def test_a_frame_header_is_sixteen_bytes_of_magic_type_flags_stream_and_length() -> None:
    header = wire.HEADER.pack(wire.MAGIC, wire.DATA, 0, 7, 3 * MiB)
    assert len(header) == 16
    assert wire.MAGIC == b"LF"
    assert struct.unpack("<2sBBIQ", header) == (b"LF", 4, 0, 7, 3 * MiB)
    assert (wire.HELLO, wire.REQUEST, wire.REPLY, wire.DATA) == (1, 2, 3, 4)
    assert (wire.STDOUT, wire.STDERR, wire.SHUTDOWN) == (5, 6, 7)


def test_a_large_bytes_value_travels_out_of_band_without_a_copy() -> None:
    payload = os.urandom(2 * MiB)
    head, buffers = wire.dumps({"op": "call", "payload": payload})
    # The pickle holds a reference to the buffer, not the bytes themselves.
    assert len(head) < 4096
    assert [memoryview(b).nbytes for b in buffers] == [2 * MiB]
    assert memoryview(buffers[0]).obj is payload
    assert wire.loads(head, [bytearray(b) for b in buffers]) == {"op": "call", "payload": payload}


def test_data_frames_are_at_most_eight_mebibytes() -> None:
    assert 8 * MiB == wire.CHUNK


def test_the_bootstrap_stub_reads_a_byte_count_line_then_the_source_in_binary() -> None:
    from letify.protocol.worker import BOOTSTRAP

    assert "sys.stdin.buffer" in BOOTSTRAP
    assert "base64" not in BOOTSTRAP
    assert "\n" not in BOOTSTRAP


def test_a_frame_with_the_wrong_magic_is_a_protocol_error() -> None:
    read, write = os.pipe()
    os.write(write, b"XX" + bytes(14))
    os.close(write)
    receiver = wire.Receiver(lambda view: os.readv(read, [view]))
    with pytest.raises(letify.ProtocolError, match="magic"):
        receiver.next_event()
    os.close(read)


# -- the calls these tests make ------------------------------------------------


def _declare(let: letify.Launcher, cpu: letify.Instance):
    @let.function(device=cpu, host=letify.remote)
    def noop() -> int:
        return 1

    @let.function(device=cpu, host=letify.remote)
    def take(payload: bytes) -> tuple[int, str]:
        import hashlib

        return len(payload), hashlib.blake2b(payload, digest_size=16).hexdigest()

    @let.function(device=cpu, host=letify.remote)
    def give(size: int) -> bytes:
        return os.urandom(size)

    return noop, take, give


# -- Spec: Frames, large values ------------------------------------------------


@pytest.mark.parametrize("size", [1 * MiB, 64 * MiB, 512 * MiB], ids=["1MiB", "64MiB", "512MiB"])
def test_arguments_and_results_round_trip_with_matching_checksums(let, cpu, size: int) -> None:
    noop, take, give = _declare(let, cpu)
    payload = os.urandom(size)
    with let.keep_alive():
        noop()
        assert take(payload) == (size, hashlib.blake2b(payload, digest_size=16).hexdigest())
        result = give(size)
        assert isinstance(result, bytes) and len(result) == size
        # A second result of the same size is different bytes, so the reply is not cached.
        assert give(size) != result


def test_a_64_mib_value_moves_at_least_300_mib_per_second_each_way_on_this_machine(
    let, cpu
) -> None:
    # A throughput floor, not a benchmark. Base64 lines managed 55 MiB/s up and 95 MiB/s
    # down here and binary frames about 1000 MiB/s, so 300 MiB/s keeps a wide margin for a
    # loaded CI machine. LETIFY_THROUGHPUT_FLOOR lowers it where a runner is slower still.
    floor = float(os.environ.get("LETIFY_THROUGHPUT_FLOOR", "300"))
    noop, take, _give = _declare(let, cpu)

    @let.function(device=cpu, host=letify.remote)
    def filled(size: int, byte: int) -> bytes:
        # Cheap to build, unlike os.urandom, so the download time is the wire's own.
        return bytes([byte]) * size

    with let.keep_alive():
        noop()
        best_up = best_down = 0.0
        for attempt in range(3):
            # A fresh value each time, so the upload is not answered from the blob table.
            payload = os.urandom(64 * MiB)
            start = time.perf_counter()
            assert take(payload)[0] == 64 * MiB
            best_up = max(best_up, 64 / (time.perf_counter() - start))
            start = time.perf_counter()
            assert len(filled(64 * MiB, attempt)) == 64 * MiB
            best_down = max(best_down, 64 / (time.perf_counter() - start))
    assert best_up >= floor, f"upload {best_up:.0f} MiB/s"
    assert best_down >= floor, f"download {best_down:.0f} MiB/s"


_RSS_SCRIPT = textwrap.dedent(
    """
    import os, resource, sys
    import letify

    project = sys.argv[1]
    let = letify.Launcher(project, home=False, announce=False)
    cpu = let.providers.local.CPU._placed("remote")

    @let.function(device=cpu, host=letify.remote)
    def worker_rss() -> int:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    @let.function(device=cpu, host=letify.remote)
    def take(payload: bytes) -> tuple:
        import resource
        return len(payload), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    size = 512 << 20
    with let.keep_alive():
        worker_before = worker_rss()
        payload = os.urandom(size)
        client_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        length, worker_after = take(payload)
        client_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert length == size
    print(client_after - client_before, worker_after - worker_before)
    """
)


@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss is KiB on Linux only")
def test_peak_memory_for_a_512_mib_argument_stays_below_two_and_a_half_copies(
    tmp_path: Path,
) -> None:
    # A separate process, so the peak is this upload's and not an earlier test's.
    project = tmp_path / "project" / ".letify"
    project.mkdir(parents=True)
    done = subprocess.run(
        [sys.executable, "-c", _RSS_SCRIPT, str(project)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    client_kib, worker_kib = (int(x) for x in done.stdout.split()[-2:])
    limit_kib = int(2.5 * 512 * 1024)
    # The client's own payload was allocated before the baseline, so its growth is what the
    # protocol added. The worker's growth includes the argument it received.
    assert client_kib < limit_kib, f"client grew {client_kib // 1024} MiB"
    assert worker_kib < limit_kib, f"worker grew {worker_kib // 1024} MiB"


# -- Spec: Worker output -------------------------------------------------------


def test_a_megabyte_of_stderr_and_5000_progress_updates_arrive_live(let, cpu) -> None:
    @let.function(device=cpu, host=letify.remote)
    def chatty() -> float:
        import sys
        import time

        sys.stderr.write("x" * (1 << 20))
        sys.stderr.flush()
        for step in range(5000):
            sys.stdout.write(f"\rstep {step:5d}")
            sys.stdout.flush()
        time.sleep(1.0)
        return time.monotonic()

    noop, _take, _give = _declare(let, cpu)
    arrivals: list[tuple[float, str, bytes]] = []

    def record(stream: str, data: bytes) -> None:
        arrivals.append((time.monotonic(), stream, bytes(data)))

    with let.keep_alive():
        noop()
        [runtime] = let.pool.live
        runtime.channel.on_output = record
        start = time.monotonic()
        returned = chatty()
        runtime.channel.on_output = None
    assert time.monotonic() - start < 60
    stderr = b"".join(data for _, stream, data in arrivals if stream == "stderr")
    stdout = b"".join(data for _, stream, data in arrivals if stream == "stdout")
    assert stderr.count(b"x") == 1 << 20
    # Raw bytes, so every carriage return survives and nothing is split into lines.
    assert stdout.count(b"\r") == 5000
    assert stdout.endswith(b"\rstep  4999")
    last = max(when for when, _, _ in arrivals)
    # Written live: everything arrived during the sleep, before the call returned.
    assert last < returned - 0.5


def test_output_is_written_to_the_local_streams_while_the_call_runs(let, cpu, capfd) -> None:
    @let.function(device=cpu, host=letify.remote)
    def speaks() -> int:
        import sys

        print("to stdout")
        print("to stderr", file=sys.stderr)
        return 1

    assert speaks() == 1
    captured = capfd.readouterr()
    assert "to stdout" in captured.out
    assert "to stderr" in captured.err


def test_output_is_not_echoed_when_the_launcher_asks_for_silence(tmp_path, capfd) -> None:
    project = tmp_path / "quiet" / ".letify"
    project.mkdir(parents=True)
    let = letify.Launcher(project, home=False, announce=False, stream_logs=False)
    cpu = let.providers.local.CPU

    @let.function(device=cpu, host=letify.remote)
    def speaks() -> int:
        print("should stay quiet")
        return 1

    assert speaks() == 1
    assert "should stay quiet" not in capfd.readouterr().out


def test_printing_on_every_step_does_not_slow_the_step(let, cpu) -> None:
    # Nothing is synchronized per step: a print is one pipe write, and the loop never waits
    # for the client. Measured over SSH it added about 3.7 microseconds per step.
    @let.function(device=cpu, host=letify.remote)
    def loop(noisy: bool) -> float:
        import statistics
        import time

        durations = []
        for step in range(3000):
            start = time.perf_counter()
            sum(range(2000))
            if noisy:
                print(f"step {step} loss 0.5", flush=True)
            durations.append(time.perf_counter() - start)
        return statistics.median(durations)

    with let.keep_alive():
        quiet = min(loop(False) for _ in range(2))
        noisy = min(loop(True) for _ in range(2))
    assert noisy - quiet < 200e-6, f"print added {(noisy - quiet) * 1e6:.0f} us per step"


# -- Spec: Waiting for a reply -------------------------------------------------


def test_stat_and_lease_are_answered_while_a_long_call_runs(let, cpu) -> None:
    @let.function(device=cpu, host=letify.remote)
    def long_call() -> int:
        import time

        time.sleep(3.0)
        return 7

    @let.function(device=cpu, host=letify.remote)
    def noop() -> int:
        return 1

    results: list[int] = []
    with let.keep_alive():
        noop()
        [runtime] = let.pool.live
        thread = threading.Thread(
            target=lambda: results.append(runtime.channel.call(long_call.fn, (), {}, timeout=60)[0])
        )
        thread.start()
        time.sleep(0.5)
        start = time.monotonic()
        stat = runtime.stat()
        runtime.request({"op": "lease", "grace": 600.0}, timeout=30)
        answered = time.monotonic() - start
        assert thread.is_alive()
        thread.join(30)
    assert isinstance(stat["pid"], int)
    assert answered < 1.0
    assert results == [7]


# -- Spec: Argument addressing -------------------------------------------------


def test_a_repeated_large_argument_is_sent_once_and_hashed_once(let, cpu, monkeypatch) -> None:
    noop, take, _give = _declare(let, cpu)
    hashed: list[int] = []
    original = codec._hasher

    def counting():
        hashed.append(1)
        return original()

    monkeypatch.setattr(codec, "_hasher", counting)
    payload = os.urandom(64 * MiB)
    with let.keep_alive():
        noop()
        [runtime] = let.pool.live
        sent: list[str] = []
        request = runtime.channel.request

        def recording(message, **kwargs):
            sent.append(message["op"])
            return request(message, **kwargs)

        runtime.channel.request = recording
        for _ in range(3):
            assert take(payload)[0] == 64 * MiB
    assert sent.count("put_blob") == 1
    assert len(hashed) == 1


def test_a_mutable_argument_is_hashed_again_and_a_mutation_does_not_leak(let, cpu) -> None:
    @let.function(device=cpu, host=letify.remote)
    def scribble(buffer: bytearray) -> int:
        first = buffer[0]
        buffer[0] = 255
        return first

    value = bytearray(1 * MiB)
    with let.keep_alive():
        assert scribble(value) == 0
        # The worker unpickled a fresh copy, so the previous call's write is not seen.
        assert scribble(value) == 0
        value[0] = 9
        assert scribble(value) == 9
