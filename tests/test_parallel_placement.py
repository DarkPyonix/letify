"""Several connections at once for a large blob.

Spec "Several connections at once". The worker side runs for real through the Local
provider, which starts the same worker behind the same framed protocol a remote runtime
uses, so ``data_put_at`` and ``data_seal`` are exercised over the real path.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import pytest

import letify

MiB = 1 << 20


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """A project root as the working directory, with a private home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = tmp_path / "study"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'study'\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def worker_of(let, cpu):
    """A live local runtime and its channel."""

    @let.function(device=cpu, host=letify.remote)
    def nothing() -> None:
        return None

    nothing()
    [runtime] = let.pool.live
    return runtime


# -- Spec: Several connections at once, the worker side ----------------------------


def test_pieces_written_out_of_order_assemble_into_the_blob(let, cpu, tmp_path) -> None:
    from letify.protocol import codec

    body = bytes(range(256)) * 400
    digest = codec.digest_of(body)
    blobs = str(tmp_path / "blobs")
    with let.keep_alive():
        runtime = worker_of(let, cpu)
        request = runtime.channel.request
        request({"op": "data_begin", "dir": blobs, "digest": digest, "size": len(body)})
        # Deliberately out of order, as several connections would arrive.
        for offset, chunk in ((60000, body[60000:]), (0, body[:30000]), (30000, body[30000:60000])):
            request(
                {"op": "data_put_at", "dir": blobs, "digest": digest, "offset": offset,
                 "chunk": chunk}
            )
        request({"op": "data_seal", "dir": blobs, "digest": digest, "size": len(body)})

    written = Path(blobs) / digest[:2] / digest
    assert written.read_bytes() == body
    assert not os.access(written, os.W_OK)


def test_a_seal_that_does_not_match_the_digest_commits_nothing(let, cpu, tmp_path) -> None:
    # Spec "Several connections at once": the hash moves to the seal, so the seal is where
    # a range that never arrived is caught.
    from letify.errors import RuntimeFailure

    digest = "f" * 16
    blobs = str(tmp_path / "blobs")
    with let.keep_alive():
        runtime = worker_of(let, cpu)
        request = runtime.channel.request
        request({"op": "data_begin", "dir": blobs, "digest": digest, "size": 40})
        request(
            {"op": "data_put_at", "dir": blobs, "digest": digest, "offset": 0,
             "chunk": b"only the first half"}
        )
        try:
            request({"op": "data_seal", "dir": blobs, "digest": digest, "size": 40})
        except (RuntimeFailure, Exception) as exc:
            assert "short" in str(exc) or "digest" in str(exc), exc
        else:
            raise AssertionError("the seal accepted a file that is not the digest")

    assert not (Path(blobs) / digest[:2] / digest).exists()


def test_a_second_placement_replaces_what_a_failed_one_left(let, cpu, tmp_path) -> None:
    # Spec "Several connections at once": a failed stream leaves a partial file, and the
    # next placement's data_begin makes it afresh. Offset zero cannot do that job, because
    # with several connections it is just another range and may arrive last.
    from letify.protocol import codec

    body = b"fresh bytes only" * 32
    digest = codec.digest_of(body)
    blobs = str(tmp_path / "blobs")
    with let.keep_alive():
        runtime = worker_of(let, cpu)
        request = runtime.channel.request
        request({"op": "data_begin", "dir": blobs, "digest": digest, "size": 2000})
        request(
            {"op": "data_put_at", "dir": blobs, "digest": digest, "offset": 0,
             "chunk": b"stale" * 400}
        )
        request({"op": "data_begin", "dir": blobs, "digest": digest, "size": len(body)})
        request({"op": "data_put_at", "dir": blobs, "digest": digest, "offset": 0,
                 "chunk": body})
        request({"op": "data_seal", "dir": blobs, "digest": digest, "size": len(body)})

    assert (Path(blobs) / digest[:2] / digest).read_bytes() == body


# -- Spec: Several connections at once, the client side ----------------------------


def test_the_split_covers_the_file_exactly() -> None:
    from letify.store.pathdata import ranges

    assert ranges(100, 4) == [(0, 25), (25, 25), (50, 25), (75, 25)]
    # Equal but for the last, which takes the remainder.
    assert ranges(10, 4) == [(0, 2), (2, 2), (4, 2), (6, 4)]
    assert ranges(3, 4) == [(0, 1), (1, 1), (2, 1)]
    assert ranges(0, 4) == []
    assert ranges(100, 1) == [(0, 100)]
    for size, count in ((1, 4), (7, 3), (1 << 20, 5), (12345, 7)):
        split = ranges(size, count)
        assert sum(length for _offset, length in split) == size
        assert split[0][0] == 0
        for before, after in itertools.pairwise(split):
            assert before[0] + before[1] == after[0]


def test_an_account_sets_the_stream_count_and_the_threshold() -> None:
    from conftest import provider_of

    from letify.providers.local import Local

    provider = provider_of(Local, "box")
    assert provider.transfer_streams == 4
    assert provider.transfer_parallel_mib == 64

    provider = provider_of(Local, "box", transfer_streams=8, transfer_parallel_mib=16)
    assert provider.transfer_streams == 8
    assert provider.transfer_parallel_mib == 16

    provider = provider_of(Local, "box", transfer_streams=1)
    assert provider.transfer_streams == 1


def test_a_transfer_channel_does_not_share_the_multiplexed_connection() -> None:
    # Spec "Several connections at once": ControlMaster=auto would put every further
    # session inside the first connection, which is the opposite of the point.
    from conftest import provider_of

    from letify.providers.shell import Shell

    provider = provider_of(Shell, "lab", address="gpu.example.edu")
    command = provider.transfer_command(runtime=None)
    joined = " ".join(command)
    assert "ControlPath=none" in joined
    assert "ControlMaster=no" in joined
    assert "ControlMaster=auto" not in joined


# -- Spec: Several connections at once, the placement --------------------------------


def persistent(launcher_from, **options):
    """A persistent local account, which sends data over the channel as a remote one does."""
    body = ["[lab]", 'kind = "local"', "persistent = true"]
    for name, value in options.items():
        rendered = str(value).lower() if isinstance(value, bool) else repr(value)
        body.append(f"{name} = {rendered}")
    return launcher_from("\n".join(body) + "\n")


def data_ops(monkeypatch) -> list[str]:
    """Every ``data_*`` request this process sends, in order, hooked at the frame layer."""
    from letify.protocol import wire

    seen: list[str] = []
    original = wire.Sender.message

    def message(self_, kind, stream, obj):
        if kind == wire.REQUEST and isinstance(obj, dict):
            name = str(obj.get("op", ""))
            if name.startswith("data_"):
                seen.append(name)
        return original(self_, kind, stream, obj)

    monkeypatch.setattr(wire.Sender, "message", message)
    return seen


def test_a_large_blob_is_placed_over_several_connections(
    project, launcher_from, monkeypatch
) -> None:
    # Spec "Several connections at once": the blob lands whole, which is what proves the
    # ranges and the seal line up, and the requests show it went out in parallel.
    from letify.store import pathdata

    monkeypatch.setattr(pathdata, "CHUNK", 1 << 16)
    # A local channel is never probed, so it has no round trip and would keep one stream,
    # which is the right answer for a local placement. The condition is pinned on its own
    # above; here the point is that the pieces assemble over several channels, so the
    # round trip of a long link is supplied.
    monkeypatch.setattr(
        pathdata,
        "_streams_for",
        lambda runtime, size: pathdata.streams_for(
            streams=4, threshold_mib=1, rtt_ms=138.0, size=size
        ),
    )
    seen = data_ops(monkeypatch)
    let = persistent(launcher_from, transfer_parallel_mib=1, transfer_streams=4)
    body = os.urandom(3 * MiB)
    source = project / "weights.bin"
    source.write_bytes(body)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_it(path) -> bytes:
        return open(path, "rb").read()

    assert read_it(source) == body
    assert "data_begin" in seen, seen
    assert "data_put_at" in seen, seen
    assert "data_seal" in seen, seen


def test_a_small_blob_keeps_the_single_stream(project, launcher_from, monkeypatch) -> None:
    # Spec "Several connections at once": below the threshold, opening connections costs
    # more than it saves, so data_put is still what goes out.
    seen = data_ops(monkeypatch)
    let = persistent(launcher_from, transfer_parallel_mib=1024, transfer_streams=4)
    source = project / "small.bin"
    source.write_bytes(b"small" * 1000)

    @let.function(device=let.providers.lab.CPU, host=letify.remote)
    def read_it(path) -> int:
        return len(open(path, "rb").read())

    assert read_it(source) == 5000
    assert "data_put" in seen, seen
    assert "data_put_at" not in seen, seen


# -- Spec: Several connections at once, the round trip condition ---------------------


def test_a_short_link_keeps_one_stream() -> None:
    # Spec "Several connections at once": measured to a lab server at 0.25 ms, one stream
    # carried 36.1 MiB/s and four carried 33.5, so splitting there is pure cost.
    from letify.store.pathdata import streams_for

    assert streams_for(streams=4, threshold_mib=64, rtt_ms=0.25, size=256 * MiB) == 1
    assert streams_for(streams=4, threshold_mib=64, rtt_ms=None, size=256 * MiB) == 1


def test_a_long_link_above_the_threshold_is_split() -> None:
    from letify.store.pathdata import streams_for

    assert streams_for(streams=4, threshold_mib=64, rtt_ms=138.0, size=256 * MiB) == 4
    assert streams_for(streams=8, threshold_mib=64, rtt_ms=138.0, size=256 * MiB) == 8


def test_a_small_blob_is_not_split_however_long_the_link() -> None:
    from letify.store.pathdata import streams_for

    assert streams_for(streams=4, threshold_mib=64, rtt_ms=138.0, size=8 * MiB) == 1


def test_one_stream_is_honoured_whatever_else_says() -> None:
    from letify.store.pathdata import streams_for

    assert streams_for(streams=1, threshold_mib=0, rtt_ms=500.0, size=1 << 30) == 1


def test_there_are_never_more_streams_than_chunks() -> None:
    from letify.store import pathdata

    # Two chunks cannot fill eight streams, and an empty range would send nothing.
    size = 2 * pathdata.CHUNK
    assert pathdata.streams_for(streams=8, threshold_mib=0, rtt_ms=138.0, size=size) == 2


def test_an_account_sets_the_round_trip_threshold() -> None:
    from conftest import provider_of

    from letify.providers.local import Local

    assert provider_of(Local, "box").transfer_parallel_rtt_ms == 20
    assert provider_of(Local, "box", transfer_parallel_rtt_ms=5).transfer_parallel_rtt_ms == 5


def test_a_provider_with_no_transfer_channel_keeps_one_stream() -> None:
    # Spec "Several connections at once": a provider whose channel is not a connection it
    # can open more of, such as Modal's sandbox pipes or Kaggle's kernel bridge, has no
    # transfer channel, and asking it for one would fail inside the placement.
    from letify.providers.kaggle import Kaggle
    from letify.providers.modal import Modal
    from letify.store import pathdata

    for cls in (Modal, Kaggle):
        assert not hasattr(cls, "transfer_channel"), cls.__name__

    class Bare:
        transfer_streams = 4
        transfer_parallel_mib = 1
        transfer_parallel_rtt_ms = 20

        def link(self, runtime):
            class Link:
                rtt_ms = 138.0

            return Link()

    class Runtime:
        provider = Bare()

    assert pathdata._streams_for(Runtime(), 256 << 20) == 1


def test_the_guard_does_not_go_through_a_provider_lookup() -> None:
    # A provider's __getattr__ answers accelerator names and raises UnknownInstance for
    # anything else, which is not an AttributeError, so hasattr on the instance raises it
    # instead of answering False. Asking the class is what answers the question.
    from letify.errors import UnknownInstance
    from letify.store import pathdata

    class Picky:
        transfer_streams = 4
        transfer_parallel_mib = 1
        transfer_parallel_rtt_ms = 0

        def __getattr__(self, name):
            raise UnknownInstance(f"no instance {name!r}")

    class Runtime:
        provider = Picky()

    assert pathdata._streams_for(Runtime(), 256 << 20) == 1
