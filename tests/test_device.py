"""PyTorch forwarding, exercised over a real device worker subprocess.

The worker runs the same executor a GPU runtime runs, on CPU tensors, behind the same
``StreamTransport`` over pipes. Nothing here needs a GPU, and everything here is skipped
when PyTorch does not import in the test environment.
"""

from __future__ import annotations

import gc
import os
import sys

import pytest

import letify
from letify.errors import RemoteError, UnsupportedMode

torch = pytest.importorskip("torch", reason="PyTorch forwarding tests need torch importable")

from letify.remoting import device as forwarding  # noqa: E402
from letify.remoting.device import frames  # noqa: E402


@pytest.fixture
def client():
    """A client connected to a worker subprocess that executes on CPU tensors."""
    connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
    try:
        with connected.activate():
            yield connected
    finally:
        connected.close()


# -- Spec: Dispatch mechanism --------------------------------------------------


def test_a_tensor_created_on_cuda_lives_on_the_remote_device(client) -> None:
    x = torch.ones(2, 3, device="cuda")
    y = torch.arange(6.0).reshape(2, 3).cuda()
    assert isinstance(x, forwarding.RemoteTensor)
    assert isinstance(y, forwarding.RemoteTensor)
    assert x.device == torch.device("cuda", 0)
    assert x.is_cuda
    assert y.cpu().tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_arithmetic_on_remote_tensors_matches_the_same_arithmetic_on_cpu(client) -> None:
    a = torch.randn(4, 5)
    b = torch.randn(5, 3)
    expected = torch.relu(a @ b * 2 - 1).sum(dim=0)
    got = torch.relu(a.cuda() @ b.cuda() * 2 - 1).sum(dim=0)
    assert torch.allclose(got.cpu(), expected)


def test_shape_inference_needs_no_round_trip(client) -> None:
    x = torch.zeros(8, 16, device="cuda")
    before = client.stats.round_trips
    y = (x.t() @ x).unsqueeze(0).expand(3, 16, 16)
    assert y.shape == (3, 16, 16)
    assert y.stride() == (0, 16, 1)
    assert y.dtype == torch.float32
    assert client.stats.round_trips == before


def test_a_data_dependent_shape_is_asked_of_the_runtime(client) -> None:
    x = torch.tensor([0.0, 1.0, 0.0, 2.0]).cuda()
    nonzero = torch.nonzero(x)
    assert nonzero.shape == (2, 1)
    assert nonzero.cpu().flatten().tolist() == [1, 3]


def test_a_module_trained_with_adam_matches_a_local_cpu_run(client) -> None:
    def train(to_device):
        torch.manual_seed(0)
        data = torch.randn(64, 8)
        model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
        model = to_device(model)
        data = to_device(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(5):
            loss = torch.nn.functional.mse_loss(model(data), data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses, [p.detach().cpu() for p in model.parameters()]

    with client.suspended():
        local_losses, local_params = train(lambda value: value)
    remote_losses, remote_params = train(lambda value: value.cuda())
    assert remote_losses == pytest.approx(local_losses, rel=1e-5, abs=1e-6)
    for remote, local in zip(remote_params, local_params, strict=True):
        assert torch.allclose(remote, local, rtol=1e-5, atol=1e-6)


def test_a_repeated_operator_reuses_its_inferred_metadata(client) -> None:
    a = torch.ones(8, 4, device="cuda")
    b = torch.ones(4, device="cuda")
    first = a + b
    before = client.stats.cached
    second = a + b
    assert client.stats.cached == before + 1
    assert second.shape == first.shape and second.stride() == first.stride()
    assert torch.equal(second.cpu(), torch.full((8, 4), 2.0))


def test_an_in_place_operator_served_from_the_cache_returns_its_input(client) -> None:
    a = torch.zeros(3, device="cuda")
    first = a.add_(1.0)
    second = a.add_(1.0)
    assert first is a and second is a
    assert a.cpu().tolist() == [2.0, 2.0, 2.0]


def test_an_optimizer_takes_its_foreach_path_on_remote_tensors(client) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.ones(2, 4, device="cuda")).sum().backward()
    optimizer.step()
    before = client.stats.ops
    optimizer.step()
    # Four parameters: the per parameter path issues well over twenty operators.
    assert client.stats.ops - before < 20


def test_a_tensor_method_enters_python_in_the_mode_only(client) -> None:
    assert forwarding.RemoteTensor.__torch_function__ is torch._C._disabled_torch_function_impl
    x = torch.zeros(2, device="cuda")
    assert x.device == torch.device("cuda", 0)
    assert x.is_cuda and x.get_device() == 0


# -- Spec: Operator templates ------------------------------------------------------


def test_a_structure_is_described_once_and_later_calls_carry_only_their_scalars(client) -> None:
    x = torch.ones(4, device="cuda")
    client.synchronize()
    before = client.stats.templates
    x.mul_(2.0)
    x.mul_(3.0)
    x.mul_(4.0)
    assert client.stats.templates == before + 1
    assert x.cpu().tolist() == [24.0, 24.0, 24.0, 24.0]


# -- Spec: Mapping cuda --------------------------------------------------------


def test_code_written_for_cuda_runs_unchanged(client) -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert torch.cuda.current_device() == 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(4, 2).to(device)
    out = model(torch.ones(3, 4, device=device))
    torch.cuda.synchronize()
    assert out.device.type == "cuda"
    assert next(model.parameters()).is_cuda
    assert out.shape == (3, 2)
    assert isinstance(torch.cuda.get_device_name(), str)


def test_a_second_cuda_device_is_refused(client) -> None:
    with pytest.raises(UnsupportedMode, match=r"cuda:1"):
        torch.zeros(1, device="cuda:1")


def test_a_cuda_api_with_no_local_counterpart_names_itself(client) -> None:
    with pytest.raises(UnsupportedMode, match=r"torch\.cuda\.Stream"):
        torch.cuda.Stream()


def _forbid_local_cuda(monkeypatch) -> None:
    """Make any call that initializes CUDA in this process raise, like a driverless CUDA build."""

    def no_driver(*args, **kwargs):
        raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")

    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: True)
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_driver)
    monkeypatch.setattr(torch.cuda.graphs, "_cuda_isCurrentStreamCapturing", no_driver)


def test_an_adam_step_never_initializes_cuda_in_this_process(client, monkeypatch) -> None:
    _forbid_local_cuda(monkeypatch)
    model = torch.nn.Linear(4, 2).cuda()
    for make in (torch.optim.Adam, torch.optim.AdamW):
        opt = make(model.parameters(), lr=1e-2)
        model(torch.ones(3, 4, device="cuda")).sum().backward()
        opt.step()
    assert torch.cuda.is_current_stream_capturing() is False
    torch.cuda.synchronize()


def test_device_capability_and_properties_are_the_runtimes(client, monkeypatch) -> None:
    _forbid_local_cuda(monkeypatch)
    assert torch.cuda.get_device_capability() == tuple(client.hello["capability"])
    properties = torch.cuda.get_device_properties(0)
    assert (properties.major, properties.minor) == tuple(client.hello["capability"])
    assert properties.name == client.hello["name"]
    assert properties.total_memory == client.hello["total_memory"]
    assert isinstance(torch.cuda.is_bf16_supported(including_emulation=False), bool)


def test_the_cuda_functions_are_restored_when_forwarding_ends() -> None:
    original = torch.cuda.is_available
    connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
    try:
        with connected.activate():
            assert torch.cuda.is_available is not original
        assert torch.cuda.is_available is original
    finally:
        connected.close()


# -- Spec: Batching and synchronization -----------------------------------------


def test_queued_operators_travel_without_a_round_trip_until_a_value_is_read(
    client, monkeypatch
) -> None:
    # A long linger leaves only the size rule and the synchronization to send the queue,
    # so the batch count does not depend on how fast this machine dispatches.
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    x = torch.ones(16, device="cuda")
    before = client.stats.snapshot()
    for _ in range(100):
        x = x * 1.0 + 0.0
    assert client.stats.round_trips == before.round_trips
    value = x.sum().item()
    assert value == 16.0
    delta = client.stats.snapshot() - before
    assert delta.round_trips == 1
    assert delta.ops >= 201
    assert delta.batches <= 3


def test_an_aged_queue_is_sent_by_the_next_dispatch_rather_than_in_the_background(
    client, monkeypatch
) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "IDLE_S", 60.0)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    before = client.stats.batches
    x.add_(1.0)
    import time

    time.sleep(0.05)
    assert client.stats.batches == before
    x.add_(1.0)
    assert client.stats.batches == before + 1


def test_a_queue_with_nothing_after_it_is_sent_once_idle(client, monkeypatch) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "IDLE_S", 0.05)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    before = client.stats.snapshot()
    x.add_(1.0)
    import time

    deadline = time.monotonic() + 5.0
    while client.stats.batches == before.batches and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.stats.batches == before.batches + 1
    assert client.stats.round_trips == before.round_trips


def test_item_and_cpu_are_synchronization_points(client) -> None:
    x = torch.full((3,), 2.5, device="cuda")
    before = client.stats.round_trips
    assert x[0].item() == 2.5
    assert client.stats.round_trips == before + 1
    assert x.cpu().tolist() == [2.5, 2.5, 2.5]
    assert client.stats.round_trips == before + 2


def test_control_flow_on_a_tensor_value_reads_it(client) -> None:
    x = torch.tensor([1.0, -1.0]).cuda()
    assert bool((x > 0).any())
    assert "tensor(" in repr(x)


def test_the_active_client_is_readable_inside_forwarding(client) -> None:
    assert forwarding.current_client() is client
    with client.suspended():
        assert forwarding.current_client() is None


def test_an_operator_is_counted_when_dispatched_rather_than_when_sent(client, monkeypatch) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    x = torch.ones(4, device="cuda")
    before = client.stats.snapshot()
    x.add_(1.0)
    delta = client.stats.snapshot() - before
    assert delta.ops == 1
    assert delta.batches == 0


def test_a_blocked_write_holds_the_sender_thread_and_not_the_step(client, monkeypatch) -> None:
    import threading
    import time

    gate = threading.Event()
    original = client.transport.send

    def held(head, buffers):
        gate.wait(10)
        original(head, buffers)

    monkeypatch.setattr(client.transport, "send", held)
    x = torch.ones(4, device="cuda")
    started = time.monotonic()
    for _ in range(400):
        x = x * 1.0 + 0.0
    elapsed = time.monotonic() - started
    gate.set()
    assert elapsed < 5.0
    assert x.sum().item() == 4.0


def test_a_read_with_nothing_else_being_sent_is_written_by_the_waiting_thread(
    client, monkeypatch
) -> None:
    import threading

    x = torch.ones(4, device="cuda")
    client.synchronize()
    writers: list[bool] = []
    original = client.transport.send

    def recorded(head, buffers):
        writers.append(threading.current_thread() is threading.main_thread())
        original(head, buffers)

    monkeypatch.setattr(client.transport, "send", recorded)
    assert (x * 2).sum().item() == 8.0
    assert writers and writers[-1] is True


def test_a_read_behind_a_batch_still_being_written_goes_through_the_sender_thread(
    client, monkeypatch
) -> None:
    import threading

    gate = threading.Event()
    writers: list[bool] = []
    original = client.transport.send

    def held(head, buffers):
        writers.append(threading.current_thread() is threading.main_thread())
        if len(writers) == 1:
            gate.wait(10)
        original(head, buffers)

    monkeypatch.setattr(client.transport, "send", held)
    x = torch.ones(4, device="cuda")
    client._flush()
    threading.Timer(0.2, gate.set).start()
    assert x.sum().item() == 4.0
    assert writers == [False, False]


def test_a_read_sends_only_the_entries_its_value_depends_on(client, monkeypatch) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    monkeypatch.setattr(client_module, "IDLE_S", 60.0)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    total = (x * 2).sum()
    later = x + 1
    assert total.item() == 8.0
    assert client.queued > 0
    assert later.cpu().tolist() == [2.0, 2.0, 2.0, 2.0]
    assert client.queued == 0


def test_an_in_place_entry_after_the_producer_sends_the_whole_queue(client, monkeypatch) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    total = x.sum()
    x.add_(1.0)
    assert total.item() == 4.0
    assert client.queued == 0
    assert x.cpu().tolist() == [2.0, 2.0, 2.0, 2.0]


# -- Spec: Step capture -------------------------------------------------------------


def _train(to_device, *, steps, batch=16, sizes=None, read_inside=False, optimizer="adam"):
    """A small training loop, returning losses, parameters, gradients and optimizer state."""
    torch.manual_seed(0)
    data = to_device(torch.randn(256, 8))
    model = to_device(
        torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
    )
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    losses = []
    for step in range(steps):
        size = sizes(step) if sizes else batch
        start = (step * 7) % (256 - size)
        chunk = data[start : start + size]
        loss = torch.nn.functional.mse_loss(model(chunk), chunk)
        if read_inside:
            losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not read_inside:
            losses.append(loss.detach())
    losses = [float(value) for value in losses]
    params = [p.detach().cpu() for p in model.parameters()]
    grads = [p.grad.detach().cpu() for p in model.parameters()]
    state = [
        value.detach().cpu()
        for p in model.parameters()
        for value in opt.state[p].values()
        if isinstance(value, torch.Tensor)
    ]
    return losses, params, grads, state


def _assert_same(remote, local) -> None:
    remote_losses, *remote_tensors = remote
    local_losses, *local_tensors = local
    assert remote_losses == pytest.approx(local_losses, rel=1e-5, abs=1e-6)
    for remote_group, local_group in zip(remote_tensors, local_tensors, strict=True):
        assert len(remote_group) == len(local_group)
        for got, want in zip(remote_group, local_group, strict=True):
            assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_a_repeated_training_step_is_captured_once_and_replayed(client) -> None:
    with client.suspended():
        local = _train(lambda value: value, steps=12)
    before = client.stats.snapshot()
    remote = _train(lambda value: value.cuda(), steps=12)
    delta = client.stats.snapshot() - before
    _assert_same(remote, local)
    assert delta.steps == 1
    assert delta.fallbacks == 0
    # Steps one to three run eagerly: the first initializes Adam, the next two are detection.
    assert delta.replayed >= delta.ops // 2


def test_a_replayed_repetition_is_one_entry_not_one_per_operator(client, monkeypatch) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    steps = 20
    before = client.stats.snapshot()
    _train(lambda value: value.cuda(), steps=steps, optimizer="sgd")
    client.synchronize()
    delta = client.stats.snapshot() - before
    eager = delta.ops - delta.replayed
    # Operators and reads both count in ops, so the step's own length divides replayed.
    size = len(client.tracer.active.ops)
    repetitions = -(-delta.replayed // size)
    assert repetitions >= steps // 2
    # Every eager operator is one entry, and every repetition adds one more.
    assert delta.entries <= eager + repetitions + 2


def test_optimizer_state_advances_through_replayed_steps_as_it_does_eagerly(client) -> None:
    with client.suspended():
        local = _train(lambda value: value, steps=10, optimizer="sgd")
    remote = _train(lambda value: value.cuda(), steps=10, optimizer="sgd")
    _assert_same(remote, local)
    assert client.stats.replayed > 0


def test_a_step_whose_shape_changes_mid_run_falls_back_and_matches_eager(client) -> None:
    def sizes(step):
        return 24 if step in (7, 8) else 16

    with client.suspended():
        local = _train(lambda value: value, steps=16, sizes=sizes)
    before = client.stats.snapshot()
    remote = _train(lambda value: value.cuda(), steps=16, sizes=sizes)
    delta = client.stats.snapshot() - before
    _assert_same(remote, local)
    assert delta.fallbacks >= 1
    assert delta.replayed > 0


def test_a_read_inside_a_repetition_sends_the_part_so_far_and_the_step_continues(client) -> None:
    with client.suspended():
        local = _train(lambda value: value, steps=12, read_inside=True)
    before = client.stats.snapshot()
    remote = _train(lambda value: value.cuda(), steps=12, read_inside=True)
    delta = client.stats.snapshot() - before
    _assert_same(remote, local)
    assert delta.replayed > 0
    assert delta.fallbacks == 0


def test_an_error_inside_a_replayed_step_names_its_operator(client) -> None:
    table = torch.arange(8.0).cuda()
    total = torch.zeros((), device="cuda")
    for step in range(12):
        position = 50 if step == 10 else step % 8
        picked = table[torch.tensor([position]).cuda()]
        total = total + picked.sum() * 1.0 + 0.0
        total = total * 1.0 - 0.0
        total = total + 0.0
    assert client.stats.replayed > 0
    with pytest.raises(RemoteError, match=r"aten\.index"):
        torch.cuda.synchronize()
    assert table.cpu().tolist() == [float(value) for value in range(8)]


def test_handles_created_in_repetitions_are_released(client) -> None:
    _train(lambda value: value.cuda(), steps=4)
    gc.collect()
    first = client.live_handles()
    _train(lambda value: value.cuda(), steps=16)
    gc.collect()
    assert client.live_handles() == first
    assert client.stats.replayed > 0


def _loop(to_device, steps, *, on_step=None, flag_at=None):
    """A training loop on one model, with a batch slice, a float scalar and a literal per step.

    ``on_step(step)`` runs before each step. ``flag_at`` is the step whose ``keepdim`` literal
    differs from every other step's.
    """
    torch.manual_seed(0)
    data = to_device(torch.randn(256, 8))
    model = to_device(
        torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for step in range(steps):
        if on_step is not None:
            on_step(step)
        # Four batch offsets in turn, so every metadata key has been seen by step 12.
        start = (step % 4) * 7
        chunk = data[start : start + 16]
        out = model(chunk)
        total = out.sum(dim=0, keepdim=step == flag_at).sum()
        loss = torch.nn.functional.mse_loss(out, chunk) + total * (1e-3 / (1 + step % 3))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    params = [p.detach().cpu() for p in model.parameters()]
    return [float(value) for value in losses], params


def test_an_operator_at_a_replayed_position_is_read_without_the_full_argument_walk(
    client, monkeypatch
) -> None:
    from letify.remoting.device import tensor as tensor_module

    walks = []
    full = tensor_module._read

    def counting(values, *rest):
        walks.append(step_now[0])
        return full(values, *rest)

    step_now = [0]

    def on_step(step):
        step_now[0] = step

    monkeypatch.setattr(tensor_module, "_read", counting)
    with client.suspended():
        local = _loop(lambda value: value, 24)
    walks.clear()
    before = client.stats.snapshot()
    remote = _loop(lambda value: value.cuda(), 24, on_step=on_step)
    delta = client.stats.snapshot() - before
    assert remote[0] == pytest.approx(local[0], rel=1e-5, abs=1e-6)
    for got, want in zip(remote[1], local[1], strict=True):
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)
    assert delta.fallbacks == 0
    assert delta.replayed > 0
    # The float scalar cycles through three values, so from step 12 every key has been seen.
    assert [step for step in walks if step >= 12] == []


def test_a_literal_that_changes_at_a_replayed_position_takes_the_full_path(client) -> None:
    with client.suspended():
        local = _loop(lambda value: value, 16, flag_at=12)
    before = client.stats.snapshot()
    remote = _loop(lambda value: value.cuda(), 16, flag_at=12)
    delta = client.stats.snapshot() - before
    assert remote[0] == pytest.approx(local[0], rel=1e-5, abs=1e-6)
    for got, want in zip(remote[1], local[1], strict=True):
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)
    assert delta.fallbacks >= 1


# -- Spec: Handles --------------------------------------------------------------


def test_dropped_tensors_release_their_remote_handles(client) -> None:
    base = client.live_handles()
    kept = [torch.zeros(4, device="cuda") for _ in range(10)]
    assert client.live_handles() == base + 10
    del kept
    gc.collect()
    assert client.live_handles() == base


def test_detach_shares_the_handle_instead_of_sending_an_operator(client) -> None:
    x = torch.zeros(4, device="cuda")
    before = client.stats.ops
    y = x.detach()
    assert client.stats.ops == before
    y.add_(1.0)
    assert x.cpu().tolist() == [1.0, 1.0, 1.0, 1.0]


# -- Spec: Transfers and Transport ------------------------------------------------


def test_a_buffer_travels_beside_the_head_rather_than_inside_it() -> None:
    read_a, write_a = os.pipe()
    sender = frames.StreamTransport(read_fd=None, write_fd=write_a)
    # The executor's end: the client sends REQUEST frames, so that is what this end reads.
    from letify.protocol import wire

    receiver = frames.StreamTransport(read_fd=read_a, write_fd=None, incoming=wire.REQUEST)
    payload = torch.arange(1 << 16, dtype=torch.int32)
    view = frames.tensor_view(payload)
    assert view.nbytes == payload.numel() * 4
    import threading

    got: list = []
    thread = threading.Thread(target=lambda: got.append(receiver.recv()))
    thread.start()
    sender.send(b"head", [view])
    thread.join(timeout=10)
    head, buffers = got[0]
    assert head == b"head"
    assert len(buffers) == 1
    assert torch.equal(torch.frombuffer(buffers[0], dtype=torch.int32), payload)
    sender.close()
    receiver.close()


def test_a_large_copy_round_trips_both_ways(client) -> None:
    data = torch.randn(1 << 20)
    back = data.cuda().cpu()
    assert torch.equal(back, data)


# -- Spec: Failure semantics -------------------------------------------------------


def test_a_remote_error_surfaces_at_the_next_synchronization_with_its_operator(client) -> None:
    good = torch.ones(3, device="cuda")
    # Meta inference accepts any index, so only the runtime can find it out of range.
    picked = good[torch.tensor([5]).cuda()]
    picked.add_(1)
    with pytest.raises(RemoteError) as caught:
        torch.cuda.synchronize()
    assert "aten.index" in str(caught.value)
    assert "Traceback" in caught.value.remote_traceback
    assert good.cpu().tolist() == [1.0, 1.0, 1.0]


def test_a_remote_operator_error_reports_the_operator_name(client) -> None:
    index = torch.tensor([7]).cuda()
    table = torch.zeros(3, device="cuda")
    table.index_fill_(0, index, 1.0)
    with pytest.raises(RemoteError, match=r"aten\.index_fill_"):
        torch.cuda.synchronize()
    assert table.sum().item() == 0.0


# -- Spec: Version guards ---------------------------------------------------------


def test_an_old_torch_is_refused_with_the_version_it_needs() -> None:
    with pytest.raises(UnsupportedMode, match=r"2\.0\.1.*2\.1"):
        forwarding.check_torch_version("2.0.1")
    forwarding.check_torch_version("2.5.1+cu121")


def test_a_worker_on_another_torch_minor_is_refused() -> None:
    with pytest.raises(UnsupportedMode, match=r"2\.4"):
        forwarding.check_worker_version(local="2.5.1", remote="2.4.0+cu121")
    forwarding.check_worker_version(local="2.5.1+cpu", remote="2.5.1+cu121")


@pytest.mark.skipif(
    not hasattr(torch.Tensor, "_make_wrapper_subclass"),
    reason="this torch has no _make_wrapper_subclass, so forwarding is refused",
)
def test_the_installed_torch_passes_the_guard() -> None:
    forwarding.check_torch_version(torch.__version__)


# -- Spec: Execution modes ---------------------------------------------------------


def test_a_host_local_function_runs_here_and_computes_on_the_runtime(let, cpu) -> None:
    @let.function(device=cpu, host="local")
    def step() -> tuple[int, float]:
        x = torch.ones(4, 4, device="cuda")
        return os.getpid(), float((x @ x).sum().item())

    pid, value = step()
    assert pid == os.getpid()
    assert value == 64.0


def test_a_session_executor_rides_the_call_channel_with_buffers_beside_the_head(let, cpu) -> None:
    @let.function(device=cpu, host="local")
    def probe() -> tuple[str, bool, bool]:
        client = forwarding.current_client()
        data = torch.randn(1 << 20)
        back = data.cuda().cpu()
        return type(client.transport).__name__, client.process is None, torch.equal(back, data)

    name, no_process, equal = probe()
    assert name == "ChannelTransport"
    assert no_process
    assert equal


def test_host_local_is_refused_without_torch(let, cpu, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(letify.UnsupportedMode, match="PyTorch"):
        cpu.provider.check_mode(cpu._placed("local"))
