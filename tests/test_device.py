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
    receiver = frames.StreamTransport(read_fd=read_a, write_fd=None)
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


def test_host_local_is_refused_without_torch(let, cpu, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(letify.UnsupportedMode, match="PyTorch"):
        cpu.provider.check_mode(cpu._placed("local"))
