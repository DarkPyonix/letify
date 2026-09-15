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


def test_an_operator_returning_new_tensors_reuses_metadata_at_a_new_storage_offset(client) -> None:
    data = torch.arange(48.0).reshape(12, 4).cuda()
    weight = torch.ones(3, 4, device="cuda")
    first = torch.mm(data[0:4], weight.t())
    before = client.stats.cached
    second = torch.mm(data[4:8], weight.t())
    # slice is a view and is inferred again; t and mm come from the cache.
    assert client.stats.cached == before + 2
    assert second.shape == first.shape and second.stride() == first.stride()
    assert second.storage_offset() == 0
    expected = torch.arange(16.0, 32.0).reshape(4, 4).sum(dim=1, keepdim=True).expand(4, 3)
    assert torch.equal(second.cpu(), expected)


def test_a_view_at_a_new_storage_offset_reports_its_own_offset(client) -> None:
    data = torch.arange(48.0).reshape(12, 4).cuda()
    first = data[0:4].t()
    second = data[4:8].t()
    assert first.storage_offset() == 0
    assert second.storage_offset() == 16
    assert torch.equal(second.cpu(), torch.arange(16.0, 32.0).reshape(4, 4).t())


def test_an_in_place_operator_served_from_the_cache_returns_its_input(client) -> None:
    a = torch.zeros(3, device="cuda")
    first = a.add_(1.0)
    second = a.add_(1.0)
    assert first is a and second is a
    assert a.cpu().tolist() == [2.0, 2.0, 2.0]


def test_an_in_place_restride_is_mirrored_on_the_wrapper_it_returns(client) -> None:
    for _ in range(2):  # the second call is served from the metadata cache
        x = torch.arange(16.0).reshape(1, 16, 1, 1).cuda()
        alias = x.detach()
        y = x.as_strided_((1, 16, 1, 1), (16, 1, 16, 16))
        assert y is x
        assert x.stride() == (16, 1, 16, 16)
        assert alias.stride() == (16, 1, 1, 1)
        assert x.contiguous().cpu().flatten().tolist() == list(range(16))


def test_a_channels_last_convolution_net_trains_as_it_does_on_cpu(client) -> None:
    def train(to_device):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 4),
        )
        model = to_device(model).to(memory_format=torch.channels_last)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        data = torch.randn(4, 3, 6, 6)
        target = torch.tensor([0, 1, 2, 3])
        losses = []
        for _ in range(4):
            x = to_device(data).contiguous(memory_format=torch.channels_last)
            loss = torch.nn.functional.cross_entropy(model(x), to_device(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        return losses, x.stride()

    expected, expected_stride = train(lambda value: value)
    got, stride = train(lambda value: value.cuda())
    assert stride == expected_stride
    assert got == pytest.approx(expected, rel=1e-5)


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


# -- Spec: Kernel selection --------------------------------------------------------


def _template_names(connected) -> set[str]:
    return {str(key[0]) for key in connected._templates}


def _batch_norm_arguments():
    x = torch.randn(4, 3, 5, 5, device="cuda")
    weight = torch.ones(3, device="cuda")
    bias = torch.zeros(3, device="cuda")
    mean = torch.zeros(3, device="cuda")
    var = torch.ones(3, device="cuda")
    return x, mean, var, weight, bias


def test_on_a_cpu_executor_batch_norm_and_attention_keep_their_ordinary_kernels(client) -> None:
    x, mean, var, weight, bias = _batch_norm_arguments()
    q = torch.randn(2, 2, 4, 8, device="cuda")
    client.synchronize()
    before = client.stats.round_trips
    normed = torch.nn.functional.batch_norm(x, mean, var, weight, bias, training=True)
    attended = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert client.stats.round_trips == before
    names = _template_names(client)
    assert "aten.native_batch_norm.default" in names
    assert not any("cudnn" in name or "flash" in name or "efficient" in name for name in names)
    assert normed.shape == x.shape and attended.shape == q.shape
    torch.cuda.synchronize()


def test_the_runtime_is_asked_for_a_kernel_once_per_signature(client) -> None:
    first = _batch_norm_arguments()
    second = _batch_norm_arguments()
    client.synchronize()
    before = client.stats.round_trips
    answers = [
        client.kernel("batch_norm", (x, weight, bias, mean, var), {"training": True})
        for x, mean, var, weight, bias in (first, second)
    ]
    assert answers == ["Native", "Native"]
    assert client.stats.round_trips == before + 1


def test_the_runtime_answers_an_attention_kernel_request(client) -> None:
    query = torch.randn(2, 2, 16, 8, device="cuda")
    flags = {"dropout_p": 0.0, "is_causal": True, "scale": None, "enable_gqa": False}
    answer = client.kernel("scaled_dot_product_attention", (query, query, query, None), flags)
    assert answer in {"MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"}


@pytest.fixture
def unsent():
    """A client whose queued operators are never executed, for kernels a CPU cannot run."""
    connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
    try:
        with connected.activate():
            yield connected
    finally:
        connected.process.kill()
        try:
            connected.close()
        except letify.RuntimeLost:
            pass


def test_a_cudnn_answer_forwards_cudnn_batch_norm(unsent, monkeypatch) -> None:
    monkeypatch.setitem(unsent.hello, "device", "cuda")
    monkeypatch.setattr(unsent, "kernel", lambda *args, **kwargs: "Cudnn")
    x, mean, var, weight, bias = _batch_norm_arguments()
    x.requires_grad_(True)
    normed = torch.nn.functional.batch_norm(x, mean, var, weight, bias, training=True)
    normed.sum().backward()
    names = _template_names(unsent)
    assert "aten.cudnn_batch_norm.default" in names
    assert "aten.cudnn_batch_norm_backward.default" in names
    assert "aten.native_batch_norm.default" not in names
    assert normed.shape == x.shape and normed.dtype == x.dtype


def test_a_flash_answer_forwards_flash_attention_and_its_backward(unsent, monkeypatch) -> None:
    monkeypatch.setitem(unsent.hello, "device", "cuda")
    monkeypatch.setattr(unsent, "kernel", lambda *args, **kwargs: "FLASH_ATTENTION")
    q = torch.randn(2, 2, 16, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attended = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    attended.float().sum().backward()
    names = _template_names(unsent)
    assert "aten._scaled_dot_product_flash_attention.default" in names
    assert "aten._scaled_dot_product_flash_attention_backward.default" in names
    assert attended.shape == q.shape and attended.dtype == q.dtype


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


def test_memory_statistics_are_the_runtimes_and_never_initialize_cuda_here(
    client, monkeypatch
) -> None:
    _forbid_local_cuda(monkeypatch)
    x = torch.ones(1024, device="cuda")
    for device in (None, 0, "cuda", "cuda:0", torch.device("cuda:0"), x.device):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.reset_max_memory_cached(device)
        torch.cuda.reset_accumulated_memory_stats(device)
        assert torch.cuda.memory_allocated(device) == 0
        assert torch.cuda.max_memory_allocated(device) == 0
        assert torch.cuda.memory_reserved(device) == 0
        assert torch.cuda.max_memory_reserved(device) == 0
        assert torch.cuda.memory_cached(device) == 0
        assert torch.cuda.max_memory_cached(device) == 0
        assert torch.cuda.memory_stats(device) == {}
        assert torch.cuda.mem_get_info(device) == (0, client.hello["total_memory"])
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    assert torch.cuda.memory.max_memory_allocated() == 0
    torch.cuda.memory.reset_peak_memory_stats()
    assert x.cpu().sum().item() == 1024.0


# -- Spec: Tensor subclasses ---------------------------------------------------------


class _Packed(torch.Tensor):
    """A wrapper subclass built the way torchao builds a quantized weight.

    Its device is its inner tensor's device, its detach aliases storage through
    ``return_and_correct_aliasing``, and every other operator runs on the unpacked value.
    """

    @staticmethod
    def __new__(cls, inner):
        return torch.Tensor._make_wrapper_subclass(
            cls, inner.shape, dtype=inner.dtype, device=inner.device, requires_grad=False
        )

    def __init__(self, inner):
        self.inner = inner

    def __tensor_flatten__(self):
        return ["inner"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        return _Packed(inner_tensors["inner"])

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from torch.utils import _pytree
        from torch.utils._python_dispatch import return_and_correct_aliasing

        kwargs = kwargs or {}
        if func is torch.ops.aten.detach.default:
            out = _Packed(args[0].inner.detach())
            return return_and_correct_aliasing(func, args, kwargs, out)
        unpacked = _pytree.tree_map_only(_Packed, lambda packed: packed.inner * 2, args)
        return func(*unpacked, **kwargs)


def test_a_wrapper_subclass_around_a_remote_tensor_runs_like_a_cuda_tensor(client) -> None:
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    x = torch.ones(2, 4)
    expected = torch.nn.functional.linear(x, values * 2)
    packed = _Packed(values.cuda())
    weight = torch.nn.Parameter(packed, requires_grad=False)
    assert type(weight) is _Packed
    assert weight.device == torch.device("cuda", 0)
    assert weight.is_cuda
    out = torch.nn.functional.linear(x.cuda(), weight)
    assert out.device.type == "cuda"
    assert torch.equal(out.cpu(), expected)


def test_memory_statistics_of_a_second_device_are_refused(client) -> None:
    with pytest.raises(UnsupportedMode, match=r"cuda:1"):
        torch.cuda.max_memory_allocated("cuda:1")


def test_the_cuda_functions_are_restored_when_forwarding_ends() -> None:
    original = torch.cuda.is_available
    connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
    try:
        with connected.activate():
            assert torch.cuda.is_available is not original
        assert torch.cuda.is_available is original
    finally:
        connected.close()


def test_a_dataloader_with_forked_workers_leaves_the_session_usable(client) -> None:
    data = torch.arange(64.0).reshape(32, 2)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data), batch_size=8, num_workers=2, timeout=60
    )
    total = torch.zeros(2, device="cuda")
    for (batch,) in loader:
        total += batch.cuda().sum(dim=0)
    assert total.cpu().tolist() == data.sum(dim=0).tolist()


# -- Spec: Pinned memory ------------------------------------------------------------


def _forbid_local_pinning(monkeypatch) -> None:
    """Make PyTorch's own pinning raise, as it does on a CUDA build without a driver."""
    _forbid_local_cuda(monkeypatch)

    def no_driver(*args, **kwargs):
        raise RuntimeError("Found no NVIDIA driver on your system")

    monkeypatch.setattr(torch.Tensor, "pin_memory", no_driver)
    monkeypatch.setattr(torch.accelerator, "current_device_index", no_driver)
    monkeypatch.setattr(torch.accelerator, "set_device_index", no_driver)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


def test_pinning_a_host_tensor_copies_it_and_reports_it_pinned(monkeypatch) -> None:
    _forbid_local_pinning(monkeypatch)
    connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
    try:
        with connected.activate():
            source = torch.arange(12.0).reshape(3, 4)
            pinned = source.pin_memory()
            assert pinned.device.type == "cpu"
            assert pinned.is_pinned() and not source.is_pinned()
            assert pinned.data_ptr() != source.data_ptr()
            assert torch.equal(pinned, source)
            assert pinned.pin_memory() is pinned
            assert torch.accelerator.is_available()
            assert torch.accelerator.current_device_index() == 0
            with pytest.raises(UnsupportedMode):
                torch.accelerator.set_device_index(1)
    finally:
        connected.close()
    with pytest.raises(RuntimeError, match="no NVIDIA driver"):
        torch.ones(2).pin_memory()


def test_a_dataloader_with_pin_memory_feeds_non_blocking_uploads(monkeypatch) -> None:
    _forbid_local_pinning(monkeypatch)
    data = torch.arange(64.0).reshape(32, 2)
    connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
    try:
        with connected.activate():
            for workers in (0, 2):
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(data),
                    batch_size=8,
                    num_workers=workers,
                    pin_memory=True,
                    timeout=60 if workers else 0,
                )
                total = torch.zeros(2, device="cuda")
                before = connected.stats.round_trips
                for (batch,) in loader:
                    assert batch.is_pinned()
                    total += batch.cuda(non_blocking=True).sum(dim=0)
                assert connected.stats.round_trips == before
                assert total.cpu().tolist() == data.sum(dim=0).tolist()
    finally:
        connected.close()


# -- Spec: Compilation --------------------------------------------------------------


def test_torch_compile_returns_the_module_and_warns_that_it_runs_eagerly(
    client, monkeypatch
) -> None:
    import warnings

    from letify.remoting.device import cuda as mapping

    _forbid_local_cuda(monkeypatch)
    monkeypatch.setattr(mapping, "_COMPILE_WARNED", [False])
    model = torch.nn.Linear(4, 2).cuda()
    with pytest.warns(UserWarning, match="host='local'"):
        compiled = torch.compile(model)
    assert compiled is model
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decorate = torch.compile(mode="max-autotune")

        def function(x):
            return x * 2

        assert decorate(function) is function
        assert torch.compile(function, fullgraph=True) is function
        assert model.compile() is None
    x = torch.ones(3, 4, device="cuda")
    assert torch.equal(compiled(x).cpu(), model(x).cpu())


def test_a_compiled_training_loop_matches_the_same_loop_run_eagerly(client) -> None:
    def train(compile_it):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)
        ).cuda()
        forward = torch.compile(model) if compile_it else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        data = torch.randn(6, 8)
        losses = []
        for _ in range(4):
            loss = torch.nn.functional.mse_loss(forward(data.cuda()), torch.zeros(6, 4).cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert train(True) == train(False)


def test_torch_compile_is_restored_when_forwarding_ends() -> None:
    original, method = torch.compile, torch.nn.Module.compile
    connected = forwarding.connect(forwarding.worker_command(sys.executable), device="cpu")
    try:
        with connected.activate():
            assert torch.compile is not original
        assert torch.compile is original
        assert torch.nn.Module.compile is method
    finally:
        connected.close()


def _child_view(queue) -> None:
    import torch

    from letify.remoting.device import current_client

    torch.manual_seed(1)
    queue.put((current_client() is None, torch.cuda.manual_seed_all.__module__))


def test_a_process_forked_inside_forwarding_sees_plain_torch_cuda(client) -> None:
    import multiprocessing

    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=_child_view, args=(queue,))
    child.start()
    no_client, module = queue.get(timeout=60)
    child.join(60)
    assert child.exitcode == 0
    assert no_client
    assert module == "torch.cuda.random"
    assert torch.ones(2, device="cuda").sum().item() == 2.0


def test_a_client_refuses_to_send_from_a_forked_process(client) -> None:
    import multiprocessing

    from letify.errors import RuntimeLost

    def attempt(queue) -> None:
        try:
            client.call("letify.seed", 1)
        except RuntimeLost as exc:
            queue.put(str(exc))
        else:
            queue.put("sent")

    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=attempt, args=(queue,))
    child.start()
    message = queue.get(timeout=60)
    child.join(60)
    assert "forked" in message
    assert torch.ones(3, device="cuda").sum().item() == 3.0


# -- Spec: Autocast ---------------------------------------------------------------


def test_autocast_runs_a_linear_layer_in_the_autocast_dtype(client) -> None:
    layer = torch.nn.Linear(4, 2).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(torch.ones(3, 4, device="cuda"))
        conv = torch.nn.functional.conv2d(
            torch.ones(1, 2, 5, 5, device="cuda"), torch.ones(3, 2, 3, 3, device="cuda")
        )
    assert out.dtype == torch.bfloat16
    assert conv.dtype == torch.bfloat16
    assert layer.weight.dtype == torch.float32
    torch.cuda.synchronize()


def test_autocast_runs_float32_functions_in_float32(client) -> None:
    half = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        normed = torch.nn.functional.layer_norm(half, (4,))
        soft = half.softmax(dim=-1)
        loss = torch.nn.functional.mse_loss(half, half)
    assert normed.dtype == soft.dtype == loss.dtype == torch.float32


def test_autocast_promotes_mixed_arguments_to_the_widest_dtype(client) -> None:
    half = torch.ones(2, device="cuda", dtype=torch.bfloat16)
    full = torch.ones(2, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        joined = torch.cat([half, full])
    assert joined.dtype == torch.float32
    assert joined.cpu().tolist() == [1.0, 1.0, 1.0, 1.0]


def test_autocast_cross_entropy_takes_log_softmax_in_the_input_dtype(client) -> None:
    # CUDA's cross_entropy_loss runs log_softmax uncast and casts only nll_loss to float32.
    torch.manual_seed(0)
    logits = torch.randn(8, 10) * 3
    target = torch.randint(0, 10, (8,))
    with client.suspended():
        half = logits.to(torch.bfloat16)
        want = torch.nn.functional.nll_loss(torch.log_softmax(half, 1).float(), target)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = torch.nn.functional.cross_entropy(logits.to(torch.bfloat16).cuda(), target.cuda())
    assert got.dtype == torch.float32
    assert got.cpu().item() == want.item()


def test_autocast_does_not_widen_index_copy(client) -> None:
    # CUDA autocast has no kernel for index_copy, so mixed dtypes raise there too.
    base = torch.zeros(4, device="cuda", dtype=torch.bfloat16)
    source = torch.ones(2, device="cuda")
    index = torch.tensor([0, 2]).cuda()
    with pytest.raises((RuntimeError, RemoteError)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base.index_copy(0, index, source)
        torch.cuda.synchronize()


def test_operators_outside_an_autocast_region_keep_their_dtype(client) -> None:
    layer = torch.nn.Linear(4, 2).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        inside = layer(torch.ones(3, 4, device="cuda"))
    after = layer(torch.ones(3, 4, device="cuda"))
    assert inside.dtype == after.dtype == torch.float32


def _mixed_step(to_device, autocast: bool):
    """One step of a small model, under autocast or with autocast's casts written out."""
    torch.manual_seed(0)
    model = to_device(torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.LayerNorm(16)))
    head = to_device(torch.nn.Linear(16, 4))
    data = to_device(torch.randn(6, 8))
    target = to_device(torch.randn(6, 4))
    if autocast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.mse_loss(head(model(data)), target)
    else:
        lin, norm = model
        bf = torch.bfloat16
        hidden = torch.nn.functional.linear(data.to(bf), lin.weight.to(bf), lin.bias.to(bf))
        hidden = torch.nn.functional.layer_norm(hidden.float(), (16,), norm.weight, norm.bias)
        out = torch.nn.functional.linear(hidden.to(bf), head.weight.to(bf), head.bias.to(bf))
        loss = torch.nn.functional.mse_loss(out.float(), target)
    loss.backward()
    params = [*model.parameters(), *head.parameters()]
    return float(loss), [p.grad.detach().cpu() for p in params]


def test_an_autocast_step_matches_the_same_casts_written_out(client) -> None:
    with client.suspended():
        want_loss, want_grads = _mixed_step(lambda value: value, autocast=False)
    got_loss, got_grads = _mixed_step(lambda value: value.cuda(), autocast=True)
    assert got_loss == want_loss
    for got, want in zip(got_grads, want_grads, strict=True):
        assert got.dtype == torch.float32
        assert torch.equal(got, want)


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


# -- Spec: Reads without waiting ---------------------------------------------------


def _hold_queue(monkeypatch) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    monkeypatch.setattr(client_module, "IDLE_S", 60.0)


def test_a_non_blocking_copy_to_the_host_returns_without_a_round_trip(client, monkeypatch) -> None:
    _hold_queue(monkeypatch)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    before = client.stats.snapshot()
    total = (x * 2).sum()
    host = total.to("cpu", non_blocking=True)
    also = x.to("cpu", non_blocking=True)
    assert type(host) is torch.Tensor and type(also) is torch.Tensor
    assert host.shape == () and host.dtype == torch.float32
    assert also.shape == (4,)
    assert (client.stats.snapshot() - before).round_trips == 0
    assert client.queued > 0
    assert host.item() == 8.0
    assert also.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_a_non_blocking_copy_holds_the_value_at_its_place_in_the_operator_order(client) -> None:
    x = torch.ones(3, device="cuda")
    early = x.to("cpu", non_blocking=True)
    x.add_(1.0)
    assert early.tolist() == [1.0, 1.0, 1.0]
    assert x.cpu().tolist() == [2.0, 2.0, 2.0]


def test_a_non_blocking_copy_into_a_host_tensor_fills_it_where_it_is_used(client) -> None:
    x = torch.arange(4.0).cuda()
    target = torch.zeros(4, dtype=torch.float64)
    returned = target.copy_(x * 2, non_blocking=True)
    assert returned is target
    assert target.tolist() == [0.0, 2.0, 4.0, 6.0]


def test_using_an_unfilled_tensor_waits_only_for_its_own_read(client, monkeypatch) -> None:
    _hold_queue(monkeypatch)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    host = (x * 3).to("cpu", non_blocking=True)
    later = x + 1
    assert host.sum().item() == 12.0
    assert client.queued > 0
    assert later.cpu().tolist() == [2.0, 2.0, 2.0, 2.0]


def test_reads_queued_together_come_back_in_one_reply(client, monkeypatch) -> None:
    _hold_queue(monkeypatch)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    replies: list[int] = []
    original = client.transport.recv

    def counted():
        replies.append(1)
        return original()

    monkeypatch.setattr(client.transport, "recv", counted)
    hosts = [(x * float(k)).sum().to("cpu", non_blocking=True) for k in range(3)]
    torch.cuda.synchronize()
    assert len(replies) == 1
    assert [host.item() for host in hosts] == [0.0, 4.0, 8.0]
    assert len(replies) == 1


def test_a_training_loop_logging_with_non_blocking_copies_matches_eager(client) -> None:
    def train(to_device):
        torch.manual_seed(0)
        data = to_device(torch.randn(128, 8))
        model = to_device(torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        logged, previous = [], None
        for step in range(20):
            if previous is not None:
                logged.append(previous.item())
            chunk = data[step : step + 16]
            loss = torch.nn.functional.mse_loss(model(chunk), chunk)
            opt.zero_grad()
            loss.backward()
            opt.step()
            previous = loss.detach().to("cpu", non_blocking=True)
        logged.append(previous.item())
        return logged

    with client.suspended():
        local = train(lambda value: value)
    before = client.stats.snapshot()
    remote = train(lambda value: value.cuda())
    delta = client.stats.snapshot() - before
    assert remote == pytest.approx(local, rel=1e-5, abs=1e-6)
    assert delta.replayed > 0


def test_fetch_awaits_a_value_while_the_event_loop_keeps_running(client, monkeypatch) -> None:
    import asyncio
    import time

    x = torch.arange(3.0).cuda()
    original = client.transport.recv

    def slow():
        time.sleep(0.2)
        return original()

    monkeypatch.setattr(client.transport, "recv", slow)

    async def main():
        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        first = letify.fetch(x * 2)
        x.add_(1.0)
        values = await asyncio.gather(first, letify.fetch(x))
        ticker.cancel()
        return ticks, [value.tolist() for value in values]

    ticks, values = asyncio.run(main())
    assert values == [[0.0, 2.0, 4.0], [1.0, 2.0, 3.0]]
    assert ticks >= 5


def test_fetch_of_a_tensor_not_on_the_runtime_resolves_without_a_round_trip(client) -> None:
    import asyncio

    before = client.stats.round_trips
    got = asyncio.run(_await(letify.fetch(torch.ones(2))))
    assert got.tolist() == [1.0, 1.0]
    assert client.stats.round_trips == before


async def _await(awaitable):
    return await awaitable


def test_a_failed_non_blocking_read_raises_where_its_tensor_is_used(client) -> None:
    good = torch.ones(3, device="cuda")
    picked = good[torch.tensor([5]).cuda()]
    host = picked.to("cpu", non_blocking=True)
    with pytest.raises(RemoteError, match=r"aten\.index"):
        host.tolist()
    assert good.cpu().tolist() == [1.0, 1.0, 1.0]


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


def _executor_with_step():
    """An in-process CPU executor holding one step of three operators.

    Position 0 makes offset 0 from an external, position 1 makes offset 1 from offset 0, and
    position 2 makes offset 2 from offset 1 and an external. Offsets 0 and 1 are last read at
    positions 1 and 2, and offset 2, which nothing in the step reads, at position 2.
    """
    from letify.remoting.device import executor as executor_module

    runner = executor_module.Executor("cpu")
    runner.define(1, "aten.mul.Tensor", ([("h", 0), ("s", 0)], {}))
    runner.define(2, "aten.add.Tensor", ([("h", 0), ("h", 1)], {}))
    runner.define_step(7, ((1, (-1,), (True,)), (1, (0,), (True,)), (2, (1, -1), (True,))))
    runner.tensors[1] = torch.ones(3)
    return runner, executor_module


def test_a_replayed_step_releases_a_temporary_after_its_last_use(client) -> None:
    runner, module = _executor_with_step()
    entry = (module.E_STEP, 7, 100, 0, 3, (1, 1), (2.0, 3.0), (), ())
    runner.run({"entries": [entry]}, [])
    assert 100 not in runner.tensors and 101 not in runner.tensors
    assert 102 not in runner.tensors
    assert runner.failure is None


def test_a_kept_offset_outlives_its_last_use_in_the_step(client) -> None:
    runner, module = _executor_with_step()
    entry = (module.E_STEP, 7, 100, 0, 3, (1, 1), (2.0, 3.0), (), (2,))
    runner.run({"entries": [entry]}, [])
    assert set(runner.tensors) == {1, 102}
    assert runner.tensors[102].tolist() == [7.0, 7.0, 7.0]


def test_a_split_repetition_releases_an_offset_in_the_part_holding_its_last_use(client) -> None:
    runner, module = _executor_with_step()
    head = (module.E_STEP, 7, 100, 0, 2, (1,), (2.0, 3.0), (), ())
    runner.run({"entries": [head]}, [])
    assert 100 not in runner.tensors and 101 in runner.tensors
    tail = (module.E_STEP, 7, 100, 2, 3, (1,), (), (), (2,))
    runner.run({"entries": [tail]}, [])
    assert set(runner.tensors) == {1, 102}


def test_replayed_steps_release_temporaries_before_the_step_ends(client, monkeypatch) -> None:
    import pickle

    from letify.remoting.device import executor as executor_module

    sent: list = []
    original = client.transport.send

    def spy(head, buffers):
        sent.append(pickle.loads(head))
        return original(head, buffers)

    monkeypatch.setattr(client.transport, "send", spy)
    with client.suspended():
        local = _train(lambda value: value, steps=12)
    remote = _train(lambda value: value.cuda(), steps=12)
    _assert_same(remote, local)
    steps = [
        entry
        for message in sent
        for entry in message.get("entries", ())
        if entry[0] == executor_module.E_STEP
    ]
    assert steps
    released_early = 0
    for entry in steps:
        step = client._steps[entry[1]]
        scheduled = sum(len(step.frees[index]) for index in range(entry[3], entry[4]))
        released_early += scheduled - len(entry[8])
    assert released_early > 0


def test_a_release_is_applied_after_the_last_entry_of_its_batch_that_reads_it(client) -> None:
    runner, module = _executor_with_step()
    runner.tensors[2] = torch.ones(3)
    live = (module.E_REQUEST, "letify.live", (), "value")
    use = (module.E_OP, 1, (1,), (2.0,), (), (5,), None)
    reply = runner.run({"entries": [live, use, live], "release": [1, 2], "reply": True}, [])
    import pickle

    results = pickle.loads(reply[0])["results"]
    # Handle 2 is read by no entry, so it goes first; handle 1 goes after the entry reading it.
    assert results == [1, 1]
    assert set(runner.tensors) == {5}


def test_a_handle_created_and_released_in_one_batch_is_not_kept(client) -> None:
    runner, module = _executor_with_step()
    make = (module.E_OP, 1, (1,), (2.0,), (), (5,), None)
    runner.run({"entries": [make], "release": [5]}, [])
    assert set(runner.tensors) == {1}


def test_a_handle_read_by_a_later_entry_is_not_released_early(client, monkeypatch) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    monkeypatch.setattr(client_module, "IDLE_S", 60.0)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    last = x * 1.0
    total = torch.zeros(4, device="cuda")
    for _ in range(16):
        carried = last * 1.0
        last = None
        gc.collect()
        a = carried * 2.0
        b = a + 1.0
        c = b - 1.0
        d = c * 0.5
        e = d + 0.0
        f = e * 1.0
        total = total + f
        last = f + 0.0
    assert client.stats.replayed > 0
    assert total.cpu().tolist() == [16.0] * 4
    assert last.cpu().tolist() == [1.0] * 4


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


def test_an_external_collected_mid_repetition_is_released_after_its_step_entry(
    client, monkeypatch
) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    monkeypatch.setattr(client_module, "IDLE_S", 60.0)
    inputs = [torch.full((4,), float(i)).cuda() for i in range(12)]
    client.synchronize()
    total = torch.zeros(4, device="cuda")
    for i in range(12):
        x, inputs[i] = inputs[i], None
        a = x * 2.0
        b = a + 1.0
        del x
        gc.collect()
        # What the idle sender thread or a collection on another thread does mid-step.
        client._flush()
        c = b * 3.0
        d = c - 1.0
        e = d + 0.5
        f = e * 1.0
        g = f + 0.0
        total = total + g
    assert client.stats.replayed > 0
    assert total.cpu().tolist() == [sum(6.0 * i + 2.5 for i in range(12))] * 4


def test_a_tensor_used_by_an_entry_a_read_leaves_queued_is_not_released_early(
    client, monkeypatch
) -> None:
    from letify.remoting.device import client as client_module

    monkeypatch.setattr(client_module, "LINGER_S", 60.0)
    monkeypatch.setattr(client_module, "IDLE_S", 60.0)
    x = torch.ones(4, device="cuda")
    client.synchronize()
    total = (x * 2).sum()
    later = x + 1
    del x
    gc.collect()
    assert total.item() == 8.0
    assert client.queued > 0
    assert later.cpu().tolist() == [2.0, 2.0, 2.0, 2.0]


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


def _hold_writes(client, monkeypatch):
    import threading

    gate = threading.Event()
    original = client.transport.send

    def held(head, buffers):
        gate.wait(10)
        original(head, buffers)

    monkeypatch.setattr(client.transport, "send", held)
    return gate


def test_an_upload_returns_before_its_bytes_are_written(client, monkeypatch) -> None:
    import time

    gate = _hold_writes(client, monkeypatch)
    data = torch.randn(1 << 20)
    started = time.monotonic()
    uploaded = data.cuda(non_blocking=True)
    doubled = uploaded * 2
    elapsed = time.monotonic() - started
    gate.set()
    assert elapsed < 2.0
    assert torch.equal(doubled.cpu(), data * 2)


def test_a_cpu_tensor_changed_after_a_blocking_upload_does_not_change_the_upload(
    client, monkeypatch
) -> None:
    gate = _hold_writes(client, monkeypatch)
    data = torch.zeros(1 << 16)
    uploaded = data.cuda()
    data.fill_(7.0)
    gate.set()
    assert uploaded.sum().item() == 0.0


def test_a_cpu_tensor_changed_after_a_pageable_non_blocking_upload_does_not_change_it(
    client, monkeypatch
) -> None:
    gate = _hold_writes(client, monkeypatch)
    data = torch.zeros(1 << 16)
    uploaded = data.to("cuda", non_blocking=True)
    data.fill_(7.0)
    gate.set()
    assert uploaded.sum().item() == 0.0


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


def test_an_async_host_local_function_awaits_its_reads(let, cpu) -> None:
    import asyncio

    @let.function(device=cpu, host="local")
    async def step() -> list[float]:
        x = torch.ones(3, device="cuda")
        return (await letify.fetch(x * 2)).tolist()

    assert asyncio.run(step()) == [2.0, 2.0, 2.0]


def test_host_local_is_refused_without_torch(let, cpu, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(letify.UnsupportedMode, match="PyTorch"):
        cpu.provider.check_mode(cpu._placed("local"))
