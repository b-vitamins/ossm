from __future__ import annotations

import pytest
import torch
from torch.utils import benchmark

from ossm.models._linoss_scan import try_run_scan as _linoss_try_run_scan
from ossm.models._rnn_scan import try_run_linear_rnn_scan
from ossm.models.rnn import _linear_rnn_reference


_CPU_SPEEDUP_THRESHOLD = 20.0
_CUDA_SPEEDUP_THRESHOLD = 4.0


def _time_function(fn, *, min_run_time: float, synchronize: bool = False) -> float:
    """Measure the average runtime of ``fn`` using ``torch.utils.benchmark``."""

    if synchronize:
        if not torch.cuda.is_available():  # pragma: no cover - defensive guard
            raise RuntimeError("CUDA synchronization requested without CUDA availability.")

        def wrapped() -> object:
            torch.cuda.synchronize()
            result = fn()
            torch.cuda.synchronize()
            return result

    else:
        def wrapped() -> object:
            return fn()

    timer = benchmark.Timer(stmt="wrapped()", globals={"wrapped": wrapped})
    return timer.blocked_autorange(min_run_time=min_run_time).mean


def _linoss_naive_scan(a_matrix: torch.Tensor, b_seq: torch.Tensor) -> torch.Tensor:
    """Reference LinOSS scan executed on the host for comparison."""

    length = b_seq.size(0)
    batch = b_seq.size(1)
    ssm = b_seq.size(2)
    state = b_seq.new_zeros((batch, ssm, 2))
    outputs = []
    for step in range(length):
        state = torch.einsum("sij,bsj->bsi", a_matrix, state) + b_seq[step]
        outputs.append(state)
    return torch.stack(outputs, dim=0)


@pytest.mark.performance
@pytest.mark.parametrize("length,batch,ssm", [(512, 8, 64)])
def test_linoss_cpu_kernel_is_faster(length: int, batch: int, ssm: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    a_matrix = torch.randn(ssm, 2, 2, dtype=torch.complex64, device=device)
    b_seq = torch.randn(length, batch, ssm, 2, dtype=torch.complex64, device=device)

    m11 = a_matrix[:, 0, 0].contiguous()
    m12 = a_matrix[:, 0, 1].contiguous()
    m21 = a_matrix[:, 1, 0].contiguous()
    m22 = a_matrix[:, 1, 1].contiguous()

    ext_out = _linoss_try_run_scan(m11, m12, m21, m22, b_seq)
    if ext_out is None:
        pytest.skip("LinOSS C++ kernel is unavailable")

    def run_extension() -> object:
        return _linoss_try_run_scan(m11, m12, m21, m22, b_seq)

    def run_reference() -> object:
        return _linoss_naive_scan(a_matrix, b_seq)

    # Warm up both paths to avoid measuring first-call overheads.
    run_extension()
    run_reference()

    optimized_mean = _time_function(run_extension, min_run_time=0.4)
    baseline_mean = _time_function(run_reference, min_run_time=0.4)

    speedup = baseline_mean / optimized_mean
    assert speedup >= _CPU_SPEEDUP_THRESHOLD, (
        f"LinOSS CPU kernel regressed: baseline {baseline_mean:.6f}s vs optimized {optimized_mean:.6f}s "
        f"(speedup {speedup:.2f}x)"
    )


@pytest.mark.performance
@pytest.mark.cuda
@pytest.mark.parametrize("length,batch,input_dim,hidden_dim", [(1024, 8, 128, 128)])
def test_linear_rnn_cuda_kernel_is_faster(
    length: int, batch: int, input_dim: int, hidden_dim: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    device = torch.device("cuda")

    weight_hh = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32, device=device)
    weight_xh = torch.randn(hidden_dim, input_dim, dtype=torch.float32, device=device)
    bias = torch.randn(hidden_dim, dtype=torch.float32, device=device)
    inputs = torch.randn(length, batch, input_dim, dtype=torch.float32, device=device)
    initial_state = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)

    ext_out = try_run_linear_rnn_scan(weight_hh, weight_xh, bias, inputs, initial_state)
    if ext_out is None:
        pytest.skip("Linear RNN CUDA kernel is unavailable")

    def run_extension() -> object:
        return try_run_linear_rnn_scan(weight_hh, weight_xh, bias, inputs, initial_state)

    def run_reference() -> object:
        return _linear_rnn_reference(weight_hh, weight_xh, bias, inputs, initial_state)

    # Warm up GPU execution paths.
    run_extension()
    run_reference()
    torch.cuda.synchronize()

    optimized_mean = _time_function(run_extension, min_run_time=0.6, synchronize=True)
    baseline_mean = _time_function(run_reference, min_run_time=0.6, synchronize=True)

    speedup = baseline_mean / optimized_mean
    assert speedup >= _CUDA_SPEEDUP_THRESHOLD, (
        f"Linear RNN CUDA kernel regressed: baseline {baseline_mean:.6f}s vs optimized {optimized_mean:.6f}s "
        f"(speedup {speedup:.2f}x)"
    )
