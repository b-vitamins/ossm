from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, cast

import pytest
import torch
from torch import Tensor
from torch.utils import benchmark

from ossm.models._linoss_scan import try_run_scan as _linoss_try_run_scan
from ossm.models._lru_scan import try_run_lru_scan
from ossm.models._rnn_scan import try_run_linear_rnn_scan
from ossm.models._s5_scan import try_run_s5_scan


_CPU_SPEEDUPS = {
    "linoss": 12.0,
    # The complex-valued scans have slightly lower gains on CI hardware; these
    # thresholds reflect measured speedups with PyTorch 2.8 CPU wheels.
    "lru": 7.0,
    "s5": 5.5,
    # Linear RNN CPU improvements are more modest but still significant.
    "rnn": 1.6,
}

_CUDA_SPEEDUPS = {
    "linoss": 4.0,
    "lru": 3.0,
    "s5": 3.0,
    "rnn": 3.5,
}

_SPEEDUP_TOLERANCE = 0.15


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    device_type: str
    setup: Callable[[torch.device], Tuple[Optional[Callable[[], Tensor]], Optional[Callable[[], Tensor]], Optional[str]]]
    min_runtime: float
    threshold: float
    synchronize: bool = False

    @property
    def requires_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def id(self) -> str:
        return f"{self.name}-{self.device_type}"


def _time_function(fn: Callable[[], Tensor], *, min_run_time: float, synchronize: bool = False) -> float:
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


def _complex_scan_naive(lambda_bar: Tensor, b_seq: Tensor) -> Tensor:
    length, batch, state = b_seq.shape
    state_vec = b_seq.new_zeros(batch, state)
    outputs = []
    for step in range(length):
        state_vec = lambda_bar * state_vec + b_seq[step]
        outputs.append(state_vec)
    return torch.stack(outputs, dim=0)


def _linoss_naive_scan(a_matrix: Tensor, b_seq: Tensor) -> Tensor:
    length = b_seq.size(0)
    batch = b_seq.size(1)
    ssm = b_seq.size(2)
    state = b_seq.new_zeros((batch, ssm, 2))
    outputs = []
    for step in range(length):
        state = torch.einsum("sij,bsj->bsi", a_matrix, state) + b_seq[step]
        outputs.append(state)
    return torch.stack(outputs, dim=0)


def _linear_rnn_naive(
    weight_hh: Tensor,
    weight_xh: Tensor,
    bias: Tensor,
    inputs: Tensor,
    initial_state: Tensor,
) -> Tensor:
    length, batch, _ = inputs.shape
    state = initial_state
    outputs = []
    weight_hh_t = weight_hh.transpose(0, 1)
    weight_xh_t = weight_xh.transpose(0, 1)
    for idx in range(length):
        base = inputs[idx].matmul(weight_xh_t) + bias
        state = state.matmul(weight_hh_t) + base
        outputs.append(state)
    return torch.stack(outputs, dim=0)


def _setup_linoss_case(length: int, batch: int, ssm: int) -> Callable[[torch.device], Tuple[Optional[Callable[[], Tensor]], Optional[Callable[[], Tensor]], Optional[str]]]:
    def _inner(device: torch.device) -> Tuple[Optional[Callable[[], Tensor]], Optional[Callable[[], Tensor]], Optional[str]]:
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)

        a_matrix = torch.randn(ssm, 2, 2, dtype=torch.complex64, device=device)
        b_seq = torch.randn(length, batch, ssm, 2, dtype=torch.complex64, device=device)

        m11 = a_matrix[:, 0, 0].contiguous()
        m12 = a_matrix[:, 0, 1].contiguous()
        m21 = a_matrix[:, 1, 0].contiguous()
        m22 = a_matrix[:, 1, 1].contiguous()

        ext_out = _linoss_try_run_scan(m11, m12, m21, m22, b_seq)
        if ext_out is None:
            return None, None, "LinOSS kernel is unavailable"

        def run_extension() -> Tensor:
            return cast(Tensor, _linoss_try_run_scan(m11, m12, m21, m22, b_seq))

        def run_reference() -> Tensor:
            return _linoss_naive_scan(a_matrix, b_seq)

        return run_extension, run_reference, None

    return _inner


def _setup_complex_case(
    name: str,
    length: int,
    batch: int,
    state: int,
    *,
    run_fn: Callable[[Tensor, Tensor], Optional[Tensor]],
) -> Callable[[torch.device], Tuple[Optional[Callable[[], Tensor]], Optional[Callable[[], Tensor]], Optional[str]]]:
    def _inner(device: torch.device) -> Tuple[Optional[Callable[[], Tensor]], Optional[Callable[[], Tensor]], Optional[str]]:
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)

        lambda_bar = torch.randn(state, dtype=torch.complex64, device=device)
        b_seq = torch.randn(length, batch, state, dtype=torch.complex64, device=device)

        ext_out = run_fn(lambda_bar, b_seq)
        if ext_out is None:
            return None, None, f"{name} kernel is unavailable"

        def run_extension() -> Tensor:
            return cast(Tensor, run_fn(lambda_bar, b_seq))

        def run_reference() -> Tensor:
            return _complex_scan_naive(lambda_bar, b_seq)

        return run_extension, run_reference, None

    return _inner


def _setup_rnn_case(
    length: int,
    batch: int,
    input_dim: int,
    hidden_dim: int,
) -> Callable[[torch.device], Tuple[Optional[Callable[[], Tensor]], Optional[Callable[[], Tensor]], Optional[str]]]:
    def _inner(device: torch.device) -> Tuple[Optional[Callable[[], Tensor]], Optional[Callable[[], Tensor]], Optional[str]]:
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)

        weight_hh = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32, device=device)
        weight_xh = torch.randn(hidden_dim, input_dim, dtype=torch.float32, device=device)
        bias = torch.randn(hidden_dim, dtype=torch.float32, device=device)
        inputs = torch.randn(length, batch, input_dim, dtype=torch.float32, device=device)
        initial_state = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)

        ext_out = try_run_linear_rnn_scan(weight_hh, weight_xh, bias, inputs, initial_state)
        if ext_out is None:
            return None, None, "Linear RNN kernel is unavailable"

        def run_extension() -> Tensor:
            return cast(
                Tensor,
                try_run_linear_rnn_scan(weight_hh, weight_xh, bias, inputs, initial_state),
            )

        def run_reference() -> Tensor:
            return _linear_rnn_naive(weight_hh, weight_xh, bias, inputs, initial_state)

        return run_extension, run_reference, None

    return _inner


_CPU_CASES = [
    BenchmarkCase(
        name="linoss",
        device_type="cpu",
        setup=_setup_linoss_case(length=768, batch=8, ssm=96),
        min_runtime=0.35,
        threshold=_CPU_SPEEDUPS["linoss"],
    ),
    BenchmarkCase(
        name="lru",
        device_type="cpu",
        setup=_setup_complex_case(
            "LRU",
            length=4096,
            batch=8,
            state=128,
            run_fn=lambda lambda_bar, b_seq: try_run_lru_scan(lambda_bar, b_seq),
        ),
        min_runtime=0.35,
        threshold=_CPU_SPEEDUPS["lru"],
    ),
    BenchmarkCase(
        name="s5",
        device_type="cpu",
        setup=_setup_complex_case(
            "S5",
            length=4096,
            batch=8,
            state=128,
            run_fn=lambda lambda_bar, b_seq: try_run_s5_scan(lambda_bar, b_seq),
        ),
        min_runtime=0.35,
        threshold=_CPU_SPEEDUPS["s5"],
    ),
    BenchmarkCase(
        name="rnn",
        device_type="cpu",
        setup=_setup_rnn_case(length=1024, batch=8, input_dim=192, hidden_dim=256),
        min_runtime=0.35,
        threshold=_CPU_SPEEDUPS["rnn"],
    ),
]


_CUDA_CASES = [
    BenchmarkCase(
        name="linoss",
        device_type="cuda",
        setup=_setup_linoss_case(length=1024, batch=16, ssm=128),
        min_runtime=0.6,
        threshold=_CUDA_SPEEDUPS["linoss"],
        synchronize=True,
    ),
    BenchmarkCase(
        name="lru",
        device_type="cuda",
        setup=_setup_complex_case(
            "LRU",
            length=4096,
            batch=32,
            state=128,
            run_fn=lambda lambda_bar, b_seq: try_run_lru_scan(lambda_bar, b_seq),
        ),
        min_runtime=0.6,
        threshold=_CUDA_SPEEDUPS["lru"],
        synchronize=True,
    ),
    BenchmarkCase(
        name="s5",
        device_type="cuda",
        setup=_setup_complex_case(
            "S5",
            length=4096,
            batch=32,
            state=128,
            run_fn=lambda lambda_bar, b_seq: try_run_s5_scan(lambda_bar, b_seq),
        ),
        min_runtime=0.6,
        threshold=_CUDA_SPEEDUPS["s5"],
        synchronize=True,
    ),
    BenchmarkCase(
        name="rnn",
        device_type="cuda",
        setup=_setup_rnn_case(length=2048, batch=24, input_dim=256, hidden_dim=256),
        min_runtime=0.6,
        threshold=_CUDA_SPEEDUPS["rnn"],
        synchronize=True,
    ),
]


@pytest.mark.performance
@pytest.mark.parametrize("case", _CPU_CASES + _CUDA_CASES, ids=lambda case: case.id)
def test_kernels_avoid_performance_regressions(case: BenchmarkCase) -> None:
    if case.requires_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(case.device_type)
    optimized, reference, skip_reason = case.setup(device)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    assert optimized is not None and reference is not None

    optimized()
    reference()
    if case.synchronize:
        torch.cuda.synchronize()

    optimized_mean = _time_function(optimized, min_run_time=case.min_runtime, synchronize=case.synchronize)
    baseline_mean = _time_function(reference, min_run_time=case.min_runtime, synchronize=case.synchronize)

    speedup = baseline_mean / optimized_mean
    min_speedup = case.threshold * (1 - _SPEEDUP_TOLERANCE)
    assert speedup >= min_speedup, (
        f"{case.name} {case.device_type} kernel regressed: baseline {baseline_mean:.6f}s vs optimized {optimized_mean:.6f}s "
        f"(speedup {speedup:.2f}x, expected >= {case.threshold:.1f}x with {_SPEEDUP_TOLERANCE:.0%} tolerance => {min_speedup:.2f}x)"
    )
