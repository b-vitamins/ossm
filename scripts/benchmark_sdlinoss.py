"""Benchmark selective D-LinOSS kernels against the pure PyTorch fallback."""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from typing import Callable, Tuple

import torch

from ossm.models._sdlinoss_scan import extension_error, has_kernels, run_sdlinoss

_VARIANTS = ("imex1", "imex2", "im", "ex")
_COMPLEX_DTYPES = {
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def _sample_inputs(
    *,
    length: int,
    batch: int,
    ssm: int,
    complex_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    A = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
    G = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
    step = torch.rand(length, batch, ssm, dtype=real_dtype, device=device) * 0.85 + 0.05
    real = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
    imag = torch.randn(length, batch, ssm, dtype=real_dtype, device=device)
    bu = torch.complex(real, imag).to(dtype=complex_dtype)
    return A, G, step, bu


@contextmanager
def _kernel_disabled() -> None:
    token = "OSSM_SDLINOSS_DISABLE_KERNEL"
    previous = os.environ.get(token)
    os.environ[token] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(token, None)
        else:
            os.environ[token] = previous


def _time_function(fn: Callable[[], torch.Tensor], *, repeats: int, synchronize: bool) -> Tuple[float, torch.Tensor]:
    for _ in range(5):
        result = fn()
        if synchronize:
            torch.cuda.synchronize()
    start = time.perf_counter()
    last: torch.Tensor | None = None
    for _ in range(repeats):
        last = fn()
        if synchronize:
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if last is None:
        raise RuntimeError("Benchmark target did not return a tensor")
    return elapsed / repeats, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=_VARIANTS, default="imex1")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ssm", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(_COMPLEX_DTYPES.keys()), default="complex64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0, help="Override torch.set_num_threads for CPU runs")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    complex_dtype = _COMPLEX_DTYPES[args.dtype]

    A, G, step, bu = _sample_inputs(
        length=args.length,
        batch=args.batch,
        ssm=args.ssm,
        complex_dtype=complex_dtype,
        device=device,
    )

    if not has_kernels(args.variant):
        error = extension_error()
        details = f"\nLast extension error: {error}" if error is not None else ""
        raise RuntimeError(
            "Selective D-LinOSS kernels are unavailable. Build the OSSM extension (e.g. `pip install -e .`) before running the benchmark."  # noqa: E501
            + details
        )

    def run_kernel() -> torch.Tensor:
        return run_sdlinoss(args.variant, A, G, step, bu)

    def run_fallback() -> torch.Tensor:
        with _kernel_disabled():
            return run_sdlinoss(args.variant, A, G, step, bu)

    synchronize = device.type == "cuda"
    kernel_time, kernel_out = _time_function(run_kernel, repeats=args.repeats, synchronize=synchronize)
    fallback_time, fallback_out = _time_function(run_fallback, repeats=args.repeats, synchronize=synchronize)

    max_error = (kernel_out - fallback_out).abs().max().item()
    speedup = fallback_time / kernel_time if kernel_time > 0 else float("inf")

    print("Selective D-LinOSS Benchmark")
    print(f"Variant: {args.variant}")
    print(f"Length={args.length}, batch={args.batch}, state={args.ssm}")
    print(f"Kernel time:    {kernel_time * 1e3:.3f} ms")
    print(f"Fallback time:  {fallback_time * 1e3:.3f} ms")
    print(f"Speedup:        {speedup:.2f}x")
    print(f"Max |kernel - fallback|: {max_error:.3e}")


if __name__ == "__main__":
    main()
