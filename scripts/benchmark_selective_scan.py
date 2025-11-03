"""Benchmark the fused CPU selective scan kernel against the reference implementation."""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch
import torch.nn.functional as F

from ossm.models._selective_scan import (
    extension_error,
    has_kernels,
    try_selective_scan,
)
from ossm.models.mambarec import _selective_scan_mamba


def _make_inputs(
    batch: int,
    channels: int,
    length: int,
    state: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch, channels, length, dtype=torch.float32, device=device)
    dt = torch.rand(batch, channels, length, dtype=torch.float32, device=device) * 0.05
    A = -torch.exp(torch.randn(channels, state, dtype=torch.float32, device=device))
    B = torch.randn(batch, length, state, dtype=torch.float32, device=device)
    C = torch.randn(batch, length, state, dtype=torch.float32, device=device)
    gate = torch.randn(batch, channels, length, dtype=torch.float32, device=device)
    return x, dt, A, B, C, gate


def _time(fn, *, repeats: int, synchronize: bool = False) -> Tuple[float, torch.Tensor]:
    for _ in range(5):
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        fn()
    start = time.perf_counter()
    result: torch.Tensor | None = None
    for _ in range(repeats):
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        result = fn()
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if result is None:
        raise RuntimeError("Benchmark function returned None")
    return elapsed / repeats, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--state", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0, help="Override torch.set_num_threads")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported device type: {device.type}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    tensors = _make_inputs(args.batch, args.channels, args.length, args.state, device=device)
    x, dt, A, B, C, gate = tensors

    if not has_kernels():
        error = extension_error()
        details = f"\nLast extension error: {error}" if error is not None else ""
        raise RuntimeError(
            "Selective scan kernel is unavailable. Build the OSSM extension (e.g. `pip install -e .`) before benchmarking." + details
        )

    def run_kernel() -> torch.Tensor:
        result = try_selective_scan(x, dt, A, B, C, gate)
        if result is None:
            raise RuntimeError("Selective scan kernel unexpectedly returned None")
        return result

    def run_reference() -> torch.Tensor:
        baseline = _selective_scan_mamba(inputs=x, dt=dt, A=A, B_t=B, C_t=C)
        return baseline * F.silu(gate)

    synchronize = device.type == "cuda"
    kernel_time, kernel_out = _time(run_kernel, repeats=args.repeats, synchronize=synchronize)
    ref_time, ref_out = _time(run_reference, repeats=args.repeats, synchronize=synchronize)

    max_error = (kernel_out - ref_out).abs().max().item()
    speedup = ref_time / kernel_time if kernel_time > 0 else float("inf")

    print("Selective Scan Benchmark")
    print(f"Batch={args.batch}, channels={args.channels}, length={args.length}, state={args.state}")
    print(f"Kernel time:    {kernel_time * 1e3:.3f} ms")
    print(f"Reference time: {ref_time * 1e3:.3f} ms")
    print(f"Speedup:        {speedup:.2f}x")
    print(f"Max |kernel - reference|: {max_error:.3e}")


if __name__ == "__main__":
    main()
