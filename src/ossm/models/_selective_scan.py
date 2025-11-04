"""Helpers for the fused selective scan kernels (CPU and CUDA)."""

from __future__ import annotations

import os
from typing import Optional, Protocol, Tuple, cast

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

__all__ = ["try_selective_scan", "has_kernels", "extension_error"]

Gradients = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]


class _SelectiveScanKernels(Protocol):
    """Protocol describing the fused selective scan kernel bindings."""

    def selective_scan(
        self,
        inputs: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        gate: Optional[Tensor],
    ) -> Tensor:
        ...

    def selective_scan_backward(
        self,
        grad_output: Tensor,
        inputs: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        gate: Optional[Tensor],
    ) -> Gradients:
        ...

    def selective_scan_cuda(
        self,
        inputs: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        gate: Optional[Tensor],
        chunk: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ...

    def selective_scan_cuda_backward(
        self,
        grad_output: Tensor,
        inputs: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        gate: Optional[Tensor],
        chunk_states: Tensor,
        raw_outputs: Tensor,
        chunk: int,
    ) -> Gradients:
        ...


_kernels: Optional[_SelectiveScanKernels]


try:  # pragma: no cover - import surface
    from ossm import _kernels as _kernels_module  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-time failure surface
    _kernels = None
    _EXTENSION_ERROR: Optional[Exception] = exc
else:
    _EXTENSION_ERROR = None
    _kernels = cast(_SelectiveScanKernels, _kernels_module)


def _trace(message: str) -> None:
    if os.environ.get("OSSM_SELECTIVE_SCAN_TRACE"):
        print(message, flush=True)


def _supports(attr: str) -> bool:
    return _kernels is not None and hasattr(_kernels, attr)


if _kernels is not None and os.environ.get("OSSM_SELECTIVE_SCAN_TRACE"):
    available = []
    if _supports("selective_scan"):
        available.append("cpu")
    if _supports("selective_scan_cuda"):
        available.append("cuda")
    if available:
        _trace(f"[OSSM] Selective scan kernels available for: {', '.join(available)}")


# Default chunk length for CUDA selective scan.
# Larger chunks reduce saved chunk-state memory (∝ ceil(L/chunk)) but increase
# backward shared memory usage ((3+chunk)*state*4 bytes). 64 is a good default
# for typical state sizes (≤128) and keeps memory low at larger batches.
_DEFAULT_CHUNK = 64
try:
    _DEFAULT_CHUNK = int(os.environ.get("OSSM_SELECTIVE_SCAN_CHUNK", _DEFAULT_CHUNK))
except ValueError:  # pragma: no cover - defensive parsing guard
    _DEFAULT_CHUNK = 64
_DEFAULT_CHUNK = max(1, _DEFAULT_CHUNK)


def has_kernels() -> bool:
    """Return ``True`` when the fused selective scan kernels are importable."""

    return _supports("selective_scan") and _supports("selective_scan_backward")


def extension_error() -> Optional[Exception]:
    """Return the cached import or runtime error for the selective scan kernel."""

    return _EXTENSION_ERROR


def try_selective_scan(
    inputs: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    gate: Optional[Tensor],
) -> Optional[Tensor]:
    """Attempt to execute the fused selective scan kernel."""

    if inputs.numel() == 0:
        return inputs
    if not has_kernels():
        return None
    if inputs.dtype != torch.float32:
        return None

    if gate is not None and gate.dtype != torch.float32:
        return None

    device_type = inputs.device.type
    if device_type == "cpu":
        if not _supports("selective_scan_backward"):
            return None
    elif device_type == "cuda":
        required = ("selective_scan_cuda", "selective_scan_cuda_backward")
        if not all(_supports(attr) for attr in required):
            return None
    else:
        return None

    requires_grad = any(
        tensor is not None and tensor.requires_grad
        for tensor in (inputs, dt, A, B, C, gate)
    )

    kernels = _kernels
    if kernels is None:
        return None

    # Choose a chunk length that fits device shared memory for the given state size.
    chunk = _DEFAULT_CHUNK
    if device_type == "cuda":
        try:
            props = torch.cuda.get_device_properties(inputs.device)
            # Shared memory per block in bytes; keep a small safety margin.
            max_smem = int(props.shared_memory_per_block)
        except Exception:  # pragma: no cover - best-effort device query
            max_smem = 48 * 1024
        state_dim = int(A.size(1)) if A.dim() == 2 else int(A.shape[-1])
        # (3 + chunk) * state_dim * 4 <= max_smem  =>  chunk <= max_smem/(4*state_dim) - 3
        if state_dim > 0:
            safe_max = max(1, int(max_smem // (4 * state_dim) - 3))
            if safe_max < 1:
                safe_max = 1
            chunk = max(1, min(chunk, safe_max))

        if not requires_grad:
            outputs, _, _ = kernels.selective_scan_cuda(inputs, dt, A, B, C, gate, chunk)
            return outputs
    elif not requires_grad:
        return kernels.selective_scan(inputs, dt, A, B, C, gate)

    try:
        return _SelectiveScanFn.apply(inputs, dt, A, B, C, gate, chunk)
    except RuntimeError as exc:  # pragma: no cover - runtime mismatch
        global _EXTENSION_ERROR
        _EXTENSION_ERROR = exc
        _trace(f"[OSSM] Selective scan kernel call failed: {exc}")
        return None


class _SelectiveScanFn(torch.autograd.Function):
    """Autograd wrapper dispatching to the fused selective scan kernels."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        inputs: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        gate: Optional[Tensor],
        chunk_length: int,
    ) -> Tensor:
        if inputs.dtype != torch.float32:
            raise RuntimeError("Selective scan kernels require float32 inputs")
        if dt.dtype != torch.float32 or A.dtype != torch.float32:
            raise RuntimeError("Selective scan kernels require float32 parameters")
        if gate is not None and gate.dtype != torch.float32:
            raise RuntimeError("Selective scan kernels require float32 gate")

        chunk = int(chunk_length)
        if chunk <= 0:
            raise RuntimeError("chunk_length must be positive")

        gate_tensor = gate.contiguous() if gate is not None else None
        gate_saved = gate_tensor if gate_tensor is not None else inputs.new_empty(0)

        inputs_c = inputs.contiguous()
        dt_c = dt.contiguous()
        A_c = A.contiguous()
        B_c = B.contiguous()
        C_c = C.contiguous()

        device_type = inputs.device.type
        ctx_attrs = cast("_SelectiveScanContext", ctx)

        ctx_attrs.use_cuda = device_type == "cuda"
        ctx_attrs.has_gate = gate_tensor is not None
        ctx_attrs.chunk = chunk

        kernels = _require_kernels()

        if ctx_attrs.use_cuda:
            outputs, chunk_states, raw_outputs = kernels.selective_scan_cuda(
                inputs_c, dt_c, A_c, B_c, C_c, gate_tensor, chunk
            )
            ctx.save_for_backward(
                inputs_c, dt_c, A_c, B_c, C_c, gate_saved, chunk_states, raw_outputs
            )
            return outputs

        outputs = kernels.selective_scan(inputs_c, dt_c, A_c, B_c, C_c, gate_tensor)
        ctx.save_for_backward(inputs_c, dt_c, A_c, B_c, C_c, gate_saved)
        return outputs

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], None]:
        ctx_attrs = cast("_SelectiveScanContext", ctx)
        kernels = _require_kernels()

        if ctx_attrs.use_cuda:
            inputs, dt, A, B, C, gate_saved, chunk_states, raw_outputs = ctx_attrs.saved_tensors
            gate_opt = gate_saved if ctx_attrs.has_gate else None
            grad_inputs, grad_dt, grad_A, grad_B, grad_C, grad_gate = (
                kernels.selective_scan_cuda_backward(
                    grad_output.contiguous(),
                    inputs,
                    dt,
                    A,
                    B,
                    C,
                    gate_opt,
                    chunk_states,
                    raw_outputs,
                    ctx_attrs.chunk,
                )
            )
        else:
            inputs, dt, A, B, C, gate_saved = ctx_attrs.saved_tensors
            gate_opt = gate_saved if ctx_attrs.has_gate else None
            grad_inputs, grad_dt, grad_A, grad_B, grad_C, grad_gate = (
                kernels.selective_scan_backward(
                    grad_output.contiguous(),
                    inputs,
                    dt,
                    A,
                    B,
                    C,
                    gate_opt,
                )
            )

        if not ctx_attrs.has_gate:
            grad_gate = None

        return grad_inputs, grad_dt, grad_A, grad_B, grad_C, grad_gate, None


class _SelectiveScanContext(Protocol):
    """Typed view over the ``torch.autograd.Function`` context."""

    saved_tensors: tuple[Tensor, ...]
    use_cuda: bool
    has_gate: bool
    chunk: int


def _require_kernels() -> _SelectiveScanKernels:
    kernels = _kernels
    if kernels is None:  # pragma: no cover - defensive runtime guard
        raise RuntimeError("Selective scan kernels are not available")
    return kernels
