"""Runtime helper bridging the selective D-LinOSS kernels."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Protocol, Tuple, cast

import torch
from torch import Tensor, autocast

__all__ = [
    "extension_error",
    "has_kernels",
    "run_sdlinoss",
]

_SUPPORTED_VARIANTS: Tuple[str, ...] = ("imex1", "imex2", "im", "ex")


def _maybe_dynamo_module() -> Optional[Any]:
    try:
        import torch._dynamo as _dynamo  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive for older torch
        return None
    return _dynamo


def _graph_break_if_compiling() -> None:
    _dynamo = _maybe_dynamo_module()
    if _dynamo is None:
        return
    try:
        if _dynamo.is_compiling():  # pragma: no cover - runtime guard
            _dynamo.graph_break()
    except RuntimeError:
        # If Dynamo is not initialized for this thread, treat as not compiling.
        pass


class _SdlinossKernels(Protocol):
    def sdlinoss_imex1_forward(self, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor: ...

    def sdlinoss_imex1_backward(
        self,
        A: Tensor,
        G: Tensor,
        step: Tensor,
        bu: Tensor,
        states: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def sdlinoss_imex2_forward(self, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor: ...

    def sdlinoss_imex2_backward(
        self,
        A: Tensor,
        G: Tensor,
        step: Tensor,
        bu: Tensor,
        states: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def sdlinoss_im_forward(self, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor: ...

    def sdlinoss_im_backward(
        self,
        A: Tensor,
        G: Tensor,
        step: Tensor,
        bu: Tensor,
        states: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def sdlinoss_ex_forward(self, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor: ...

    def sdlinoss_ex_backward(
        self,
        A: Tensor,
        G: Tensor,
        step: Tensor,
        bu: Tensor,
        states: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


try:
    from ossm import _kernels as _kernels  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-time failure surface
    _kernels: Optional[_SdlinossKernels] = None
    _EXTENSION_ERROR: Optional[Exception] = exc
else:
    _kernels = cast("_SdlinossKernels", _kernels)
    _EXTENSION_ERROR = None

def _infer_real_dtype_from_complex(z: Tensor) -> torch.dtype:
    if z.dtype == torch.complex64:
        return torch.float32
    if z.dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Expected complex64/complex128, got {z.dtype}.")


def _normalize_param(
    param: Tensor,
    name: str,
    length: int,
    batch: int,
    ssm: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if param.device != device:
        raise ValueError(f"{name} must be on device {device}, got {param.device}.")
    if param.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {param.dtype}.")
    if not torch.is_floating_point(param):
        raise TypeError(f"{name} must be a floating point tensor, got {param.dtype}.")

    if param.dim() == 1 and param.shape[0] == ssm:
        return param.unsqueeze(0).unsqueeze(0)

    if param.dim() == 2 and param.shape[1] == ssm:
        if param.shape[0] == length:
            return param.unsqueeze(1)
        if param.shape[0] == batch:
            return param.unsqueeze(0)

    if param.dim() == 3 and param.shape[2] == ssm:
        len_ok = param.shape[0] in (1, length)
        batch_ok = param.shape[1] in (1, batch)
        if len_ok and batch_ok:
            return param

    raise ValueError(
        f"{name} must be shaped as (M,), (L,M), (B,M), or (L,B,M); got {tuple(param.shape)}."
    )


def _validate_kernel_inputs(A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> None:
    length, batch, ssm = bu.shape
    for name, tensor in ("A", A), ("G", G), ("step", step):
        if tensor.dim() != 3 or tensor.shape[2] != ssm:
            raise ValueError(
                f"{name} must be a 3D tensor with trailing dimension {ssm}; got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] not in (1, length):
            raise ValueError(
                f"{name} has incompatible length dimension {tensor.shape[0]} for sequence length {length}."
            )
        if tensor.shape[1] not in (1, batch):
            raise ValueError(
                f"{name} has incompatible batch dimension {tensor.shape[1]} for batch size {batch}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous before launching the CUDA kernels.")


def _reference_sdlinoss_states_from_views(
    variant: str,
    A: Tensor,
    G: Tensor,
    step: Tensor,
    bu: Tensor,
) -> Tensor:
    length, batch, ssm = bu.shape
    device = bu.device
    complex_dtype = bu.dtype

    dt = torch.clamp(step, min=1e-6, max=1.0)
    bu = bu.contiguous()

    aux = torch.zeros(batch, ssm, dtype=complex_dtype, device=device)
    x = torch.zeros(batch, ssm, dtype=complex_dtype, device=device)
    states = torch.empty(length, batch, ssm, 2, dtype=complex_dtype, device=device)

    for t in range(length):
        idx_t = 0 if dt.size(0) == 1 else t
        a_idx = 0 if A.size(0) == 1 else t
        g_idx = 0 if G.size(0) == 1 else t

        dt_t = dt[idx_t]
        a_t = A[a_idx]
        g_t = G[g_idx]
        bu_t = bu[t]

        if variant == "imex1":
            S = torch.clamp(1.0 + dt_t * g_t, min=1e-6)
            comb = -a_t * x + bu_t
            aux = (aux + (dt_t * dt_t) * comb) / S
            x = x + aux
        elif variant == "imex2":
            S = torch.clamp(1.0 + (dt_t * dt_t) * a_t, min=1e-6)
            aux = (aux + dt_t * (-a_t * x - g_t * aux + bu_t)) / S
            x = x + dt_t * aux
        elif variant == "im":
            S = torch.clamp(1.0 + dt_t * g_t + (dt_t * dt_t) * a_t, min=1e-6)
            comb = -a_t * x + bu_t
            aux = (aux + (dt_t * dt_t) * comb) / S
            x = x + aux
        else:  # ex
            comb = -a_t * x + bu_t
            alpha = 1.0 - dt_t * g_t
            aux = alpha * aux + (dt_t * dt_t) * comb
            x = x + aux

        states[t, :, :, 0] = aux
        states[t, :, :, 1] = x

    return states


def _reference_sdlinoss_states(variant: str, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    variant = variant.lower()
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown selective D-LinOSS variant '{variant}'.")

    if bu.dim() != 3:
        raise ValueError("Selective D-LinOSS expects (length, batch, state) inputs.")

    length, batch, ssm = bu.shape
    device = bu.device
    real_dtype = _infer_real_dtype_from_complex(bu)

    if not bu.is_complex():
        raise TypeError("Selective D-LinOSS expects complex-valued bu inputs.")

    A = _normalize_param(a_diag, "A", length, batch, ssm, device=device, dtype=real_dtype)
    G = _normalize_param(g_diag, "G", length, batch, ssm, device=device, dtype=real_dtype)
    dt = _normalize_param(step, "step", length, batch, ssm, device=device, dtype=real_dtype)

    return _reference_sdlinoss_states_from_views(variant, A, G, dt, bu)


def _use_kernels(variant: str) -> bool:
    if os.environ.get("OSSM_SDLINOSS_DISABLE_KERNEL"):
        return False
    if _kernels is None:
        return False
    _dynamo = _maybe_dynamo_module()
    if _dynamo is not None:
        try:
            if _dynamo.is_compiling():  # pragma: no cover - runtime guard
                return False
        except RuntimeError:
            pass
    return variant.lower() in _SUPPORTED_VARIANTS


def has_kernels(variant: Optional[str] = None) -> bool:
    if _kernels is None:
        return False
    if variant is None:
        return all(has_kernels(name) for name in _SUPPORTED_VARIANTS)
    variant = variant.lower()
    if variant == "imex1":
        return hasattr(_kernels, "sdlinoss_imex1_forward")
    if variant == "imex2":
        return hasattr(_kernels, "sdlinoss_imex2_forward")
    if variant == "im":
        return hasattr(_kernels, "sdlinoss_im_forward")
    if variant == "ex":
        return hasattr(_kernels, "sdlinoss_ex_forward")
    raise ValueError(f"Unknown selective D-LinOSS variant '{variant}'.")


def extension_error() -> Optional[Exception]:
    return _EXTENSION_ERROR


def _fallback_sdlinoss(variant: str, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    states = _reference_sdlinoss_states(variant, A, G, step, bu)
    return states[..., 1]


class _SdlinossImex1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        _validate_kernel_inputs(A, G, step, bu)
        states = kernels.sdlinoss_imex1_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        return states[..., 1]

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        A, G, step, bu, states = ctx.saved_tensors
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        grad_output = grad_output.contiguous()
        grad_A, grad_G, grad_step, grad_bu = kernels.sdlinoss_imex1_backward(A, G, step, bu, states, grad_output)
        return grad_A, grad_G, grad_step, grad_bu


class _SdlinossImex2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        _validate_kernel_inputs(A, G, step, bu)
        states = kernels.sdlinoss_imex2_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        return states[..., 1]

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        A, G, step, bu, states = ctx.saved_tensors
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        grad_output = grad_output.contiguous()
        grad_A, grad_G, grad_step, grad_bu = kernels.sdlinoss_imex2_backward(A, G, step, bu, states, grad_output)
        return grad_A, grad_G, grad_step, grad_bu


class _SdlinossImFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        _validate_kernel_inputs(A, G, step, bu)
        states = kernels.sdlinoss_im_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        return states[..., 1]

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        A, G, step, bu, states = ctx.saved_tensors
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        grad_output = grad_output.contiguous()
        grad_A, grad_G, grad_step, grad_bu = kernels.sdlinoss_im_backward(A, G, step, bu, states, grad_output)
        return grad_A, grad_G, grad_step, grad_bu


class _SdlinossExFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        _validate_kernel_inputs(A, G, step, bu)
        states = kernels.sdlinoss_ex_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        return states[..., 1]

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        A, G, step, bu, states = ctx.saved_tensors
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("Selective D-LinOSS kernels are unavailable; cannot use optimized path.")

        grad_output = grad_output.contiguous()
        grad_A, grad_G, grad_step, grad_bu = kernels.sdlinoss_ex_backward(A, G, step, bu, states, grad_output)
        return grad_A, grad_G, grad_step, grad_bu


def run_sdlinoss(variant: str, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    variant = variant.lower()
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant '{variant}'. "
            f"Expected one of {', '.join(_SUPPORTED_VARIANTS)}."
        )

    if bu.dim() != 3:
        raise ValueError("bu must be (length, batch, ssm_size) complex.")

    length, batch, ssm = bu.shape
    device = bu.device
    complex_dtype = bu.dtype
    real_dtype = _infer_real_dtype_from_complex(bu)

    if not bu.is_complex():
        raise TypeError("bu must be complex valued.")

    A_view = _normalize_param(a_diag, "A", length, batch, ssm, device=device, dtype=real_dtype)
    G_view = _normalize_param(g_diag, "G", length, batch, ssm, device=device, dtype=real_dtype)
    step_view = _normalize_param(step, "step", length, batch, ssm, device=device, dtype=real_dtype)

    if length == 0:
        return bu.new_empty((0, batch, ssm))

    def _fallback_from_views() -> Tensor:
        states = _reference_sdlinoss_states_from_views(
            variant, A_view, G_view, step_view, bu.to(dtype=complex_dtype)
        )
        return states[..., 1]

    if not _use_kernels(variant):
        _graph_break_if_compiling()
        def _fallback() -> Tensor:
            return _fallback_from_views()
        _dynamo = _maybe_dynamo_module()
        if _dynamo is not None:
            try:
                if _dynamo.is_compiling():  # pragma: no cover - runtime guard
                    disable = getattr(_dynamo, "disable", None)
                    if disable is not None and callable(disable):
                        disable_fn = cast(
                            "Callable[[Callable[[], Tensor]], Callable[[], Tensor]]",
                            disable,
                        )
                        return cast(Tensor, disable_fn(_fallback)())
            except RuntimeError:
                pass
        with autocast("cuda", enabled=False):
            return _fallback()

    if variant == "imex1":
        fn = _SdlinossImex1Fn
    elif variant == "imex2":
        fn = _SdlinossImex2Fn
    elif variant == "im":
        fn = _SdlinossImFn
    else:
        fn = _SdlinossExFn

    # The CUDA kernels index the parameter tensors as contiguous (L, B, M) buffers.
    # Copy the broadcast-normalized views so the fast path cannot read stray memory
    # even if the caller provides lower-dimensional inputs.
    A_ctg = A_view.contiguous()
    G_ctg = G_view.contiguous()
    step_ctg = step_view.contiguous()
    bu_ctg = bu.contiguous()

    with autocast("cuda", enabled=False):
        return cast(Tensor, fn.apply(A_ctg, G_ctg, step_ctg, bu_ctg))

