"""Feature flags and fast selective D-LinOSS bindings."""

from __future__ import annotations

import os
from typing import Any, Optional, cast

import torch
from torch import Tensor

__all__ = [
    "USE_FAST",
    "X_ONLY",
    "TILE",
    "has_fast_kernels",
    "run_sdlinoss_fast",
    "SdlinossExFastFn",
    "SdlinossImex1FastFn",
    "SdlinossImex2FastFn",
    "SdlinossImFastFn",
    "_im_forward",
]

USE_FAST = os.getenv("OSSM_SDLINOSS_FAST", "0") == "1"
X_ONLY = os.getenv("OSSM_SDLINOSS_FAST_X_ONLY", "0") == "1"
TILE = int(os.getenv("OSSM_SDLINOSS_FAST_TILE", "128"))

try:
    from ossm import _kernels as _kernels  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - build-time failure surface
    _kernels = None


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


def _normalize_inputs(A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if bu.dim() != 3:
        raise ValueError("bu must be (length, batch, ssm_size) complex.")
    if not bu.is_complex():
        raise TypeError("bu must be complex valued.")

    length, batch, ssm = bu.shape
    device = bu.device
    real_dtype = _infer_real_dtype_from_complex(bu)

    A_view = _normalize_param(A, "A", length, batch, ssm, device=device, dtype=real_dtype)
    G_view = _normalize_param(G, "G", length, batch, ssm, device=device, dtype=real_dtype)
    step_view = _normalize_param(step, "step", length, batch, ssm, device=device, dtype=real_dtype)

    def _materialize(param: Tensor) -> Tensor:
        """Return a contiguous tensor whose length/batch match ``bu`` exactly."""

        length_dim = length if param.size(0) == 1 else param.size(0)
        batch_dim = batch if param.size(1) == 1 else param.size(1)

        if length_dim == param.size(0) and batch_dim == param.size(1):
            return param.contiguous()

        return param.expand(length_dim, batch_dim, ssm).contiguous()

    return (
        _materialize(A_view),
        _materialize(G_view),
        _materialize(step_view),
        bu.contiguous(),
    )


def has_fast_kernels(variant: Optional[str] = None) -> bool:
    """Return whether the fast selective D-LinOSS kernels are available."""

    if _kernels is None or not hasattr(_kernels, "sdlinoss_fast_has_kernels"):
        return False

    available = bool(_kernels.sdlinoss_fast_has_kernels())
    if not available:
        return False

    if variant is None:
        return True

    variant = variant.lower()
    if variant not in {"ex", "imex1", "imex2", "im"}:
        return False

    attr = f"sdlinoss_fast_{variant}_forward"
    if not hasattr(_kernels, attr):
        return False
    if X_ONLY:
        x_attr = f"sdlinoss_fast_{variant}_forward_xonly"
        if not hasattr(_kernels, x_attr):
            return False
    return True


def _im_forward(A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    if _kernels is None:
        raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

    if X_ONLY:
        return _kernels.sdlinoss_fast_im_forward_xonly(A, G, step, bu)

    return _kernels.sdlinoss_fast_im_forward(A, G, step, bu)


class SdlinossExFastFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        if X_ONLY:
            x_only = _kernels.sdlinoss_fast_ex_forward_xonly(A, G, step, bu)
            ctx.save_for_backward(A, G, step, bu, x_only)
            ctx.x_only = True
            return x_only

        states = _kernels.sdlinoss_fast_ex_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        ctx.x_only = False
        return states[..., 1]

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        grad_out = grad_out.contiguous()

        A, G, step, bu, stash = ctx.saved_tensors
        if getattr(ctx, "x_only", False):
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_ex_backward_xonly(
                A, G, step, bu, stash, grad_out
            )
        else:
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_ex_backward(
                A, G, step, bu, stash, grad_out
            )
        return grad_A, grad_G, grad_step, grad_bu


class SdlinossImex1FastFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        if X_ONLY:
            x_only = _kernels.sdlinoss_fast_imex1_forward_xonly(A, G, step, bu)
            ctx.save_for_backward(A, G, step, bu, x_only)
            ctx.x_only = True
            return x_only

        states = _kernels.sdlinoss_fast_imex1_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        ctx.x_only = False
        return states[..., 1]

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        grad_out = grad_out.contiguous()
        A, G, step, bu, stash = ctx.saved_tensors

        if getattr(ctx, "x_only", False):
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_imex1_backward_xonly(
                A, G, step, bu, stash, grad_out
            )
        else:
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_imex1_backward(
                A, G, step, bu, stash, grad_out
            )
        return grad_A, grad_G, grad_step, grad_bu


class SdlinossImex2FastFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        if X_ONLY:
            x_only = _kernels.sdlinoss_fast_imex2_forward_xonly(A, G, step, bu)
            ctx.save_for_backward(A, G, step, bu, x_only)
            ctx.x_only = True
            return x_only

        states = _kernels.sdlinoss_fast_imex2_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        ctx.x_only = False
        return states[..., 1]

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        grad_out = grad_out.contiguous()

        A, G, step, bu, stash = ctx.saved_tensors
        if getattr(ctx, "x_only", False):
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_imex2_backward_xonly(
                A, G, step, bu, stash, grad_out
            )
        else:
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_imex2_backward(A, G, step, bu, stash, grad_out)
        return grad_A, grad_G, grad_step, grad_bu


class SdlinossImFastFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, A: Tensor, G: Tensor, step: Tensor, bu: Tensor) -> Tensor:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        if X_ONLY:
            x = _kernels.sdlinoss_fast_im_forward_xonly(A, G, step, bu)
            ctx.save_for_backward(A, G, step, bu, x)
            ctx.x_only = True
            return x

        states = _kernels.sdlinoss_fast_im_forward(A, G, step, bu)
        ctx.save_for_backward(A, G, step, bu, states)
        ctx.x_only = False
        return states[..., 1]

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore[override]
        if _kernels is None:
            raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

        grad_out = grad_out.contiguous()

        if getattr(ctx, "x_only", False):
            A, G, step, bu, x_only = ctx.saved_tensors
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_im_backward_xonly(
                A, G, step, bu, x_only, grad_out
            )
        else:
            A, G, step, bu, states = ctx.saved_tensors
            grad_A, grad_G, grad_step, grad_bu = _kernels.sdlinoss_fast_im_backward(
                A, G, step, bu, states, grad_out
            )
        return grad_A, grad_G, grad_step, grad_bu


def run_sdlinoss_fast(
    variant: str,
    A: Tensor,
    G: Tensor,
    step: Tensor,
    bu: Tensor,
) -> Tensor:
    """Execute the fast selective D-LinOSS kernels and return the ``x`` state."""

    if _kernels is None:
        raise RuntimeError("Selective D-LinOSS fast kernels are unavailable; extension not loaded.")

    variant_normalized = variant.lower()
    dispatch: dict[str, type[torch.autograd.Function]] = {
        "ex": SdlinossExFastFn,
        "imex1": SdlinossImex1FastFn,
        "imex2": SdlinossImex2FastFn,
        "im": SdlinossImFastFn,
    }

    if variant_normalized not in dispatch:
        raise ValueError(
            "Only the EX, IMEX1, IMEX2, and IM variants are supported by the fast kernels."
        )

    if not has_fast_kernels(variant_normalized):
        raise RuntimeError(f"Fast kernels for variant '{variant}' are unavailable in this build.")

    A_norm, G_norm, step_norm, bu_norm = _normalize_inputs(A, G, step, bu)

    fn = dispatch[variant_normalized]
    return cast(Tensor, fn.apply(A_norm, G_norm, step_norm, bu_norm))

