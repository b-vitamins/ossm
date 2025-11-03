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

    fn = dispatch[variant_normalized]
    return cast(Tensor, fn.apply(A, G, step, bu))

