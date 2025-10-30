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


def _expand_param(param: Tensor, length: int, batch: int, ssm: int, *, device, dtype) -> Tensor:
    if param.dim() == 1 and param.shape[0] == ssm:
        return param.view(1, 1, ssm).to(device=device, dtype=dtype).expand(length, batch, ssm)
    if param.dim() == 2 and param.shape == (length, ssm):
        return param.view(length, 1, ssm).to(device=device, dtype=dtype).expand(length, batch, ssm)
    if param.dim() == 2 and param.shape == (batch, ssm):
        return param.view(1, batch, ssm).to(device=device, dtype=dtype).expand(length, batch, ssm)
    if param.shape == (length, batch, ssm):
        return param.to(device=device, dtype=dtype)
    raise ValueError(
        "Parameter must be (M,), (L,M), (B,M) or (L,B,M); "
        f"got {tuple(param.shape)}."
    )


def _reference_sdlinoss_states(variant: str, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    variant = variant.lower()
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown selective D-LinOSS variant '{variant}'.")

    if bu.dim() != 3:
        raise ValueError("Selective D-LinOSS expects (length, batch, state) inputs.")

    length, batch, ssm = bu.shape
    device = bu.device
    complex_dtype = bu.dtype
    real_dtype = _infer_real_dtype_from_complex(bu)

    A = _expand_param(a_diag, length, batch, ssm, device=device, dtype=real_dtype).contiguous()
    G = _expand_param(g_diag, length, batch, ssm, device=device, dtype=real_dtype).contiguous()
    dt = _expand_param(step, length, batch, ssm, device=device, dtype=real_dtype)
    dt = torch.clamp(dt, min=1e-6, max=1.0).contiguous()
    bu = bu.contiguous()

    z = torch.zeros(batch, ssm, dtype=complex_dtype, device=device)
    x = torch.zeros(batch, ssm, dtype=complex_dtype, device=device)
    states = torch.empty(length, batch, ssm, 2, dtype=complex_dtype, device=device)

    for t in range(length):
        dt_t = dt[t]
        a_t = A[t]
        g_t = G[t]
        bu_t = bu[t]

        if variant == "imex1":
            S = torch.clamp(1.0 + dt_t * g_t, min=1e-6)
            z = (z + dt_t * (-a_t * x + bu_t)) / S
        elif variant == "imex2":
            S = torch.clamp(1.0 + (dt_t * dt_t) * a_t, min=1e-6)
            z = (z + dt_t * (-a_t * x - g_t * z + bu_t)) / S
        elif variant == "im":
            S = torch.clamp(1.0 + dt_t * g_t + (dt_t * dt_t) * a_t, min=1e-6)
            z = (z + dt_t * (-a_t * x + bu_t)) / S
        else:  # ex
            z = z + dt_t * (-a_t * x - g_t * z + bu_t)

        x = x + dt_t * z
        states[t, :, :, 0] = z
        states[t, :, :, 1] = x

    return states


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

    A = _expand_param(a_diag, length, batch, ssm, device=device, dtype=real_dtype)
    G = _expand_param(g_diag, length, batch, ssm, device=device, dtype=real_dtype)
    step_expanded = _expand_param(step, length, batch, ssm, device=device, dtype=real_dtype)

    if not bu.is_complex():
        raise TypeError("bu must be complex valued.")

    if length == 0:
        return bu.new_empty((0, batch, ssm))

    if not _use_kernels(variant):
        _graph_break_if_compiling()
        def _fallback() -> Tensor:
            return _fallback_sdlinoss(
                variant,
                A.contiguous(),
                G.contiguous(),
                step_expanded.contiguous(),
                bu.to(complex_dtype).contiguous(),
            )
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

    bu_ctg = bu.contiguous()

    if variant == "imex1":
        fn = _SdlinossImex1Fn
    elif variant == "imex2":
        fn = _SdlinossImex2Fn
    elif variant == "im":
        fn = _SdlinossImFn
    else:
        fn = _SdlinossExFn

    with autocast("cuda", enabled=False):
        return cast(Tensor, fn.apply(A, G, step_expanded, bu_ctg))

