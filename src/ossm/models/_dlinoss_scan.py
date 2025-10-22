"""Runtime helper for the Damped LinOSS IMEX1 custom kernels."""

from __future__ import annotations

import os
from typing import Optional, Protocol, cast

import torch
from torch import Tensor

from .linoss import _run_associative_scan

__all__ = ["run_dlinoss_imex1", "has_kernels", "extension_error"]

class _DlinossKernels(Protocol):
    def dlinoss_imex1_forward(
        self, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
    ) -> Tensor: ...

    def dlinoss_imex1_backward(
        self,
        a_diag: Tensor,
        g_diag: Tensor,
        step: Tensor,
        bu: Tensor,
        states: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


try:
    from ossm import _kernels as _kernels  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-time failure surface
    _kernels: Optional[_DlinossKernels] = None
    _EXTENSION_ERROR: Optional[Exception] = exc
else:
    _kernels = cast("_DlinossKernels", _kernels)
    _EXTENSION_ERROR = None

    # Enable ``torch.compile`` and Dynamo to treat the PyCapsule entry points as
    # graph-friendly primitives instead of emitting loud warnings during
    # tracing.  The helpers are optional so we guard them defensively to keep
    # import-time behavior identical on older PyTorch wheels.
    try:  # pragma: no cover - import guard only exercised with recent PyTorch
        import torch._dynamo as _torch_dynamo  # type: ignore[attr-defined]

        try:
            _torch_dynamo.allow_in_graph(_kernels.dlinoss_imex1_forward)
            _torch_dynamo.allow_in_graph(_kernels.dlinoss_imex1_backward)
        except AttributeError:
            pass
    except ImportError:  # pragma: no cover - dependency not present
        pass

    try:  # pragma: no cover - ``torch.compiler`` may be unavailable
        import torch.compiler as _torch_compiler  # type: ignore[attr-defined]

        try:
            _torch_compiler.allow_in_graph(_kernels.dlinoss_imex1_forward)
            _torch_compiler.allow_in_graph(_kernels.dlinoss_imex1_backward)
        except AttributeError:
            pass
    except ImportError:
        pass


def _trace(message: str) -> None:
    if os.environ.get("OSSM_DLINOSS_TRACE"):
        print(message, flush=True)


def has_kernels() -> bool:
    """Return ``True`` when the D-LinOSS kernels are importable."""

    return _kernels is not None and hasattr(_kernels, "dlinoss_imex1_forward")


def extension_error() -> Optional[Exception]:
    """Return the cached import/build error for the D-LinOSS kernels, if any."""

    return _EXTENSION_ERROR


def _use_kernels() -> bool:
    if os.environ.get("OSSM_DLINOSS_DISABLE_KERNEL"):
        return False

    try:  # pragma: no cover - ``torch._dynamo`` may be absent on older builds
        import torch._dynamo as _torch_dynamo  # type: ignore[attr-defined]

        if _torch_dynamo.is_compiling():
            # ``torch.compile`` drives tracing with ``FakeTensor`` instances
            # that deliberately lack storage.  The custom kernels expect real
            # storage so we fall back to the pure PyTorch reference when the
            # graph is being captured.  ``torch`` will then trace the fallback
            # implementation directly, producing a graph that is agnostic to
            # the extension while the eager path continues to benefit from the
            # optimized kernels.
            return False
    except ImportError:
        pass

    return has_kernels()


def _fallback_dlinoss_imex1(a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    """Pure PyTorch fallback matching the reference recurrence."""

    device = bu.device
    dtype = bu.dtype

    a_diag_c = a_diag.to(device=device, dtype=dtype)
    g_diag_c = g_diag.to(device=device, dtype=dtype)
    step_c = step.to(device=device, dtype=dtype)

    denom = 1.0 + step_c * g_diag_c
    m11 = 1.0 / denom
    m12 = -(step_c * a_diag_c) / denom
    m21 = step_c / denom
    m22 = 1.0 - (step_c * step_c * a_diag_c) / denom

    step_broadcast = step_c.view(1, 1, -1)
    denom_broadcast = denom.view(1, 1, -1)
    f1 = bu * (step_broadcast / denom_broadcast)
    f2 = bu * (step_broadcast * step_broadcast / denom_broadcast)

    a_matrix = torch.stack(
        (
            torch.stack((m11, m12), dim=-1),
            torch.stack((m21, m22), dim=-1),
        ),
        dim=-2,
    )
    b_elems = torch.stack((f1, f2), dim=-1)
    states = _run_associative_scan(a_matrix, b_elems)
    return states[..., 1]


class _DlinossImex1Fn(torch.autograd.Function):
    """Autograd bridge around the optimized D-LinOSS IMEX1 kernels."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        a_diag: Tensor,
        g_diag: Tensor,
        step: Tensor,
        bu: Tensor,
    ) -> Tensor:
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("D-LinOSS kernels are unavailable; cannot use optimized path.")

        states = kernels.dlinoss_imex1_forward(a_diag, g_diag, step, bu)
        ctx.save_for_backward(a_diag, g_diag, step, bu, states)
        return states[..., 1]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a_diag, g_diag, step, bu, states = ctx.saved_tensors
        kernels = _kernels
        if kernels is None:
            raise RuntimeError("D-LinOSS kernels are unavailable; cannot use optimized path.")

        grad_a, grad_g, grad_step, grad_bu = kernels.dlinoss_imex1_backward(
            a_diag, g_diag, step, bu, states, grad_output.contiguous()
        )
        return grad_a, grad_g, grad_step, grad_bu


def run_dlinoss_imex1(a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    """Evaluate the D-LinOSS IMEX1 recurrence, preferring custom kernels when available."""

    if bu.numel() == 0:
        return bu

    if _use_kernels():
        try:
            return cast(Tensor, _DlinossImex1Fn.apply(a_diag, g_diag, step, bu))
        except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
            global _EXTENSION_ERROR
            _EXTENSION_ERROR = exc
            _trace(f"[OSSM] D-LinOSS kernel call failed: {exc}")
    return _fallback_dlinoss_imex1(a_diag, g_diag, step, bu)


if has_kernels() and os.environ.get("OSSM_DLINOSS_TRACE"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _trace(f"[OSSM] Using D-LinOSS C++ extension (device={device}).")
