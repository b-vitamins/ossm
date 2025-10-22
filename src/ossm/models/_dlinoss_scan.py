"""Runtime helper for the Damped LinOSS IMEX1 custom kernels."""

from __future__ import annotations

import os
from typing import Optional, Protocol, cast

import torch
from torch import Tensor

from .linoss import _run_associative_scan

__all__ = ["run_dlinoss_imex1", "has_kernels", "extension_error"]


def _reference_dlinoss_states(
    a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
) -> Tensor:
    """Pure PyTorch recurrence returning the full latent state trajectory."""

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
    return _run_associative_scan(a_matrix, b_elems)

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


_CUSTOM_OP_AVAILABLE = False


try:
    from ossm import _kernels as _kernels  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-time failure surface
    _kernels: Optional[_DlinossKernels] = None
    _EXTENSION_ERROR: Optional[Exception] = exc
else:
    _kernels = cast("_DlinossKernels", _kernels)
    kernels_ref: _DlinossKernels = _kernels
    _EXTENSION_ERROR = None

    # Register custom operators so ``torch.compile`` can trace the kernels using
    # fake tensors while still lowering to the optimized extension at runtime.
    try:  # pragma: no cover - ``torch.library`` is unavailable on very old wheels
        from torch.library import Library

        _CUSTOM_OP_LIBRARY: Optional[Library] = None

        try:
            _CUSTOM_OP_LIBRARY = Library("ossm", "DEF")
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_imex1_forward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu) -> Tensor"
            )
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_imex1_backward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu, Tensor states, Tensor grad_output) -> (Tensor, Tensor, Tensor, Tensor)"
            )
        except RuntimeError:
            # The operators may already be defined when multiple OSSM modules are
            # imported in the same interpreter.  In that case we simply look them
            # up and continue registering implementations.
            _CUSTOM_OP_LIBRARY = Library("ossm", "IMPL")

        if _CUSTOM_OP_LIBRARY is not None:
            def _dlinoss_forward_composite(
                a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
            ) -> Tensor:
                return kernels_ref.dlinoss_imex1_forward(a_diag, g_diag, step, bu)

            def _dlinoss_backward_composite(
                a_diag: Tensor,
                g_diag: Tensor,
                step: Tensor,
                bu: Tensor,
                states: Tensor,
                grad_output: Tensor,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                return kernels_ref.dlinoss_imex1_backward(
                    a_diag, g_diag, step, bu, states, grad_output
                )

            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_imex1_forward",
                _dlinoss_forward_composite,
                "CompositeExplicitAutograd",
            )
            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_imex1_backward",
                _dlinoss_backward_composite,
                "CompositeExplicitAutograd",
            )

            def _dlinoss_forward_meta(
                a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
            ) -> Tensor:
                return _reference_dlinoss_states(a_diag, g_diag, step, bu)

            def _dlinoss_backward_meta(
                a_diag: Tensor,
                g_diag: Tensor,
                step: Tensor,
                bu: Tensor,
                states: Tensor,
                grad_output: Tensor,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                del states, grad_output
                return (
                    torch.empty_like(a_diag),
                    torch.empty_like(g_diag),
                    torch.empty_like(step),
                    torch.empty_like(bu),
                )

            try:
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_imex1_forward", _dlinoss_forward_meta, "Meta"
                )
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_imex1_backward", _dlinoss_backward_meta, "Meta"
                )
            except RuntimeError:
                # Meta kernels may already be present; that's acceptable.
                pass

            _CUSTOM_OP_AVAILABLE = True
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

    return has_kernels()


def _fallback_dlinoss_imex1(a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    """Pure PyTorch fallback matching the reference recurrence."""

    states = _reference_dlinoss_states(a_diag, g_diag, step, bu)
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

        if _CUSTOM_OP_AVAILABLE:
            states = torch.ops.ossm.dlinoss_imex1_forward(a_diag, g_diag, step, bu)
        else:
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

        if _CUSTOM_OP_AVAILABLE:
            grad_a, grad_g, grad_step, grad_bu = torch.ops.ossm.dlinoss_imex1_backward(
                a_diag, g_diag, step, bu, states, grad_output.contiguous()
            )
        else:
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
