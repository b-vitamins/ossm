"""Runtime helper for the Damped LinOSS recurrences and custom kernels."""

from __future__ import annotations

import os
import shutil
from typing import Optional, Protocol, Tuple, cast

import torch
from torch import Tensor, autocast

from .linoss import _run_associative_scan

__all__ = ["extension_error", "has_kernels", "run_dlinoss", "run_dlinoss_imex1"]


_SUPPORTED_VARIANTS: Tuple[str, ...] = ("imex1", "imex2", "im", "ex")


if os.environ.get("TORCHINDUCTOR_USE_OPENSSL") is None and shutil.which("openssl") is None:
    # TorchInductor shells out to ``openssl`` for hashing; fall back to the builtin
    # Python implementation when the binary is unavailable to avoid runtime errors.
    os.environ["TORCHINDUCTOR_USE_OPENSSL"] = "0"


def _reference_dlinoss_states(
    variant: str, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
) -> Tensor:
    """Pure PyTorch recurrences returning the full latent state trajectory."""

    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown D-LinOSS variant '{variant}'.")

    # Ensure the math runs with supported CUDA kernels
    with autocast("cuda", enabled=False):
        # Use fp32/complex64 by default, but honor higher precision inputs.
        real_dtype = torch.float32
        if (
            a_diag.dtype == torch.float64
            or g_diag.dtype == torch.float64
            or step.dtype == torch.float64
        ):
            real_dtype = torch.float64
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

        a_diag_f = a_diag.to(dtype=real_dtype)
        g_diag_f = g_diag.to(dtype=real_dtype)
        step_f = step.to(dtype=real_dtype)
        bu_c = bu.to(dtype=complex_dtype)

        ones = torch.ones_like(a_diag_f)
        step_broadcast = step_f.view(1, 1, -1)

        if variant == "imex1":
            denom = ones + step_f * g_diag_f
            # The official D-LinOSS recurrence (JAX implementation) keeps the
            # auxiliary variable z.  We evolve w = dt * z instead, which is
            # algebraically equivalent for fixed per-mode dt and avoids the
            # 1/dt amplification discussed in the conditioning analysis.  The
            # associated state transition maps directly onto the same x sequence
            # because x_{k+1} = x_k + w_{k+1} and w_{k+1} = dt * z_{k+1}.
            m11 = 1.0 / denom
            m12 = -(step_f * step_f * a_diag_f) / denom
            m21 = m11
            m22 = ones + m12
            scale = (step_broadcast * step_broadcast) / denom.view(1, 1, -1)
            f1 = bu_c * scale
            f2 = f1
        elif variant == "imex2":
            m11 = ones - step_f * g_diag_f
            m12 = -step_f * a_diag_f
            m21 = step_f * (ones - step_f * g_diag_f)
            m22 = ones - step_f.pow(2) * a_diag_f
            f1 = bu_c * step_broadcast
            f2 = bu_c * (step_broadcast * step_broadcast)
        elif variant == "im":
            denom = ones + step_f * g_diag_f + step_f.pow(2) * a_diag_f
            m11 = 1.0 / denom
            m12 = -(step_f * a_diag_f) / denom
            m21 = step_f / denom
            m22 = (ones + step_f * g_diag_f) / denom
            scale = (step_broadcast / denom.view(1, 1, -1))
            f1 = bu_c * scale
            f2 = bu_c * (scale * step_broadcast)
        else:  # variant == "ex"
            m11 = ones - step_f * g_diag_f
            m12 = -step_f * a_diag_f
            m21 = step_f
            m22 = ones
            f1 = bu_c * step_broadcast
            f2 = torch.zeros_like(f1)

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

    def dlinoss_imex2_forward(
        self, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
    ) -> Tensor: ...

    def dlinoss_imex2_backward(
        self,
        a_diag: Tensor,
        g_diag: Tensor,
        step: Tensor,
        bu: Tensor,
        states: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def dlinoss_im_forward(
        self, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
    ) -> Tensor: ...

    def dlinoss_im_backward(
        self,
        a_diag: Tensor,
        g_diag: Tensor,
        step: Tensor,
        bu: Tensor,
        states: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...

    def dlinoss_ex_forward(
        self, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
    ) -> Tensor: ...

    def dlinoss_ex_backward(
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
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_imex2_forward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu) -> Tensor"
            )
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_imex2_backward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu, Tensor states, Tensor grad_output) -> (Tensor, Tensor, Tensor, Tensor)"
            )
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_im_forward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu) -> Tensor"
            )
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_im_backward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu, Tensor states, Tensor grad_output) -> (Tensor, Tensor, Tensor, Tensor)"
            )
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_ex_forward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu) -> Tensor"
            )
            _CUSTOM_OP_LIBRARY.define(
                "dlinoss_ex_backward(Tensor a_diag, Tensor g_diag, Tensor step, Tensor bu, Tensor states, Tensor grad_output) -> (Tensor, Tensor, Tensor, Tensor)"
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

            def _dlinoss_imex2_forward_composite(
                a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
            ) -> Tensor:
                return kernels_ref.dlinoss_imex2_forward(a_diag, g_diag, step, bu)

            def _dlinoss_imex2_backward_composite(
                a_diag: Tensor,
                g_diag: Tensor,
                step: Tensor,
                bu: Tensor,
                states: Tensor,
                grad_output: Tensor,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                return kernels_ref.dlinoss_imex2_backward(
                    a_diag, g_diag, step, bu, states, grad_output
                )

            def _dlinoss_im_forward_composite(
                a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
            ) -> Tensor:
                return kernels_ref.dlinoss_im_forward(a_diag, g_diag, step, bu)

            def _dlinoss_im_backward_composite(
                a_diag: Tensor,
                g_diag: Tensor,
                step: Tensor,
                bu: Tensor,
                states: Tensor,
                grad_output: Tensor,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                return kernels_ref.dlinoss_im_backward(
                    a_diag, g_diag, step, bu, states, grad_output
                )

            def _dlinoss_ex_forward_composite(
                a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
            ) -> Tensor:
                return kernels_ref.dlinoss_ex_forward(a_diag, g_diag, step, bu)

            def _dlinoss_ex_backward_composite(
                a_diag: Tensor,
                g_diag: Tensor,
                step: Tensor,
                bu: Tensor,
                states: Tensor,
                grad_output: Tensor,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                return kernels_ref.dlinoss_ex_backward(
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
            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_imex2_forward",
                _dlinoss_imex2_forward_composite,
                "CompositeExplicitAutograd",
            )
            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_imex2_backward",
                _dlinoss_imex2_backward_composite,
                "CompositeExplicitAutograd",
            )
            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_im_forward",
                _dlinoss_im_forward_composite,
                "CompositeExplicitAutograd",
            )
            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_im_backward",
                _dlinoss_im_backward_composite,
                "CompositeExplicitAutograd",
            )
            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_ex_forward",
                _dlinoss_ex_forward_composite,
                "CompositeExplicitAutograd",
            )
            _CUSTOM_OP_LIBRARY.impl(
                "dlinoss_ex_backward",
                _dlinoss_ex_backward_composite,
                "CompositeExplicitAutograd",
            )

            def _dlinoss_forward_meta(
                a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
            ) -> Tensor:
                return _reference_dlinoss_states("imex1", a_diag, g_diag, step, bu)

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
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_imex2_forward",
                    lambda a_diag, g_diag, step, bu: _reference_dlinoss_states(
                        "imex2", a_diag, g_diag, step, bu
                    ),
                    "Meta",
                )
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_imex2_backward",
                    lambda a_diag, g_diag, step, bu, states, grad_output: (
                        torch.empty_like(a_diag),
                        torch.empty_like(g_diag),
                        torch.empty_like(step),
                        torch.empty_like(bu),
                    ),
                    "Meta",
                )
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_im_forward",
                    lambda a_diag, g_diag, step, bu: _reference_dlinoss_states(
                        "im", a_diag, g_diag, step, bu
                    ),
                    "Meta",
                )
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_im_backward",
                    lambda a_diag, g_diag, step, bu, states, grad_output: (
                        torch.empty_like(a_diag),
                        torch.empty_like(g_diag),
                        torch.empty_like(step),
                        torch.empty_like(bu),
                    ),
                    "Meta",
                )
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_ex_forward",
                    lambda a_diag, g_diag, step, bu: _reference_dlinoss_states(
                        "ex", a_diag, g_diag, step, bu
                    ),
                    "Meta",
                )
                _CUSTOM_OP_LIBRARY.impl(
                    "dlinoss_ex_backward",
                    lambda a_diag, g_diag, step, bu, states, grad_output: (
                        torch.empty_like(a_diag),
                        torch.empty_like(g_diag),
                        torch.empty_like(step),
                        torch.empty_like(bu),
                    ),
                    "Meta",
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


def has_kernels(variant: Optional[str] = None) -> bool:
    """Return ``True`` when kernels for ``variant`` are importable."""

    if _kernels is None:
        return False
    if variant is None:
        return all(
            has_kernels(name) for name in _SUPPORTED_VARIANTS
        )

    variant = variant.lower()
    if variant == "imex1":
        return hasattr(_kernels, "dlinoss_imex1_forward")
    if variant == "imex2":
        return hasattr(_kernels, "dlinoss_imex2_forward")
    if variant == "im":
        return hasattr(_kernels, "dlinoss_im_forward")
    if variant == "ex":
        return hasattr(_kernels, "dlinoss_ex_forward")
    raise ValueError(f"Unknown D-LinOSS variant '{variant}'.")


def extension_error() -> Optional[Exception]:
    """Return the cached import/build error for the D-LinOSS kernels, if any."""

    return _EXTENSION_ERROR


def _use_kernels(variant: str) -> bool:
    if os.environ.get("OSSM_DLINOSS_DISABLE_KERNEL"):
        return False

    return has_kernels(variant)


def _fallback_dlinoss(variant: str, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    """Pure PyTorch fallback matching the reference recurrence."""

    states = _reference_dlinoss_states(variant, a_diag, g_diag, step, bu)
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


class _DlinossImex2Fn(torch.autograd.Function):
    """Autograd bridge around the optimized D-LinOSS IMEX2 kernels."""

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
            states = torch.ops.ossm.dlinoss_imex2_forward(a_diag, g_diag, step, bu)
        else:
            states = kernels.dlinoss_imex2_forward(a_diag, g_diag, step, bu)
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
            grad_a, grad_g, grad_step, grad_bu = torch.ops.ossm.dlinoss_imex2_backward(
                a_diag, g_diag, step, bu, states, grad_output.contiguous()
            )
        else:
            grad_a, grad_g, grad_step, grad_bu = kernels.dlinoss_imex2_backward(
                a_diag, g_diag, step, bu, states, grad_output.contiguous()
            )
        return grad_a, grad_g, grad_step, grad_bu


class _DlinossImFn(torch.autograd.Function):
    """Autograd bridge around the optimized D-LinOSS IM kernels."""

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
            states = torch.ops.ossm.dlinoss_im_forward(a_diag, g_diag, step, bu)
        else:
            states = kernels.dlinoss_im_forward(a_diag, g_diag, step, bu)
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
            grad_a, grad_g, grad_step, grad_bu = torch.ops.ossm.dlinoss_im_backward(
                a_diag, g_diag, step, bu, states, grad_output.contiguous()
            )
        else:
            grad_a, grad_g, grad_step, grad_bu = kernels.dlinoss_im_backward(
                a_diag, g_diag, step, bu, states, grad_output.contiguous()
            )
        return grad_a, grad_g, grad_step, grad_bu


class _DlinossExFn(torch.autograd.Function):
    """Autograd bridge around the optimized D-LinOSS EX kernels."""

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
            states = torch.ops.ossm.dlinoss_ex_forward(a_diag, g_diag, step, bu)
        else:
            states = kernels.dlinoss_ex_forward(a_diag, g_diag, step, bu)
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
            grad_a, grad_g, grad_step, grad_bu = torch.ops.ossm.dlinoss_ex_backward(
                a_diag, g_diag, step, bu, states, grad_output.contiguous()
            )
        else:
            grad_a, grad_g, grad_step, grad_bu = kernels.dlinoss_ex_backward(
                a_diag, g_diag, step, bu, states, grad_output.contiguous()
            )
        return grad_a, grad_g, grad_step, grad_bu


def run_dlinoss(
    variant: str, a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor
) -> Tensor:
    """Evaluate the D-LinOSS recurrence for ``variant`` with kernel fallbacks."""

    global _EXTENSION_ERROR

    variant = variant.lower()
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown D-LinOSS variant '{variant}'.")

    if bu.numel() == 0:
        return bu

    # The optimized kernels currently operate in single precision.  When
    # higher-precision inputs are provided we fall back to the reference
    # implementation to avoid silent downcasting and preserve numerical
    # parity with the pure PyTorch path.
    if (
        a_diag.dtype == torch.float64
        or g_diag.dtype == torch.float64
        or step.dtype == torch.float64
        or bu.dtype == torch.complex128
    ):
        return _fallback_dlinoss(variant, a_diag, g_diag, step, bu)

    if variant == "imex1" and _use_kernels(variant):
        try:
            return cast(Tensor, _DlinossImex1Fn.apply(a_diag, g_diag, step, bu))
        except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
            _EXTENSION_ERROR = exc
            _trace(f"[OSSM] D-LinOSS kernel call failed: {exc}")
    elif variant == "imex2" and _use_kernels(variant):
        try:
            return cast(Tensor, _DlinossImex2Fn.apply(a_diag, g_diag, step, bu))
        except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
            _EXTENSION_ERROR = exc
            _trace(f"[OSSM] D-LinOSS kernel call failed: {exc}")
    elif variant == "im" and _use_kernels(variant):
        try:
            return cast(Tensor, _DlinossImFn.apply(a_diag, g_diag, step, bu))
        except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
            _EXTENSION_ERROR = exc
            _trace(f"[OSSM] D-LinOSS kernel call failed: {exc}")
    elif variant == "ex" and _use_kernels(variant):
        try:
            return cast(Tensor, _DlinossExFn.apply(a_diag, g_diag, step, bu))
        except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
            _EXTENSION_ERROR = exc
            _trace(f"[OSSM] D-LinOSS kernel call failed: {exc}")
    return _fallback_dlinoss(variant, a_diag, g_diag, step, bu)


def run_dlinoss_imex1(a_diag: Tensor, g_diag: Tensor, step: Tensor, bu: Tensor) -> Tensor:
    """Backward-compatible wrapper around :func:`run_dlinoss` for IMEX1."""

    return run_dlinoss("imex1", a_diag, g_diag, step, bu)


if has_kernels() and os.environ.get("OSSM_DLINOSS_TRACE"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _trace(f"[OSSM] Using D-LinOSS C++ extension (device={device}).")
