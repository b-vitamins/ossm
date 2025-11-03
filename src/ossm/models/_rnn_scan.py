"""Runtime helper for the fused linear RNN scan kernel."""

from __future__ import annotations

from typing import Optional
import os

from torch import Tensor

__all__ = ["try_run_linear_rnn_scan", "is_available", "extension_error"]

try:
    from ossm import _kernels as _kernels  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-time failure surface
    _kernels = None
    _EXTENSION_ERROR: Optional[Exception] = exc
else:
    _EXTENSION_ERROR = None


def try_run_linear_rnn_scan(
    weight_hh: Tensor,
    weight_xh: Tensor,
    bias: Tensor,
    inputs: Tensor,
    initial_state: Tensor,
) -> Optional[Tensor]:
    """Attempt to execute the fused linear RNN scan."""

    if inputs.numel() == 0:
        return inputs.new_empty((0, inputs.size(1), weight_hh.size(0)))
    # Allow disabling the kernel at runtime (stability/perf variability across envs)
    if os.environ.get("OSSM_DISABLE_RNN_KERNEL") == "1":
        return None
    if _kernels is None:
        return None

    try:
        return _kernels.linear_rnn_scan(
            weight_hh.contiguous(),
            weight_xh.contiguous(),
            bias.contiguous(),
            inputs.contiguous(),
            initial_state.contiguous(),
        )
    except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
        global _EXTENSION_ERROR
        _EXTENSION_ERROR = exc
        return None


def is_available() -> bool:
    """Return ``True`` when the compiled linear RNN kernels are importable."""

    return _kernels is not None


def extension_error() -> Optional[Exception]:
    """Return the cached import/build error, if any."""

    return _EXTENSION_ERROR
