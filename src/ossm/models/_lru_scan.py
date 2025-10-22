"""Runtime helper for the LRU custom scan kernel."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["try_run_lru_scan", "is_available", "extension_error"]

try:
    from ossm import _kernels as _kernels  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-time failure surface
    _kernels = None
    _EXTENSION_ERROR: Optional[Exception] = exc
else:
    _EXTENSION_ERROR = None


def try_run_lru_scan(lambda_bar: Tensor, b_seq: Tensor) -> Optional[Tensor]:
    """Attempt to execute the custom LRU scan; return ``None`` if unavailable."""

    if b_seq.numel() == 0:
        return b_seq
    if _kernels is None:
        return None

    lambda_real = lambda_bar.real.contiguous()
    lambda_imag = lambda_bar.imag.contiguous()
    b_seq = b_seq.contiguous()

    try:
        result = _kernels.lru_scan(lambda_real, lambda_imag, torch.view_as_real(b_seq))
    except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
        global _EXTENSION_ERROR
        _EXTENSION_ERROR = exc
        return None
    return torch.view_as_complex(result.contiguous())


def is_available() -> bool:
    """Return ``True`` when the compiled LRU kernels are importable."""

    return _kernels is not None


def extension_error() -> Optional[Exception]:
    """Return the cached import/build error, if any."""

    return _EXTENSION_ERROR
