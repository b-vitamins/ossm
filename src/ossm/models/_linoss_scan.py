"""Runtime helper for the LinOSS custom scan kernel."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import Tensor

__all__ = ["try_run_scan", "is_available", "extension_error"]

try:
    from ossm import _kernels as _kernels  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-time failure surface
    _kernels = None
    _EXTENSION_ERROR: Optional[Exception] = exc
else:
    _EXTENSION_ERROR = None


def _trace(message: str) -> None:
    if os.environ.get("OSSM_LINOSS_TRACE"):
        print(message, flush=True)


if _kernels is not None and os.environ.get("OSSM_LINOSS_TRACE"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _trace(f"[OSSM] Using LinOSS C++ extension (device={device}).")


def is_available() -> bool:
    """Return ``True`` when the compiled LinOSS kernels are importable."""

    return _kernels is not None


def extension_error() -> Optional[Exception]:
    """Return the cached import/build error, if any."""

    return _EXTENSION_ERROR


def try_run_scan(
    m11: Tensor,
    m12: Tensor,
    m21: Tensor,
    m22: Tensor,
    b_seq: Tensor,
) -> Optional[Tensor]:
    """Attempt to execute the custom LinOSS scan; return ``None`` if unavailable."""

    if b_seq.numel() == 0:
        return b_seq
    if _kernels is None:
        return None
    try:
        return _kernels.linoss_scan(m11, m12, m21, m22, b_seq)
    except RuntimeError as exc:  # pragma: no cover - runtime device mismatch
        global _EXTENSION_ERROR
        _EXTENSION_ERROR = exc
        _trace(f"[OSSM] LinOSS extension call failed: {exc}")
        return None
