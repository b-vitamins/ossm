"""Runtime helper for the Linear RNN custom scan kernel."""

from __future__ import annotations

import hashlib
import pathlib
from types import ModuleType
from typing import Optional

import torch
from torch import Tensor
from torch.utils.cpp_extension import CUDA_HOME, load as load_extension

__all__ = ["try_run_linear_rnn_scan"]

_EXTENSION: Optional[ModuleType] = None
_EXTENSION_ERROR: Optional[Exception] = None


def _load_extension() -> Optional[ModuleType]:
    """Compile and cache the linear RNN scan extension if available."""

    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None or _EXTENSION_ERROR is not None:
        return _EXTENSION

    base = pathlib.Path(__file__).with_name("_rnn_scan.cpp")
    cuda_source = base.with_name("_rnn_scan_cuda.cu")

    sources = [str(base)]
    extra_cuda_cflags = None
    with_cuda = False
    if CUDA_HOME is not None and torch.cuda.is_available() and cuda_source.exists():
        sources.append(str(cuda_source))
        extra_cuda_cflags = ["-O3"]
        with_cuda = True

    hasher = hashlib.md5()
    for src in sources:
        hasher.update(pathlib.Path(src).read_bytes())
    hasher.update("-O3-ffast-math-march=native".encode())
    digest = hasher.hexdigest()

    try:
        _EXTENSION = load_extension(
            name=f"ossm_rnn_scan_{digest}",
            sources=sources,
            extra_cflags=["-O3", "-ffast-math", "-march=native"],
            extra_cuda_cflags=extra_cuda_cflags,
            with_cuda=with_cuda,
            verbose=False,
        )
    except (OSError, RuntimeError) as exc:  # pragma: no cover - build environment dependent
        _EXTENSION_ERROR = exc
        _EXTENSION = None
    return _EXTENSION


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

    extension = _load_extension()
    if extension is None:
        return None

    return extension.linear_rnn_scan(
        weight_hh.contiguous(),
        weight_xh.contiguous(),
        bias.contiguous(),
        inputs.contiguous(),
        initial_state.contiguous(),
    )
