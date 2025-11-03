"""OSSM package public exports."""

from __future__ import annotations

import os
import shutil

if os.environ.get("TORCHINDUCTOR_USE_OPENSSL") is None and shutil.which("openssl") is None:
    os.environ["TORCHINDUCTOR_USE_OPENSSL"] = "0"

from . import data, metrics
from .models import (
    AbstractRNNCell,
    Backbone,
    BatchOnDevice,
    ClassificationHead,
    DampedLinOSSBackbone,
    DampedLinOSSBlock,
    DampedLinOSSLayer,
    Dlinoss4Rec,
    GRURNNCell,
    Head,
    ItemEmbeddingEncoder,
    Sdlinoss4Rec,
    LRUBackbone,
    LRUBlock,
    LRULayer,
    LSTMRNNCell,
    LinOSSBackbone,
    LinOSSBlock,
    LinOSSLayer,
    LinearRNNCell,
    Mamba4Rec,
    MambaLayer,
    MLPRNNCell,
    NCDEBackbone,
    NCDELayer,
    NCDEVectorField,
    NRDELayer,
    RegressionHead,
    ResidualSSMBlock,
    RNNBackbone,
    RNNLayer,
    S5Backbone,
    S5Block,
    S5Layer,
    SequenceBackboneOutput,
    SelectiveDLinOSSBackbone,
    SelectiveDLinOSSBlock,
    SelectiveDLinOSSLayer,
    TiedSoftmaxHead,
    run_sdlinoss,
)
__all__: tuple[str, ...] = (
    "data",
    "metrics",
    "AbstractRNNCell",
    "Backbone",
    "BatchOnDevice",
    "ClassificationHead",
    "DampedLinOSSBackbone",
    "DampedLinOSSBlock",
    "DampedLinOSSLayer",
    "Dlinoss4Rec",
    "GRURNNCell",
    "Head",
    "ItemEmbeddingEncoder",
    "Sdlinoss4Rec",
    "LinOSSBackbone",
    "LinOSSBlock",
    "LinOSSLayer",
    "LinearRNNCell",
    "LRUBackbone",
    "LRUBlock",
    "LRULayer",
    "LSTMRNNCell",
    "Mamba4Rec",
    "MambaLayer",
    "MLPRNNCell",
    "NCDEBackbone",
    "NCDELayer",
    "NCDEVectorField",
    "NRDELayer",
    "RegressionHead",
    "ResidualSSMBlock",
    "RNNBackbone",
    "RNNLayer",
    "S5Backbone",
    "S5Block",
    "S5Layer",
    "SequenceBackboneOutput",
    "SelectiveDLinOSSBackbone",
    "SelectiveDLinOSSBlock",
    "SelectiveDLinOSSLayer",
    "TiedSoftmaxHead",
    "run_sdlinoss",
)

# Optionally disable torch.compile to avoid environment-specific failures when
# Inductor/Triton are unavailable or mismatched. This keeps tests stable on
# systems without a working compiler stack. Controlled via
# OSSM_DISABLE_TORCH_COMPILE=1 (default enabled).
try:  # pragma: no cover - environment guard
    import torch as _torch
    if os.environ.get("OSSM_DISABLE_TORCH_COMPILE", "1") == "1" and hasattr(_torch, "compile"):
        try:
            delattr(_torch, "compile")
        except Exception:
            def _identity_compile(module, *args, **kwargs):
                return module
            _torch.compile = _identity_compile  # type: ignore[attr-defined]
except Exception:
    pass
