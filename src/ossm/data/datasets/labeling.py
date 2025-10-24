from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch import Tensor

__all__ = [
    "encode_labels",
    "fit_label_encoder",
    "transform_labels",
]

def _ensure_numpy(labels: Iterable) -> np.ndarray:
    if isinstance(labels, np.ndarray):
        return labels
    return np.asarray(list(labels))


def encode_labels(labels: Iterable) -> Tensor:
    """Encode labels to a 1D ``torch.long`` tensor."""

    arr = _ensure_numpy(labels)
    if arr.dtype.kind in {"i", "u"}:
        return torch.as_tensor(arr, dtype=torch.long)
    encoder = LabelEncoder()
    y = encoder.fit_transform(arr)
    return torch.as_tensor(y, dtype=torch.long)


def fit_label_encoder(train_labels: Iterable) -> LabelEncoder:
    """Fit a :class:`~sklearn.preprocessing.LabelEncoder` on ``train_labels``."""

    encoder = LabelEncoder()
    encoder.fit(_ensure_numpy(train_labels))
    return encoder


def transform_labels(encoder: LabelEncoder, labels: Iterable) -> np.ndarray:
    """Transform ``labels`` using a fitted encoder into ``int64`` NumPy array."""

    transformed = encoder.transform(_ensure_numpy(labels))
    return transformed.astype(np.int64)
