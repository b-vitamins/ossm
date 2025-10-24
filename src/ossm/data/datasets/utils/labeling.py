from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch import Tensor

__all__ = ["encode_labels", "fit_label_encoder", "transform_labels"]


def encode_labels(labels: Iterable) -> Tensor:
    """Encode labels to a 1D ``torch.long`` tensor.

    Numeric labels are forwarded directly to PyTorch while non-numeric inputs use
    ``sklearn``'s :class:`LabelEncoder`. This mirrors the historical behaviour of
    the dataset utilities.
    """

    arr = np.asarray(labels)
    if arr.dtype.kind in {"i", "u"}:
        return torch.as_tensor(arr, dtype=torch.long)
    enc = LabelEncoder()
    y = enc.fit_transform(arr)
    return torch.as_tensor(y, dtype=torch.long)


def fit_label_encoder(train_labels: np.ndarray) -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(train_labels)
    return enc


def transform_labels(encoder: LabelEncoder, labels: np.ndarray) -> np.ndarray:
    return encoder.transform(labels).astype(np.int64)
