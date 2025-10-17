from __future__ import annotations
import os
import warnings
import importlib
import importlib.util
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch import Tensor


def _arff_path(root: str, name: str, split: str) -> str:
    split = split.upper()
    return os.path.join(
        root, "raw", "UEA", "Multivariate_arff", name, f"{name}_{split}.arff"
    )


def _df_to_numpy(data_df: pd.DataFrame) -> np.ndarray:
    data_expand = data_df.map(lambda x: x.values).values
    return np.stack([np.vstack(x).T for x in data_expand]).astype(np.float32)


def load_uea_numpy(root: str, name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    sktime_spec = importlib.util.find_spec("sktime")
    if sktime_spec is None:
        raise RuntimeError(
            "sktime is required to parse ARFF files. Install with `pip install sktime`."
        )
    from sktime.datasets import (  # pyright: ignore[reportMissingImports]
        load_from_arff_to_dataframe,
    )

    file_path = _arff_path(root, name, split)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        data_df, labels = load_from_arff_to_dataframe(file_path)
    X = _df_to_numpy(data_df)
    y = np.asarray(labels)
    return X, y


def ensure_uea_layout(root: str) -> None:
    """Create the expected UEA folder layout under `root` if missing."""
    path = os.path.join(root, "raw", "UEA", "Multivariate_arff")
    os.makedirs(path, exist_ok=True)


def encode_labels(labels) -> Tensor:
    """Encode labels to a 1D torch.long tensor.

    If numeric labels are provided, they're returned as-is (long). Otherwise,
    a sklearn LabelEncoder is fit on the provided labels.
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


def deduplicate_pairs(X: np.ndarray, y: np.ndarray):
    flat = X.reshape(X.shape[0], -1)
    _, idx = np.unique(flat, axis=0, return_index=True)
    return X[idx], y[idx]


def cache_key(*parts: str) -> str:
    return "_".join(parts).replace(os.sep, "-")


def maybe_save_cache(dirpath: str, key: str, obj: dict) -> None:
    try:
        os.makedirs(dirpath, exist_ok=True)
        path = os.path.join(dirpath, f"{key}.pt")
        import torch

        torch.save(obj, path)
    except Exception:
        # Best-effort cache.
        pass
