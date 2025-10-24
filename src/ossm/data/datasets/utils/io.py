from __future__ import annotations

import importlib
import importlib.util
import os
import pickle
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

__all__ = [
    "ensure_uea_layout",
    "load_uea_numpy",
    "load_uea_tensors",
    "deduplicate_pairs",
]


def _arff_path(root: str, name: str, split: str) -> str:
    split = split.upper()
    return os.path.join(
        root, "raw", "UEA", "Multivariate_arff", name, f"{name}_{split}.arff"
    )


def _df_to_numpy(data_df: pd.DataFrame) -> np.ndarray:
    data_expand = data_df.map(lambda x: x.values).values
    return np.stack([np.vstack(x).T for x in data_expand]).astype(np.float32)


def _load_processed_split(root: str, name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset split from the processed LinOSS layout if available."""

    split = split.lower()
    base = os.path.join(root, "processed", "UEA", name)
    if not os.path.isdir(base):
        raise FileNotFoundError

    if split in {"train", "test", "val", "validation"}:
        if split == "validation":
            split = "val"
        x_file = os.path.join(base, f"X_{split}.pkl")
        y_file = os.path.join(base, f"y_{split}.pkl")
    elif split in {"all", "full"}:
        x_file = os.path.join(base, "data.pkl")
        y_file = os.path.join(base, "labels.pkl")
    else:
        raise ValueError(f"Unknown processed split '{split}' for dataset '{name}'.")

    if not os.path.exists(x_file) or not os.path.exists(y_file):
        raise FileNotFoundError

    with open(x_file, "rb") as f:
        X = pickle.load(f)
    with open(y_file, "rb") as f:
        y = pickle.load(f)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.argmax(axis=-1)
    return X, y


def load_uea_numpy(root: str, name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a UEA/UCR dataset split as NumPy arrays.

    The helper first tries the processed LinOSS layout which avoids the heavy
    dependency chain. Falling back to raw ARFF parsing requires NumPy, Pandas,
    and ``sktime`` to be available in the environment.
    """

    try:
        return _load_processed_split(root, name, split)
    except FileNotFoundError:
        pass

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


def load_uea_tensors(
    root: str,
    name: str,
    split: str,
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Torch-centric wrapper around :func:`load_uea_numpy`.

    The values tensor is materialised via ``torch.as_tensor`` to respect the
    requested ``device`` and ``dtype``. Labels are only converted when they are
    already numeric; string labels should be encoded separately via
    :func:`ossm.data.datasets.utils.labeling.encode_labels`.
    """

    values_np, labels_np = load_uea_numpy(root, name, split)
    values = torch.as_tensor(values_np, dtype=dtype, device=device)
    try:
        labels = torch.as_tensor(labels_np, device=device)
    except (TypeError, ValueError) as err:
        raise TypeError(
            "load_uea_tensors requires numeric labels. "
            "Use labeling.encode_labels before converting to tensors."
        ) from err
    return values, labels


def ensure_uea_layout(root: str) -> None:
    """Create the expected UEA folder layout under ``root`` if missing."""

    path = os.path.join(root, "raw", "UEA", "Multivariate_arff")
    os.makedirs(path, exist_ok=True)


def deduplicate_pairs(X: np.ndarray, y: np.ndarray):
    flat = X.reshape(X.shape[0], -1)
    _, idx = np.unique(flat, axis=0, return_index=True)
    return X[idx], y[idx]
