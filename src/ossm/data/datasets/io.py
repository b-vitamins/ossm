from __future__ import annotations

import importlib.util
import os
import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - import-time guard
    import pandas as pd

__all__ = [
    "ensure_uea_layout",
    "load_uea_numpy",
    "load_uea_tensors",
    "deduplicate_pairs",
]


def _arff_path(root: str | os.PathLike[str], name: str, split: str) -> Path:
    """Return the path to a raw ARFF file for *name*/*split*."""

    root_path = Path(root)
    return (
        root_path
        / "raw"
        / "UEA"
        / "Multivariate_arff"
        / name
        / f"{name}_{split.upper()}.arff"
    )


def _df_to_numpy(data_df: "pd.DataFrame") -> np.ndarray:
    """Convert a sktime dataframe to a (N, T, C) float32 NumPy array."""

    data_expand = data_df.map(lambda x: x.values).values
    return np.stack([np.vstack(x).T for x in data_expand]).astype(np.float32)


def _load_processed_split(root: str | os.PathLike[str], name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset split from the processed LinOSS layout if available."""

    split_key = split.lower()
    base = Path(root) / "processed" / "UEA" / name
    if not base.is_dir():
        raise FileNotFoundError

    if split_key in {"train", "test", "val", "validation"}:
        if split_key == "validation":
            split_key = "val"
        x_file = base / f"X_{split_key}.pkl"
        y_file = base / f"y_{split_key}.pkl"
    elif split_key in {"all", "full"}:
        x_file = base / "data.pkl"
        y_file = base / "labels.pkl"
    else:
        raise ValueError(f"Unknown processed split '{split}' for dataset '{name}'.")

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError

    with x_file.open("rb") as f:
        X = pickle.load(f)
    with y_file.open("rb") as f:
        y = pickle.load(f)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.argmax(axis=-1)
    return X, y


def ensure_uea_layout(root: str | os.PathLike[str]) -> None:
    """Create the expected UEA folder layout under *root* if missing."""

    path = Path(root) / "raw" / "UEA" / "Multivariate_arff"
    path.mkdir(parents=True, exist_ok=True)


def load_uea_numpy(root: str | os.PathLike[str], name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a UEA dataset split as NumPy arrays.

    Parameters
    ----------
    root:
        Dataset root directory containing the ``processed``/``raw`` layout.
    name:
        Name of the dataset inside the archive.
    split:
        Dataset split. Processed layouts support ``train``, ``test``, ``val``/``validation``
        and ``all``/``full``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(values, labels)`` as ``float32`` values and integer labels.

    Notes
    -----
    * If a processed pickle layout is present, no optional dependencies are required.
    * When falling back to raw ARFF files, :mod:`pandas` and ``sktime`` must be
      installed to parse the inputs.
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
    import pandas as pd  # Imported lazily to avoid hard dependency.

    file_path = _arff_path(root, name, split)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        data_df, labels = load_from_arff_to_dataframe(str(file_path))
    X = _df_to_numpy(data_df)
    y = np.asarray(labels)
    return X, y


def load_uea_tensors(root: str | os.PathLike[str], name: str, split: str) -> Tuple[Tensor, Tensor]:
    """Load a UEA dataset split directly into :class:`torch.Tensor` objects."""

    values, labels = load_uea_numpy(root, name, split)
    label_array = np.asarray(labels)
    if label_array.dtype.kind in {"i", "u"}:
        label_tensor = torch.as_tensor(label_array, dtype=torch.long)
    else:  # Fall back to sklearn encoder to handle string labels.
        from .labeling import encode_labels as _encode_labels

        label_tensor = _encode_labels(label_array)
    value_tensor = torch.as_tensor(values, dtype=torch.float32)
    return value_tensor, label_tensor


def deduplicate_pairs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove duplicate samples while keeping the first occurrence."""

    flat = X.reshape(X.shape[0], -1)
    _, idx = np.unique(flat, axis=0, return_index=True)
    return X[idx], y[idx]
