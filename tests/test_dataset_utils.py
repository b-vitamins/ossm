from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from ossm.data.datasets.utils import cache, io


def test_load_uea_numpy_prefers_processed(tmp_path: Path) -> None:
    root = tmp_path
    dataset_root = root / "processed" / "UEA" / "TestDataset"
    dataset_root.mkdir(parents=True)

    X_train = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)
    with open(dataset_root / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(dataset_root / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)

    X, y = io.load_uea_numpy(str(root), "TestDataset", "train")

    np.testing.assert_array_equal(X, X_train)
    np.testing.assert_array_equal(y, y_train)


def test_load_uea_numpy_requires_sktime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(io.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError) as excinfo:
        io.load_uea_numpy(str(tmp_path), "TestDataset", "train")

    assert str(excinfo.value) == (
        "sktime is required to parse ARFF files. Install with `pip install sktime`."
    )


def test_load_uea_tensors_respects_dtype(tmp_path: Path) -> None:
    dataset_root = tmp_path / "processed" / "UEA" / "TensorDataset"
    dataset_root.mkdir(parents=True)

    X_train = np.array([[[1.0], [2.0]]], dtype=np.float32)
    y_train = np.array([5], dtype=np.int64)
    with open(dataset_root / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(dataset_root / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)

    values, labels = io.load_uea_tensors(str(tmp_path), "TensorDataset", "train", dtype=torch.float64)

    assert values.dtype == torch.float64
    assert labels.dtype == torch.int64


def test_maybe_save_cache_logs_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    captured: Dict[str, object] = {"called": False}

    def failing_save(*args, **kwargs):
        captured["called"] = True
        raise RuntimeError("serialisation exploded")

    monkeypatch.setattr(torch, "save", failing_save)

    logger = logging.getLogger("ossm.tests.cache")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    cache_dir = tmp_path / "cache"
    cache.maybe_save_cache(str(cache_dir), "key", {"foo": torch.tensor([1.0])}, logger=logger)

    assert captured["called"] is True
    assert "serialisation exploded" in caplog.text
    assert not any(cache_dir.iterdir())
