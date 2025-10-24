from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from ossm.data.datasets import cache as cache_utils
from ossm.data.datasets import io as io_utils


@pytest.fixture
def processed_root(tmp_path: Path) -> Path:
    base = tmp_path / "processed" / "UEA" / "Dummy"
    base.mkdir(parents=True, exist_ok=True)

    X = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    y = np.array([1, 0], dtype=np.int64)

    with (base / "X_train.pkl").open("wb") as handle:
        pickle.dump(X, handle)
    with (base / "y_train.pkl").open("wb") as handle:
        pickle.dump(y, handle)

    return tmp_path


def test_load_uea_numpy_prefers_processed(monkeypatch, processed_root: Path) -> None:
    def _fail(*_args, **_kwargs):  # pragma: no cover - ensures branch isolation
        raise AssertionError("raw loader should not be invoked when processed data exist")

    monkeypatch.setattr(io_utils.importlib.util, "find_spec", _fail)

    values, labels = io_utils.load_uea_numpy(processed_root, "Dummy", "train")
    assert values.shape == (2, 3, 2)
    assert labels.tolist() == [1, 0]


def test_load_uea_numpy_requires_sktime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(io_utils.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match="sktime is required"):
        io_utils.load_uea_numpy(tmp_path, "Missing", "train")


def test_load_uea_tensors_returns_torch(processed_root: Path) -> None:
    values, labels = io_utils.load_uea_tensors(processed_root, "Dummy", "train")
    assert isinstance(values, torch.Tensor) and values.dtype == torch.float32
    assert isinstance(labels, torch.Tensor)


def test_maybe_save_cache_success(tmp_path: Path) -> None:
    payload = {"weights": torch.ones(2)}
    cache_utils.maybe_save_cache(tmp_path / "cache", "demo", payload)

    saved = torch.load(tmp_path / "cache" / "demo.pt")
    assert torch.equal(saved["weights"], payload["weights"])


def test_maybe_save_cache_directory_failure(tmp_path: Path, caplog) -> None:
    block = tmp_path / "cache"
    block.write_text("busy")

    with caplog.at_level(logging.DEBUG):
        cache_utils.maybe_save_cache(block, "demo", {"x": 1})

    assert "Failed to create cache directory" in caplog.text
    assert not (tmp_path / "cache" / "demo.pt").exists()


def test_maybe_save_cache_serialization_failure(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        cache_utils.maybe_save_cache(tmp_path / "cache", "bad", {"fn": lambda: None})

    assert "Failed to save cache file" in caplog.text
    assert not (tmp_path / "cache" / "bad.pt").exists()
