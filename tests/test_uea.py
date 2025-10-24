from __future__ import annotations
import numpy as np
import pytest
import torch

from ossm.data.datasets.uea import UEA
import ossm.data.datasets.utils as utils


def _fake_loader(root, name, split):
    rs = np.random.RandomState(0 if split.lower() == "train" else 1)
    N = 8 if split.lower() == "train" else 4
    T, C = 33, 3
    X = rs.randn(N, T, C).astype(np.float32)
    y = np.array([0, 1] * (N // 2) + ([0] if N % 2 else []))
    return X, y


_TRAIN_VALUES = np.stack(
    [np.full((5, 2), fill_value=float(i), dtype=np.float32) for i in range(4)], axis=0
)
_TRAIN_TIMES = np.stack(
    [np.linspace(0.0, 2.0, 5, dtype=np.float32) for _ in range(4)], axis=0
)
_TRAIN_LABELS = np.array([0, 1, 2, 3], dtype=np.int64)

_TEST_VALUES = np.stack(
    [
        np.full((5, 2), fill_value=0.0, dtype=np.float32),
        np.full((5, 2), fill_value=10.0, dtype=np.float32),
        np.full((5, 2), fill_value=2.0, dtype=np.float32),
    ],
    axis=0,
)
_TEST_TIMES = np.stack(
    [
        np.linspace(0.0, 2.0, 5, dtype=np.float32),
        np.linspace(0.0, 3.0, 5, dtype=np.float32),
        np.linspace(0.0, 2.0, 5, dtype=np.float32),
    ],
    axis=0,
)
_TEST_LABELS = np.array([0, 4, 2], dtype=np.int64)


def _loader_with_times(root, name, split):
    if split.lower() == "train":
        return _TRAIN_TIMES, _TRAIN_VALUES, _TRAIN_LABELS
    if split.lower() == "test":
        return _TEST_TIMES, _TEST_VALUES, _TEST_LABELS
    raise ValueError(f"Unsupported split {split}")


def test_uea_raw_monkeypatch(monkeypatch):
    monkeypatch.setattr(utils, "load_uea_numpy", _fake_loader)
    ds = UEA(root=".", name="Dummy", split="train", view="raw")
    item = ds[0]
    assert set(item.keys()) == {"times", "values", "label"}
    assert item["values"].shape[-1] == 3
    assert item["times"].ndim == 1


@pytest.mark.parametrize("view", ["raw", "coeff", "path"])
def test_uea_views_smoke(monkeypatch, view):
    monkeypatch.setattr(utils, "load_uea_numpy", _fake_loader)
    kwargs = {}
    if view == "path":
        kwargs.update(dict(depth=2, steps=8))
    ds = UEA(root=".", name="Dummy", split="train", view=view, **kwargs)
    item = ds[0]
    assert "label" in item


def test_uea_all_split(monkeypatch):
    monkeypatch.setattr(utils, "load_uea_numpy", _fake_loader)
    ds_all = UEA(root=".", name="Dummy", split="all", view="raw")
    assert len(ds_all) == 12

    ds_pair = UEA(root=".", name="Dummy", split=("train", "test"), view="raw")
    assert len(ds_pair) == 12


def test_uea_deduplicate_across_splits(monkeypatch):
    monkeypatch.setattr(utils, "load_uea_numpy", _loader_with_times)

    baseline = UEA(root=".", name="Dummy", split="all", view="raw")
    assert len(baseline) == 7

    deduped = UEA(
        root=".",
        name="Dummy",
        split="all",
        view="raw",
        deduplicate=True,
        record_source=True,
    )
    assert len(deduped) == 5

    inv = {v: k for k, v in deduped.source_split_encoding.items()}
    sources = set()
    for i in range(len(deduped)):
        item = deduped[i]
        split = inv[int(item["source_split"].item())]
        index = int(item["source_index"].item())
        sources.add((split, index))
    assert ("test", 1) in sources


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_uea_deduplicate_cuda(monkeypatch):
    monkeypatch.setattr(utils, "load_uea_numpy", _loader_with_times)

    ds = UEA(
        root=".",
        name="Dummy",
        split="all",
        view="raw",
        deduplicate=True,
        device="cuda",
    )

    assert len(ds) == 5
    assert ds.times.device.type == "cuda"
    assert ds.values.device.type == "cuda"
    assert ds.labels.device.type == "cuda"

    sample = ds[0]
    assert sample["times"].device.type == "cuda"
    assert sample["values"].device.type == "cuda"
    assert sample["label"].device.type == "cuda"


def test_uea_resample_partition(monkeypatch):
    monkeypatch.setattr(utils, "load_uea_numpy", _fake_loader)
    resample_cfg = {"train": 6, "val": 3, "test": 3}
    common_kwargs = dict(
        root=".",
        name="Dummy",
        view="raw",
        resample=resample_cfg,
        resample_seed=7,
        record_source=True,
    )
    train_ds = UEA(split="train", **common_kwargs)
    val_ds = UEA(split="val", **common_kwargs)
    test_ds = UEA(split="test", **common_kwargs)

    assert len(train_ds) == 6
    assert len(val_ds) == 3
    assert len(test_ds) == 3

    def collect(ds):
        inv_map = {v: k for k, v in ds.source_split_encoding.items()}
        seen = set()
        for i in range(len(ds)):
            item = ds[i]
            split = inv_map[int(item["source_split"].item())]
            index = int(item["source_index"].item())
            seen.add((split, index))
        return seen

    train_items = collect(train_ds)
    val_items = collect(val_ds)
    test_items = collect(test_ds)

    assert train_items.isdisjoint(val_items)
    assert train_items.isdisjoint(test_items)
    assert val_items.isdisjoint(test_items)

    full = collect(UEA(root=".", name="Dummy", split="all", view="raw", record_source=True))
    assert train_items | val_items | test_items == full


def test_uea_record_grid_and_source(monkeypatch):
    monkeypatch.setattr(utils, "load_uea_numpy", _loader_with_times)
    ds = UEA(
        root=".",
        name="Dummy",
        split="train",
        view="coeff",
        record_grid=True,
        record_source=True,
    )
    sample = ds[0]
    assert {"times", "grid", "coeffs", "label", "values", "source_index", "source_split"} <= set(sample.keys())

    original = torch.from_numpy(_TRAIN_TIMES[0])
    assert torch.allclose(sample["grid"], original)
    assert not torch.allclose(sample["grid"], sample["times"])
    assert sample["source_index"].item() == 0
    assert sample["source_split"].item() == ds.source_split_encoding["train"]
