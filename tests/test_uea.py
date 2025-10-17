from __future__ import annotations
import numpy as np
import pytest

from ossm.data.datasets.uea import UEA
import ossm.data.datasets.utils as utils


def _fake_loader(root, name, split):
    rs = np.random.RandomState(0 if split.lower() == "train" else 1)
    N = 8 if split.lower() == "train" else 4
    T, C = 33, 3
    X = rs.randn(N, T, C).astype(np.float32)
    y = np.array([0, 1] * (N // 2) + ([0] if N % 2 else []))
    return X, y


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
