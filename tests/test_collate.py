from __future__ import annotations
import torch
from ossm.data.datasets.collate import pad_collate, path_collate, coeff_collate


def test_pad_collate_shapes():
    b = [
        {
            "values": torch.randn(5, 3),
            "times": torch.linspace(0, 1, 5),
            "label": torch.tensor(0),
        },
        {
            "values": torch.randn(7, 3),
            "times": torch.linspace(0, 1, 7),
            "label": torch.tensor(1),
        },
    ]
    out = pad_collate(b)
    assert out["values"].shape == (2, 7, 3)
    assert out["mask"].dtype == torch.bool


def test_path_collate_shapes():
    b = [
        {"features": torch.randn(4, 16), "label": torch.tensor(0)},
        {"features": torch.randn(6, 16), "label": torch.tensor(1)},
    ]
    out = path_collate(b)
    assert out["features"].shape == (2, 6, 16)


def test_coeff_collate_shapes():
    b = [
        {
            "times": torch.linspace(0, 1, 6),
            "coeffs": torch.randn(5, 12),
            "label": torch.tensor(0),
        },
        {
            "times": torch.linspace(0, 1, 8),
            "coeffs": torch.randn(7, 12),
            "label": torch.tensor(1),
        },
    ]
    out = coeff_collate(b)
    assert out["times"].shape == (2, 8)
    assert out["coeffs"].shape == (2, 7, 12)
