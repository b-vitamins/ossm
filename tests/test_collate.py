from __future__ import annotations

import pytest
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
            "coeffs": torch.randn(5, 3, 4),
            "initial": torch.randn(3),
            "label": torch.tensor(0),
        },
        {
            "times": torch.linspace(0, 1, 8),
            "coeffs": torch.randn(7, 3, 4),
            "initial": torch.randn(3),
            "label": torch.tensor(1),
        },
    ]
    out = coeff_collate(b)
    assert out["times"].shape == (2, 8)
    assert out["coeffs"].shape == (2, 7, 3, 4)
    assert out["initial"].shape == (2, 3)
    assert out["mask"].dtype == torch.bool
    assert out["mask"].shape == (2, 8)


def test_coeff_collate_flat_coeffs():
    b = [
        {
            "times": torch.linspace(0, 1, 4),
            "coeffs": torch.randn(3, 12),
            "initial": torch.randn(3),
            "label": torch.tensor(0),
        },
        {
            "times": torch.linspace(0, 1, 5),
            "coeffs": torch.randn(4, 12),
            "initial": torch.randn(3),
            "label": torch.tensor(1),
        },
    ]
    out = coeff_collate(b)
    assert out["times"].shape == (2, 5)
    assert out["coeffs"].shape == (2, 4, 12)
    assert out["initial"].shape == (2, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pad_collate_preserves_device_cuda():
    device = torch.device("cuda")
    b = [
        {
            "values": torch.randn(5, 3, device=device),
            "times": torch.linspace(0, 1, 5, device=device),
            "label": torch.tensor(0, device=device),
        },
        {
            "values": torch.randn(7, 3, device=device),
            "times": torch.linspace(0, 1, 7, device=device),
            "label": torch.tensor(1, device=device),
        },
    ]
    out = pad_collate(b)
    assert out["values"].device == device
    assert out["times"].device == device
    assert out["mask"].device == device
    assert out["label"].device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_path_collate_time_grid_device_cuda():
    device = torch.device("cuda")
    b = [
        {
            "features": torch.randn(4, 16, device=device, dtype=torch.float64),
            "label": torch.tensor(0, device=device),
        },
        {
            "features": torch.randn(6, 16, device=device, dtype=torch.float64),
            "label": torch.tensor(1, device=device),
        },
    ]
    out = path_collate(b)
    assert out["logsig"].device == device
    assert out["times"].device == device
    assert out["times"].dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_coeff_collate_mask_device_cuda():
    device = torch.device("cuda")
    b = [
        {
            "times": torch.linspace(0, 1, 6, device=device),
            "coeffs": torch.randn(5, 3, 4, device=device),
            "initial": torch.randn(3, device=device),
            "label": torch.tensor(0, device=device),
        },
        {
            "times": torch.linspace(0, 1, 8, device=device),
            "coeffs": torch.randn(7, 3, 4, device=device),
            "initial": torch.randn(3, device=device),
            "label": torch.tensor(1, device=device),
        },
    ]
    out = coeff_collate(b)
    assert out["times"].device == device
    assert out["coeffs"].device == device
    assert out["initial"].device == device
    assert out["mask"].device == device
