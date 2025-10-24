from __future__ import annotations

import importlib

import pytest
import torch

from ossm.data.transforms.cde import ToCubicSplineCoeffs


def test_coefficients_match_torchcde_output() -> None:
    torchcde = pytest.importorskip("torchcde")
    times = torch.tensor([0.0, 0.5, 1.0, 1.75, 2.5], dtype=torch.float32)
    values = torch.tensor(
        [
            [0.0, 1.0, -1.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [4.0, -1.0, 3.0],
            [6.0, -2.0, 5.0],
        ],
        dtype=torch.float32,
    )
    sample = {"times": times, "values": values.clone(), "label": torch.tensor(0)}

    tfm = ToCubicSplineCoeffs()
    out = tfm(sample)

    expected = torchcde.natural_cubic_coeffs(values.unsqueeze(0), t=times).squeeze(0)

    assert "coeffs" in out
    assert out["coeffs"].shape == expected.shape
    torch.testing.assert_close(out["coeffs"], expected)


def test_to_cubic_spline_missing_torchcde(monkeypatch) -> None:
    from ossm.data.transforms import cde as cde_mod

    original_import_module = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "torchcde":
            raise ModuleNotFoundError("No module named 'torchcde'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(cde_mod.importlib, "import_module", fake_import)

    tfm = ToCubicSplineCoeffs()
    times = torch.tensor([0.0, 0.5], dtype=torch.float32)
    values = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    sample = {"times": times, "values": values}

    with pytest.raises(RuntimeError, match="torchcde is required for Neural CDE support"):
        tfm(sample)
