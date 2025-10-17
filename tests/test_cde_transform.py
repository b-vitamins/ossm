from __future__ import annotations
import importlib
import pytest
import torch
from ossm.data.transforms.cde import ToCubicSplineCoeffs

cde_spec = importlib.util.find_spec("torchcde")


@pytest.mark.skipif(cde_spec is None, reason="torchcde not installed")
def test_hermite_coeffs_smoke():
    t = torch.linspace(0, 1, 17)
    x = torch.randn(17, 4)
    sample = {"times": t, "values": x, "label": torch.tensor(0)}
    tfm = ToCubicSplineCoeffs()
    out = tfm(sample)
    assert "coeffs" in out
    assert out["coeffs"].shape[0] == x.size(0) - 1
