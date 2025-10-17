from __future__ import annotations
import importlib
import pytest
import torch
from ossm.data.transforms.signature import ToWindowedLogSignature

ts_spec = importlib.util.find_spec("torchsignature")


@pytest.mark.skipif(ts_spec is None, reason="torchsignature not installed")
def test_windowed_logsig_smoke():
    t = torch.linspace(0, 1, 33)
    x = torch.randn(33, 3)
    sample = {"times": t, "values": x, "label": torch.tensor(0)}
    tfm = ToWindowedLogSignature(depth=2, steps=8, basis="lyndon")
    out = tfm(sample)
    assert "features" in out
    assert out["features"].dim() == 2
