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
    tfm = ToWindowedLogSignature(depth=2, steps=8, basis="hall")
    out = tfm(sample)
    assert "features" in out
    assert out["features"].dim() == 2


@pytest.mark.skipif(ts_spec is None, reason="torchsignature not installed")
def test_hall_basis_matches_manual():
    t = torch.linspace(0, 1, 29)
    values = torch.randn(29, 4)
    sample = {"times": t, "values": values.clone(), "label": torch.tensor(1)}
    steps = 7
    depth = 2

    tfm = ToWindowedLogSignature(depth=depth, steps=steps, basis="hall")
    out = tfm(sample)

    from torchsignature.signatures import signature
    from torchsignature.functional import log as log_tensor

    def hall_coords(path: torch.Tensor) -> torch.Tensor:
        sig_lvls = signature(path, depth=depth, stream=False, flatten=False)
        log_lvls = log_tensor(sig_lvls)
        lvl1 = log_lvls[0].reshape(-1)
        lvl2 = log_lvls[1]
        comms = []
        C = lvl2.size(0)
        for i in range(C):
            for j in range(i + 1, C):
                comms.append(0.5 * (lvl2[i, j] - lvl2[j, i]))
        lie2 = torch.stack(comms) if comms else torch.empty(0, dtype=lvl1.dtype, device=path.device)
        scalar = torch.zeros(1, dtype=lvl1.dtype, device=path.device)
        return torch.cat([scalar, lvl1, lie2])

    dtype = values.dtype
    device = values.device
    augmented = torch.cat([torch.zeros(1, values.size(-1), dtype=dtype, device=device), values], dim=0)
    window = min(steps, augmented.size(0))
    remainder = augmented.size(0) % window
    bulk_len = augmented.size(0) - remainder

    feats = []
    if bulk_len:
        blocks = augmented[:bulk_len].reshape(-1, window, values.size(-1))
        prev_last = torch.zeros(blocks.size(0), 1, values.size(-1), dtype=dtype, device=device)
        if blocks.size(0) > 1:
            prev_last[1:, 0, :] = blocks[:-1, -1, :]
        blocks = torch.cat([prev_last, blocks], dim=1)
        for block in blocks:
            feats.append(hall_coords(block))

    if remainder:
        base = torch.zeros(1, values.size(-1), dtype=dtype, device=device)
        tail = torch.cat([base, augmented[-(remainder + 1) :]], dim=0)
        feats.append(hall_coords(tail))

    assert feats  # sanity: at least one window
    expected = torch.stack(feats)
    assert expected.shape == out["features"].shape
    assert torch.allclose(expected, out["features"], atol=1e-5)
