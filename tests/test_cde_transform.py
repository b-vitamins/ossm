from __future__ import annotations

import torch
import torchcde

from ossm.data.transforms.cde import ToCubicSplineCoeffs


def test_coefficients_match_torchcde_output() -> None:
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
