from __future__ import annotations

import pytest
import torch

from ossm.data.transforms.cde import ToCubicSplineCoeffs


def _manual_backward_coeffs(times: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    values_ = values.unsqueeze(-1) if values.ndim == 1 else values
    dt = times.diff()
    view_shape = (dt.shape[0],) + (1,) * (values_.ndim - 1)
    dt_view = dt.view(view_shape)
    forward_deriv = (values_[1:] - values_[:-1]) / dt_view

    deriv0 = torch.empty_like(values_)
    deriv0[0] = forward_deriv[0]
    deriv0[1:] = forward_deriv

    delta_deriv = forward_deriv - deriv0[:-1]

    d = -delta_deriv / dt_view.square()
    c = 2.0 * delta_deriv / dt_view
    b = deriv0[:-1]
    a = values_[:-1]

    coeffs = torch.cat([d, c, b, a], dim=-1)
    if values.ndim == 1:
        return coeffs.squeeze(-1)
    return coeffs


def test_hermite_coeffs_match_manual_formula() -> None:
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

    expected = _manual_backward_coeffs(times, values)

    assert "coeffs" in out
    assert out["coeffs"].shape == expected.shape
    torch.testing.assert_close(out["coeffs"], expected)


def test_coeffs_align_with_diffrax_when_available() -> None:
    diffrax = pytest.importorskip("diffrax", reason="Diffrax is required for parity check")
    import jax.numpy as jnp
    import numpy as np

    times = torch.linspace(0, 1, 7, dtype=torch.float32)
    values = torch.tensor(
        [
            [0.0, 1.0],
            [0.2, 1.5],
            [0.4, 1.4],
            [0.8, 0.9],
            [1.4, 0.5],
            [1.6, 0.0],
            [1.8, -0.2],
        ],
        dtype=torch.float32,
    )

    expected = diffrax.backward_hermite_coefficients(jnp.array(times.numpy()), jnp.array(values.numpy()))
    expected_concat = np.concatenate([np.asarray(block) for block in expected], axis=-1)

    tfm = ToCubicSplineCoeffs()
    actual = tfm({"times": times, "values": values, "label": torch.tensor(0)})["coeffs"]

    torch.testing.assert_close(actual, torch.from_numpy(expected_concat).to(actual))
