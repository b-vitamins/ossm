from __future__ import annotations

from typing import Dict

import torch


def _backward_hermite_coefficients(times: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Compute backward Hermite cubic coefficients matching Diffrax.

    Args:
        times: Monotonically increasing time grid of shape ``(T,)``.
        values: Observations of shape ``(T, C)`` (or more generally ``(T, ...)``).

    Returns:
        Tensor of shape ``(T-1, C * 4)`` containing the ``d, c, b, a`` blocks
        concatenated along the channel dimension for each interval.
    """

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if values.shape[0] != times.shape[0]:
        raise ValueError("times and values must share the leading dimension")
    if times.numel() < 2:
        raise ValueError("at least two time points are required")

    # Promote to at least two dimensions so concatenation along the channel axis
    # is well-defined for univariate series.
    values_ = values.unsqueeze(-1) if values.ndim == 1 else values

    deltas_t = times.diff().to(values_.dtype)
    if torch.any(deltas_t <= 0):
        raise ValueError("times must be strictly increasing")

    view_shape = (deltas_t.shape[0],) + (1,) * (values_.ndim - 1)
    deltas_t_view = deltas_t.view(view_shape)

    forward_deriv = (values_[1:] - values_[:-1]) / deltas_t_view

    deriv0 = torch.empty_like(values_)
    deriv0[0] = forward_deriv[0]
    deriv0[1:] = forward_deriv

    delta_deriv = forward_deriv - deriv0[:-1]
    deltas_t_sq = deltas_t_view.square()

    d = -delta_deriv / deltas_t_sq
    c = 2.0 * delta_deriv / deltas_t_view
    b = deriv0[:-1]
    a = values_[:-1]

    coeffs = torch.cat([d, c, b, a], dim=-1)
    if values.ndim == 1:
        return coeffs.squeeze(-1)
    return coeffs


class ToCubicSplineCoeffs:
    """Annotate samples with Hermite cubic coefficients computed in torch."""

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        coeffs = _backward_hermite_coefficients(sample["times"], sample["values"])
        sample["coeffs"] = coeffs
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

