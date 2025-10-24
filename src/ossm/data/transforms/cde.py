from __future__ import annotations

from typing import cast

import importlib

import torch

from .compose import TimeSeriesSample


def _ensure_time_series(sample: TimeSeriesSample) -> tuple[torch.Tensor, torch.Tensor]:
    times = sample.get("times")
    values = sample.get("values")
    if times is None or values is None:
        raise KeyError("sample must contain 'times' and 'values'")
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if values.shape[0] != times.shape[0]:
        raise ValueError("times and values must share the leading dimension")
    if times.numel() < 2:
        raise ValueError("at least two time points are required")
    if not torch.all(times[1:] > times[:-1]):
        raise ValueError("times must be strictly increasing")
    return times, values


class ToCubicSplineCoeffs:
    """Annotate samples with torchcde-native natural cubic coefficients."""

    def __call__(self, sample: TimeSeriesSample) -> TimeSeriesSample:
        times, values = _ensure_time_series(sample)
        times = times.to(dtype=torch.float32)
        values = values.to(dtype=torch.float32)

        if values.ndim == 1:
            values = values.unsqueeze(-1)

        torchcde = _lazy_torchcde()
        coeffs = torchcde.natural_cubic_coeffs(values.unsqueeze(0), t=times)
        updated = sample.copy()
        updated["coeffs"] = coeffs.squeeze(0)
        if "initial" not in updated:
            updated["initial"] = values[0]
        return cast(TimeSeriesSample, updated)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


_TORCHCDE_MISSING = (
    "torchcde is required for Neural CDE support. Install it via "
    "`pip install ossm[cde]` or `pip install torchcde`."
)


def _lazy_torchcde():
    try:
        return importlib.import_module("torchcde")
    except ModuleNotFoundError as err:  # pragma: no cover - defensive
        raise RuntimeError(_TORCHCDE_MISSING) from err
