from __future__ import annotations

from typing import Dict

import torch
import torchcde


def _ensure_time_series(sample: Dict[str, torch.Tensor]) -> None:
    if "times" not in sample or "values" not in sample:
        raise KeyError("sample must contain 'times' and 'values'")
    times = sample["times"]
    values = sample["values"]
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if values.shape[0] != times.shape[0]:
        raise ValueError("times and values must share the leading dimension")
    if times.numel() < 2:
        raise ValueError("at least two time points are required")
    if not torch.all(times[1:] > times[:-1]):
        raise ValueError("times must be strictly increasing")


class ToCubicSplineCoeffs:
    """Annotate samples with torchcde-native natural cubic coefficients."""

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        _ensure_time_series(sample)
        times = sample["times"].to(dtype=torch.float32)
        values = sample["values"].to(dtype=torch.float32)

        if values.ndim == 1:
            values = values.unsqueeze(-1)

        coeffs = torchcde.natural_cubic_coeffs(values.unsqueeze(0), t=times)
        sample["coeffs"] = coeffs.squeeze(0)
        sample.setdefault("initial", values[0])
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

