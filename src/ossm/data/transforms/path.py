from __future__ import annotations

from typing import cast

import torch

from .compose import TimeSeriesSample


class AddTime:
    def __init__(self, T: float = 1.0) -> None:
        self.T = float(T)

    def __call__(self, sample: TimeSeriesSample) -> TimeSeriesSample:
        values = sample.get("values")
        if values is None:
            raise KeyError("sample must contain 'values'")

        times = sample.get("times")
        if times is None or times.numel() == 0:
            new_times = torch.linspace(
                0.0, self.T, values.size(0), device=values.device, dtype=values.dtype
            )
        else:
            new_times = times.to(values.dtype)
            if new_times.dim() > 1:
                new_times = new_times.squeeze(0)

        updated = sample.copy()
        updated["times"] = new_times
        return cast(TimeSeriesSample, updated)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(T={self.T})"


class NormalizeTime:
    def __init__(self, T: float = 1.0) -> None:
        self.T = float(T)

    def __call__(self, sample: TimeSeriesSample) -> TimeSeriesSample:
        times = sample.get("times")
        if times is None:
            raise KeyError("sample must contain 'times'")
        start, end = times[..., :1], times[..., -1:]
        normalized = self.T * (times - start) / (end - start + 1e-12)

        updated = sample.copy()
        updated["times"] = normalized
        return cast(TimeSeriesSample, updated)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(T={self.T})"


class SegmentFixedLength:
    def __init__(self, steps: int) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.steps = int(steps)

    def __call__(self, sample: TimeSeriesSample) -> TimeSeriesSample:
        values = sample.get("values")
        times = sample.get("times")
        if values is None or times is None:
            raise KeyError("sample must contain 'values' and 'times'")
        length = values.size(0)
        chunks = length // self.steps + int(length % self.steps != 0)
        pad = chunks * self.steps - length
        if pad:
            values = torch.cat([values, values[-1:].expand(pad, -1)], dim=0)
            times = torch.cat([times, times[-1:].expand(pad)], dim=0)

        segmented_values = values.view(chunks, self.steps, values.size(-1))
        segmented_times = times.view(chunks, self.steps)

        updated = sample.copy()
        updated["values"] = segmented_values
        updated["times"] = segmented_times
        return cast(TimeSeriesSample, updated)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(steps={self.steps})"
