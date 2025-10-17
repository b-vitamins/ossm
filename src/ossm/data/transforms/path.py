from __future__ import annotations
import torch


class AddTime:
    def __init__(self, T: float = 1.0):
        self.T = T

    def __call__(self, sample):
        x = sample["values"]
        if "times" not in sample or sample["times"].numel() == 0:
            t = torch.linspace(0.0, self.T, x.size(0), device=x.device, dtype=x.dtype)
        else:
            t = sample["times"].to(x.dtype)
            if t.dim() > 1:
                t = t.squeeze(0)
        sample["times"] = t
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(T={self.T})"


class NormalizeTime:
    def __init__(self, T: float = 1.0):
        self.T = T

    def __call__(self, sample):
        t = sample["times"]
        t0, t1 = t[..., :1], t[..., -1:]
        sample["times"] = self.T * (t - t0) / (t1 - t0 + 1e-12)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(T={self.T})"


class SegmentFixedLength:
    def __init__(self, steps: int):
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.steps = int(steps)

    def __call__(self, sample):
        x = sample["values"]
        t = sample["times"]
        T = x.size(0)
        chunks = T // self.steps + int(T % self.steps != 0)
        pad = chunks * self.steps - T
        if pad:
            x = torch.cat([x, x[-1:].expand(pad, -1)], dim=0)
            t = torch.cat([t, t[-1:].expand(pad)], dim=0)
        sample["values"] = x.view(chunks, self.steps, x.size(-1))
        sample["times"] = t.view(chunks, self.steps)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(steps={self.steps})"
