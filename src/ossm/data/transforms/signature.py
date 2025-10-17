from __future__ import annotations

import importlib
import importlib.util
from typing import List, Tuple, cast

import torch


Tensor = torch.Tensor


def _windowed_logsig_fn():
    mod = importlib.import_module("torchsignature.windowed")
    fn = getattr(mod, "windowed_logsignature", None)
    if fn is not None:
        return fn
    top = importlib.import_module("torchsignature")
    fn = getattr(top, "windowed_logsignature", None)
    if fn is None:
        raise RuntimeError(
            "torchsignature.windowed_logsignature not found; update torchsignature."
        )
    return fn


def _require_torchsignature():
    if importlib.util.find_spec("torchsignature") is None:
        raise RuntimeError(
            "torchsignature is required for log-signature features. Install it first."
        )
    return _windowed_logsig_fn()


def _segment_windows(x: Tensor, steps: int) -> Tuple[Tensor, Tensor | None, bool]:
    """Replicate LINOSS window segmentation with basepoint carry-over."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.dim() == 3:
        squeeze = False
    else:
        raise ValueError(f"x must be (T,C) or (B,T,C); got {tuple(x.shape)}")

    B, T, C = x.shape
    dtype, device = x.dtype, x.device
    data = torch.cat([torch.zeros(B, 1, C, dtype=dtype, device=device), x], dim=1)

    steps = min(int(steps), data.size(1))
    remainder = data.size(1) % steps
    bulk_len = data.size(1) - remainder

    if bulk_len:
        blocks = data[:, :bulk_len, :].reshape(B, -1, steps, C)
        prev_last = torch.zeros(B, blocks.size(1), 1, C, dtype=dtype, device=device)
        if blocks.size(1) > 1:
            prev_last[:, 1:, 0, :] = blocks[:, :-1, -1, :]
        blocks = torch.cat([prev_last, blocks], dim=2)
    else:
        blocks = torch.zeros(B, 0, steps + 1, C, dtype=dtype, device=device)

    tail = None
    if remainder:
        tail_vals = data[:, -(remainder) - 1 :, :]
        base = torch.zeros(B, 1, C, dtype=dtype, device=device)
        if B > 1:
            base[1:, 0, :] = tail_vals[:-1, -1, :]
        tail = torch.cat([base, tail_vals], dim=1)

    return blocks, tail, squeeze


def _hall_coordinates(path: Tensor) -> Tensor:
    from torchsignature.signatures import signature
    from torchsignature.functional import log as log_tensor

    sig_lvls = cast(List[Tensor], signature(path, depth=2, stream=False, flatten=False))
    log_lvls = log_tensor(sig_lvls)

    lvl1 = log_lvls[0]
    lvl2 = log_lvls[1]

    linear = lvl1.reshape(-1)
    comms = []
    C = lvl2.size(0)
    for i in range(C):
        for j in range(i + 1, C):
            comms.append(0.5 * (lvl2[i, j] - lvl2[j, i]))
    if comms:
        lie2 = torch.stack(comms)
        return torch.cat([torch.zeros(1, dtype=linear.dtype, device=linear.device), linear, lie2])
    return torch.cat([torch.zeros(1, dtype=linear.dtype, device=linear.device), linear])


def _windowed_hall_logsignature(x: Tensor, steps: int) -> Tensor:
    blocks, tail, squeeze = _segment_windows(x, steps)

    B = blocks.size(0)

    outs = []
    for b in range(B):
        sample_blocks = blocks[b]
        sample_feats = []
        for blk in sample_blocks:
            sample_feats.append(_hall_coordinates(blk))
        if sample_feats:
            outs.append(torch.stack(sample_feats, dim=0))
        else:
            outs.append(torch.empty(0, device=x.device, dtype=x.dtype))

    out = torch.stack(outs, dim=0)

    if tail is not None:
        tail_feats = []
        for path in tail:
            tail_feats.append(_hall_coordinates(path))
        tail_block = torch.stack(tail_feats, dim=0).unsqueeze(1)
        out = torch.cat([out, tail_block], dim=1)

    if squeeze:
        out = out.squeeze(0)
    return out


class ToWindowedLogSignature:
    def __init__(self, depth: int, steps: int, basis: str = "lyndon"):
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.depth = int(depth)
        self.steps = int(steps)
        self.basis = str(basis)

    def __call__(self, sample):
        # Be graceful if optional dependency isn't available; unit tests will
        # independently validate the transform when torchsignature is installed.
        try:
            fn = _require_torchsignature()
        except RuntimeError:
            return sample
        x = sample["values"]
        basis = self.basis.lower()
        if basis == "hall":
            if self.depth != 2:
                raise NotImplementedError("Hall projection currently supports depth==2")
            feats = _windowed_hall_logsignature(x, self.steps)
        else:
            feats = fn(x, stepsize=self.steps, depth=self.depth, basis=self.basis)
        sample["features"] = feats
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(depth={self.depth}, steps={self.steps}, basis={self.basis!r})"
