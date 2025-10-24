from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Callable, List, Protocol, cast

import torch


Tensor = torch.Tensor

LOGGER = logging.getLogger(__name__)


class TorchSignatureUnavailable(RuntimeError):
    """Raised when the optional torchsignature dependency is missing."""


class TorchSignatureBackend:
    """Lazy loader for torchsignature utilities."""

    __slots__ = ("_loaded", "signature", "windowed_logsignature")

    def __init__(self) -> None:
        self._loaded = False
        self.signature: Callable[..., List[Tensor]] | None = None
        self.windowed_logsignature: Callable[..., Tensor] | None = None

    def ensure(self) -> TorchSignatureBackend:
        if self._loaded:
            if self.signature is None or self.windowed_logsignature is None:
                raise TorchSignatureUnavailable(
                    "torchsignature functions unavailable despite cached load"
                )
            return self

        if importlib.util.find_spec("torchsignature") is None:
            raise TorchSignatureUnavailable(
                "torchsignature is required for log-signature features. Install it first."
            )

        windowed_mod = importlib.import_module("torchsignature.windowed")
        windowed_fn = getattr(windowed_mod, "windowed_logsignature", None)
        if windowed_fn is None:
            top = importlib.import_module("torchsignature")
            windowed_fn = getattr(top, "windowed_logsignature", None)
            if windowed_fn is None:
                raise RuntimeError(
                    "torchsignature.windowed_logsignature not found; update torchsignature."
                )

        sig_mod = importlib.import_module("torchsignature.signatures")
        signature_fn = getattr(sig_mod, "signature")

        self.windowed_logsignature = windowed_fn
        self.signature = signature_fn
        self._loaded = True
        return self


_TORCHSIGNATURE_BACKEND = TorchSignatureBackend()


@dataclass(frozen=True)
class WindowSegments:
    bulk: Tensor
    tail: Tensor | None
    squeeze: bool


def _segment_windows(x: Tensor, steps: int) -> WindowSegments:
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

    return WindowSegments(blocks, tail, squeeze)


def _hall_coordinate_dim(channels: int) -> int:
    return 1 + channels + (channels * (channels - 1)) // 2


def _hall_coordinates_batch(
    paths: Tensor, signature_fn: Callable[..., List[Tensor]], depth: int
) -> Tensor:
    if paths.size(0) == 0:
        feature_dim = _hall_coordinate_dim(paths.size(-1))
        return torch.empty(0, feature_dim, dtype=paths.dtype, device=paths.device)

    sig_lvls = cast(
        List[Tensor], signature_fn(paths, depth=depth, stream=False, flatten=False)
    )
    lvl1 = sig_lvls[0]
    lvl2 = sig_lvls[1]

    idx = torch.triu_indices(lvl2.size(-2), lvl2.size(-1), offset=1, device=paths.device)
    lie2 = 0.5 * (lvl2[:, idx[0], idx[1]] - lvl2[:, idx[1], idx[0]])
    scalar = torch.zeros(lvl1.size(0), 1, dtype=lvl1.dtype, device=lvl1.device)
    return torch.cat([scalar, lvl1.reshape(lvl1.size(0), -1), lie2], dim=1)


def _windowed_hall_logsignature(
    x: Tensor, steps: int, signature_fn: Callable[..., List[Tensor]], depth: int
) -> Tensor:
    segments = _segment_windows(x, steps)

    bulk = segments.bulk
    B, num_blocks, _, channels = bulk.shape
    feature_dim = _hall_coordinate_dim(channels)
    dtype, device = bulk.dtype, bulk.device

    if num_blocks:
        flat_blocks = bulk.reshape(-1, bulk.size(2), channels)
        block_feats = _hall_coordinates_batch(flat_blocks, signature_fn, depth)
        block_feats = block_feats.reshape(B, num_blocks, -1)
    else:
        block_feats = torch.empty(B, 0, feature_dim, dtype=dtype, device=device)

    feats = block_feats
    if segments.tail is not None:
        tail = segments.tail
        tail_flat = tail.reshape(-1, tail.size(1), tail.size(2))
        tail_feats = _hall_coordinates_batch(tail_flat, signature_fn, depth)
        tail_feats = tail_feats.reshape(B, 1, -1)
        feats = torch.cat([feats, tail_feats], dim=1)

    if segments.squeeze:
        feats = feats.squeeze(0)
    return feats


class BasisStrategy(Protocol):
    def transform(self, values: Tensor, steps: int, depth: int) -> Tensor: ...


class TorchSignatureBasisStrategy:
    def __init__(self, basis: str, backend: TorchSignatureBackend) -> None:
        self._basis = basis
        self._backend = backend

    def transform(self, values: Tensor, steps: int, depth: int) -> Tensor:
        backend = self._backend.ensure()
        fn = backend.windowed_logsignature
        assert fn is not None  # for type checkers
        return fn(values, stepsize=steps, depth=depth, basis=self._basis)


class HallBasisStrategy:
    def __init__(self, backend: TorchSignatureBackend) -> None:
        self._backend = backend

    def transform(self, values: Tensor, steps: int, depth: int) -> Tensor:
        backend = self._backend.ensure()
        signature_fn = backend.signature
        assert signature_fn is not None  # for type checkers
        return _windowed_hall_logsignature(values, steps, signature_fn, depth)


def _resolve_basis_strategy(
    basis: str, depth: int, backend: TorchSignatureBackend
) -> BasisStrategy:
    if basis == "hall":
        if depth != 2:
            raise NotImplementedError("Hall projection currently supports depth==2")
        return HallBasisStrategy(backend)
    return TorchSignatureBasisStrategy(basis, backend)


from .compose import TimeSeriesSample


class ToWindowedLogSignature:
    def __init__(self, depth: int, steps: int, basis: str = "lyndon"):
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.depth = int(depth)
        self.steps = int(steps)
        self.basis = str(basis)
        self._basis_key = self.basis.lower()
        self._strategy = _resolve_basis_strategy(
            self._basis_key, self.depth, _TORCHSIGNATURE_BACKEND
        )
        self._warned_missing = False

    def __call__(self, sample: TimeSeriesSample) -> TimeSeriesSample:
        values = sample.get("values")
        if values is None:
            raise KeyError("sample must contain 'values'")

        try:
            feats = self._strategy.transform(values, self.steps, self.depth)
        except TorchSignatureUnavailable:
            if not self._warned_missing:
                LOGGER.warning(
                    "torchsignature is not available; returning sample unchanged from %s",
                    self.__class__.__name__,
                )
                self._warned_missing = True
            return sample

        updated = sample.copy()
        updated["features"] = feats
        return cast(TimeSeriesSample, updated)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(depth={self.depth}, steps={self.steps}, basis={self.basis!r})"
