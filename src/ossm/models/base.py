"""Base abstractions for OSSM models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from torch import nn


@dataclass
class SequenceBackboneOutput:
    """Container for backbone outputs.

    Attributes:
        features: Sequence features of shape ``(batch, length, hidden)``.
        pooled: Optional pooled representation (e.g. for classification).
    """

    features: torch.Tensor
    pooled: Optional[torch.Tensor] = None


class Backbone(nn.Module):
    """Abstract base class for model backbones."""

    def prepare_batch(self, batch: Mapping[str, Any]) -> Any:
        """Return the tensor inputs expected by the backbone."""

        if "values" not in batch:
            raise KeyError("Batch must contain 'values' for this backbone")
        values = batch["values"]
        if not isinstance(values, torch.Tensor):
            raise TypeError("'values' entry must be a tensor")
        return values

    def forward(self, x: torch.Tensor) -> SequenceBackboneOutput:  # pragma: no cover - interface
        raise NotImplementedError


class Head(nn.Module):
    """Abstract task-specific head."""

    def prepare_batch(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Return the supervised targets expected by the head."""

        if "label" not in batch:
            raise KeyError("Batch must contain 'label' for this head")
        target = batch["label"]
        if not isinstance(target, torch.Tensor):
            raise TypeError("'label' entry must be a tensor")
        return target

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


def _to_device(obj: Any, *, device: torch.device, dtype: torch.dtype | None) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype if dtype is not None else obj.dtype)
    if isinstance(obj, Mapping):
        return {key: _to_device(value, device=device, dtype=dtype) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_device(value, device=device, dtype=dtype) for value in obj)
    if isinstance(obj, list):
        return [_to_device(value, device=device, dtype=dtype) for value in obj]
    return obj


class BatchOnDevice(Mapping[str, Any]):
    """Mapping-like container holding a batch that already lives on a device."""

    def __init__(self, batch: Mapping[str, Any]) -> None:
        self._batch = dict(batch)

    def __getitem__(self, key: str) -> Any:
        return self._batch[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._batch)

    def __len__(self) -> int:
        return len(self._batch)

    def get(self, key: str, default: Any | None = None) -> Any:
        return self._batch.get(key, default)

    @classmethod
    def from_batch(
        cls,
        batch: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> BatchOnDevice:
        moved = _to_device(batch, device=device, dtype=dtype)
        return cls(moved)
