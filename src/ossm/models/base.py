"""Base abstractions for OSSM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

    def forward(self, x: torch.Tensor) -> SequenceBackboneOutput:  # pragma: no cover - interface
        raise NotImplementedError


class Head(nn.Module):
    """Abstract task-specific head."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError
