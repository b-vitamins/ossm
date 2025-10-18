"""Task-specific heads for OSSM backbones."""

from __future__ import annotations

import torch
from torch import nn

from .base import Head


class ClassificationHead(Head):
    """Applies a linear classifier on pooled features."""

    def __init__(self, hidden_dim: int, num_classes: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        logits = self.linear(self.dropout(pooled))
        return logits


class RegressionHead(Head):
    """Applies a linear regressor on sequence features."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)
