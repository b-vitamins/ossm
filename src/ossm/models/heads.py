"""Task-specific heads for OSSM backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .base import Head

__all__ = [
    "ClassificationHead",
    "RegressionHead",
    "TiedSoftmaxHead",
]


class ClassificationHead(Head):
    """Applies a linear classifier on pooled features."""

    def __init__(self, hidden_dim: int, num_classes: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(pooled))


class RegressionHead(Head):
    """Applies a linear regressor on sequence features."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


@dataclass(frozen=True)
class _TiedSoftmaxConfig:
    temperature: float
    offset: int


class TiedSoftmaxHead(Head):
    """Shared-embedding softmax head for next-item prediction."""

    def __init__(
        self,
        embedding: nn.Embedding,
        *,
        bias: bool = True,
        temperature: float = 1.0,
        padding_idx: Optional[int] = 0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if padding_idx is not None:
            if embedding.padding_idx is not None and embedding.padding_idx != padding_idx:
                raise ValueError(
                    "embedding padding_idx does not match the requested padding_idx"
                )
            offset = int(padding_idx) + 1
        else:
            offset = 0

        self.embedding = embedding
        self._config = _TiedSoftmaxConfig(temperature=float(temperature), offset=offset)
        if bias:
            self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))
        else:
            self.register_parameter("bias", None)

    @property
    def vocabulary_size(self) -> int:
        """Total number of logits, including any padding entry."""

        return int(self.embedding.num_embeddings)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """Return logits against the tied vocabulary."""

        logits = representations @ self.embedding.weight.transpose(0, 1)
        if self.bias is not None:
            logits = logits + self.bias
        if self._config.temperature != 1.0:
            logits = logits / self._config.temperature
        return logits

    def logits(
        self,
        representations: torch.Tensor,
        *,
        exclude_padding: bool = False,
    ) -> torch.Tensor:
        """Convenience wrapper returning logits with optional padding removal."""

        logits = self.forward(representations)
        if exclude_padding and self._config.offset:
            logits = logits[..., self._config.offset :]
        return logits

    def loss(
        self,
        representations: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Cross-entropy loss tying weights to the input embeddings."""

        logits = self.logits(representations, exclude_padding=self._config.offset > 0)
        if self._config.offset:
            adjusted = target - self._config.offset
            if torch.any(adjusted < 0):
                raise ValueError("targets must be greater than the padding index")
        else:
            adjusted = target
        return F.cross_entropy(logits, adjusted, reduction=reduction)
