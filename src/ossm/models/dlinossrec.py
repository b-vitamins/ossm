"""Composable D-LinOSS sequential recommender."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import ModuleList

from .dlinoss import DampedLinOSSBlock
from .heads import TiedSoftmaxHead

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..data.datasets.seqrec import SeqRecBatch

__all__ = ["ItemEmbeddingEncoder", "Dlinoss4Rec"]


class ItemEmbeddingEncoder(nn.Module):
    """Encode item identifiers with optional positional context."""

    def __init__(
        self,
        num_items: int,
        d_model: int,
        *,
        dropout: float,
        max_len: int,
        use_layernorm: bool = True,
        use_pos_emb: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.use_positional = use_pos_emb
        self.max_len = int(max_len)
        self.position_embedding: nn.Embedding | None
        self.position_ids: torch.Tensor | None
        if use_pos_emb:
            self.position_embedding = nn.Embedding(self.max_len, d_model)
            position_ids = torch.arange(self.max_len).unsqueeze(0)
            self.register_buffer("position_ids", position_ids, persistent=False)
            self.position_ids = position_ids
        else:
            self.position_embedding = None
            self.register_buffer("position_ids", None, persistent=False)
            self.position_ids = None

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        if self.use_positional and self.position_embedding is not None and self.position_ids is not None:
            seq_len = input_ids.size(1)
            positions = self.position_ids[:, :seq_len]
            positional = self.position_embedding(positions)
            mask = input_ids.ne(0).unsqueeze(-1)
            embeddings = embeddings + positional * mask
        return self.norm(self.dropout(embeddings))


class _FeedForwardBlock(nn.Module):
    """Pre-norm feed-forward block with residual connection."""

    def __init__(self, d_model: int, dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.norm(inputs)
        outputs = self.ffn(outputs)
        return outputs + residual


class Dlinoss4Rec(nn.Module):
    """D-LinOSS-based sequential recommender."""

    def __init__(
        self,
        *,
        num_items: int,
        d_model: int,
        ssm_size: int,
        blocks: int,
        dropout: float,
        max_len: int,
        use_pffn: bool = True,
        use_pos_emb: bool = False,
        use_layernorm: bool = True,
        head_bias: bool = True,
        head_temperature: float = 1.0,
        **dlinoss_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = ItemEmbeddingEncoder(
            num_items,
            d_model,
            dropout=dropout,
            max_len=max_len,
            use_layernorm=use_layernorm,
            use_pos_emb=use_pos_emb,
        )
        self.blocks = ModuleList(
            [
                DampedLinOSSBlock(
                    ssm_size,
                    d_model,
                    dropout=dropout,
                    **dlinoss_kwargs,
                )
                for _ in range(blocks)
            ]
        )
        self.pffn_blocks: ModuleList | None
        if use_pffn:
            self.pffn_blocks = ModuleList(
                [_FeedForwardBlock(d_model, dropout, use_layernorm) for _ in range(blocks)]
            )
        else:
            self.pffn_blocks = None
        self.head = TiedSoftmaxHead(
            self.encoder.embedding,
            bias=head_bias,
            temperature=head_temperature,
            padding_idx=0,
        )

    def forward_features(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.encoder(input_ids)
        if self.pffn_blocks is None:
            for block in self.blocks:
                features = block(features)
        else:
            for block, ffn in zip(self.blocks, self.pffn_blocks):
                features = ffn(block(features))
        lengths = mask.sum(dim=1)
        last_index = lengths.clamp(min=1) - 1
        batch_indices = torch.arange(features.size(0), device=features.device)
        return features[batch_indices, last_index]

    def forward_loss(self, batch: "SeqRecBatch") -> torch.Tensor:
        last_hidden = self.forward_features(batch.input_ids, batch.mask)
        return self.head.loss(last_hidden, batch.target)

    def predict_scores(
        self, batch: "SeqRecBatch", *, include_padding: bool = False
    ) -> torch.Tensor:
        last_hidden = self.forward_features(batch.input_ids, batch.mask)
        return self.head.logits(last_hidden, exclude_padding=not include_padding)
