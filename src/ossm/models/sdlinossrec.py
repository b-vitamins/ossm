# sdlinossrec.py
# Composable Selective D‑LinOSS sequential recommender.
#
# This file mirrors dlinossrec.py but replaces DampedLinOSSBlock with
# SelectiveDLinOSSBlock (spectral-selective, time-varying). The overall
# embedding → SSM stack → optional PFFN → tied softmax head pipeline
# follows the same template used in Mamba4Rec-style recommenders. See:
#   • D‑LinOSS (time-invariant oscillatory SSM, inverse map & IMEX): arXiv:2505.12171
#   • Mamba4Rec (selective SSMs in recommender plumbing): arXiv:2403.03900

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import ModuleList

from .sdlinoss import SelectiveDLinOSSBlock
from .heads import TiedSoftmaxHead

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from ..data.datasets.seqrec import SeqRecBatch

__all__ = ["ItemEmbeddingEncoder", "Sdlinoss4Rec"]


def _last_mask_index(mask: torch.Tensor) -> torch.LongTensor:
    """Return the last valid timestep index for each sequence in ``mask``.

    The mask is expected to mark valid (non-padding) positions with truthy values.
    Fully padded sequences default to index ``0`` so callers can safely gather
    without branching.
    """

    if mask.ndim != 2:
        raise ValueError("Mask must have shape (batch, length)")

    batch, seq_len = mask.shape
    if seq_len == 0:
        return cast(torch.LongTensor, torch.zeros(batch, dtype=torch.long, device=mask.device))

    mask_bool = mask.to(dtype=torch.bool)
    positions = torch.arange(seq_len, device=mask.device, dtype=torch.long).unsqueeze(0)
    masked_positions = torch.where(mask_bool, positions, positions.new_full((1, seq_len), -1))
    last_index = masked_positions.max(dim=1).values
    return cast(torch.LongTensor, last_index.clamp(min=0).long())


class ItemEmbeddingEncoder(nn.Module):
    """Encode item identifiers with optional positional context.

    Matches the encoder used in the D‑LinOSS recommender so this module can
    be swapped without touching data loaders or heads. See §3 in Mamba4Rec for
    the standard stack (embedding + dropout + norm).
    """

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
    """Pre‑norm feed‑forward block with residual connection."""

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


class Sdlinoss4Rec(nn.Module):
    """Selective D‑LinOSS–based sequential recommender.

    This is the Selective D‑LinOSS mirror of ``Dlinoss4Rec``. It keeps the same
    external interface and head, but the sequence modeling blocks are
    ``SelectiveDLinOSSBlock`` instead of ``DampedLinOSSBlock``. The selective
    backbone exposes per‑token spectral parameters while preserving the same
    (B, L, D) I/O and dropout/norm/FFN plumbing. For background on the
    oscillatory IMEX discretization and the eigenvalue inverse map, see D‑LinOSS.
    """

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
        # Pass-through kwargs for SelectiveDLinOSSLayer/Block (e.g., variant, per_step_dt, selective_injection, etc.)
        **sdlinoss_kwargs,
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
        # Sequence modeling stack: Selective D‑LinOSS blocks
        self.blocks = ModuleList(
            [
                SelectiveDLinOSSBlock(
                    ssm_size,
                    d_model,
                    dropout=dropout,
                    **sdlinoss_kwargs,
                )
                for _ in range(blocks)
            ]
        )
        # Optional position‑wise FFN after each SSM block (matches original wiring)
        self.pffn_blocks: ModuleList | None
        if use_pffn:
            self.pffn_blocks = ModuleList(
                [_FeedForwardBlock(d_model, dropout, use_layernorm) for _ in range(blocks)]
            )
        else:
            self.pffn_blocks = None

        # Tied softmax prediction head (same as before)
        self.head = TiedSoftmaxHead(
            self.encoder.embedding,
            bias=head_bias,
            temperature=head_temperature,
            padding_idx=0,
        )

    # ---- Public API identical to Dlinoss4Rec --------------------------------

    def forward_features(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the sequence and return the final non‑pad hidden state."""
        features = self.encoder(input_ids)
        if self.pffn_blocks is None:
            for block in self.blocks:
                features = block(features)
        else:
            for block, ffn in zip(self.blocks, self.pffn_blocks):
                features = ffn(block(features))
        last_index = _last_mask_index(mask).to(device=features.device)
        batch_indices = torch.arange(features.size(0), device=features.device)
        return features[batch_indices, last_index]

    def forward_loss(self, batch: "SeqRecBatch") -> torch.Tensor:
        last_hidden = self.forward_features(batch.input_ids, batch.mask)
        return self.head.loss(last_hidden, batch.target)

    def predict_logits(
        self, batch: "SeqRecBatch", *, include_padding: bool = False
    ) -> torch.Tensor:
        last_hidden = self.forward_features(batch.input_ids, batch.mask)
        return self.head.logits(last_hidden, exclude_padding=not include_padding)
