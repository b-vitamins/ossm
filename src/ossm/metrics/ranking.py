"""Ranking metrics and helpers for recommendation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

__all__ = [
    "mask_history_inplace",
    "TopKMetricAccumulator",
    "compute_topk_metrics",
]


def mask_history_inplace(
    scores: torch.Tensor,
    user_ids: torch.Tensor,
    history: Mapping[int, torch.Tensor],
    *,
    offset: int = 0,
) -> None:
    """Mask previously seen items in-place by setting their scores to ``-inf``.

    Args:
        scores: Score tensor of shape ``(batch, vocab_size)``.
        user_ids: Tensor of user identifiers of shape ``(batch,)``.
        history: Mapping from user id to 1-D tensor of seen item ids.
        offset: Optional offset subtracted from item ids before indexing.
    """

    if not history:
        return
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D tensor")
    if user_ids.ndim != 1:
        raise ValueError("user_ids must be a 1D tensor")
    if scores.size(0) != user_ids.size(0):
        raise ValueError("scores and user_ids batch dimensions must match")

    vocab_size = scores.size(1)
    device = scores.device
    for row, user_id in enumerate(user_ids.tolist()):
        seen = history.get(int(user_id))
        if seen is None or seen.numel() == 0:
            continue
        indices = (seen.to(device=device, dtype=torch.long) - offset).clamp(min=0)
        indices = indices[indices < vocab_size]
        if indices.numel():
            scores[row, indices] = float("-inf")


def compute_topk_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-example HR, NDCG, and MRR at ``topk``.

    Returns tensors of shape ``(batch,)`` corresponding to each metric.
    """

    if topk <= 0:
        raise ValueError("topk must be positive")
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D tensor")
    if target.ndim != 1:
        raise ValueError("target must be a 1D tensor")
    if scores.size(0) != target.size(0):
        raise ValueError("scores and target batch dimensions must match")

    batch_size = scores.size(0)
    device = scores.device
    topk = min(topk, scores.size(1))
    top_indices = torch.topk(scores, topk, dim=1).indices
    positions = torch.arange(1, topk + 1, device=device, dtype=torch.long)
    expanded_positions = positions.unsqueeze(0).expand(batch_size, topk)
    default = torch.full_like(expanded_positions, topk + 1)
    matches = top_indices.eq(target.view(-1, 1))
    ranks = torch.where(matches, expanded_positions, default)
    best_ranks = ranks.min(dim=1).values

    hits = best_ranks <= topk
    best_ranks_f = best_ranks.to(scores.dtype)
    hits_f = hits.to(scores.dtype)
    reciprocal_rank = torch.where(hits, 1.0 / best_ranks_f, torch.zeros_like(best_ranks_f))
    denom = torch.log2(best_ranks_f + 1.0)
    ndcg = torch.where(hits, 1.0 / denom, torch.zeros_like(best_ranks_f))
    return hits_f, ndcg, reciprocal_rank


@dataclass
class TopKMetricAccumulator:
    """Accumulate ranking metrics over multiple batches."""

    topk: int
    hit_total: float = 0.0
    ndcg_total: float = 0.0
    mrr_total: float = 0.0
    count: int = 0
    effective_topk: int | None = None

    def update(self, scores: torch.Tensor, target: torch.Tensor) -> None:
        if scores.numel() == 0:
            return
        if scores.ndim != 2:
            raise ValueError("scores must be a 2D tensor")
        if target.ndim != 1:
            raise ValueError("target must be a 1D tensor")
        if scores.size(0) != target.size(0):
            raise ValueError("scores and target batch dimensions must match")

        finite_counts = torch.isfinite(scores).sum(dim=1)
        if finite_counts.numel() == 0:
            return
        min_candidates = int(finite_counts.min().item())
        if min_candidates <= 0:
            return

        k = min(self.topk, scores.size(1), min_candidates)
        if k <= 0:
            return

        hits, ndcg, mrr = compute_topk_metrics(scores, target, topk=k)
        self.effective_topk = k if self.effective_topk is None else min(self.effective_topk, k)
        self.hit_total += float(hits.sum().item())
        self.ndcg_total += float(ndcg.sum().item())
        self.mrr_total += float(mrr.sum().item())
        self.count += int(scores.size(0))

    def compute(self) -> dict[str, float]:
        denom = max(self.count, 1)
        return {
            f"HR@{self.topk}": self.hit_total / denom,
            f"NDCG@{self.topk}": self.ndcg_total / denom,
            f"MRR@{self.topk}": self.mrr_total / denom,
        }
