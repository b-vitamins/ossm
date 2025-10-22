from __future__ import annotations

from functools import partial
from typing import List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ossm.data.datasets.seqrec import collate_left_pad
from ossm.training.seqrec import evaluate_fullsort


class _ToyDataset(Dataset[Tuple[int, List[int], int]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Tuple[int, List[int], int]:
        return 0, [1, 2], 2


class _DummyModel:
    def eval(self) -> None:  # pragma: no cover - no-op
        pass

    def to(self, device: torch.device) -> "_DummyModel":  # pragma: no cover
        return self

    def predict_scores(self, batch, *, include_padding: bool = False) -> torch.Tensor:
        _ = include_padding
        return torch.tensor([[0.2, 0.6, 0.3]], device=batch.input_ids.device)


def test_evaluate_fullsort_masks_history() -> None:
    dataset = _ToyDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=partial(collate_left_pad, max_len=4))
    model = _DummyModel()
    seen_items = {0: torch.tensor([2])}
    metrics = evaluate_fullsort(model, loader, seen_items, torch.device("cpu"), topk=1)
    assert metrics["HR@1"] == 0.0


def test_evaluate_fullsort_raises_when_topk_exceeds_candidates() -> None:
    dataset = _ToyDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=partial(collate_left_pad, max_len=4))
    model = _DummyModel()
    seen_items = {0: torch.tensor([1, 2])}
    with pytest.raises(RuntimeError, match="candidate pool is smaller"):
        evaluate_fullsort(model, loader, seen_items, torch.device("cpu"), topk=5)
