from __future__ import annotations

import torch

from ossm.metrics import TopKMetricAccumulator


def test_topk_accumulator_clamps_to_available_candidates() -> None:
    accumulator = TopKMetricAccumulator(topk=5)
    scores = torch.tensor([[0.9, 0.4, 0.2]])
    target = torch.tensor([0])

    accumulator.update(scores, target)
    metrics = accumulator.compute()

    assert metrics["HR@5"] == 1.0
    assert accumulator.effective_topk == 3
