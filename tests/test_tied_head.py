from __future__ import annotations

import torch

from ossm.models import TiedSoftmaxHead


def test_tied_head_prefers_matching_item() -> None:
    embedding = torch.nn.Embedding(4, 3, padding_idx=0)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
    head = TiedSoftmaxHead(embedding, bias=False)
    features = embedding.weight[3].unsqueeze(0)
    good_loss = head.loss(features, torch.tensor([3]))
    bad_loss = head.loss(features, torch.tensor([2]))
    assert good_loss < bad_loss
