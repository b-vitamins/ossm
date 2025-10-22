from __future__ import annotations

import torch

from ossm.data.datasets.seqrec import SeqRecBatch
from ossm.models.dlinossrec import Dlinoss4Rec


def test_dlinoss4rec_forward_loss() -> None:
    model = Dlinoss4Rec(
        num_items=50,
        d_model=16,
        ssm_size=32,
        blocks=1,
        dropout=0.1,
        max_len=6,
    )
    batch = SeqRecBatch(
        input_ids=torch.tensor(
            [
                [0, 0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8, 9],
            ],
            dtype=torch.long,
        ),
        target=torch.tensor([5, 10], dtype=torch.long),
        mask=torch.tensor(
            [
                [False, False, True, True, True, True],
                [False, True, True, True, True, True],
            ]
        ),
        user_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    loss = model.forward_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
