"""Regression tests for residual SSM blocks."""

from __future__ import annotations

import pytest
import torch

from ossm.models.dlinoss import DampedLinOSSBlock
from ossm.models.linoss import LinOSSBlock
from ossm.models.lru import LRUBlock
from ossm.models.s5 import S5Block


def _manual_block_forward(block: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    batch, length, hidden = inputs.shape
    normed = block.norm(inputs.reshape(-1, hidden)).reshape(batch, length, hidden)
    outputs = block.layer(normed)
    outputs = block.activation(outputs)
    outputs = block.dropout(outputs)
    outputs = block.glu(outputs)
    outputs = block.dropout(outputs)
    return outputs + inputs


DLINOSS_VARIANTS = ("imex1", "imex2", "im", "ex")


@pytest.mark.parametrize(
    ("block", "shape"),
    [
        (LinOSSBlock(ssm_size=2, hidden_dim=4, discretization="IM"), (2, 3, 4)),
        *[
            (
                DampedLinOSSBlock(
                    ssm_size=2,
                    hidden_dim=4,
                    variant=variant,
                    initialization="ring",
                    dropout=0.1,
                ),
                (2, 3, 4),
            )
            for variant in DLINOSS_VARIANTS
        ],
        (LRUBlock(ssm_size=2, hidden_dim=4, dropout=0.2), (2, 3, 4)),
        (
            S5Block(
                ssm_size=2,
                hidden_dim=4,
                blocks=1,
                discretization="zoh",
                dropout=0.1,
            ),
            (2, 3, 4),
        ),
    ],
)
def test_residual_block_matches_manual_flow(block: torch.nn.Module, shape: tuple[int, int, int]) -> None:
    block.eval()
    inputs = torch.randn(*shape, requires_grad=True)
    with torch.no_grad():
        expected = _manual_block_forward(block, inputs.detach())

    outputs = block(inputs)
    assert torch.allclose(outputs, expected, atol=1e-6, rtol=1e-5)
    assert outputs.shape == inputs.shape
    assert outputs.dtype == inputs.dtype

    loss = outputs.sum()
    loss.backward()
    assert inputs.grad is not None
