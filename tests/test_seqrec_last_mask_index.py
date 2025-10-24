from __future__ import annotations

import pytest
import torch

from ossm.models.dlinossrec import Dlinoss4Rec, _last_mask_index
from ossm.models.mambarec import Mamba4Rec


def _build_dlinoss() -> Dlinoss4Rec:
    return Dlinoss4Rec(
        num_items=32,
        d_model=8,
        ssm_size=8,
        blocks=1,
        dropout=0.0,
        max_len=4,
        use_pffn=True,
    )


def _build_mamba() -> Mamba4Rec:
    return Mamba4Rec(
        num_items=32,
        d_model=8,
        ssm_size=8,
        blocks=1,
        dropout=0.0,
        max_len=4,
        use_pffn=True,
        d_conv=2,
        expand=2,
    )


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (
            torch.tensor(
                [
                    [False, False, True, True],
                    [False, True, True, False],
                ]
            ),
            [3, 2],
        ),
        (
            torch.tensor(
                [
                    [False, False, False, False],
                    [True, True, True, True],
                ]
            ),
            [0, 3],
        ),
    ],
)
def test_last_mask_index(mask: torch.Tensor, expected: list[int]) -> None:
    indices = _last_mask_index(mask)
    assert indices.dtype == torch.long
    assert indices.tolist() == expected


@pytest.mark.parametrize(
    "model_builder",
    [_build_dlinoss, _build_mamba],
    ids=["dlinoss", "mamba"],
)
def test_forward_features_gathers_last_valid_step(model_builder) -> None:
    model = model_builder()
    model.eval()
    input_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [False, False, False, False],
        ]
    )

    with torch.no_grad():
        hidden = model.encoder(input_ids)
        if hasattr(model, "blocks") and getattr(model, "pffn_blocks", None) is not None:
            for block, ffn in zip(model.blocks, model.pffn_blocks):
                hidden = ffn(block(hidden))
        elif hasattr(model, "blocks"):
            for block in model.blocks:
                hidden = block(hidden)
        else:
            for layer in model.layers:
                hidden = layer(hidden)

        expected_index = _last_mask_index(mask).to(device=hidden.device)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        expected = hidden[batch_indices, expected_index]

        output = model.forward_features(input_ids, mask)

    torch.testing.assert_close(output, expected)
