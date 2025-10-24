from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from ossm.data.datasets.seqrec import SeqRecEvalDataset, SeqRecTrainDataset, collate_left_pad


def _create_seqrec_dataset(root: Path) -> None:
    user_ptr = np.array([0, 3, 6], dtype=np.int64)
    train_items = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    pd.DataFrame({"user_id": [0, 0, 0, 1, 1, 1], "item_id": [1, 2, 3, 4, 5, 6], "ts": [1, 2, 3, 4, 5, 6]}).to_parquet(
        root / "train.parquet", index=False
    )
    pd.DataFrame({"user_id": [0, 1], "prefix_len": [3, 3], "target_id": [7, 8]}).to_parquet(root / "val.parquet", index=False)
    pd.DataFrame({"user_id": [0, 1], "prefix_len": [4, 4], "target_id": [9, 10]}).to_parquet(
        root / "test.parquet", index=False
    )
    np.save(root / "user_ptr.npy", user_ptr)
    np.save(root / "train_items.npy", train_items)
    (root / "item_count.txt").write_text("12")


def test_collate_left_pad(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    _create_seqrec_dataset(tmp_path)
    train_dataset = SeqRecTrainDataset(tmp_path, max_len=5)
    sample0 = train_dataset[0]
    sample1 = train_dataset[1]
    batch = collate_left_pad([sample0, sample1], max_len=5)
    assert batch.input_ids.shape == (2, 5)
    assert batch.mask.shape == (2, 5)
    assert batch.target.shape == (2,)
    assert batch.mask[0].sum() == len(sample0[1])

    contexts = [sample0[1], sample1[1]]
    expected_inputs = []
    expected_mask = []
    for context in contexts:
        trimmed = context[-5:]
        pad = [0] * (5 - len(trimmed))
        expected_inputs.append(pad + trimmed)
        expected_mask.append([0] * len(pad) + [1] * len(trimmed))

    expected_inputs_tensor = torch.tensor(
        expected_inputs, dtype=batch.input_ids.dtype
    )
    expected_mask_tensor = torch.tensor(expected_mask, dtype=torch.bool)

    assert torch.equal(batch.input_ids.cpu(), expected_inputs_tensor)
    assert torch.equal(batch.mask.cpu(), expected_mask_tensor)


def test_eval_dataset_context(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    _create_seqrec_dataset(tmp_path)
    val_dataset = SeqRecEvalDataset(tmp_path, split="val", max_len=5)
    test_dataset = SeqRecEvalDataset(tmp_path, split="test", max_len=5)
    user0_context = val_dataset.context_for_user(0)
    assert user0_context == [1, 2, 3]
    user1_context = test_dataset.context_for_user(1)
    assert user1_context[-1] == 8


def test_collate_left_pad_pin_memory_behavior() -> None:
    samples = [
        (0, [1, 2, 3], 4),
        (1, [5], 6),
    ]
    batch = collate_left_pad(samples, max_len=4, pin_memory=True, dtype=torch.int16)

    assert batch.input_ids.dtype == torch.int16
    assert batch.target.dtype == torch.long
    assert batch.user_ids.dtype == torch.long

    if torch.cuda.is_available():
        assert batch.input_ids.is_pinned()
        assert batch.target.is_pinned()
        assert batch.mask.is_pinned()
        assert batch.user_ids.is_pinned()
    else:
        assert not batch.input_ids.is_pinned()
        assert not batch.mask.is_pinned()

    expected_inputs = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 0, 5]], dtype=batch.input_ids.dtype
    )
    expected_mask = torch.tensor(
        [[0, 1, 1, 1], [0, 0, 0, 1]], dtype=torch.bool
    )
    assert torch.equal(batch.input_ids.cpu(), expected_inputs)
    assert torch.equal(batch.mask.cpu(), expected_mask)


def test_collate_left_pad_no_pin_memory_even_if_available() -> None:
    samples = [
        (0, [1, 2], 3),
        (1, [4, 5, 6], 7),
    ]

    batch = collate_left_pad(samples, max_len=4, pin_memory=False)

    assert not batch.input_ids.is_pinned()
    assert not batch.mask.is_pinned()
    assert not batch.target.is_pinned()
    assert not batch.user_ids.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_collate_left_pad_cuda_dtype_and_to() -> None:
    device = torch.device("cuda")
    samples = [
        (
            torch.tensor(0, device=device, dtype=torch.int32),
            torch.tensor([1, 2, 3, 4], device=device, dtype=torch.int32),
            torch.tensor(9, device=device, dtype=torch.int16),
        ),
        (
            torch.tensor(1, device=device, dtype=torch.int32),
            torch.tensor([5, 6], device=device, dtype=torch.int32),
            torch.tensor(10, device=device, dtype=torch.int16),
        ),
    ]

    batch = collate_left_pad(samples, max_len=4)

    assert batch.input_ids.device == device
    assert batch.mask.device == device
    assert batch.input_ids.dtype == torch.int32
    assert batch.target.dtype == torch.int16
    assert batch.user_ids.dtype == torch.int32

    expected_inputs = torch.tensor(
        [[1, 2, 3, 4], [0, 0, 5, 6]], device=device, dtype=torch.int32
    )
    expected_mask = torch.tensor(
        [[1, 1, 1, 1], [0, 0, 1, 1]], device=device, dtype=torch.bool
    )
    assert torch.equal(batch.input_ids, expected_inputs)
    assert torch.equal(batch.mask, expected_mask)

    moved = batch.to("cpu")
    assert moved.input_ids.device.type == "cpu"
    assert moved.input_ids.dtype == batch.input_ids.dtype
    assert torch.equal(moved.target.cpu(), batch.target.cpu())
