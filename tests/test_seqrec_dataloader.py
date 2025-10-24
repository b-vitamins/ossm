from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ossm.data.datasets.seqrec import SeqRecEvalDataset, SeqRecTrainDataset, collate_left_pad

pytest.importorskip("pyarrow")


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
    _create_seqrec_dataset(tmp_path)
    train_dataset = SeqRecTrainDataset(tmp_path, max_len=5)
    sample0 = train_dataset[0]
    sample1 = train_dataset[1]
    batch = collate_left_pad([sample0, sample1], max_len=5)
    assert batch.input_ids.shape == (2, 5)
    assert batch.mask.shape == (2, 5)
    assert batch.target.shape == (2,)
    assert batch.mask[0].sum() == len(sample0[1])


def test_collate_left_pad_pin_memory(tmp_path: Path) -> None:
    _create_seqrec_dataset(tmp_path)
    train_dataset = SeqRecTrainDataset(tmp_path, max_len=5)
    sample0 = train_dataset[0]
    sample1 = train_dataset[1]
    batch = collate_left_pad([sample0, sample1], max_len=5, pin_memory=True)
    assert batch.input_ids.is_pinned()
    assert batch.target.is_pinned()
    assert batch.user_ids.is_pinned()
    assert batch.mask.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_collate_left_pad_cuda(tmp_path: Path) -> None:
    _create_seqrec_dataset(tmp_path)
    train_dataset = SeqRecTrainDataset(tmp_path, max_len=5)
    sample0 = train_dataset[0]
    sample1 = train_dataset[1]
    device = torch.device("cuda")
    batch = collate_left_pad([sample0, sample1], max_len=5, device=device)
    assert batch.input_ids.device == device
    assert batch.mask.device == device
    assert batch.target.device == device
    assert batch.user_ids.device == device


def test_seqrec_batch_to_roundtrip(tmp_path: Path) -> None:
    _create_seqrec_dataset(tmp_path)
    train_dataset = SeqRecTrainDataset(tmp_path, max_len=5)
    sample = train_dataset[0]
    batch = collate_left_pad([sample], max_len=5, pin_memory=True)
    moved = batch.to(torch.device("cpu"))
    assert isinstance(moved, type(batch))
    assert moved.input_ids.device.type == "cpu"


def test_eval_dataset_context(tmp_path: Path) -> None:
    _create_seqrec_dataset(tmp_path)
    val_dataset = SeqRecEvalDataset(tmp_path, split="val", max_len=5)
    test_dataset = SeqRecEvalDataset(tmp_path, split="test", max_len=5)
    user0_context = val_dataset.context_for_user(0)
    assert user0_context == [1, 2, 3]
    user1_context = test_dataset.context_for_user(1)
    assert user1_context[-1] == 8
