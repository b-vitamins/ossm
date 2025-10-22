from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_eval_dataset_context(tmp_path: Path) -> None:
    _create_seqrec_dataset(tmp_path)
    val_dataset = SeqRecEvalDataset(tmp_path, split="val", max_len=5)
    test_dataset = SeqRecEvalDataset(tmp_path, split="test", max_len=5)
    user0_context = val_dataset.context_for_user(0)
    assert user0_context == [1, 2, 3]
    user1_context = test_dataset.context_for_user(1)
    assert user1_context[-1] == 8
