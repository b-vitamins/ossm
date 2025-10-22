from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")


def _make_ml1m_raw(root: Path) -> None:
    data = []
    for user in range(3):
        for idx in range(6):
            data.append(
                {
                    "userId": user + 1,
                    "movieId": (user * 10) + idx + 1,
                    "rating": 4.5,
                    "timestamp": 1_600_000_000 + idx,
                }
            )
    df = pd.DataFrame(data)
    df.to_csv(root / "ratings.csv", index=False)


def test_prepare_ml1m_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_ml1m_raw(raw_dir)
    out_dir = tmp_path / "processed"

    subprocess.check_call(
        [
            sys.executable,
            str(Path("scripts/prepare_ml1m.py")),
            "--raw",
            str(raw_dir),
            "--out",
            str(out_dir),
            "--min-interactions",
            "3",
        ]
    )

    required_files = [
        out_dir / "train.parquet",
        out_dir / "val.parquet",
        out_dir / "test.parquet",
        out_dir / "user_ptr.npy",
        out_dir / "train_items.npy",
        out_dir / "item_count.txt",
    ]
    for path in required_files:
        assert path.exists(), f"Missing expected output {path}"

    user_ptr = np.load(out_dir / "user_ptr.npy")
    assert np.all(np.diff(user_ptr) >= 0)
    assert user_ptr[0] == 0

    train_items = np.load(out_dir / "train_items.npy")
    item_count = int((out_dir / "item_count.txt").read_text())
    assert train_items.max() <= item_count

    val_df = pd.read_parquet(out_dir / "val.parquet")
    test_df = pd.read_parquet(out_dir / "test.parquet")
    assert (val_df["prefix_len"] >= 1).all()
    assert (test_df["prefix_len"] >= val_df["prefix_len"].min()).all()
    assert (val_df["target_id"] >= 1).all()
    assert (test_df["target_id"] >= 1).all()
