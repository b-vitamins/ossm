"""Inspect a processed sequential recommendation dataset.

Prints user/item counts, train/eval sizes, total interactions, and basic
sequence length stats for a dataset prepared with ``prepare_ml1m.py`` or
``prepare_amazon.py``.

Usage:
  guix shell -m ossm/manifest.scm -- \
    python3 ossm/scripts/inspect_seqrec.py --root data/seqrec/ml1m
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _stats(root: Path) -> Dict[str, float | int]:
    val = pd.read_parquet(root / "val.parquet")
    test = pd.read_parquet(root / "test.parquet")
    ptr = np.load(root / "user_ptr.npy")
    train_items = np.load(root / "train_items.npy")
    users = int(ptr.size - 1)
    items = int((root / "item_count.txt").read_text().strip())
    total = int(train_items.size + 2 * users)
    lengths = np.diff(ptr)
    avg_train = float(lengths.mean()) if lengths.size else 0.0
    return {
        "users": users,
        "items": items,
        "train_interactions": int(train_items.size),
        "total_interactions": total,
        "avg_train_len": avg_train,
        "avg_total_len": avg_train + 2.0,
        "val_examples": int(len(val)),
        "test_examples": int(len(test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a processed seqrec dataset")
    parser.add_argument("--root", type=Path, required=True, help="Path to the processed dataset root")
    args = parser.parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist")
    stats = _stats(root)
    print("Dataset • root=", root)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

