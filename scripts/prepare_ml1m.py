"""Preprocess MovieLens-1M for sequential recommendation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MIN_RATING = 4.0


def _load_raw(path: Path) -> pd.DataFrame:
    ratings_dat = path / "ratings.dat"
    if ratings_dat.exists():
        df = pd.read_csv(
            ratings_dat,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return df
    ratings_csv = path / "ratings.csv"
    if ratings_csv.exists():
        df = pd.read_csv(ratings_csv)
        expected = {"userId", "movieId", "rating", "timestamp"}
        if not expected.issubset(df.columns):
            raise ValueError(f"ratings.csv must contain columns {expected}")
        return df.rename(columns={"userId": "user_id", "movieId": "item_id"})
    raise FileNotFoundError("Unable to locate ratings.dat or ratings.csv in the provided directory")


def _build_splits(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    df = df[df["rating"] >= MIN_RATING].copy()
    df.sort_values(["user_id", "timestamp"], inplace=True)

    item_counts = df["item_id"].value_counts()
    keep_items = set(item_counts[item_counts >= max(min_item_interactions, 1)].index)
    df = df[df["item_id"].isin(keep_items)]

    grouped = df.groupby("user_id", sort=False)

    kept_users: List[int] = []
    frames: List[pd.DataFrame] = []
    for user_id, group in grouped:
        if len(group) < max(min_user_interactions, 3):
            continue
        frames.append(group)
        kept_users.append(user_id)
    if not frames:
        raise ValueError("No users meet the minimum interaction threshold")

    df_kept = pd.concat(frames, ignore_index=True)
    user_mapping = {user: idx for idx, user in enumerate(sorted(set(kept_users)))}
    item_mapping = {item: idx + 1 for idx, item in enumerate(sorted(df_kept["item_id"].unique()))}
    num_items = len(item_mapping)

    df_kept["user_id"] = df_kept["user_id"].map(user_mapping)
    df_kept["item_id"] = df_kept["item_id"].map(item_mapping)

    train_records: List[Dict[str, int]] = []
    val_records: List[Dict[str, int]] = []
    test_records: List[Dict[str, int]] = []
    train_items: List[int] = []
    user_ptr = [0]

    for user_id, group in df_kept.groupby("user_id", sort=True):
        items = group["item_id"].tolist()
        timestamps = group["timestamp"].tolist()
        if len(items) < 3:
            continue
        train_seq = items[:-2]
        val_target = items[-2]
        test_target = items[-1]
        for item, ts in zip(train_seq, timestamps[:-2]):
            train_records.append({"user_id": int(user_id), "item_id": int(item), "ts": int(ts)})
            train_items.append(int(item))
        user_ptr.append(len(train_items))
        val_records.append({"user_id": int(user_id), "prefix_len": len(train_seq), "target_id": int(val_target)})
        test_records.append({"user_id": int(user_id), "prefix_len": len(train_seq) + 1, "target_id": int(test_target)})

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    test_df = pd.DataFrame(test_records)
    return train_df, val_df, test_df, np.asarray(user_ptr, dtype=np.int64), np.asarray(train_items, dtype=np.int32), num_items


def _write_outputs(
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_ptr: np.ndarray,
    train_items: np.ndarray,
    num_items: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    np.save(out_dir / "user_ptr.npy", user_ptr)
    np.save(out_dir / "train_items.npy", train_items)
    (out_dir / "item_count.txt").write_text(str(num_items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MovieLens-1M for seqrec experiments")
    parser.add_argument("--raw", type=Path, required=True, help="Path to the raw ml-1m directory")
    parser.add_argument("--out", type=Path, required=True, help="Directory where the processed data will be stored")
    parser.add_argument("--min-interactions", type=int, default=5, help="Minimum interactions per user")
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=None,
        help="Minimum interactions per item (defaults to --min-interactions)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_raw(args.raw)
    min_item = args.min_item_interactions if args.min_item_interactions is not None else args.min_interactions
    train_df, val_df, test_df, user_ptr, train_items, num_items = _build_splits(
        df,
        min_user_interactions=args.min_interactions,
        min_item_interactions=min_item,
    )
    _write_outputs(args.out, train_df, val_df, test_df, user_ptr, train_items, num_items)
    summary = {
        "users": int(user_ptr.size - 1),
        "items": int(num_items),
        "train_interactions": int(train_items.size),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
