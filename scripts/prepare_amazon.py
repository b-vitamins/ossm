"""Preprocess Amazon review subsets for sequential recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

VALID_SUBSETS = {"beauty": "Beauty", "videogames": "Video_Games"}


def _find_file(raw_dir: Path, subset_key: str) -> Path:
    pretty = VALID_SUBSETS[subset_key]
    candidates = [
        raw_dir / f"reviews_{pretty}.json.gz",
        raw_dir / f"reviews_{pretty}.json",
        raw_dir / f"reviews_{pretty}_5.json.gz",
        raw_dir / f"reviews_{pretty}_5.json",
        raw_dir / f"{subset_key}.json.gz",
        raw_dir / f"{subset_key}.json",
        raw_dir / f"{subset_key}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to find a data file for subset '{subset_key}'. Checked: {[str(c) for c in candidates]}"
    )


def _load_raw(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True, compression="infer")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {}
    if "reviewerID" in df.columns:
        column_map["reviewerID"] = "user_id"
    elif "user_id" in df.columns:
        column_map["user_id"] = "user_id"
    else:
        raise ValueError("Could not infer user id column")

    if "asin" in df.columns:
        column_map["asin"] = "item_id"
    elif "item_id" in df.columns:
        column_map["item_id"] = "item_id"
    else:
        raise ValueError("Could not infer item id column")

    if "unixReviewTime" in df.columns:
        column_map["unixReviewTime"] = "timestamp"
    elif "unix_review_time" in df.columns:
        column_map["unix_review_time"] = "timestamp"
    elif "timestamp" in df.columns:
        column_map["timestamp"] = "timestamp"
    else:
        raise ValueError("Could not infer timestamp column")

    normalized = df.rename(columns=column_map)
    missing = {"user_id", "item_id", "timestamp"} - set(normalized.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")
    return normalized[["user_id", "item_id", "timestamp"]]


def _build_splits(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    df = df.copy()
    df.sort_values(["user_id", "timestamp"], inplace=True)

    # Apply k-core style filtering so that both user and item thresholds hold
    # simultaneously (mirrors RecBole-style preprocessing used by Mamba4Rec).
    changed = True
    while changed:
        before = len(df)
        if min_item_interactions > 1:
            item_counts = df["item_id"].value_counts()
            keep_items = set(item_counts[item_counts >= min_item_interactions].index)
            df = df[df["item_id"].isin(keep_items)]
        if min_user_interactions > 1:
            user_counts = df["user_id"].value_counts()
            keep_users = set(user_counts[user_counts >= min_user_interactions].index)
            df = df[df["user_id"].isin(keep_users)]
        changed = len(df) != before

    grouped = df.groupby("user_id", sort=False)

    frames: List[pd.DataFrame] = []
    for _, group in grouped:
        if len(group) < max(min_user_interactions, 3):
            continue
        frames.append(group)
    if not frames:
        raise ValueError("No users meet the minimum interaction threshold")

    df_kept = pd.concat(frames, ignore_index=True)
    user_mapping = {user: idx for idx, user in enumerate(sorted(df_kept["user_id"].unique()))}
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
    parser = argparse.ArgumentParser(description="Prepare Amazon review subset for seqrec experiments")
    parser.add_argument("--subset", choices=sorted(VALID_SUBSETS.keys()), required=True, help="Amazon subset to preprocess")
    parser.add_argument("--raw", type=Path, required=True, help="Directory containing the raw Amazon data files")
    parser.add_argument("--out", type=Path, required=True, help="Directory to store the processed outputs")
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
    file_path = _find_file(args.raw, args.subset)
    raw_df = _load_raw(file_path)
    df = _normalize_columns(raw_df)
    min_item = args.min_item_interactions if args.min_item_interactions is not None else args.min_interactions
    train_df, val_df, test_df, user_ptr, train_items, num_items = _build_splits(
        df,
        min_user_interactions=args.min_interactions,
        min_item_interactions=min_item,
    )
    _write_outputs(args.out, train_df, val_df, test_df, user_ptr, train_items, num_items)
    summary = {
        "subset": args.subset,
        "users": int(user_ptr.size - 1),
        "items": int(num_items),
        "train_interactions": int(train_items.size),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
