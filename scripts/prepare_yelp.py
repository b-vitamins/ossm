"""Preprocess the Yelp Open Dataset reviews for sequential recommendation.

This matches the format expected by the seqrec dataloaders and mirrors the
protocol used for the Amazon/MovieLens helpers:

  - Treat all reviews as implicit interactions (no rating threshold).
  - Apply k-core filtering so that both users and items meet the minimum
    interaction counts simultaneously.
  - Sort by timestamp per user and perform a leave-one-out split: last item →
    validation target, last two items → test target.

Input sources
-------------
The Yelp download distributed as ``Yelp-JSON.zip`` contains a TAR archive with
``yelp_academic_dataset_review.json``. This script can operate either on the
ZIP directly (``--zip``), a directory containing the JSON file (``--raw``), or
the path to the JSONL file itself (``--json``).

Example usage
-------------
  guix shell -m manifest.scm -- \
    python3 scripts/prepare_yelp.py \
      --json data/raw/yelp/yelp_academic_dataset_review.json \
      --out data/seqrec/yelp \
      --min-interactions 5

Notes
-----
Reading/writing Parquet requires either ``pyarrow`` or ``fastparquet``. The
project's ``manifest.scm`` includes these dependencies by default; if running
outside Guix, ``pip install pyarrow`` is sufficient.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _resolve_json_path(zip_path: Path | None, raw_dir: Path | None, json_path: Path | None) -> Path:
    """Locate (and optionally extract) the JSONL file with Yelp reviews.

    Preference order: ``--json`` > ``--raw`` > ``--zip``.
    """

    if json_path is not None:
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        return json_path

    if raw_dir is not None:
        candidates = [
            raw_dir / "yelp_academic_dataset_review.json",
            raw_dir / "Yelp JSON" / "yelp_academic_dataset_review.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"Unable to find yelp_academic_dataset_review.json under {raw_dir}"
        )

    if zip_path is None:
        raise ValueError("One of --json, --raw, or --zip must be provided")

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")

    # Extract the inner TAR archive from the ZIP into a temporary folder next
    # to the ZIP file to avoid polluting the repository root.
    extract_root = zip_path.parent / ".yelp_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = {info.filename: info for info in zf.infolist()}
        tar_name = None
        # Look for the canonical path first, then fall back to any *.tar entry.
        for cand in (
            "Yelp JSON/yelp_dataset.tar",
            "yelp_dataset.tar",
        ):
            if cand in members:
                tar_name = cand
                break
        if tar_name is None:
            for name in members:
                if name.lower().endswith(".tar"):
                    tar_name = name
                    break
        if tar_name is None:
            raise FileNotFoundError("Could not locate yelp_dataset.tar inside the zip")

        tar_path = extract_root / Path(tar_name).name
        if not tar_path.exists():
            with zf.open(tar_name) as src, open(tar_path, "wb") as dst:
                dst.write(src.read())

    # Now extract just the review JSON from the TAR.
    with tarfile.open(tar_path) as tf:
        member = None
        for cand in (
            "yelp_academic_dataset_review.json",
            "Yelp JSON/yelp_academic_dataset_review.json",
        ):
            try:
                member = tf.getmember(cand)
                break
            except KeyError:
                continue
        if member is None:
            # Fall back: search members for the expected filename.
            for m in tf.getmembers():
                if Path(m.name).name == "yelp_academic_dataset_review.json":
                    member = m
                    break
        if member is None:
            raise FileNotFoundError(
                "yelp_academic_dataset_review.json is missing from yelp_dataset.tar"
            )

        out_json = extract_root / "yelp_academic_dataset_review.json"
        if not out_json.exists():
            tf.extract(member, path=extract_root)
            extracted = extract_root / member.name
            if extracted != out_json:
                out_json.write_bytes(extracted.read_bytes())
        return out_json


def _load_reviews(json_path: Path, min_stars: float | None = None) -> pd.DataFrame:
    """Load minimal columns from the Yelp review JSONL.

    Returns a DataFrame with columns: ``user_id``, ``item_id``, ``timestamp``.
    """

    # Prefer pyarrow for speed and memory usage; fall back to pandas if
    # pyarrow.json is unavailable.
    try:
        import pyarrow as pa
        import pyarrow.json as pajson

        # Only parse the columns we need.
        schema = pa.schema(
            [
                ("user_id", pa.string()),
                ("business_id", pa.string()),
                ("date", pa.string()),
            ]
        )
        table = pajson.read_json(
            str(json_path),
            read_options=pajson.ReadOptions(),
            parse_options=pajson.ParseOptions(explicit_schema=schema),
        )
        cols = ["user_id", "business_id", "date"]
        if "stars" in table.column_names:
            cols.append("stars")
        df = table.select(cols).to_pandas(types_mapper={})
    except Exception:  # pragma: no cover - runtime dependency gate
        df = pd.read_json(json_path, lines=True)
        keep = [c for c in ("user_id", "business_id", "date", "stars") if c in df.columns]
        df = df[keep]

    df = df.rename(columns={"business_id": "item_id"})
    if min_stars is not None and "stars" in df.columns:
        df = df[df["stars"] >= float(min_stars)]
    # Convert date to a UNIX timestamp (seconds since epoch) to allow
    # deterministic sorting and consistent parity with other preprocessors.
    ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
    # Fill NaT with a sentinel so sort is still defined (rare/malformed rows).
    ts = ts.fillna(pd.Timestamp(0, unit="s", tz="UTC"))
    df["timestamp"] = (ts.view("int64") // 1_000_000_000).astype(np.int64)
    return df[["user_id", "item_id", "timestamp"]]


def _kcore_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Apply k-core filtering until both constraints hold simultaneously."""

    pruned = df
    changed = True
    while changed:
        before = len(pruned)
        if min_item > 1:
            item_counts = pruned["item_id"].value_counts()
            keep_items = set(item_counts[item_counts >= min_item].index)
            pruned = pruned[pruned["item_id"].isin(keep_items)]
        if min_user > 1:
            user_counts = pruned["user_id"].value_counts()
            keep_users = set(user_counts[user_counts >= min_user].index)
            pruned = pruned[pruned["user_id"].isin(keep_users)]
        changed = len(pruned) != before
    return pruned


def _build_splits(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    # Sort by user then timestamp to build per-user streams.
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df = _kcore_filter(df, min_user_interactions, min_item_interactions)

    # Drop users with fewer than 3 total interactions to allow leave-one-out.
    frames: List[pd.DataFrame] = []
    kept_users: List[str] = []
    for user, group in df.groupby("user_id", sort=False):
        if len(group) < max(min_user_interactions, 3):
            continue
        frames.append(group)
        kept_users.append(str(user))
    if not frames:
        raise ValueError("No users meet the minimum interaction threshold")

    df_kept = pd.concat(frames, ignore_index=True)

    # Map to contiguous integer ids (users: 0..U-1, items: 1..I).
    user_mapping = {user: idx for idx, user in enumerate(sorted(set(kept_users)))}
    item_mapping = {
        item: idx + 1 for idx, item in enumerate(sorted(df_kept["item_id"].unique()))
    }
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
    return (
        train_df,
        val_df,
        test_df,
        np.asarray(user_ptr, dtype=np.int64),
        np.asarray(train_items, dtype=np.int32),
        num_items,
    )


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
    p = argparse.ArgumentParser(description="Prepare Yelp reviews for seqrec experiments")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=Path, help="Path to yelp_academic_dataset_review.json")
    src.add_argument("--raw", type=Path, help="Directory that contains the JSON file")
    src.add_argument("--zip", type=Path, help="Path to Yelp-JSON.zip (contains yelp_dataset.tar)")
    p.add_argument("--out", type=Path, required=True, help="Directory to store the processed outputs")
    p.add_argument("--min-interactions", type=int, default=5, help="Minimum interactions per user")
    p.add_argument(
        "--min-item-interactions",
        type=int,
        default=None,
        help="Minimum interactions per item (defaults to --min-interactions)",
    )
    p.add_argument(
        "--min-stars",
        type=float,
        default=None,
        help="Filter to reviews with at least this star rating (e.g., 4.0)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    json_path = _resolve_json_path(args.zip, args.raw, args.json)
    raw_df = _load_reviews(json_path, min_stars=args.min_stars)
    min_item = args.min_item_interactions if args.min_item_interactions is not None else args.min_interactions
    train_df, val_df, test_df, user_ptr, train_items, num_items = _build_splits(
        raw_df,
        min_user_interactions=args.min_interactions,
        min_item_interactions=min_item,
    )
    _write_outputs(args.out, train_df, val_df, test_df, user_ptr, train_items, num_items)
    summary = {
        "dataset": "yelp",
        "users": int(user_ptr.size - 1),
        "items": int(num_items),
        "train_interactions": int(train_items.size),
        "avg_total_len": float(np.diff(user_ptr).mean() + 2.0) if user_ptr.size > 1 else 0.0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
