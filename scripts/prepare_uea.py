"""Download and preprocess the UEA Multivariate Time-Series Archive."""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_from_arff_to_dataframe  # type: ignore[import]
from tqdm import tqdm

ARCHIVE_URL = (
    "http://www.timeseriesclassification.com/aeon-toolkit/Archives"
    "/Multivariate2018_arff.zip"
)
DEFAULT_ROOT = Path.home() / ".cache" / "torch" / "datasets" / "ossm"


class DownloadError(RuntimeError):
    """Raised when the archive download fails."""


def _download(url: str, dest: Path, *, force: bool = False) -> None:
    """Download *url* to *dest*, streaming bytes to disk."""

    if dest.exists():
        if not force:
            return
        dest.unlink()

    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(dest)
    except Exception as exc:  # pragma: no cover - defensive
        if tmp_path.exists():
            tmp_path.unlink()
        raise DownloadError(f"Failed to download '{url}': {exc}") from exc


def _ensure_archive(url: str, root: Path, *, force: bool = False) -> Path:
    """Ensure the compressed archive is present under *root* and return its path."""

    archive_path = root / "raw" / "UEA" / "Multivariate2018_arff.zip"
    _download(url, archive_path, force=force)
    return archive_path


def _extract_archive(archive_path: Path, target_dir: Path, *, force: bool = False) -> None:
    """Extract *archive_path* into *target_dir* if needed."""

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        return

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)

    nested_root = target_dir / "Multivariate_arff"
    if nested_root.is_dir():
        for child in nested_root.iterdir():
            destination = target_dir / child.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            child.rename(destination)
        nested_root.rmdir()


def _convert_dataframe(data_frame) -> np.ndarray:
    """Convert a sktime dataframe into a (N, T, C) float32 array."""

    data_expand = data_frame.map(lambda x: x.values).values
    stacked = [np.vstack(sample).T for sample in data_expand]
    return np.stack(stacked).astype(np.float32)


def _load_split(arff_dir: Path, dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    file_path = arff_dir / dataset / f"{dataset}_{split.upper()}.arff"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing {split} file for dataset '{dataset}'")

    data_frame, labels = load_from_arff_to_dataframe(str(file_path))
    values = _convert_dataframe(data_frame)
    return values, np.asarray(labels)


def _encode_labels(
    train_labels: np.ndarray, test_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    return encoder.transform(train_labels).astype(np.int64), encoder.transform(test_labels).astype(
        np.int64
    )


def _save_pickle(obj, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _process_dataset(dataset: str, arff_dir: Path, processed_dir: Path, *, force: bool = False) -> None:
    target_dir = processed_dir / dataset
    if target_dir.exists() and not force:
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train_raw = _load_split(arff_dir, dataset, "train")
    X_test, y_test_raw = _load_split(arff_dir, dataset, "test")
    y_train, y_test = _encode_labels(y_train_raw, y_test_raw)

    data = np.concatenate([X_train, X_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0)

    original_indices = (
        np.arange(0, X_train.shape[0], dtype=np.int64),
        np.arange(X_train.shape[0], data.shape[0], dtype=np.int64),
    )

    _save_pickle(X_train, target_dir / "X_train.pkl")
    _save_pickle(y_train, target_dir / "y_train.pkl")
    _save_pickle(X_test, target_dir / "X_test.pkl")
    _save_pickle(y_test, target_dir / "y_test.pkl")
    _save_pickle(data, target_dir / "data.pkl")
    _save_pickle(labels, target_dir / "labels.pkl")
    _save_pickle(original_indices, target_dir / "original_idxs.pkl")

def list_datasets(arff_dir: Path) -> Sequence[str]:
    datasets = []
    for entry in sorted(arff_dir.iterdir()):
        if entry.is_dir():
            train_file = entry / f"{entry.name}_TRAIN.arff"
            test_file = entry / f"{entry.name}_TEST.arff"
            if train_file.exists() and test_file.exists():
                datasets.append(entry.name)
    return datasets


def prepare_uea(
    *,
    root: Path,
    datasets: Optional[Sequence[str]] = None,
    force_download: bool = False,
    force_extract: bool = False,
    force_process: bool = False,
) -> None:
    """Download and preprocess the requested UEA datasets."""

    archive_path = _ensure_archive(ARCHIVE_URL, root, force=force_download)
    raw_arff_dir = root / "raw" / "UEA" / "Multivariate_arff"
    _extract_archive(archive_path, raw_arff_dir, force=force_extract or force_download)

    available = list_datasets(raw_arff_dir)
    if not available:
        raise RuntimeError("Archive extraction succeeded but no datasets were found.")

    if datasets is None or len(datasets) == 0:
        selected = available
    else:
        missing = sorted(set(datasets) - set(available))
        if missing:
            raise ValueError(f"Requested datasets not found in archive: {', '.join(missing)}")
        selected = list(datasets)

    processed_dir = root / "processed" / "UEA"
    for name in tqdm(selected, desc="Processing UEA datasets"):
        _process_dataset(name, raw_arff_dir, processed_dir, force=force_process)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("OSSM_DATA_ROOT", DEFAULT_ROOT)),
        help=(
            "Root directory for the dataset layout. Defaults to $OSSM_DATA_ROOT if set, "
            "otherwise ~/.cache/torch/datasets/ossm"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Specific dataset names to process (default: all datasets in the archive).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the archive, overwriting existing raw files.",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Re-process datasets even if processed files already exist.",
    )
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    root = args.root.expanduser().resolve()

    prepare_uea(
        root=root,
        datasets=args.datasets,
        force_download=args.force_download,
        force_extract=args.force_extract,
        force_process=args.force_process,
    )


if __name__ == "__main__":
    main()
