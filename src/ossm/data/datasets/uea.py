# src/ossm/data/datasets/uea.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from . import utils  # keep module import so monkeypatch works
from ..transforms.compose import Compose, TimeSeriesSample
from ..transforms.path import AddTime, NormalizeTime
from ..transforms.cde import ToCubicSplineCoeffs
from ..transforms.signature import ToWindowedLogSignature


class UEA(Dataset):
    """UEA/UCR multivariate time-series dataset with torchvision-style API.

    Views:
      - raw   : returns {'times',(T,), 'values',(T,C), 'label'}
      - coeff : adds cubic-spline coefficients computed in torch
      - path  : windowed log-signature features via torchsignature (Hall basis by default)

    Loader compatibility:
      - Accepts loaders returning (times, values, labels) OR (values, labels).
        If times are omitted, a normalized [0,1] grid is synthesized.
    """

    def __init__(
        self,
        root: str,
        name: str,
        split: Union[str, Sequence[str]] = "train",
        view: str = "raw",
        *,
        steps: int = 32,
        depth: int = 2,
        download: bool = False,
        loader: Optional[Callable[..., Tuple]] = None,
        basis: str = "hall",
        source_splits: Optional[Sequence[str]] = None,
        deduplicate: bool = False,
        resample: Optional[Dict[str, Union[int, float]]] = None,
        resample_seed: Optional[int] = None,
        record_grid: bool = False,
        record_source: bool = False,
    ) -> None:
        super().__init__()
        self.root, self.name = root, name
        self.record_grid = bool(record_grid)
        self.record_source = bool(record_source)
        self.view = view.lower()
        self.steps, self.depth = int(steps), int(depth)
        self.basis = str(basis)

        if download:
            # Parity with torchvision-style datasets; creates expected folder layout.
            utils.ensure_uea_layout(root)

        self._target_split = self._normalize_target(split)
        base_splits = self._normalize_sources(split, source_splits, resample)
        self._source_encoding = {"train": 0, "test": 1}
        self.source_split_encoding = dict(self._source_encoding)

        loader_fn = loader if loader is not None else utils.load_uea_numpy

        times_parts: List[torch.Tensor] = []
        values_parts: List[torch.Tensor] = []
        labels_parts: List[torch.Tensor] = []
        split_labels: List[str] = []
        index_parts: List[torch.Tensor] = []

        for base_split in base_splits:
            out = loader_fn(root, name, base_split)
            if not isinstance(out, tuple) or len(out) not in (2, 3):
                raise TypeError(
                    "UEA loader must return (values, labels) or (times, values, labels)"
                )

            times_tensor: Optional[torch.Tensor] = None
            if len(out) == 3:
                times_np, values_np, labels_np = out
                times_tensor = torch.as_tensor(times_np, dtype=torch.float32)
            else:
                values_np, labels_np = out

            values = torch.as_tensor(values_np, dtype=torch.float32)
            labels = utils.encode_labels(labels_np)

            if times_tensor is None:
                N, T, _ = values.shape
                base = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
                times_tensor = base.expand(N, T).clone()

            times_parts.append(times_tensor)
            values_parts.append(values)
            labels_parts.append(labels)
            split_labels.extend([base_split] * values.shape[0])
            index_parts.append(torch.arange(values.shape[0], dtype=torch.long))

        self.times = (
            torch.cat(times_parts, dim=0)
            if times_parts
            else torch.empty((0, 0), dtype=torch.float32)
        )
        self.values = (
            torch.cat(values_parts, dim=0)
            if values_parts
            else torch.empty((0, 0, 0), dtype=torch.float32)
        )
        self.labels = (
            torch.cat(labels_parts, dim=0)
            if labels_parts
            else torch.empty((0,), dtype=torch.long)
        )
        self._source_split_names = split_labels
        self._source_indices = (
            torch.cat(index_parts, dim=0)
            if index_parts
            else torch.empty((0,), dtype=torch.long)
        )

        if deduplicate and self.values.numel():
            keep = self._deduplicate_indices()
            self._apply_index(keep)

        self._resampled_indices = None
        if resample is not None and self.values.numel():
            resample_dict = resample
            if not isinstance(resample_dict, dict):
                resample_dict = dict(resample_dict)
            self._resampled_indices = self._build_resample_indices(resample_dict, resample_seed)

        selection = self._select_indices(self._target_split)
        if selection is not None:
            self._apply_index(selection)

        self.split = self._target_split

        self.transform: Compose[TimeSeriesSample] = self._build_pipeline(self.view)

    def _build_pipeline(self, view: str) -> Compose[TimeSeriesSample]:
        transforms: List[Callable[[TimeSeriesSample], TimeSeriesSample]]
        if view == "raw":
            transforms = [AddTime(), NormalizeTime()]
        elif view == "coeff":
            transforms = [AddTime(), NormalizeTime(), ToCubicSplineCoeffs()]
        elif view == "path":
            # Let torchsignature do the windowing; don't pre-segment here.
            basis = self.basis if self.depth == 2 else "lyndon"
            transforms = [
                AddTime(),
                NormalizeTime(),
                ToWindowedLogSignature(
                    depth=self.depth,
                    steps=self.steps,
                    basis=basis,
                ),
            ]
        else:
            raise ValueError(f"Unknown view '{view}'. Expected 'raw'|'coeff'|'path'.")
        return Compose(transforms)

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, idx: int) -> TimeSeriesSample:
        t = self.times[idx]  # (T,)
        x = self.values[idx]  # (T, C)
        y = self.labels[idx]  # ()
        sample: TimeSeriesSample = {"times": t, "values": x, "label": y}
        if self.record_grid:
            sample["grid"] = t.clone()
        if self.record_source:
            sample["source_index"] = self._source_indices[idx]
            sample["source_split"] = torch.tensor(
                self._source_encoding[self._source_split_names[idx]],
                dtype=torch.long,
            )
        return self.transform(sample)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_target(self, split: Union[str, Sequence[str]]) -> str:
        if isinstance(split, (list, tuple, set)):
            return "all"
        if not isinstance(split, str):
            raise TypeError("split must be a string or sequence of strings")
        normalized = split.lower()
        if normalized in {"train", "test", "val", "all"}:
            return normalized
        if normalized in {"full", "train+test", "train_test"}:
            return "all"
        raise ValueError(f"Unknown split '{split}'.")

    def _normalize_sources(
        self,
        split: Union[str, Sequence[str]],
        source_splits: Optional[Sequence[str]],
        resample: Optional[Dict[str, Union[int, float]]],
    ) -> List[str]:
        if source_splits is not None:
            sources = [self._validate_base_split(s) for s in source_splits]
            if not sources:
                raise ValueError("source_splits must contain at least one split")
            return sources

        if isinstance(split, (list, tuple, set)):
            return [self._validate_base_split(s) for s in split]

        target = split.lower() if isinstance(split, str) else "all"
        if resample is not None or target in {"val", "all", "full", "train+test", "train_test"}:
            return ["train", "test"]
        return [self._validate_base_split(target)]

    @staticmethod
    def _validate_base_split(split: str) -> str:
        normalized = split.lower()
        if normalized not in {"train", "test"}:
            raise ValueError("Only 'train' and 'test' splits exist in the UEA archive")
        return normalized

    def _apply_index(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            self.times = self.times.new_empty((0, self.times.shape[-1]))
            self.values = self.values.new_empty((0,) + self.values.shape[1:])
            self.labels = self.labels.new_empty((0,))
            self._source_split_names = []
            self._source_indices = torch.empty(0, dtype=torch.long)
            return

        indices = indices.to(dtype=torch.long)
        self.times = self.times.index_select(0, indices)
        self.values = self.values.index_select(0, indices)
        self.labels = self.labels.index_select(0, indices)
        self._source_indices = self._source_indices.index_select(0, indices)
        self._source_split_names = [self._source_split_names[i] for i in indices.tolist()]

    def _deduplicate_indices(self) -> torch.Tensor:
        flat_times = self.times.reshape(self.times.size(0), -1).cpu().numpy()
        flat_values = self.values.reshape(self.values.size(0), -1).cpu().numpy()
        signature = np.concatenate([flat_times, flat_values], axis=1)
        _, keep = np.unique(signature, axis=0, return_index=True)
        keep.sort()
        return torch.as_tensor(keep, dtype=torch.long)

    def _build_resample_indices(
        self,
        resample: Dict[str, Union[int, float]],
        seed: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(resample, dict) or not resample:
            raise TypeError("resample must be a non-empty dict")

        keys: List[str] = []
        values: List[Union[int, float]] = []
        for key, value in resample.items():
            normalized = key.lower()
            if normalized not in {"train", "val", "test"}:
                raise ValueError("Resample keys must be 'train', 'val', or 'test'")
            keys.append(normalized)
            values.append(value)

        total = self.values.shape[0]
        if all(isinstance(v, int) for v in values):
            counts = [int(v) for v in values]
            if sum(counts) != total:
                raise ValueError("Sum of resample counts must equal dataset size")
        else:
            weights = torch.tensor([float(v) for v in values], dtype=torch.float64)
            if torch.any(weights <= 0):
                raise ValueError("Resample proportions must be positive")
            weights = weights / weights.sum()
            base = torch.floor(weights * total).to(dtype=torch.long)
            remainder = total - int(base.sum().item())
            if remainder > 0:
                fractions = weights * total - base.to(weights.dtype)
                order = torch.argsort(fractions, descending=True)
                for idx in order.tolist():
                    if remainder <= 0:
                        break
                    base[idx] += 1
                    remainder -= 1
            counts = base.tolist()
            if sum(counts) != total:
                counts[-1] += total - sum(counts)

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        permutation = torch.randperm(total, generator=generator)

        out: Dict[str, torch.Tensor] = {}
        start = 0
        for key, count in zip(keys, counts):
            end = start + count
            out[key] = permutation[start:end]
            start = end
        if start != total:
            raise RuntimeError("Resampling did not allocate all samples")
        return out

    def _select_indices(self, target: str) -> Optional[torch.Tensor]:
        if target == "all":
            return torch.arange(self.values.size(0), dtype=torch.long)
        if self._resampled_indices is not None:
            if target not in self._resampled_indices:
                raise ValueError(
                    f"Requested split '{target}' missing from resample specification"
                )
            return self._resampled_indices[target]

        mask = [i for i, split in enumerate(self._source_split_names) if split == target]
        if not mask:
            raise ValueError(f"Split '{target}' has no samples under the current configuration")
        return torch.as_tensor(mask, dtype=torch.long)
