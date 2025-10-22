"""Sequential recommendation datasets and collate utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = [
    "SeqRecBatch",
    "SeqRecTrainDataset",
    "SeqRecEvalDataset",
    "collate_left_pad",
]


@dataclass
class SeqRecBatch:
    """Container holding a mini-batch for sequential recommendation."""

    input_ids: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    user_ids: torch.Tensor

    def to(
        self, device: torch.device | str, *, non_blocking: bool = True
    ) -> "SeqRecBatch":
        """Move tensors to the requested device."""

        device = torch.device(device)
        return SeqRecBatch(
            input_ids=self.input_ids.to(device=device, non_blocking=non_blocking),
            target=self.target.to(device=device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
            user_ids=self.user_ids.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "SeqRecBatch":  # pragma: no cover - DataLoader hook
        return SeqRecBatch(
            input_ids=self.input_ids.pin_memory(),
            target=self.target.pin_memory(),
            mask=self.mask.pin_memory(),
            user_ids=self.user_ids.pin_memory(),
        )


class SeqRecTrainDataset(Dataset[Tuple[int, List[int], int]]):
    """Next-item training examples built from per-user interaction streams."""

    def __init__(self, root: str | Path, max_len: int) -> None:
        self.root = Path(root)
        self.max_len = int(max_len)

        user_ptr_path = self.root / "user_ptr.npy"
        items_path = self.root / "train_items.npy"
        vocab_path = self.root / "item_count.txt"
        if not user_ptr_path.exists() or not items_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(
                "Expected 'user_ptr.npy', 'train_items.npy', and 'item_count.txt' to exist under "
                f"{self.root}"
            )

        self._user_ptr = np.load(user_ptr_path)
        self._train_items = np.load(items_path)
        self.num_items = int(Path(vocab_path).read_text().strip())
        if self._user_ptr.ndim != 1:
            raise ValueError("user_ptr.npy must be a 1D array of offsets")
        if self._user_ptr.size == 0:
            raise ValueError("user_ptr.npy is empty")
        if self._user_ptr[-1] != self._train_items.size:
            raise ValueError("Mismatch between user_ptr offsets and train_items length")
        if not np.all(self._user_ptr[1:] >= self._user_ptr[:-1]):
            raise ValueError("user_ptr offsets must be non-decreasing")

        self._user_sequences: List[Tuple[int, ...]] = []
        self._index: List[Tuple[int, int]] = []
        for user_id in range(self.num_users):
            start = int(self._user_ptr[user_id])
            end = int(self._user_ptr[user_id + 1])
            seq = tuple(int(item) for item in self._train_items[start:end].tolist())
            self._user_sequences.append(seq)
            for target_pos in range(1, len(seq)):
                self._index.append((user_id, target_pos))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[int, List[int], int]:  # type: ignore[override]
        user_id, target_pos = self._index[idx]
        seq = self._user_sequences[user_id]
        context = list(seq[:target_pos])
        target = int(seq[target_pos])
        return user_id, context, target

    @property
    def num_users(self) -> int:
        return self._user_ptr.size - 1

    def user_sequence(self, user_id: int) -> List[int]:
        return list(self._user_sequences[user_id])


class SeqRecEvalDataset(Dataset[Tuple[int, List[int], int]]):
    """Evaluation dataset exposing a single next-item example per user."""

    def __init__(self, root: str | Path, split: str, max_len: int) -> None:
        if split not in {"val", "test"}:
            raise ValueError("split must be 'val' or 'test'")
        self.root = Path(root)
        self.split = split
        self.max_len = int(max_len)

        user_ptr_path = self.root / "user_ptr.npy"
        items_path = self.root / "train_items.npy"
        vocab_path = self.root / "item_count.txt"
        self._user_ptr = np.load(user_ptr_path)
        self._train_items = np.load(items_path)
        self.num_items = int(Path(vocab_path).read_text().strip())
        self._user_sequences: List[Tuple[int, ...]] = []
        for user_id in range(self.num_users):
            start = int(self._user_ptr[user_id])
            end = int(self._user_ptr[user_id + 1])
            seq = tuple(int(item) for item in self._train_items[start:end].tolist())
            self._user_sequences.append(seq)

        if split == "test":
            val_path = self.root / "val.parquet"
            if not val_path.exists():
                raise FileNotFoundError("val.parquet is required to build test contexts")
            val_df = pd.read_parquet(val_path)
            val_user_ids = val_df["user_id"].to_numpy(dtype=np.int64, copy=False)
            val_targets = val_df["target_id"].to_numpy(dtype=np.int64, copy=False)
            self._val_targets: Dict[int, int] = {
                int(user_id): int(target) for user_id, target in zip(val_user_ids, val_targets)
            }
        else:
            self._val_targets = {}

        data_path = self.root / f"{split}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing split file '{data_path}'")
        df = pd.read_parquet(data_path)

        self._examples: List[Tuple[int, List[int], int]] = []
        self._contexts: Dict[int, Tuple[int, ...]] = {}
        self._histories: Dict[int, torch.Tensor] = {}
        user_column = df["user_id"].to_numpy(dtype=np.int64, copy=False)
        prefix_column = df["prefix_len"].to_numpy(dtype=np.int64, copy=False)
        target_column = df["target_id"].to_numpy(dtype=np.int64, copy=False)
        for user_id_raw, prefix_raw, target_raw in zip(
            user_column, prefix_column, target_column
        ):
            user_id = int(user_id_raw)
            prefix_len = int(prefix_raw)
            target = int(target_raw)
            train_seq = self._user_sequences[user_id]
            context = train_seq[:min(prefix_len, len(train_seq))]
            if self.split == "test" and prefix_len > len(train_seq):
                extra = prefix_len - len(train_seq)
                if extra == 1:
                    val_item = self._val_targets.get(user_id)
                    if val_item is None:
                        raise KeyError(f"No validation target recorded for user {user_id}")
                    context = list(train_seq) + [val_item]
                else:
                    raise ValueError(
                        "Test prefix length exceeds training sequence by more than one interaction"
                    )
            context_tuple = tuple(context)
            self._examples.append((user_id, list(context_tuple), target))
            self._contexts[user_id] = context_tuple
            self._histories[user_id] = torch.as_tensor(context_tuple, dtype=torch.long)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._examples)

    def __getitem__(self, idx: int) -> Tuple[int, List[int], int]:  # type: ignore[override]
        user_id, context, target = self._examples[idx]
        truncated = context[-self.max_len :]
        return user_id, truncated, target

    @property
    def num_users(self) -> int:
        return self._user_ptr.size - 1

    def context_for_user(self, user_id: int) -> List[int]:
        return list(self._contexts[user_id])

    def history_tensor(self, user_id: int) -> torch.Tensor:
        return self._histories[user_id]


def collate_left_pad(samples: Iterable[Tuple[int, Sequence[int], int]], max_len: int) -> SeqRecBatch:
    """Left-pad item sequences to a fixed maximum length."""

    samples_list = list(samples)
    batch_size = len(samples_list)
    if batch_size == 0:
        raise ValueError("collate_left_pad received an empty batch")

    padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)
    user_ids = torch.zeros(batch_size, dtype=torch.long)

    for row, (user_id, context, target) in enumerate(samples_list):
        trimmed = list(context)[-max_len:]
        length = len(trimmed)
        if length:
            padded[row, max_len - length :].copy_(
                torch.as_tensor(trimmed, dtype=torch.long)
            )
            mask[row, max_len - length :] = True
        targets[row] = int(target)
        user_ids[row] = int(user_id)

    return SeqRecBatch(input_ids=padded, target=targets, mask=mask, user_ids=user_ids)

