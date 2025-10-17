# src/ossm/data/datasets/uea.py
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from . import utils  # keep module import so monkeypatch works
from ..transforms.compose import Compose
from ..transforms.path import AddTime, NormalizeTime
from ..transforms.cde import ToCubicSplineCoeffs
from ..transforms.signature import ToWindowedLogSignature


class UEA(Dataset):
    """UEA/UCR multivariate time-series dataset with torchvision-style API.

    Views:
      - raw   : returns {'times',(T,), 'values',(T,C), 'label'}
      - coeff : adds cubic-spline coefficients computed in torch
      - path  : windowed log-signature features via torchsignature

    Loader compatibility:
      - Accepts loaders returning (times, values, labels) OR (values, labels).
        If times are omitted, a normalized [0,1] grid is synthesized.
    """

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "train",
        view: str = "raw",
        *,
        steps: int = 32,
        depth: int = 2,
        download: bool = False,
        loader: Optional[Callable[..., Tuple]] = None,
    ) -> None:
        super().__init__()
        self.root, self.name, self.split = root, name, split
        self.view = view.lower()
        self.steps, self.depth = int(steps), int(depth)

        if download:
            # Parity with torchvision-style datasets; creates expected folder layout.
            utils.ensure_uea_layout(root)

        # Resolve loader at runtime so monkeypatching `utils.load_uea_numpy` works
        loader_fn = loader if loader is not None else utils.load_uea_numpy

        # Call the loader; support both 2- and 3-tuple returns
        out = loader_fn(root, name, split)
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

        # Convert values/labels
        values = torch.as_tensor(values_np, dtype=torch.float32)  # (N, T, C)
        labels = utils.encode_labels(labels_np)  # (N,) long

        # If times missing, synthesize per-sample normalized grid [0,1]
        if times_tensor is None:
            N, T, _ = values.shape
            base = torch.linspace(0.0, 1.0, T, dtype=torch.float32)
            times_tensor = base.expand(N, T).clone()

        self.times = times_tensor  # (N, T)
        self.values = values  # (N, T, C)
        self.labels = labels  # (N,)

        # Build transform pipeline per view
        self.transform = self._build_pipeline(self.view)

    def _build_pipeline(self, view: str) -> Compose:
        if view == "raw":
            tfms = [AddTime(), NormalizeTime()]
        elif view == "coeff":
            tfms = [AddTime(), NormalizeTime(), ToCubicSplineCoeffs()]
        elif view == "path":
            # Let torchsignature do the windowing; don't pre-segment here.
            tfms = [
                AddTime(),
                NormalizeTime(),
                ToWindowedLogSignature(depth=self.depth, steps=self.steps),
            ]
        else:
            raise ValueError(f"Unknown view '{view}'. Expected 'raw'|'coeff'|'path'.")
        return Compose(tfms)

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.times[idx]  # (T,)
        x = self.values[idx]  # (T, C)
        y = self.labels[idx]  # ()
        sample = {"times": t, "values": x, "label": y}
        return self.transform(sample)
