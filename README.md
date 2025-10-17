# Oscillatory state-space models

PyTorch implementation for oscillatory state-space models. The project will include datasets, transforms, models, and training. **Currently implemented:** UEA data pipeline with three views (`raw`, `coeff`, `path`).

## Install

```bash
pip install -e .[uea,signature]
````

## Quickstart

```python
from torch.utils.data import DataLoader
from ossm.data.datasets import UEA, pad_collate, coeff_collate, path_collate

# Raw (RNN/SSM)
train_raw = UEA(root="data_dir", name="GunPoint", split="train", view="raw")
raw_loader = DataLoader(train_raw, batch_size=64, shuffle=True, collate_fn=pad_collate)

# NCDE (cubic spline coefficients)
train_cde = UEA(root="data_dir", name="GunPoint", split="train", view="coeff")
cde_loader = DataLoader(train_cde, batch_size=64, shuffle=True, collate_fn=coeff_collate)

# NRDE / Log-NCDE (torchsignature)
train_path = UEA(
    root="data_dir",
    name="GunPoint",
    split="train",
    view="path",
    depth=2,
    steps=32,
)
path_loader = DataLoader(train_path, batch_size=64, shuffle=True, collate_fn=path_collate)
```

## Notes

* UEA ARFF expected at: `<root>/raw/UEA/Multivariate_arff/<DatasetName>/<DatasetName>_{TRAIN|TEST}.arff`
* `view="path"` uses `torchsignature` (no-op if not installed) and emits depth-2 Hall-basis log-signatures with the scalar coordinate included, matching the Signax/LinOSS preprocessing.
