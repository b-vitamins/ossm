# Oscillatory state-space models

PyTorch implementation for oscillatory state-space models. The project will include datasets, transforms, models, and training.
**Currently implemented:** UEA data pipeline with three views (`raw`, `coeff`, `path`).

## UEA multivariate archive

The [UEA/UCR multivariate time-series archive](https://timeseriesclassification.com)
collects benchmark classification problems with aligned sampling grids. Each
problem ships with disjoint training and testing splits stored as ARFF files;
after parsing the data live in tensors shaped `(num_samples, seq_len, channels)`.

### Mathematical views

* **`raw`** – exposes the observation matrix together with a normalized
time-grid `t_k` in `[0, 1]` and the encoded class label. This view is suitable
for sequence models that ingest the original measurements.
* **`coeff`** – augments `raw` with the backward Hermite cubic coefficients
`(d, c, b, a)` produced by `diffrax.backward_hermite_coefficients`. These
coefficients parameterise the unique cubic spline that interpolates the
trajectory and are flattened along the channel axis in the PyTorch view.
* **`path`** – segments the path into fixed windows, inserts basepoints, and
exports Hall-basis log-signatures of depth 2. The feature layout (scalar term,
linear words, then Lie brackets) matches the Signax/LinOSS preprocessing.

### Layout and usage

The loader mirrors torchvision datasets. Point `root` at a directory with the
following layout, or pass `download=True` to create it before populating the
ARFF files yourself:

```
<root>/
  raw/UEA/Multivariate_arff/<DatasetName>/
    <DatasetName>_TRAIN.arff
    <DatasetName>_TEST.arff
```

The dataset constructor accepts

* `name`: dataset key from the archive,
* `split`: `"train"` or `"test"` (no resampling performed),
* `view`: one of `"raw"`, `"coeff"`, or `"path"`,
* log-signature hyper-parameters (`steps`, `depth`, `basis`) when `view="path"`.

Each item is a dictionary with `times`, `values`, `label`, and when applicable
either `coeffs` or `features`. Collate helpers in `ossm.data.datasets` pad,
stack, and batch these tensors for PyTorch dataloaders.

## Install

```bash
pip install -e .[uea,signature]
```

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
