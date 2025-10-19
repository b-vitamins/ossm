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
* `split`: final subset to materialise (`"train"`, `"test"`, `"val"`, `"all"`, or a
  sequence such as `("train", "test")`),
* `view`: one of `"raw"`, `"coeff"`, or `"path"`,
* log-signature hyper-parameters (`steps`, `depth`, `basis`) when `view="path"`,
* `source_splits`: base splits to ingest before further manipulation (defaults to
  the requested `split` or both train/test when resampling),
* `deduplicate`: drop repeated trajectories by comparing the original time grids
  and observation matrices across the loaded source splits,
* `resample` + `resample_seed`: deterministically repartition the aggregated pool
  into new `train`/`val`/`test` subsets using either counts or proportions,
* `record_grid`: attach the pre-normalisation time grid as `sample["grid"]`,
* `record_source`: include `sample["source_index"]` and `sample["source_split"]`
  (encoded via `dataset.source_split_encoding`) tracing back to the original
  ARFF location.

Each item is a dictionary with `times`, `values`, `label`, and when applicable
either `coeffs` or `features`. Optional metadata (`grid`, `source_*`) survives the
transform pipeline so downstream code can faithfully reproduce the LinOSS data
layout while staying within idiomatic PyTorch workflows. Collate helpers in
`ossm.data.datasets` pad, stack, and batch these tensors for PyTorch dataloaders.

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

## Training with `train.py`

The training entrypoint is implemented in [`train.py`](./train.py) and is configured
with [Hydra](https://hydra.cc). The default configuration (`configs/config.yaml`)
trains a LinOSS backbone on the EthanolConcentration coefficients view and logs
artifacts under `./outputs`.

### Basic invocation

1. Download the desired UEA dataset so that the ARFF files live under
   `<data_root>/raw/UEA/Multivariate_arff/<DatasetName>/` as shown above.
2. Point the `OSSM_DATA_ROOT` environment variable at that directory (or accept
   the default `./data`).
3. Run the training script:

   ```bash
   OSSM_DATA_ROOT=/path/to/data python train.py
   ```

Hydra mirrors its working directory to `${OSSM_WORK_DIR:-./outputs}` and writes a
separate timestamped folder for each run containing the resolved configuration,
logs, and any checkpoints saved by callbacks.

### Customising runs with Hydra overrides

Any field inside the configuration tree can be overridden from the command line.
Common adjustments include:

* Switching the dataset split or representation:

  ```bash
  python train.py \
      dataset.name=GunPoint \
      validation_dataset.name=GunPoint \
      dataset.split=train \
      validation_dataset.split=validation \
      dataset.view=raw \
      validation_dataset.view=raw \
      dataloader.collate=pad \
      training.max_steps=2000
  ```

* Training on GPU with a different backbone:

  ```bash
  python train.py model=ncde training.device=cuda:0 optimizer.lr=3e-4
  ```

  When running in NRDE mode, remember to request the log-signature view and its
  collate helper:

  ```bash
  python train.py \
      model=ncde \
      model.params.mode=nrde \
      dataset.view=path \
      dataloader.collate=path \
      dataset.steps=16 \
      dataset.depth=2
  ```

* Sweeping over multiple backbones with Hydra's multirun support:

  ```bash
  python train.py --multirun model=linoss_im,lru training.max_steps=5000
  ```

Refer to the `configs/` directory for the available model, optimiser, and data
presets. Hydra will compose the requested defaults, apply CLI overrides, and
emit the fully-resolved configuration alongside the run logs.

## Notes

* UEA ARFF expected at: `<root>/raw/UEA/Multivariate_arff/<DatasetName>/<DatasetName>_{TRAIN|TEST}.arff`
* `view="path"` uses `torchsignature` (no-op if not installed) and emits depth-2 Hall-basis log-signatures with the scalar coordinate included, matching the Signax/LinOSS preprocessing.
