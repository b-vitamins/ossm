# Oscillatory state-space models

PyTorch utilities for oscillatory state-space models (OSSM) that power RNN, SSM,
NCDE, and NRDE experiments. The library bundles reusable dataset loaders,
transform pipelines, and training entrypoints so that downstream projects can
focus on modelling instead of bespoke preprocessing.

## Highlights

- **UEA multivariate archive pipeline** with three ready-to-use representations
  (`raw`, `coeff`, and `path`) that mirror the torchvision dataset API.
- **Compositional data transforms** for cubic-spline NCDE features and
  log-signature NRDE inputs.
- **Training tooling** driven by [Hydra](https://hydra.cc) configurations and a
  batteries-included `train.py` script.
- **Tested and type-checked** via the accompanying GitHub Actions workflow that
  enforces `ruff`, `pyright`, and `pytest` on every push and pull request.

## Installation

> **Requirements:** Python 3.9+ and a recent PyTorch CPU wheel (GPU wheels work
> too). The default extras install `torchsignature` from GitHub; make sure build
> tooling such as `git` is available on the runner.

```bash
pip install -e .[uea,signature]
```

### Prebuilt wheels from CI

The GitHub Actions [CI workflow](.github/workflows/ci.yml) builds and publishes
downloadable artifacts for both the pure Python package and the compiled
`ossm._kernels` extension on every successful run:

1. Navigate to **Actions → CI** on the repository and open the latest green run.
2. Download the `ossm-core-dist` archive for the source distribution and pure
   Python wheel.
3. Choose the `ossm-kernels-cpu` artifact for CPU-only environments or
   `ossm-kernels-cu121` for CUDA 12.1 targets. The wheel filenames encode the
   `+cpu` or `+cu121` local version tag.
4. Install the wheels locally (order matters so that the Python package is
   available before the extension):

   ```bash
   pip install path/to/ossm-*.whl
   pip install path/to/ossm_kernels-*.whl
   ```

Additional extras are defined in `pyproject.toml`:

- `uea` — dependencies for loading the UEA archive (`sktime`).
- `signature` — installs the log-signature backend (`torchsignature`).
- `linoss` — optional JAX stack used by the LinOSS parity checks.

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

## Working with the UEA multivariate archive

The [UEA/UCR multivariate time-series archive](https://timeseriesclassification.com)
collects benchmark classification problems with aligned sampling grids. Each
problem ships with disjoint training and testing splits stored as ARFF files;
after parsing the data live in tensors shaped `(num_samples, seq_len, channels)`.

### Mathematical views

- **`raw`** – exposes the observation matrix together with a normalized time
  grid `t_k` in `[0, 1]` and the encoded class label. This view is suitable for
  sequence models that ingest the original measurements.
- **`coeff`** – augments `raw` with the backward Hermite cubic coefficients
  `(d, c, b, a)` produced by `diffrax.backward_hermite_coefficients`. These
  coefficients parameterise the unique cubic spline that interpolates the
  trajectory and are flattened along the channel axis in the PyTorch view.
- **`path`** – segments the path into fixed windows, inserts basepoints, and
  exports Hall-basis log-signatures of depth 2. The feature layout (scalar term,
  linear words, then Lie brackets) matches the Signax/LinOSS preprocessing.

### Dataset layout

Point `root` at a directory with the following layout, or pass `download=True`
to create it before populating the ARFF files yourself:

```
<root>/
  raw/UEA/Multivariate_arff/<DatasetName>/
    <DatasetName>_TRAIN.arff
    <DatasetName>_TEST.arff
```

The dataset constructor accepts:

- `name`: dataset key from the archive.
- `split`: final subset to materialise (`"train"`, `"test"`, `"val"`, `"all"`, or a
  sequence such as `("train", "test")`).
- `view`: one of `"raw"`, `"coeff"`, or `"path"`.
- Log-signature hyper-parameters (`steps`, `depth`, `basis`) when `view="path"`.
- `source_splits`: base splits to ingest before further manipulation (defaults to
  the requested `split` or both train/test when resampling).
- `deduplicate`: drop repeated trajectories by comparing the original time grids
  and observation matrices across the loaded source splits.
- `resample` + `resample_seed`: deterministically repartition the aggregated pool
  into new `train`/`val`/`test` subsets using either counts or proportions.
- `record_grid`: attach the pre-normalisation time grid as `sample["grid"]`.
- `record_source`: include `sample["source_index"]` and `sample["source_split"]`
  (encoded via `dataset.source_split_encoding`) tracing back to the original
  ARFF location.

Each item is a dictionary with `times`, `values`, `label`, and when applicable
either `coeffs` or `features`. Optional metadata (`grid`, `source_*`) survives the
transform pipeline so downstream code can faithfully reproduce the LinOSS data
layout while staying within idiomatic PyTorch workflows. Collate helpers in
`ossm.data.datasets` pad, stack, and batch these tensors for PyTorch dataloaders.

## Training with `train.py`

The training entrypoint is implemented in [`train.py`](./train.py) and is
configured with [Hydra](https://hydra.cc). The default configuration
(`configs/config.yaml`) trains a LinOSS backbone on the EthanolConcentration
coefficients view and logs artifacts under `./outputs`.

### Basic invocation

1. Prepare the desired UEA dataset so that the ARFF files live under
   `<data_root>/raw/UEA/Multivariate_arff/<DatasetName>/` as shown above. The
   [`scripts/prepare_uea.py`](./scripts/prepare_uea.py) helper downloads the
   archive to `~/.cache/torch/datasets/ossm` by default and materialises the
   processed pickles expected by the dataset loader.
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

* Pre-populating the cache using the helper script:

  ```bash
  # Downloads the archive to ~/.cache/torch/datasets/ossm by default and
  # materialises train/test pickles for the requested datasets.
  python scripts/prepare_uea.py --datasets EthanolConcentration

  # Point OSSM_DATA_ROOT at the processed cache before training.
  OSSM_DATA_ROOT=~/.cache/torch/datasets/ossm python train.py
  ```

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

## Development workflow

We rely on `pytest` for testing, `ruff` for linting/formatting, and `pyright` for
static type analysis. The recommended developer setup mirrors the CI job:

```bash
pip install -e .[uea,signature]
pip install ruff pyright

ruff check .        # or `ruff format .` to auto-format
pyright             # type checking
pytest              # run the test suite
```

## Continuous integration

The repository ships with a GitHub Actions workflow at
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). It runs on pushes to
`main`, pull requests, and manual dispatches. Each job matrix covers Python 3.9
and 3.11 on Ubuntu and executes the following steps:

1. Install the project in editable mode with the UEA and log-signature extras.
2. Lint with `ruff`.
3. Type-check with `pyright`.
4. Execute the `pytest` suite.

Keeping local changes green against the same commands will help ensure clean CI
runs.

## Notes

* UEA ARFF expected at: `<root>/raw/UEA/Multivariate_arff/<DatasetName>/<DatasetName>_{TRAIN|TEST}.arff`
* `view="path"` uses `torchsignature` (no-op if not installed) and emits depth-2
  Hall-basis log-signatures with the scalar coordinate included, matching the
  Signax/LinOSS preprocessing.

## License

Licensed under the [MIT License](./LICENSE).
