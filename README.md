# Oscillatory state-space models

PyTorch utilities for oscillatory state-space models (OSSM) that power RNN, SSM,
NCDE, and NRDE experiments. The library bundles reusable dataset loaders,
transform pipelines, and a Hydra-driven training entrypoint so that downstream
projects can focus on modelling instead of bespoke preprocessing.

## Highlights

- **UEA multivariate archive pipeline** with three ready-to-use representations
  (`raw`, `coeff`, and `path`) that mirror the torchvision dataset API.
- **Compositional data transforms** for cubic-spline NCDE features and
  log-signature NRDE inputs.
- **Training tooling** with a polished CLI on top of
  [Hydra](https://hydra.cc), providing discoverable switches for the most common
  OSSM experiments.
- **Tested and type-checked** via the accompanying GitHub Actions workflow that
  enforces `ruff`, `pyright`, and `pytest` on every push and pull request.

---

## Installation

### 1. Install PyTorch

OSSM targets Python 3.11+ and PyTorch 2.8.0. Pick the wheel that matches your
hardware:

```bash
# CPU-only
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0

# CUDA 12.6 (requires a compatible NVIDIA driver)
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.8.0
```

### 2. Install OSSM

You can either build the package locally (recommended while developing) or use
the prebuilt artifacts produced by GitHub Actions.

#### Editable/development install

```bash
git clone https://github.com/<your-org>/ossm.git
cd ossm

# Installs the Python package together with optional extras.
pip install -e .[uea,signature]
```

This compiles the optimised kernels locally and makes the source tree importable
in-place.

#### GitHub Actions wheels

Every CI run uploads wheels for both the pure Python package and the compiled
`ossm._kernels` extension:

1. Navigate to **Actions → CI** and open the latest successful run.
2. Download the `ossm-core-dist` artifact for the source distribution and pure
   Python wheel.
3. Download the matching kernel wheel (`ossm-kernels-cpu` or
   `ossm-kernels-cu121` depending on your hardware).
4. Install the wheels locally (install the Python package before the kernels):

   ```bash
   pip install path/to/ossm-*.whl
   pip install path/to/ossm_kernels-*.whl
   ```

Optional extras are defined in `pyproject.toml`:

- `uea` — dependencies for loading the UEA archive (`sktime`).
- `signature` — installs the log-signature backend (`torchsignature`).
- `linoss` — optional JAX stack used by the LinOSS parity checks.

### 3. Verify the extension

The parity tests compile the CUDA/CPU kernels on the fly and confirm that the
installation is functional:

```bash
pytest tests/test_scan_parity.py -q
```

On GPU runners ensure `CUDA_VISIBLE_DEVICES` exposes a device that supports the
architectures listed in the CI matrix (`sm_80`, `sm_86`, `sm_89`, `sm_90`).

---

## Using the library

OSSM mirrors the torchvision dataset ergonomics. Each item is a dictionary with
`times`, `values`, and `label` fields, plus representation-specific keys:

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
    steps=32,
    depth=2,
)
path_loader = DataLoader(train_path, batch_size=64, shuffle=True, collate_fn=path_collate)
```

Collate helpers in `ossm.data.datasets` pad, stack, and batch these tensors for
PyTorch dataloaders, while the transform pipeline keeps optional metadata such
as the original time grid or source indices.

---

## Preparing the UEA archive

The [UEA/UCR multivariate time-series archive](https://timeseriesclassification.com)
ships datasets as ARFF files with predefined train/test splits. OSSM expects the
following layout:

```
<root>/
  raw/UEA/Multivariate_arff/<DatasetName>/
    <DatasetName>_TRAIN.arff
    <DatasetName>_TEST.arff
```

Use the helper script to download and preprocess the archive:

```bash
# Downloads to ~/.cache/torch/datasets/ossm by default and materialises pickles.
python scripts/prepare_uea.py --datasets EthanolConcentration GunPoint

# Point OSSM_DATA_ROOT at the processed cache before training (optional).
export OSSM_DATA_ROOT=~/.cache/torch/datasets/ossm
```

You can also populate the layout manually; the dataset constructor accepts
parameters for resampling, deduplication, log-signature depth, and more. See
`src/ossm/data/datasets/uea.py` for the exhaustive list.

---

## Training with `train.py`

The training entrypoint wraps Hydra with a more discoverable CLI. All switches
have sensible defaults that map to the provided configuration presets while
remaining overridable via standard Hydra overrides.

### Quick start

```bash
python train.py \
  --backbone linoss_im \
  --head classification \
  --dataset-name EthanolConcentration \
  --dataset-root ~/.cache/torch/datasets/ossm \
  --dataset-view coeff \
  --batch-size 128 \
  --num-workers 8 \
  --epochs 20 \
  --optimizer adamw \
  --lr 2.5e-4 \
  --prefetch-gpu \
  --prefetch-depth 2 \
  --cudnn-benchmark
```

At launch the script prints a concise summary:

```
Training • task=classification steps=25,600 device=cuda:0 batch_size=128 batches/epoch=128 train_samples=1636 prefetch_gpu(depth=2)
```

Progress logs report loss, accuracy (when applicable), learning rate,
throughput (`Samples/s`), and step latency, followed by validation/test metrics
at the configured evaluation interval.

### Discoverable toggles

`python train.py --help` documents the full surface area. The most common flags
are grouped by intent:

- **Model & task**: `--backbone`, `--head`, `--task`, `--hidden-dim`,
  `--ssm-size`, `--num-blocks`.
- **Dataset**: `--dataset-name`, `--dataset-view`, `--train-split`,
  `--val-name`, `--test-name`, `--window-steps`, `--window-depth`,
  `--logsig-basis`, `--record-grid`, `--record-source`, `--download`.
- **Dataloader**: `--batch-size`, `--num-workers`, `--prefetch-factor`,
  `--persistent-workers`, `--pin-memory`, `--drop-last`, `--collate`.
- **Optimisation**: `--optimizer`, `--scheduler`, `--lr`, `--weight-decay`,
  `--grad-clip`.
- **Training schedule**: `--max-steps`, `--epochs`, `--log-interval`,
  `--eval-interval`, `--prefetch-gpu`, `--prefetch-depth`, `--cudnn-benchmark`.
- **Runtime**: `--device`, `--seed`, `--work-dir`, `--dataset-root`.

When the CLI is not sufficient you can still fall back to Hydra overrides:

```bash
python train.py training.max_steps=5000 model.params.hidden_dim=256
```

Hydra resolves the requested defaults, applies CLI/override modifications, and
stores the full configuration under `${OSSM_WORK_DIR:-./outputs}` alongside
training logs.

### Sequential recommendation with D-LinOSSRec

The same `train.py` entrypoint also orchestrates the D-LinOSS4Rec experiments.
The workflow mirrors the MovieLens/Amazon evaluation protocol from the paper
and can be reproduced on CPU (for smoke tests) or GPU (for full-scale runs).

1. **Download & preprocess datasets**

   ```bash
   # MovieLens-1M
   wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
   unzip ml-1m.zip -d data/raw/ml-1m
   python scripts/prepare_ml1m.py \
     --raw data/raw/ml-1m/ml-1m \
     --out data/seqrec/ml1m \
     --min-interactions 5

   # Amazon category dumps (Beauty & Video Games)
   mkdir -p data/raw/amazon
   wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz \
     -O data/raw/amazon/reviews_Beauty_5.json.gz
   wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz \
     -O data/raw/amazon/reviews_Video_Games_5.json.gz
   python scripts/prepare_amazon.py \
     --subset beauty \
     --raw data/raw/amazon \
     --out data/seqrec/amazonbeauty \
     --min-interactions 5
   python scripts/prepare_amazon.py \
     --subset videogames \
     --raw data/raw/amazon \
     --out data/seqrec/amazonvideogames \
     --min-interactions 5
   ```

   Increase `--min-interactions` when you need a smaller CPU-only benchmark; the
   script always performs leave-one-out splits and writes the parquet/NumPy
   artefacts expected by the trainer.

2. **Train D-LinOSSRec while monitoring telemetry**

   ```bash
   # Example: CPU-friendly sanity check on Amazon Beauty
   python train.py \
     training=seqrec \
     dataset=amazonbeauty validation_dataset=amazonbeauty test_dataset=amazonbeauty \
     model=dlinossrec head=tiedsoftmax \
     training.device=cpu training.amp=false training.epochs=2 training.batch_size=64 \
     dataset.num_workers=0 validation_dataset.num_workers=0 test_dataset.num_workers=0 \
     model.d_model=32 model.ssm_size=64 \
     | tee reports/amazonbeauty_seqrec.log
   ```

   The progress reporter prints step-level loss, throughput, validation ranking
   metrics (HR@10/NDCG@10/MRR@10), and a final summary that mirrors the
   classification telemetry style.

3. **Aggregate metrics into publication-style tables**

   Every training run appends a row to `${OSSM_WORK_DIR}/seqrec/.../summary.jsonl`.
   You can consolidate arbitrary runs into the report tables with:

   ```bash
   python scripts/make_tables.py outputs/seqrec/**/summary.jsonl
   ```

   The utility prints Table 1 (overall recommendation quality), Table 3
   (efficiency on ML-1M), and Table 4 (ablation study) in the same format used by
   the paper.

---

## Development workflow

We rely on `pytest` for testing, `ruff` for linting/formatting, and `pyright`
for static type analysis. The recommended developer setup mirrors the CI job:

```bash
pip install -e .[uea,signature]
pip install ruff pyright

ruff check .        # or `ruff format .` to auto-format
pyright             # type checking
pytest              # run the test suite
```

---

## Continuous integration

The repository ships with a GitHub Actions workflow at
`.github/workflows/ci.yml`. It runs on pushes to `main`, pull requests, and
manual dispatches. Each job matrix covers Python 3.11 on Ubuntu and executes the
following steps:

1. Install the project in editable mode with the UEA and log-signature extras.
2. Lint with `ruff`.
3. Type-check with `pyright`.
4. Execute the `pytest` suite.

Keeping local changes green against the same commands helps ensure smooth CI
runs.

---

## License

Licensed under the [MIT License](./LICENSE).
