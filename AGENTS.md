# Repository Guidelines

## Project Structure & Module Organization

```
.
├─ README.md
├─ AGENTS.md
├─ pyproject.toml
├─ src/ossm/
│  ├─ __init__.py
│  └─ data/
│     ├─ __init__.py
│     ├─ datasets/
│     │  ├─ __init__.py  ├─ uea.py  ├─ collate.py  └─ loader_utils.py
│     └─ transforms/
│        ├─ __init__.py  ├─ compose.py  ├─ path.py  ├─ cde.py  └─ signature.py
└─ tests/
   ├─ test_uea.py  ├─ test_collate.py  ├─ test_loader_utils.py
   ├─ test_cde_transform.py  └─ test_signature_transform.py
```

* Public symbols must be exported via `src/ossm/__init__.py` and per-module `__all__`.
* Keep modules import-clean (no side effects).

## Build, Test, and Development Commands

If you use Guix, run inside the project shell (when `manifest.scm` exists):

```
guix shell -m manifest.scm -- <command>
```

Otherwise:

```bash
# setup (editable)
pip install -e .[uea,cde,signature]

# tests
pytest -q

# lint & type-check (must be clean)
ruff check . --fix
pyright

# optional formatting
ruff format .

# before opening a PR or requesting review, run all of the following
ruff check .
ruff format .
pyright
pytest
```

## Coding Style & Naming Conventions

* Python ≥ 3.11 with explicit type hints; use `torch.Tensor` in signatures.
* Files: `snake_case.py`; Classes: `PascalCase`; functions/vars: `snake_case`.
* Torch-first: preserve `device`/`dtype`, avoid NumPy in hot paths.
* Transforms should be stateless and composable; no hidden globals.
* Prefer out-of-place ops to avoid autograd surprises.

## Testing Guidelines

* Framework: `pytest`; tests live in `tests/` as `test_*.py`.
* Deterministic inputs where possible; validate shapes, dtypes, and basic numerics.
* Optional deps:

  * `torchcde` tests are skipped if not installed.
  * `torchsignature` tests are skipped if not installed.

## Commit & Pull Request Guidelines

* **Conventional Commits**: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `chore:`, `build:`, `ci:`.
  Use scopes like `(data)`, `(datasets)`, `(uea)`, `(transforms)`, `(cde)`, `(signature)`, `(models)`, `(training)`.
  Example: `feat(uea): add normalized time grid fallback`
* PR checklist:

  * Clear motivation and summary; call out API/shape changes
  * Tests covering the change
  * `ruff` and `pyright` both **zero errors**
  * `pytest` passing locally
  * Linked issues (e.g., “Closes #123”)

## Agent-Specific Instructions

* Keep `__all__` and package `__init__.py` updated when adding exports.
* Do not introduce import-time side effects (e.g., filesystem writes, network).
* Watch algorithmic complexity in core loops; avoid accidental quadratic scans.
* Maintain consistent error messages for missing optional deps (`torchcde`, `torchsignature`).

## D-LinOSS Bench (PGO) Utility

The script `scripts/bench_dlinoss.py` is a lean, iteration-friendly benchmark for PGO-style tuning of the D-LinOSS CUDA kernels. It times the pure PyTorch fallback and the optimized kernel path, checks numerical agreement, and writes JSONL results suitable for quick diffs across kernel revisions.

What it measures
- Forward timings per case (ms), throughput (elements/s)
- Approximate bandwidth (GB/s) and compute rate (GFLOP/s)
- Optional backward timings and gradient error stats
- Peak CUDA memory for forward-only and forward+backward

Key concepts
- Cases are defined by `(variant, T, B, S)` where `variant ∈ {imex1, imex2, im, ex}`.
- “Reference” path disables kernels and uses the pure PyTorch recurrence.
- “Kernel” path calls `run_dlinoss` which uses the extension if available, else falls back.

Basic usage
- CUDA run that writes results to slot files (see below):
  - `python scripts/bench_dlinoss.py --device cuda --variants imex1 imex2 im ex --lengths 128 256 512 1024 --batches 2 8 32 --ssms 64 256 --repeats 20 --warmup 5`
- CPU sanity run (kernel path falls back to reference):
  - `python scripts/bench_dlinoss.py --device cpu --variants ex --lengths 4 --batches 1 --ssms 1 --repeats 1 --warmup 0`

Output and slot rotation
- Default output directory: `outputs/bench_dlinoss/` (ignored by git).
- Two slot files: `old.jsonl` and `new.jsonl`.
  1) If neither exists: the script populates `new.jsonl` and copies it to `old.jsonl`.
  2) If `new.jsonl` exists: it is moved/copied to `old.jsonl`, then a fresh `new.jsonl` is written.
- Quick perf diff is printed after each run comparing `old.jsonl` vs `new.jsonl`.
- You can override the output directory via `--out-dir <dir>`.
- You can write to a custom path (bypassing slot rotation) via `--output results.jsonl`.

Diff-only mode
- To view a quick diff without measuring, use:
  - `python scripts/bench_dlinoss.py --diff-only`
- It prints aggregate counts and top improvements/regressions by kernel ms.
- Use `--top N` to control how many entries are shown (default 5).

JSONL schema (per line)
- Minimal example:
  - `{ "device": "cuda", "variant": "ex", "T": 1024, "B": 8, "S": 256,
       "ref": {"ms": ..., "std": ..., "tps": ..., "GBs": ..., "GFLOPs": ...},
       "ker": {"ms": ..., "std": ..., "tps": ..., "GBs": ..., "GFLOPs": ...},
       "err": {"out_rel_max": ..., "max|y_ref|": ..., "max|y_ker|": ..., "finite_ref": 1.0, "finite_ker": 1.0},
       "bwd_ref_ms": ..., "bwd_ker_ms": ..., "bwd_speedup": ..., "bwd_finite": { ... },
       "out_err": { ... }, "grad_err": { ... },
       "peak_alloc_fwd": <bytes>, "peak_alloc_fwbw": <bytes> }`
- Backward/memory fields only appear if you omit `--no_backward` and run on CUDA.

Reproducibility controls
- `--seed N`: set the PyTorch RNG seed used for case inputs.
- `--raw_params`: use raw `a, g, step` samples; omit to project into a stable region.

NVTX ranges (for Nsight profiling)
- The script emits NVTX ranges by default on CUDA. Disable via `--no-nvtx`.
- Range names include case parameters to make the timeline easy to read:
  - `case var=<v> T=<T> B=<B> S=<S>`
  - `fwd_ref_warmup var=...`, `fwd_ref_timing var=...`
  - `fwd_ker_warmup var=...`, `fwd_ker_timing var=...`
  - `bwd_ref_timing var=...`, `bwd_ker_timing var=...`
  - `peak_fwd var=...`, `peak_fwbw var=...`

Profiling examples
- Nsight Systems (timeline):
  - `nsys profile --trace=cuda,osrt,nvtx --sample=none -o nsys_dlinoss python scripts/bench_dlinoss.py --device cuda --variants ex --lengths 2048 --batches 8 --ssms 256 --repeats 50 --warmup 10`
  - Open the `.qdrep` in Nsight Systems GUI and filter by NVTX ranges (e.g., `fwd_ker_timing`).
- Nsight Compute (kernel metrics):
  - `ncu --target-processes all --nvtx --set full --nvtx-include "fwd_ker_timing*" python scripts/bench_dlinoss.py --device cuda --variants imex1 --lengths 2048 --batches 8 --ssms 256 --repeats 5 --warmup 5`
  - Adjust `--nvtx-include` to focus on specific ranges; disable timer noise by minimizing `--repeats`.

Tips for PGO-style iteration
- Start with a small sweep (1–2 variants, a couple of T/B/S combos) to shorten the loop.
- Use the printed `[diff]` summary to quickly spot improvements/regressions.
- Drill down with Nsight Systems for timeline stalls; confirm bottlenecks with Nsight Compute metrics.
- Commit only meaningful performance changes; the slot files live in `outputs/bench_dlinoss/` and are ignored by git.
