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
```

## Coding Style & Naming Conventions

* Python ≥ 3.9 with explicit type hints; use `torch.Tensor` in signatures.
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
