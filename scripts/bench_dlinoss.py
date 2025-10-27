#!/usr/bin/env python3
"""
Lean D-LinOSS benchmark for PGO-style tuning.

It times the reference (pure PyTorch) and kernel paths, checks numerical
agreement, and can record JSONL for later analysis. It avoids speculative
kernel heuristics and unused env toggles.

Measures per case (variant, T, B, S):
- Forward timings (ms), throughput (elements/s), approx GB/s and GFLOP/s
- Optional backward timings and gradient error stats
- Peak CUDA memory for forward and forward+backward

Usage examples:
  python scripts/bench_dlinoss.py \
    --device cuda --variants imex1 imex2 im ex \
    --lengths 128 256 512 1024 2048 --batches 8 32 --ssms 256 \
    --repeats 20 --warmup 5 --output results.jsonl

Notes:
- Both reference and kernel paths run on the selected device; on CPU, the
  "kernel" call falls back to the reference implementation.
- Uses CUDA events for accurate GPU timings when device=cuda.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import shutil
import statistics

import torch


# Ensure repository src/ is on sys.path so `ossm` can be imported without installation
def _ensure_repo_src_on_path() -> None:
    try:
        here = Path(__file__).resolve()
    except Exception:
        return
    # scripts/bench_dlinoss.py -> repo_root/src
    repo_root = here.parents[1]
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

_ensure_repo_src_on_path()


def _repo_root() -> Path:
    try:
        here = Path(__file__).resolve()
    except Exception:
        return Path.cwd()
    return here.parents[1]


def _nvtx_available() -> bool:
    try:
        return bool(torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"))
    except Exception:
        return False


@contextmanager
def nvtx_range(name: str, *, enabled: bool = True):
    active = enabled and _nvtx_available()
    if active:
        try:
            torch.cuda.nvtx.range_push(name)
        except Exception:
            active = False
    try:
        yield
    finally:
        if active and _nvtx_available():
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    # Skip malformed lines to keep robustness in iterative runs
                    continue
    except Exception:
        pass
    return records


def _index_by_case(records: list[dict]) -> dict[tuple[str, int, int, int], dict]:
    index: dict[tuple[str, int, int, int], dict] = {}
    for r in records:
        try:
            key = (str(r["variant"]), int(r["T"]), int(r["B"]), int(r["S"]))
        except Exception:
            continue
        index[key] = r
    return index


def _print_perf_diff(old_path: Path, new_path: Path, *, top: int = 5) -> None:
    old = _load_jsonl(old_path)
    new = _load_jsonl(new_path)
    if not old or not new:
        print(f"[diff] missing data: old={old_path.exists()} new={new_path.exists()}")
        return
    idx_old = _index_by_case(old)
    idx_new = _index_by_case(new)

    improvements: list[tuple[tuple[str, int, int, int], float, float, float]] = []
    regressions: list[tuple[tuple[str, int, int, int], float, float, float]] = []
    unchanged = 0

    for key, rec_new in idx_new.items():
        rec_old = idx_old.get(key)
        if not rec_old:
            continue
        ker_old = float(rec_old.get("ker", {}).get("ms", float("nan")))
        ker_new = float(rec_new.get("ker", {}).get("ms", float("nan")))
        if not (ker_old == ker_old and ker_new == ker_new):  # NaN check
            continue
        delta = ker_new - ker_old
        rel = (delta / ker_old) if ker_old != 0.0 else float("inf")
        entry = (key, ker_old, ker_new, rel)
        if abs(rel) < 1e-6:
            unchanged += 1
        elif delta < 0:
            improvements.append(entry)
        else:
            regressions.append(entry)

    improvements.sort(key=lambda x: x[3])  # most negative rel first
    regressions.sort(key=lambda x: x[3], reverse=True)  # largest positive rel first

    print("[diff] vs previous new.jsonl -> old.jsonl")
    print(f"[diff] cases: old={len(idx_old)} new={len(idx_new)} improved={len(improvements)} regressed={len(regressions)} unchanged={unchanged}")
    if improvements:
        print("[diff] top improvements (ker_ms rel%):")
        for key, ko, kn, rel in improvements[:top]:
            v, T, B, S = key
            print(f"  var={v} T={T} B={B} S={S}: {ko:.3f} -> {kn:.3f} ms ({rel*100:.1f}%)")
    if regressions:
        print("[diff] top regressions (ker_ms rel%):")
        for key, ko, kn, rel in regressions[:top]:
            v, T, B, S = key
            print(f"  var={v} T={T} B={B} S={S}: {ko:.3f} -> {kn:.3f} ms (+{rel*100:.1f}%)")

# Import OSSM helpers
try:
    from ossm.models._dlinoss_scan import (
        run_dlinoss,
        has_kernels as dlinoss_has_kernels,
        extension_error as dlinoss_extension_error,
        _reference_dlinoss_states,  # type: ignore[attr-defined]
    )
except Exception as exc:  # pragma: no cover
    print(f"[bench] ERROR: failed to import ossm.models._dlinoss_scan: {exc}", file=sys.stderr)
    sys.exit(2)


ALL_VARIANTS = ("imex1", "imex2", "im", "ex")


@dataclass
class Case:
    variant: str
    T: int
    B: int
    S: int  # ssm size


def setup_env(device: str) -> None:
    torch.set_float32_matmul_precision("high")
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def env_info() -> str:
    parts: List[str] = []
    parts.append(f"python={sys.version.split()[0]}")
    parts.append(f"torch={torch.__version__}")
    parts.append(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(dev)
        cc = f"{prop.major}.{prop.minor}"
        parts.append(f"gpu_name={prop.name.strip()}")
        parts.append(f"sm={cc}")
        parts.append(f"sm_count={prop.multi_processor_count}")
        parts.append(f"total_mem_gb={prop.total_memory/1024**3:.2f}")
        try:
            parts.append(f"driver={torch.version.driver}")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            parts.append(f"cuda_runtime={torch.version.cuda}")  # type: ignore[attr-defined]
        except Exception:
            pass
    parts.append(f"matmul_precision={torch.get_float32_matmul_precision()}")
    return " ".join(parts)


def kernel_info() -> str:
    status = []
    for v in ALL_VARIANTS:
        ok = dlinoss_has_kernels(v)
        status.append(f"{v}:{'Y' if ok else 'N'}")
    err = dlinoss_extension_error()
    if err is not None:
        return f"kernels=({', '.join(status)}) extension_error={repr(err)}"
    return f"kernels=({', '.join(status)})"


def _project_stable_params(a: torch.Tensor, g: torch.Tensor, step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Mirror DampedLinOSSLayer._project_parameters
    step_p = torch.sigmoid(step)
    g_p = torch.relu(g)
    denom = torch.clamp(step_p * step_p, min=1e-6)
    s = step_p * g_p
    base = torch.sqrt(torch.clamp(1.0 + s, min=1e-6))
    a_low = (2.0 + s - 2.0 * base) / denom
    a_high = (2.0 + s + 2.0 * base) / denom
    a_p = a_low + torch.relu(a - a_low) - torch.relu(a - a_high)
    return a_p, g_p, step_p


def make_inputs(T: int, B: int, S: int, device: torch.device, *, raw_params: bool = False) -> Tuple[torch.Tensor, ...]:
    # Sample and optionally project to a stable region
    g = torch.empty(S, device=device, dtype=torch.float32).uniform_(-0.25, 0.25)
    a = torch.empty(S, device=device, dtype=torch.float32).uniform_(0.0, 1.0)
    step = torch.randn(S, device=device, dtype=torch.float32) * 0.5
    if not raw_params:
        a, g, step = _project_stable_params(a, g, step)
    bu = torch.randn(T, B, S, device=device, dtype=torch.complex64) * 0.5
    return a, g, step, bu


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (len(sorted_vals) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_vals[int(rank)])
    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    return float(lower_val + (upper_val - lower_val) * (rank - lower))


def _timing_stats(times: List[float]) -> Dict[str, Any]:
    if not times:
        return {
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "p05_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "mad_ms": float("nan"),
            "cv_pct": float("nan"),
            "samples_ms": [],
            "repeats": 0,
        }
    sorted_vals = sorted(float(t) for t in times)
    mean_ms = float(statistics.mean(sorted_vals))
    std_ms = float(statistics.pstdev(sorted_vals)) if len(sorted_vals) > 1 else 0.0
    median_ms = float(statistics.median(sorted_vals))
    mad_ms = float(
        statistics.mean(abs(x - median_ms) for x in sorted_vals)
        if len(sorted_vals) > 1
        else 0.0
    )
    cv_pct = float((std_ms / mean_ms) * 100.0) if mean_ms > 0 else float("nan")
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": float(sorted_vals[0]),
        "max_ms": float(sorted_vals[-1]),
        "p05_ms": _percentile(sorted_vals, 0.05),
        "p50_ms": median_ms,
        "p95_ms": _percentile(sorted_vals, 0.95),
        "p99_ms": _percentile(sorted_vals, 0.99),
        "mad_ms": mad_ms,
        "cv_pct": cv_pct,
        "samples_ms": list(sorted_vals),
        "repeats": len(sorted_vals),
    }


def cuda_timing(repeats: int, fn) -> Dict[str, Any]:
    times: List[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return _timing_stats(times)


def cpu_timing(repeats: int, fn) -> Dict[str, Any]:
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return _timing_stats(times)


def _ops_and_bytes_per_step(variant: str) -> Tuple[int, int]:
    # Rough counts per series per timestep (complex treated as two reals).
    # IM/IMEX1/IMEX2: n0 and n1 each do ~3 mul + 2 add per real => 20 flops total.
    # EX: n0 ~5 per real (10), n1 ~2 per real (4) => 14 flops total.
    flops = 20 if variant in ("imex1", "imex2", "im") else 14
    # Memory: read bu (complex64 8B), write 2 states (2 * complex64 = 16B) => ~24B per step per series
    bytes_ = 24
    return flops, bytes_


def _add_timing_enrichments(stats: Dict[str, Any], *, elems: int, steps: int) -> Dict[str, Any]:
    mean_ms = float(stats.get("mean_ms", float("nan")))
    stats = dict(stats)
    stats["ms"] = mean_ms
    stats.setdefault("std", stats.get("std_ms", float("nan")))
    if elems > 0 and math.isfinite(mean_ms):
        stats["ns_per_elem"] = (mean_ms * 1_000_000.0) / elems
    else:
        stats["ns_per_elem"] = float("nan")
    if steps > 0 and math.isfinite(mean_ms):
        stats["ns_per_step"] = (mean_ms * 1_000_000.0) / steps
    else:
        stats["ns_per_step"] = float("nan")
    stats.setdefault("repeats", len(stats.get("samples_ms", [])))
    return stats


def forward_benchmark(
    case: Case,
    device: torch.device,
    warmup: int,
    repeats: int,
    *,
    raw_params: bool,
    nvtx: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    a, g, step, bu = make_inputs(case.T, case.B, case.S, device, raw_params=raw_params)

    # Fallback reference (disable kernels explicitly)
    os.environ["OSSM_DLINOSS_DISABLE_KERNEL"] = "1"
    def ref_fn() -> torch.Tensor:
        return _reference_dlinoss_states(case.variant, a, g, step, bu)[..., 1]
    # Warmup
    with nvtx_range(f"fwd_ref_warmup var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
        for _ in range(max(1, warmup)):
            _ = ref_fn()
    with nvtx_range(f"fwd_ref_timing var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
        ref_stats = cuda_timing(repeats, ref_fn) if device.type == "cuda" else cpu_timing(repeats, ref_fn)

    # Kernel path
    os.environ.pop("OSSM_DLINOSS_DISABLE_KERNEL", None)
    def ker_fn() -> torch.Tensor:
        return run_dlinoss(case.variant, a, g, step, bu)
    with nvtx_range(f"fwd_ker_warmup var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
        for _ in range(max(1, warmup)):
            _ = ker_fn()
    with nvtx_range(f"fwd_ker_timing var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
        ker_stats = cuda_timing(repeats, ker_fn) if device.type == "cuda" else cpu_timing(repeats, ker_fn)

    elems = case.T * case.B * case.S
    flops_step, bytes_step = _ops_and_bytes_per_step(case.variant)
    arith_intensity = flops_step / bytes_step if bytes_step else float("inf")
    ref_stats = _add_timing_enrichments(ref_stats, elems=elems, steps=case.T)
    ker_stats = _add_timing_enrichments(ker_stats, elems=elems, steps=case.T)

    # Timings
    ref_ms = ref_stats["ms"]
    ref_std = ref_stats["std"]
    ker_ms = ker_stats["ms"]
    ker_std = ker_stats["std"]
    # Throughput
    ref_tps = elems / (ref_ms / 1000.0)
    ker_tps = elems / (ker_ms / 1000.0)
    # Bandwidth/compute (approx)
    total_bytes = elems * bytes_step
    total_flops = elems * flops_step
    ref_gbs = (total_bytes / (ref_ms / 1000.0)) / 1e9
    ker_gbs = (total_bytes / (ker_ms / 1000.0)) / 1e9
    ref_gflops = (total_flops / (ref_ms / 1000.0)) / 1e9
    ker_gflops = (total_flops / (ker_ms / 1000.0)) / 1e9

    # Output relative error and magnitude checks
    with torch.no_grad():
        y_ref = _reference_dlinoss_states(case.variant, a, g, step, bu)[..., 1]
        y_ker = run_dlinoss(case.variant, a, g, step, bu)
        max_ref = y_ref.abs().max().item()
        max_ker = y_ker.abs().max().item()
        diff = (y_ker - y_ref).abs()
        out_rel = (diff.max() / (y_ref.abs().max().clamp_min(1e-8))).item()
        finite_ref = torch.isfinite(y_ref).float().mean().item()
        finite_ker = torch.isfinite(y_ker).float().mean().item()

    ref = {
        **ref_stats,
        "ms": ref_ms,
        "std": ref_std,
        "tps": ref_tps,
        "GBs": ref_gbs,
        "GFLOPs": ref_gflops,
    }
    ker = {
        **ker_stats,
        "ms": ker_ms,
        "std": ker_std,
        "tps": ker_tps,
        "GBs": ker_gbs,
        "GFLOPs": ker_gflops,
    }
    ker["speedup_vs_ref"] = (ref_ms / ker_ms) if ker_ms > 0 else float("inf")
    err = {
        "out_rel_max": out_rel,
        "max|y_ref|": max_ref,
        "max|y_ker|": max_ker,
        "finite_ref": finite_ref,
        "finite_ker": finite_ker,
    }
    extras = {
        "elems": elems,
        "total_bytes": total_bytes,
        "total_flops": total_flops,
        "flops_per_step": flops_step,
        "bytes_per_step": bytes_step,
        "arith_intensity": arith_intensity,
    }
    return ref, ker, err, extras


def numeric_and_backward(case: Case, device: torch.device, *, raw_params: bool, nvtx: bool) -> Tuple[dict, dict, dict, dict]:
    a, g, step, bu = make_inputs(case.T, case.B, case.S, device, raw_params=raw_params)

    # Reference outputs and grads
    a_ref = a.clone().requires_grad_(True)
    g_ref = g.clone().requires_grad_(True)
    s_ref = step.clone().requires_grad_(True)
    bu_ref = bu.clone().requires_grad_(True)
    os.environ["OSSM_DLINOSS_DISABLE_KERNEL"] = "1"
    y_ref = _reference_dlinoss_states(case.variant, a_ref, g_ref, s_ref, bu_ref)[..., 1]
    # Backward timing (reference)
    if device.type == "cuda":
        with nvtx_range(f"bwd_ref_timing var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            loss_ref = (y_ref.real.square() + y_ref.imag.square()).mean()
            loss_ref.backward()
            t1.record()
            torch.cuda.synchronize()
            bwd_ref_ms = float(t0.elapsed_time(t1))
    else:
        with nvtx_range(f"bwd_ref_timing_cpu var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
            t0 = time.perf_counter()
            loss_ref = (y_ref.real.square() + y_ref.imag.square()).mean()
            loss_ref.backward()
            t1 = time.perf_counter()
            bwd_ref_ms = (t1 - t0) * 1000.0
    assert a_ref.grad is not None and g_ref.grad is not None and s_ref.grad is not None and bu_ref.grad is not None
    grads_ref = {
        "a": a_ref.grad.detach().clone(),
        "g": g_ref.grad.detach().clone(),
        "step": s_ref.grad.detach().clone(),
        "bu": bu_ref.grad.detach().clone(),
    }

    # Kernel outputs and grads
    a_k = a.clone().requires_grad_(True)
    g_k = g.clone().requires_grad_(True)
    s_k = step.clone().requires_grad_(True)
    bu_k = bu.clone().requires_grad_(True)
    os.environ.pop("OSSM_DLINOSS_DISABLE_KERNEL", None)
    y_k = run_dlinoss(case.variant, a_k, g_k, s_k, bu_k)
    # Backward timing (kernel)
    if device.type == "cuda":
        with nvtx_range(f"bwd_ker_timing var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
            torch.cuda.synchronize()
            k0 = torch.cuda.Event(enable_timing=True)
            k1 = torch.cuda.Event(enable_timing=True)
            k0.record()
            loss_k = (y_k.real.square() + y_k.imag.square()).mean()
            loss_k.backward()
            k1.record()
            torch.cuda.synchronize()
            bwd_ker_ms = float(k0.elapsed_time(k1))
    else:
        with nvtx_range(f"bwd_ker_timing_cpu var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
            k0 = time.perf_counter()
            loss_k = (y_k.real.square() + y_k.imag.square()).mean()
            loss_k.backward()
            k1 = time.perf_counter()
            bwd_ker_ms = (k1 - k0) * 1000.0
    assert a_k.grad is not None and g_k.grad is not None and s_k.grad is not None and bu_k.grad is not None
    grads_k = {
        "a": a_k.grad.detach().clone(),
        "g": g_k.grad.detach().clone(),
        "step": s_k.grad.detach().clone(),
        "bu": bu_k.grad.detach().clone(),
    }

    # Output error metrics
    out_err = {
        "max_abs": (y_k - y_ref).abs().max().item(),
        "mean_abs": (y_k - y_ref).abs().mean().item(),
        "rmse": torch.sqrt(((y_k - y_ref).abs() ** 2).mean()).item(),
    }

    # Gradient error metrics per tensor
    def grad_stats(g0: torch.Tensor, g1: torch.Tensor) -> dict:
        diff = (g1 - g0)
        denom = g0.abs().max().clamp_min(1e-8)
        return {
            "max_abs": diff.abs().max().item(),
            "mean_abs": diff.abs().mean().item(),
            "rel_max": (diff.abs().max() / denom).item(),
        }

    grad_err = {k: grad_stats(grads_ref[k], grads_k[k]) for k in grads_ref.keys()}

    def frac_finite(x: torch.Tensor) -> float:
        return float(torch.isfinite(x).float().mean().item())

    fin = {
        "a": frac_finite(grads_k["a"]),
        "g": frac_finite(grads_k["g"]),
        "step": frac_finite(grads_k["step"]),
        "bu": frac_finite(grads_k["bu"]),
    }

    bwd_ref = {"ms": bwd_ref_ms}
    bwd_ker = {"ms": bwd_ker_ms, "finite": fin}
    return out_err, grad_err, bwd_ref, bwd_ker


def peak_memory(case: Case, device: torch.device, *, raw_params: bool, nvtx: bool) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats()
    a, g, step, bu = make_inputs(case.T, case.B, case.S, device, raw_params=raw_params)
    a.requires_grad_(True)
    g.requires_grad_(True)
    step.requires_grad_(True)
    bu.requires_grad_(True)
    # Force kernel path if available
    os.environ.pop("OSSM_DLINOSS_DISABLE_KERNEL", None)
    with nvtx_range(f"peak_fwbw var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
        y = run_dlinoss(case.variant, a, g, step, bu)
        loss = (y.real.square() + y.imag.square()).mean()
        loss.backward()
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated()
    return int(alloc)

def peak_memory_forward(case: Case, device: torch.device, *, raw_params: bool, nvtx: bool) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats()
    a, g, step, bu = make_inputs(case.T, case.B, case.S, device, raw_params=raw_params)
    # Force kernel path if available
    os.environ.pop("OSSM_DLINOSS_DISABLE_KERNEL", None)
    with nvtx_range(f"peak_fwd var={case.variant} T={case.T} B={case.B} S={case.S}", enabled=nvtx):
        _ = run_dlinoss(case.variant, a, g, step, bu)
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated()
    return int(alloc)


def human_bytes(n: int) -> str:
    if n <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f}{units[i]}"


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _case_label(case: Case) -> str:
    return f"var={case.variant} T={case.T} B={case.B} S={case.S}"


def _fmt_value(value: float, unit: str = "", precision: int = 3) -> str:
    if not math.isfinite(value):
        return f"nan{unit}"
    return f"{format(value, f'.{precision}f')}{unit}"


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}x"


def _print_summary_tables(results: List[Dict[str, Any]], *, top: int) -> None:
    if top <= 0:
        return
    print("\n[summary] slowest kernel cases (mean ms)")
    ranked = sorted(results, key=lambda r: r["ker"].get("ms", float("nan")), reverse=True)
    for row in ranked[:top]:
        case = row["case"]
        ker = row["ker"]
        extras = row.get("extras", {})
        ai = extras.get("arith_intensity", float("nan"))
        print(
            f"  {_case_label(case)} ker={_fmt_value(ker.get('ms', float('nan')), 'ms')} "
            f"p95={_fmt_value(ker.get('p95_ms', float('nan')), 'ms')} cv={_fmt_value(ker.get('cv_pct', float('nan')), '%', precision=1)} "
            f"ns/elem={_fmt_value(ker.get('ns_per_elem', float('nan')), 'ns', precision=1)} speedup={_fmt_ratio(ker.get('speedup_vs_ref', float('nan')))} "
            f"GB/s={_fmt_value(ker.get('GBs', float('nan')), 'GB/s')} GF/s={_fmt_value(ker.get('GFLOPs', float('nan')), 'GF/s')} AI={_fmt_value(ai, 'F/B', precision=2)}"
        )

    jitter_rank = [
        (
            row["case"],
            row["ker"].get("p95_ms", float("nan")) - row["ker"].get("p05_ms", float("nan")),
            row["ker"].get("cv_pct", float("nan")),
            row["ker"],
        )
        for row in results
    ]
    jitter_rank = [entry for entry in jitter_rank if math.isfinite(entry[1])]
    if jitter_rank:
        jitter_rank.sort(key=lambda x: x[1], reverse=True)
        print("\n[summary] highest kernel jitter (p95-p05 ms)")
        for case, span, cv_pct, ker in jitter_rank[:top]:
            print(
                f"  {_case_label(case)} span={_fmt_value(span, 'ms')} cv={_fmt_value(cv_pct, '%', precision=1)} "
                f"min={_fmt_value(ker.get('min_ms', float('nan')), 'ms')} max={_fmt_value(ker.get('max_ms', float('nan')), 'ms')}"
            )

    err_rank = [
        (row["case"], row["err"].get("out_rel_max", 0.0), row["err"])
        for row in results
    ]
    err_rank = [entry for entry in err_rank if math.isfinite(entry[1])]
    if err_rank:
        err_rank.sort(key=lambda x: x[1], reverse=True)
        print("\n[summary] largest forward relative error (kernel vs reference)")
        for case, rel_err, err in err_rank[:top]:
            print(
                f"  {_case_label(case)} rel_err={rel_err:.2e} max|ref|={err.get('max|y_ref|', float('nan')):.2e} "
                f"max|ker|={err.get('max|y_ker|', float('nan')):.2e}"
            )

    backward_rows = [row for row in results if row.get("backward")]
    if backward_rows:
        slowest_bwd = sorted(
            backward_rows,
            key=lambda r: r["backward"]["bwd_ker"].get("ms", float("nan")),
            reverse=True,
        )
        print("\n[summary] slowest backward cases (kernel path)")
        for row in slowest_bwd[:top]:
            case = row["case"]
            bwd = row["backward"]
            ker_ms = bwd["bwd_ker"].get("ms", float("nan"))
            speed = bwd.get("bwd_speedup", float("nan"))
            alloc = bwd.get("alloc", 0)
            print(
                f"  {_case_label(case)} ker={_fmt_value(ker_ms, 'ms')} speedup={_fmt_ratio(speed)} alloc={human_bytes(int(alloc))}"
            )

        grad_rank = []
        for row in backward_rows:
            grad_err = row["backward"].get("grad_err")
            if not grad_err:
                continue
            max_rel = max((val.get("rel_max", 0.0) for val in grad_err.values()), default=0.0)
            grad_rank.append((row["case"], max_rel, grad_err))
        grad_rank = [entry for entry in grad_rank if math.isfinite(entry[1])]
        if grad_rank:
            grad_rank.sort(key=lambda x: x[1], reverse=True)
            print("\n[summary] largest gradient relative error (kernel vs reference)")
            for case, rel, grad_err in grad_rank[:top]:
                parts = ", ".join(f"{name}={vals.get('rel_max', float('nan')):.2e}" for name, vals in grad_err.items())
                print(f"  {_case_label(case)} rel_max={rel:.2e} [{parts}]")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark D-LinOSS kernels (forward/backward, bandwidth, agreement)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Device to benchmark on")
    parser.add_argument("--variants", nargs="*", default=list(ALL_VARIANTS), help="Variants to run")
    parser.add_argument("--lengths", nargs="*", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--batches", nargs="*", type=int, default=[2, 8, 32], help="Batch sizes to sweep")
    parser.add_argument("--ssms", nargs="*", type=int, default=[64, 256], help="SSM sizes to sweep")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--no_backward", action="store_true", help="Skip backward checks/timing")
    parser.add_argument("--raw_params", action="store_true", help="Use raw randomly sampled a,g,step (unstable). Default uses stable projection.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Optional JSONL path to append per-case results (disables slot rotation)")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for slot files old.jsonl/new.jsonl; defaults to outputs/bench_dlinoss")
    parser.add_argument("--diff-only", action="store_true", help="Only print performance diff between slot files and exit")
    parser.add_argument("--top", type=int, default=5, help="Top-N improvements/regressions to display in diffs")
    parser.add_argument("--summary-top", type=int, default=8, help="Number of cases to show per summary section")
    parser.add_argument("--no-summary", action="store_true", help="Disable aggregated summary tables")
    parser.add_argument("--no-nvtx", action="store_true", help="Disable NVTX ranges emission")
    args = parser.parse_args(list(argv) if argv is not None else None)

    setup_env(args.device)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    print("[env]", env_info())
    print("[kernels]", kernel_info())
    summary_cfg = "off" if args.no_summary else args.summary_top
    print(
        f"[config] variants={args.variants} lengths={args.lengths} batches={args.batches} ssms={args.ssms} repeats={args.repeats} "
        f"warmup={args.warmup} stable_params={not args.raw_params} seed={args.seed} nvtx={'off' if args.no_nvtx else 'on'} "
        f"summary_top={summary_cfg}"
    )

    if device.type == "cpu" and any(dlinoss_has_kernels(v) for v in args.variants):
        print("[note] CUDA kernels exist but device=cpu; kernel path will fall back to reference.")

    # Resolve output/slot paths
    default_out_dir = _repo_root() / "outputs" / "bench_dlinoss"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir

    new_path: Optional[Path] = None
    old_path: Optional[Path] = None
    copy_new_to_old_after = False

    if args.output:
        # Explicit target file; no slot management
        out_path: Optional[Path] = Path(args.output).resolve()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        new_path = out_dir / "new.jsonl"
        old_path = out_dir / "old.jsonl"
        if new_path.exists():
            # Shift new -> old (overwrite)
            try:
                new_path.replace(old_path)
            except Exception:
                # Best-effort overwrite
                try:
                    if old_path.exists():
                        old_path.unlink()
                except Exception:
                    pass
                shutil.copyfile(new_path, old_path)
                new_path.unlink(missing_ok=True)
            out_path = new_path
        else:
            out_path = new_path
            # If neither exists, we will populate both
            if not old_path.exists():
                copy_new_to_old_after = True

    # Diff-only mode: just compare slot files and exit
    if args.diff_only:
        if not out_dir.exists():
            print(f"[diff] no directory: {out_dir}")
            return
        oldf = out_dir / "old.jsonl"
        newf = out_dir / "new.jsonl"
        _print_perf_diff(oldf, newf, top=args.top)
        return

    nvtx_on = not args.no_nvtx
    all_results: List[Dict[str, Any]] = []
    for v in args.variants:
        for S in args.ssms:
            for B in args.batches:
                for T in args.lengths:
                    case = Case(variant=v, T=T, B=B, S=S)
                    with nvtx_range(f"case var={v} T={T} B={B} S={S}", enabled=nvtx_on):
                        ref, ker, err, extras = forward_benchmark(
                            case,
                            device,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            raw_params=args.raw_params,
                            nvtx=nvtx_on,
                        )
                        speed = ker.get("speedup_vs_ref", float("nan"))
                        ref_p95 = ref.get("p95_ms", float("nan"))
                        ker_p95 = ker.get("p95_ms", float("nan"))
                        ref_cv = ref.get("cv_pct", float("nan"))
                        ker_cv = ker.get("cv_pct", float("nan"))
                        ker_ns = ker.get("ns_per_elem", float("nan"))
                        ai = extras.get("arith_intensity", float("nan"))
                        print(
                            f"[fwd] var={v} T={T} B={B} S={S} "
                            f"ref={ref['ms']:.3f}±{ref['std']:.3f}ms(p95={ref_p95:.3f} cv={ref_cv:.1f}% {ref['GBs']:.2f}GB/s {ref['GFLOPs']:.2f}GF/s) "
                            f"ker={ker['ms']:.3f}±{ker['std']:.3f}ms(p95={ker_p95:.3f} cv={ker_cv:.1f}% ns/elem={ker_ns:.1f} {ker['GBs']:.2f}GB/s {ker['GFLOPs']:.2f}GF/s) "
                            f"speedup={speed:.2f}x AI={ai:.2f}F/B rel_err={err['out_rel_max']:.2e} |y|ref={err['max|y_ref|']:.2e} |y|ker={err['max|y_ker|']:.2e} fin(ref,ker)=({err['finite_ref']:.2f},{err['finite_ker']:.2f})"
                        )

                    backward_payload: Optional[Dict[str, Any]] = None
                    if not args.no_backward:
                        out_err, grad_err, bwd_ref, bwd_ker = numeric_and_backward(case, device, raw_params=args.raw_params, nvtx=nvtx_on)
                        alloc_fwd = peak_memory_forward(case, device, raw_params=args.raw_params, nvtx=nvtx_on) if device.type == "cuda" else 0
                        alloc = peak_memory(case, device, raw_params=args.raw_params, nvtx=nvtx_on) if device.type == "cuda" else 0
                        ge = grad_err
                        bwd_speed = (bwd_ref["ms"] / bwd_ker["ms"]) if bwd_ker["ms"] > 0 else float('inf')
                        ge_s = (
                            f"a(max={ge['a']['max_abs']:.2e},rel={ge['a']['rel_max']:.2e}) "
                            f"g(max={ge['g']['max_abs']:.2e},rel={ge['g']['rel_max']:.2e}) "
                            f"step(max={ge['step']['max_abs']:.2e},rel={ge['step']['rel_max']:.2e}) "
                            f"bu(max={ge['bu']['max_abs']:.2e},rel={ge['bu']['rel_max']:.2e})"
                        )
                        fin_k = bwd_ker.get("finite", {})
                        fin_s = f" fin(a,g,step,bu)=({fin_k.get('a',0.0):.2f},{fin_k.get('g',0.0):.2f},{fin_k.get('step',0.0):.2f},{fin_k.get('bu',0.0):.2f})"
                        print(
                            f"[bwd] var={v} T={T} B={B} S={S} ref={bwd_ref['ms']:.3f}ms ker={bwd_ker['ms']:.3f}ms speedup={bwd_speed:.2f}x "
                            f"out(max={out_err['max_abs']:.2e},mean={out_err['mean_abs']:.2e},rmse={out_err['rmse']:.2e}) "
                            f"grads[{ge_s}]{fin_s} peak_alloc_fwd={human_bytes(alloc_fwd)} peak_alloc_fwbw={human_bytes(alloc)}"
                        )
                        backward_payload = {
                            "out_err": out_err,
                            "grad_err": grad_err,
                            "bwd_ref": bwd_ref,
                            "bwd_ker": bwd_ker,
                            "bwd_speedup": bwd_speed,
                            "alloc_fwd": alloc_fwd,
                            "alloc": alloc,
                        }

                    if out_path is not None:
                        rec: Dict[str, Any] = {
                            "device": str(device),
                            "variant": v,
                            "T": T,
                            "B": B,
                            "S": S,
                            "ref": ref,
                            "ker": ker,
                            "err": err,
                            "workload": extras,
                        }
                        if not args.no_backward:
                            rec.update(
                                {
                                    "bwd_ref_ms": bwd_ref["ms"],
                                    "bwd_ker_ms": bwd_ker["ms"],
                                    "bwd_speedup": bwd_speed,
                                    "bwd_finite": bwd_ker.get("finite", {}),
                                    "out_err": out_err,
                                    "grad_err": grad_err,
                                    "peak_alloc_fwd": alloc_fwd,
                                    "peak_alloc_fwbw": alloc,
                                }
                            )
                        _append_jsonl(out_path, rec)

                    all_results.append({
                        "case": case,
                        "ref": ref,
                        "ker": ker,
                        "err": err,
                        "extras": extras,
                        "backward": backward_payload,
                    })

    if all_results and not args.no_summary:
        _print_summary_tables(all_results, top=args.summary_top)

    # If requested, bootstrap both slots with the same run
    if copy_new_to_old_after and new_path is not None and old_path is not None:
        try:
            shutil.copyfile(new_path, old_path)
            print(f"[slots] initialized: copied {new_path} -> {old_path}")
        except Exception as exc:
            print(f"[slots] WARN: failed copying to old slot: {exc}")

    # After writing new, show a quick diff vs old
    if old_path is not None and new_path is not None and old_path.exists() and new_path.exists():
        _print_perf_diff(old_path, new_path, top=args.top)


if __name__ == "__main__":  # pragma: no cover
    main()
