#!/usr/bin/env python
"""Benchmark LinOSS PyTorch kernels against reference JAX implementation."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLUGINS_DISABLED"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import torch

jax.config.update("jax_enable_x64", True)

_JAX_ARRAY_TYPE = getattr(jax, "Array", type(jnp.asarray(0.0)))


def _ensure_linoss_repo() -> None:
    candidates: list[Path] = []
    repo_env = os.environ.get("LINOSS_REPO")
    if repo_env:
        candidates.append(Path(repo_env).expanduser())
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates.extend(
        [
            repo_root.parent / "linoss",
            repo_root / "linoss",
            repo_root / "external" / "linoss",
        ]
    )
    for path in candidates:
        models_dir = path / "models"
        if models_dir.is_dir():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return
    raise ModuleNotFoundError(
        "Unable to locate tk-rusch/linoss repository. Clone it locally and either "
        "set LINOSS_REPO to its path or add it to PYTHONPATH."
    )


_ensure_linoss_repo()

from models.LinOSS import LinOSSLayer as JaxLinOSSLayer  # noqa: E402  (requires repo path)

from ossm.models import linoss as torch_linoss  # noqa: E402
from ossm.models import _linoss_scan  # noqa: E402


@contextmanager
def _disable_kernel(flag: bool):
    if not flag:
        yield
        return
    old_kernels = getattr(_linoss_scan, "_kernels", None)
    _linoss_scan._kernels = None  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _linoss_scan._kernels = old_kernels  # type: ignore[attr-defined]


def _to_jax_array(tensor: torch.Tensor, *, dtype: jnp.dtype | None = None) -> jnp.ndarray:
    array = jnp.asarray(tensor.detach().cpu().numpy())
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return array


def _promote_tree_to_64bits(tree):
    def _promote(value):
        if isinstance(value, _JAX_ARRAY_TYPE):
            if jnp.issubdtype(value.dtype, jnp.floating):
                return value.astype(jnp.float64)
            if jnp.issubdtype(value.dtype, jnp.complexfloating):
                return value.astype(jnp.complex128)
        return value

    return jtu.tree_map(_promote, tree)


def _build_layers(ssm_size: int, hidden_dim: int, discretization: str):
    torch.manual_seed(42)
    layer = torch_linoss.LinOSSLayer(ssm_size=ssm_size, hidden_dim=hidden_dim, discretization=discretization).eval()

    torch_params = {
        "A_diag": layer.A_diag,
        "steps": layer.steps,
        "B": layer.B,
        "C": layer.C,
        "D": layer.D,
    }

    key = jr.PRNGKey(0)
    jax_layer = JaxLinOSSLayer(ssm_size, hidden_dim, discretization, key=key)
    for name, tensor in torch_params.items():
        target = getattr(jax_layer, name)
        target_dtype = getattr(target, "dtype", None)
        value = _to_jax_array(tensor, dtype=target_dtype)
        object.__setattr__(jax_layer, name, value)

    return layer, jax_layer


def _measure_torch(layer: torch_linoss.LinOSSLayer, inputs: torch.Tensor, *, disable_kernel: bool) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        with _disable_kernel(disable_kernel):
            for _ in range(10):
                layer(inputs)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                output = layer(inputs)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
    return elapsed / 100.0, output


def _measure_jax(jax_layer: JaxLinOSSLayer, inputs: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        return jax_layer(x)

    compiled = jax.jit(forward)
    warm = compiled(inputs)
    warm.block_until_ready()
    start = time.perf_counter()
    for _ in range(100):
        out = compiled(inputs)
        out.block_until_ready()
    elapsed = time.perf_counter() - start
    return elapsed / 100.0, out


def main() -> None:
    if not _linoss_scan.is_available():
        error = _linoss_scan.extension_error()
        details = f"\nLast extension error: {error}" if error is not None else ""
        raise RuntimeError(
            "LinOSS custom kernels are unavailable. Install OSSM with the `linoss` extra "
            "(e.g. `pip install -e .[linoss]`) so the C++ extension is built before benchmarking." + details
        )

    ssm_size = 64
    hidden_dim = 128
    seq_len = 256
    discretization = "IMEX"

    layer, jax_layer = _build_layers(ssm_size, hidden_dim, discretization)

    torch_input = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)
    jax_input = _to_jax_array(torch_input.squeeze(0))

    torch_kernel_time, torch_kernel_out = _measure_torch(layer, torch_input, disable_kernel=False)
    torch_fallback_time, torch_fallback_out = _measure_torch(layer, torch_input, disable_kernel=True)

    max_diff_kernel_vs_fallback = (torch_kernel_out - torch_fallback_out).abs().max().item()

    jax_time, jax_out = _measure_jax(jax_layer, jax_input)

    torch_kernel_jnp = _to_jax_array(torch_kernel_out.squeeze(0), dtype=jax_out.dtype)
    max_diff_kernel_vs_jax = jnp.max(jnp.abs(torch_kernel_jnp - jax_out)).item()

    layer_double = torch_linoss.LinOSSLayer(ssm_size=ssm_size, hidden_dim=hidden_dim, discretization=discretization).double().eval()
    layer_double.load_state_dict({k: v.double() for k, v in layer.state_dict().items()})
    torch_input_double = torch_input.double()
    with torch.no_grad():
        torch_double_out = layer_double(torch_input_double)

    jax_layer64 = _promote_tree_to_64bits(jax_layer)

    jax_double_in = _to_jax_array(torch_input_double.squeeze(0), dtype=jnp.float64)
    jax_double_out = jax_layer64(jax_double_in)
    torch_double_jnp = _to_jax_array(torch_double_out.squeeze(0), dtype=jnp.float64)
    max_diff_double = jnp.max(jnp.abs(torch_double_jnp - jax_double_out)).item()

    print("LinOSS IMEX Benchmark")
    print(f"Sequence length: {seq_len}, state dim: {ssm_size}, hidden dim: {hidden_dim}")
    print(f"PyTorch kernel time:   {torch_kernel_time * 1e3:.3f} ms/step")
    print(f"PyTorch fallback time: {torch_fallback_time * 1e3:.3f} ms/step")
    print(f"JAX jit time:          {jax_time * 1e3:.3f} ms/step")
    print(f"Max |PyTorch kernel - PyTorch fallback|: {max_diff_kernel_vs_fallback:.3e}")
    print(f"Max |PyTorch kernel - JAX|:              {max_diff_kernel_vs_jax:.3e}")
    print(f"Max |PyTorch kernel64 - JAX64|:          {max_diff_double:.3e}")


if __name__ == "__main__":
    main()
