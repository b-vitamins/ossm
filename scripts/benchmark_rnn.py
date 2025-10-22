#!/usr/bin/env python
"""Benchmark linear RNN PyTorch kernels against reference JAX implementation."""

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
import torch


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

from models.RNN import LinearCell as JaxLinearCell  # noqa: E402

from ossm.models import rnn as torch_rnn  # noqa: E402
from ossm.models import _rnn_scan  # noqa: E402


@contextmanager
def _disable_kernel(flag: bool):
    if not flag:
        yield
        return
    old_kernels = getattr(_rnn_scan, "_kernels", None)
    _rnn_scan._kernels = None  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _rnn_scan._kernels = old_kernels  # type: ignore[attr-defined]


def _to_numpy(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(tensor.detach().cpu().numpy())


def _build_layers(input_dim: int, hidden_dim: int):
    torch.manual_seed(42)
    cell = torch_rnn.LinearRNNCell(input_dim, hidden_dim)
    layer = torch_rnn.RNNLayer(cell).eval()

    params = {
        "weight": _to_numpy(cell.linear.weight),
        "bias": _to_numpy(cell.linear.bias),
    }

    key = jr.PRNGKey(0)
    jax_cell = JaxLinearCell(input_dim, hidden_dim, key=key)
    object.__setattr__(jax_cell.cell, "weight", params["weight"])
    object.__setattr__(jax_cell.cell, "bias", params["bias"])

    return layer, jax_cell, params


def _measure_torch(layer: torch_rnn.RNNLayer, inputs: torch.Tensor, *, disable_kernel: bool) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        with _disable_kernel(disable_kernel):
            for _ in range(10):
                layer(inputs)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                output, _ = layer(inputs)
            if inputs.is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
    return elapsed / 100.0, output


def _jax_linear_forward(jax_cell: JaxLinearCell, inputs: jnp.ndarray) -> jnp.ndarray:
    hidden_dim = jax_cell.hidden_size

    def step(state: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        new_state = jax_cell(state, x)
        return new_state, new_state

    init_state = jnp.zeros((hidden_dim,), dtype=inputs.dtype)
    _, states = jax.lax.scan(step, init_state, inputs)
    return states


def _measure_jax(jax_cell: JaxLinearCell, inputs: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    compiled = jax.jit(lambda x: _jax_linear_forward(jax_cell, x))
    warm = compiled(inputs)
    warm.block_until_ready()
    start = time.perf_counter()
    for _ in range(100):
        out = compiled(inputs)
        out.block_until_ready()
    elapsed = time.perf_counter() - start
    return elapsed / 100.0, out


def main() -> None:
    if not _rnn_scan.is_available():
        error = _rnn_scan.extension_error()
        details = f"\nLast extension error: {error}" if error is not None else ""
        raise RuntimeError(
            "Linear RNN custom kernels are unavailable. Install OSSM with the `linoss` extra "
            "(e.g. `pip install -e .[linoss]`) so the C++ extension is built before benchmarking." + details
        )

    input_dim = 128
    hidden_dim = 128
    seq_len = 256

    layer, jax_cell, params = _build_layers(input_dim, hidden_dim)

    torch_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)
    jax_input = _to_numpy(torch_input.squeeze(0))

    torch_kernel_time, torch_kernel_out = _measure_torch(layer, torch_input, disable_kernel=False)
    torch_fallback_time, torch_fallback_out = _measure_torch(layer, torch_input, disable_kernel=True)

    max_diff_kernel_vs_fallback = (torch_kernel_out - torch_fallback_out).abs().max().item()

    jax_time, jax_out = _measure_jax(jax_cell, jax_input)

    torch_kernel_jnp = _to_numpy(torch_kernel_out.squeeze(0))
    max_diff_kernel_vs_jax = jnp.max(jnp.abs(torch_kernel_jnp - jax_out)).item()

    layer_double = torch_rnn.RNNLayer(torch_rnn.LinearRNNCell(input_dim, hidden_dim)).double().eval()
    layer_double.load_state_dict({k: v.double() for k, v in layer.state_dict().items()})
    torch_input_double = torch_input.double()
    with torch.no_grad():
        torch_double_out, _ = layer_double(torch_input_double)

    prev_x64 = jax.config.read("jax_enable_x64") if hasattr(jax.config, "read") else None
    jax.config.update("jax_enable_x64", True)
    params64 = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in params.items()}
    jax_cell64 = JaxLinearCell(input_dim, hidden_dim, key=jr.PRNGKey(1))
    object.__setattr__(jax_cell64.cell, "weight", params64["weight"])
    object.__setattr__(jax_cell64.cell, "bias", params64["bias"])
    jax_double_in = jnp.asarray(torch_input_double.squeeze(0).detach().cpu().numpy())
    jax_double_out = _jax_linear_forward(jax_cell64, jax_double_in)
    torch_double_jnp = jnp.asarray(torch_double_out.squeeze(0).detach().cpu().numpy())
    max_diff_double = jnp.max(jnp.abs(torch_double_jnp - jax_double_out)).item()
    if prev_x64 is not None:
        jax.config.update("jax_enable_x64", prev_x64)

    print("Linear RNN Benchmark")
    print(f"Sequence length: {seq_len}, input dim: {input_dim}, hidden dim: {hidden_dim}")
    print(f"PyTorch kernel time:   {torch_kernel_time * 1e3:.3f} ms/step")
    print(f"PyTorch fallback time: {torch_fallback_time * 1e3:.3f} ms/step")
    print(f"JAX jit time:          {jax_time * 1e3:.3f} ms/step")
    print(f"Max |PyTorch kernel - PyTorch fallback|: {max_diff_kernel_vs_fallback:.3e}")
    print(f"Max |PyTorch kernel - JAX|:              {max_diff_kernel_vs_jax:.3e}")
    print(f"Max |PyTorch kernel64 - JAX64|:          {max_diff_double:.3e}")


if __name__ == "__main__":
    main()
