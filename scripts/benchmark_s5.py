#!/usr/bin/env python
"""Benchmark S5 PyTorch kernels against reference JAX implementation."""

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

from models.S5 import S5Layer as JaxS5Layer  # noqa: E402

from ossm.models import s5 as torch_s5  # noqa: E402
from ossm.models import _s5_scan  # noqa: E402


@contextmanager
def _disable_kernel(flag: bool):
    if not flag:
        yield
        return
    old_kernels = getattr(_s5_scan, "_kernels", None)
    _s5_scan._kernels = None  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _s5_scan._kernels = old_kernels  # type: ignore[attr-defined]


def _to_numpy(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(tensor.detach().cpu().numpy())


def _build_layers(ssm_size: int, hidden_dim: int, *, blocks: int, discretization: str):
    torch.manual_seed(42)
    layer = torch_s5.S5Layer(ssm_size=ssm_size, hidden_dim=hidden_dim, blocks=blocks, discretization=discretization).eval()

    params = {
        "Lambda_re": _to_numpy(layer.lambda_real),
        "Lambda_im": _to_numpy(layer.lambda_imag),
        "B": _to_numpy(layer.B),
        "C": _to_numpy(layer.C),
        "D": _to_numpy(layer.D),
        "log_step": _to_numpy(layer.log_step),
    }

    key = jr.PRNGKey(0)
    jax_layer = JaxS5Layer(
        ssm_size,
        blocks,
        hidden_dim,
        "lecun_normal",
        True,
        False,
        discretization,
        0.001,
        0.1,
        1.0,
        key=key,
    )
    for name, value in params.items():
        object.__setattr__(jax_layer, name, value)

    return layer, jax_layer, params


def _measure_torch(layer: torch_s5.S5Layer, inputs: torch.Tensor, *, disable_kernel: bool) -> tuple[float, torch.Tensor]:
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


def _measure_jax(jax_layer: JaxS5Layer, inputs: jnp.ndarray) -> tuple[float, jnp.ndarray]:
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
    if not _s5_scan.is_available():
        error = _s5_scan.extension_error()
        details = f"\nLast extension error: {error}" if error is not None else ""
        raise RuntimeError(
            "S5 custom kernels are unavailable. Install OSSM with the `linoss` extra "
            "(e.g. `pip install -e .[linoss]`) so the C++ extension is built before benchmarking." + details
        )

    ssm_size = 64
    hidden_dim = 128
    seq_len = 256
    blocks = 1
    discretization = "zoh"

    layer, jax_layer, params = _build_layers(ssm_size, hidden_dim, blocks=blocks, discretization=discretization)

    torch_input = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)
    jax_input = _to_numpy(torch_input.squeeze(0))

    torch_kernel_time, torch_kernel_out = _measure_torch(layer, torch_input, disable_kernel=False)
    torch_fallback_time, torch_fallback_out = _measure_torch(layer, torch_input, disable_kernel=True)

    max_diff_kernel_vs_fallback = (torch_kernel_out - torch_fallback_out).abs().max().item()

    jax_time, jax_out = _measure_jax(jax_layer, jax_input)

    torch_kernel_jnp = _to_numpy(torch_kernel_out.squeeze(0))
    max_diff_kernel_vs_jax = jnp.max(jnp.abs(torch_kernel_jnp - jax_out)).item()

    layer_double = torch_s5.S5Layer(ssm_size=ssm_size, hidden_dim=hidden_dim, blocks=blocks, discretization=discretization).double().eval()
    layer_double.load_state_dict({k: v.double() for k, v in layer.state_dict().items()})
    torch_input_double = torch_input.double()
    with torch.no_grad():
        torch_double_out = layer_double(torch_input_double)

    prev_x64 = jax.config.read("jax_enable_x64") if hasattr(jax.config, "read") else None
    jax.config.update("jax_enable_x64", True)
    params64 = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in params.items()}
    jax_layer64 = JaxS5Layer(
        ssm_size,
        blocks,
        hidden_dim,
        "lecun_normal",
        True,
        False,
        discretization,
        0.001,
        0.1,
        1.0,
        key=jr.PRNGKey(1),
    )
    for name, value in params64.items():
        object.__setattr__(jax_layer64, name, value)
    jax_double_in = jnp.asarray(torch_input_double.squeeze(0).detach().cpu().numpy())
    jax_double_out = jax_layer64(jax_double_in)
    torch_double_jnp = jnp.asarray(torch_double_out.squeeze(0).detach().cpu().numpy())
    max_diff_double = jnp.max(jnp.abs(torch_double_jnp - jax_double_out)).item()
    if prev_x64 is not None:
        jax.config.update("jax_enable_x64", prev_x64)

    print("S5 Benchmark")
    print(f"Sequence length: {seq_len}, state dim: {ssm_size}, hidden dim: {hidden_dim}")
    print(f"PyTorch kernel time:   {torch_kernel_time * 1e3:.3f} ms/step")
    print(f"PyTorch fallback time: {torch_fallback_time * 1e3:.3f} ms/step")
    print(f"JAX jit time:          {jax_time * 1e3:.3f} ms/step")
    print(f"Max |PyTorch kernel - PyTorch fallback|: {max_diff_kernel_vs_fallback:.3e}")
    print(f"Max |PyTorch kernel - JAX|:              {max_diff_kernel_vs_jax:.3e}")
    print(f"Max |PyTorch kernel64 - JAX64|:          {max_diff_double:.3e}")


if __name__ == "__main__":
    main()
