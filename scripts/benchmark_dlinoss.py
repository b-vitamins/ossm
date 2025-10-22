#!/usr/bin/env python
"""Benchmark D-LinOSS PyTorch kernels against JIT-compiled JAX."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLUGINS_DISABLED"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import torch

from ossm.models._dlinoss_scan import extension_error, has_kernels

try:
    from damped_linoss.models.LinOSS import DampedIMEX1Layer as JaxDampedIMEX1Layer
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "damped_linoss is required for benchmarking; install with "
        "`pip install -e .[linoss,linoss-ref]` (and consider adding --no-deps to the latter)."
    ) from exc

from ossm.models.dlinoss import DampedLinOSSLayer

jax.config.update("jax_enable_x64", True)

_JAX_ARRAY_TYPE = getattr(jax, "Array", type(jnp.asarray(0.0)))


@contextmanager
def _disable_kernel(flag: bool):
    token = "OSSM_DLINOSS_DISABLE_KERNEL"
    old_value = os.environ.get(token)
    if flag:
        os.environ[token] = "1"
    else:
        if token in os.environ:
            del os.environ[token]
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(token, None)
        else:
            os.environ[token] = old_value


def _to_numpy(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(tensor.detach().cpu().numpy())


def _promote_tree_to_64bits(tree):
    def _promote(value):
        if isinstance(value, _JAX_ARRAY_TYPE):
            if jnp.issubdtype(value.dtype, jnp.floating):
                return value.astype(jnp.float64)
            if jnp.issubdtype(value.dtype, jnp.complexfloating):
                return value.astype(jnp.complex128)
        return value

    return jtu.tree_map(_promote, tree)


def _build_layers(ssm_size: int, hidden_dim: int):
    torch.manual_seed(42)
    layer = DampedLinOSSLayer(ssm_size=ssm_size, hidden_dim=hidden_dim).eval()

    params = {
        "A_diag": _to_numpy(layer.A_diag),
        "G_diag": _to_numpy(layer.G_diag),
        "dt": _to_numpy(layer.steps),
        "B": _to_numpy(layer.B),
        "C": _to_numpy(layer.C),
        "D": _to_numpy(layer.D),
    }

    key = jr.PRNGKey(0)
    jax_layer = JaxDampedIMEX1Layer(
        state_dim=ssm_size,
        hidden_dim=hidden_dim,
        initialization="ring",
        r_min=0.9,
        r_max=1.0,
        theta_min=0.0,
        theta_max=jnp.pi,
        G_min=0.0,
        G_max=1.0,
        A_min=0.0,
        A_max=1.0,
        dt_std=0.5,
        key=key,
    )

    object.__setattr__(jax_layer, "A_diag", params["A_diag"])
    object.__setattr__(jax_layer, "G_diag", params["G_diag"])
    object.__setattr__(jax_layer, "dt", params["dt"])
    object.__setattr__(jax_layer, "B", params["B"])
    object.__setattr__(jax_layer, "C", params["C"])
    object.__setattr__(jax_layer, "D", params["D"])

    return layer, jax_layer


def _measure_torch(layer: DampedLinOSSLayer, inputs: torch.Tensor, *, disable_kernel: bool) -> tuple[float, torch.Tensor]:
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


def _measure_jax(jax_layer: JaxDampedIMEX1Layer, inputs: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    forward = jax.jit(lambda x: jax_layer(x))
    warm = forward(inputs)
    warm.block_until_ready()
    start = time.perf_counter()
    for _ in range(100):
        out = forward(inputs)
        out.block_until_ready()
    elapsed = time.perf_counter() - start
    return elapsed / 100.0, out


def main() -> None:
    if not has_kernels():
        error = extension_error()
        details = f"\nLast extension error: {error}" if error is not None else ""
        raise RuntimeError(
            "D-LinOSS custom kernels are unavailable. "
            "Install OSSM with the `linoss` extra (e.g. `pip install -e .[linoss]`) "
            "so the C++ extension is built before benchmarking." + details
        )

    ssm_size = 64
    hidden_dim = 128
    seq_len = 256

    layer, jax_layer = _build_layers(ssm_size, hidden_dim)

    torch_input = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)
    jax_input = _to_numpy(torch_input.squeeze(0))

    torch_kernel_time, torch_kernel_out = _measure_torch(layer, torch_input, disable_kernel=False)
    torch_fallback_time, torch_fallback_out = _measure_torch(layer, torch_input, disable_kernel=True)

    max_diff_kernel_vs_fallback = (torch_kernel_out - torch_fallback_out).abs().max().item()

    jax_time, jax_out = _measure_jax(jax_layer, jax_input)

    torch_kernel_jnp = jnp.asarray(torch_kernel_out.squeeze(0).detach().cpu().numpy())
    max_diff_kernel_vs_jax = jnp.max(jnp.abs(torch_kernel_jnp - jax_out)).item()

    # High-precision parity check
    layer_double = DampedLinOSSLayer(ssm_size=ssm_size, hidden_dim=hidden_dim).double().eval()
    layer_double.load_state_dict({k: v.double() for k, v in layer.state_dict().items()})
    torch_input_double = torch_input.double()
    with torch.no_grad():
        torch_double_out = layer_double(torch_input_double)

    jax_layer64 = _promote_tree_to_64bits(jax_layer)
    jax_double_in = jnp.asarray(
        torch_input_double.squeeze(0).detach().cpu().numpy(), dtype=jnp.float64
    )
    jax_double_out = jax_layer64(jax_double_in)
    torch_double_jnp = jnp.asarray(
        torch_double_out.squeeze(0).detach().cpu().numpy(), dtype=jnp.float64
    )
    max_diff_double = jnp.max(jnp.abs(torch_double_jnp - jax_double_out)).item()

    print("Damped LinOSS IMEX1 Benchmark")
    print(f"Sequence length: {seq_len}, state dim: {ssm_size}, hidden dim: {hidden_dim}")
    print(f"PyTorch kernel time:   {torch_kernel_time * 1e3:.3f} ms/step")
    print(f"PyTorch fallback time: {torch_fallback_time * 1e3:.3f} ms/step")
    print(f"JAX jit time:          {jax_time * 1e3:.3f} ms/step")
    print(f"Max |PyTorch kernel - PyTorch fallback|: {max_diff_kernel_vs_fallback:.3e}")
    print(f"Max |PyTorch kernel - JAX|:              {max_diff_kernel_vs_jax:.3e}")
    print(f"Max |PyTorch kernel64 - JAX64|:          {max_diff_double:.3e}")


if __name__ == "__main__":
    main()
