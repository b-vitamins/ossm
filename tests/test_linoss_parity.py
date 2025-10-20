from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from unittest import mock

from ossm.models._linoss_scan import is_available as linoss_extension_available
from ossm.models.linoss import LinOSSLayer, _LinossScanFn


def _naive_scan(a_matrix: torch.Tensor, b_seq: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the associative LinOSS scan using PyTorch ops."""

    length, batch, ssm, _ = b_seq.shape
    state = b_seq.new_zeros((batch, ssm, 2))
    outputs = []
    for t in range(length):
        state = torch.einsum("sij,bsj->bsi", a_matrix, state) + b_seq[t]
        outputs.append(state)
    return torch.stack(outputs, dim=0)


def _random_inputs(
    *,
    length: int,
    batch: int,
    ssm: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a_matrix = torch.randn(ssm, 2, 2, dtype=torch.complex64, device=device)
    b_seq = torch.randn(length, batch, ssm, 2, dtype=torch.complex64, device=device)
    return a_matrix, b_seq


@pytest.mark.parametrize("length,batch,ssm", [(4, 2, 3), (8, 1, 5)])
def test_linoss_scan_matches_naive(length: int, batch: int, ssm: int) -> None:
    device = torch.device("cpu")
    a_matrix, b_seq = _random_inputs(length=length, batch=batch, ssm=ssm, device=device)
    a_matrix.requires_grad_(True)
    b_seq.requires_grad_(True)

    out_custom = _LinossScanFn.apply(a_matrix, b_seq)
    out_naive = _naive_scan(a_matrix, b_seq)
    torch.testing.assert_close(out_custom, out_naive, atol=1e-6, rtol=1e-6)

    grad_fn: Callable[[torch.Tensor], torch.Tensor] = lambda out: out.real.sum() + out.imag.sum()
    grad_custom = torch.autograd.grad(grad_fn(out_custom), (a_matrix, b_seq))
    grad_naive = torch.autograd.grad(grad_fn(out_naive), (a_matrix, b_seq))
    for lhs, rhs in zip(grad_custom, grad_naive):
        torch.testing.assert_close(lhs, rhs, atol=5e-5, rtol=5e-5)


def test_linoss_extension_matches_python() -> None:
    if not linoss_extension_available():
        pytest.skip("compiled LinOSS extension is unavailable")

    device = torch.device("cpu")
    a_matrix, b_seq = _random_inputs(length=6, batch=2, ssm=4, device=device)

    a_matrix.requires_grad_(True)
    b_seq.requires_grad_(True)

    out_ext = _LinossScanFn.apply(a_matrix, b_seq)

    with mock.patch("ossm.models._linoss_scan.try_run_scan", return_value=None):
        out_fallback = _LinossScanFn.apply(a_matrix, b_seq)

    torch.testing.assert_close(out_ext, out_fallback, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("ssm_size,hidden_dim,length,batch", [(8, 6, 12, 2), (16, 16, 9, 3)])
def test_linoss_layer_matches_jax(
    ssm_size: int, hidden_dim: int, length: int, batch: int
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    eqx = pytest.importorskip("equinox")
    jax_linoss_module = pytest.importorskip("models.LinOSS")
    if not hasattr(jax_linoss_module, "LinOSSLayer"):
        pytest.skip("JAX LinOSS reference module does not expose LinOSSLayer")
    JaxLinossLayer = getattr(jax_linoss_module, "LinOSSLayer")

    jax.config.update("jax_enable_x64", False)

    key = jax.random.PRNGKey(0)
    jax_layer = JaxLinossLayer(ssm_size, hidden_dim, "IM", key=key)

    torch_layer = LinOSSLayer(ssm_size, hidden_dim, "IM").eval()
    with torch.no_grad():
        torch_layer.A_diag.copy_(torch.from_numpy(np.array(jax_layer.A_diag)))
        torch_layer.B.copy_(torch.from_numpy(np.array(jax_layer.B)))
        torch_layer.C.copy_(torch.from_numpy(np.array(jax_layer.C)))
        torch_layer.D.copy_(torch.from_numpy(np.array(jax_layer.D)))
        torch_layer.steps.copy_(torch.from_numpy(np.array(jax_layer.steps)))

    torch.manual_seed(0)
    inputs = torch.randn(batch, length, hidden_dim)
    with torch.no_grad():
        torch_out = torch_layer(inputs).numpy()

    jax_inputs = jnp.array(inputs.numpy())
    jax_out = np.array(jax.vmap(jax_layer)(jax_inputs))

    torch.testing.assert_close(torch.tensor(torch_out), torch.tensor(jax_out), atol=5e-6, rtol=5e-6)
