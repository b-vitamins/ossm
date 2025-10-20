from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from unittest import mock

from ossm.models.lru import _LRUScanFn
from ossm.models.rnn import _LinearRNNScanFn
from ossm.models.s5 import _S5ScanFn


def _has_kernel(attr: str) -> bool:
    try:
        from ossm import _kernels as kernels  # type: ignore[attr-defined]
    except ImportError:
        return False
    return hasattr(kernels, attr)


def _complex_scan_naive(lambda_bar: torch.Tensor, b_seq: torch.Tensor) -> torch.Tensor:
    length, batch, state = b_seq.shape
    state_vec = b_seq.new_zeros(batch, state)
    outputs = []
    for idx in range(length):
        state_vec = lambda_bar * state_vec + b_seq[idx]
        outputs.append(state_vec)
    return torch.stack(outputs, dim=0)


def _linear_rnn_naive(
    weight_hh: torch.Tensor,
    weight_xh: torch.Tensor,
    bias: torch.Tensor,
    inputs: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    length, batch, _ = inputs.shape
    state = initial_state
    outputs = []
    weight_hh_t = weight_hh.transpose(0, 1)
    weight_xh_t = weight_xh.transpose(0, 1)
    for idx in range(length):
        base = inputs[idx].matmul(weight_xh_t) + bias
        state = state.matmul(weight_hh_t) + base
        outputs.append(state)
    return torch.stack(outputs, dim=0)


@pytest.mark.parametrize("length,batch,state", [(6, 3, 4), (10, 2, 8)])
def test_lru_scan_matches_naive(length: int, batch: int, state: int) -> None:
    lambda_bar = torch.randn(state, dtype=torch.complex64, requires_grad=True)
    b_seq = torch.randn(length, batch, state, dtype=torch.complex64, requires_grad=True)

    out_ext = _LRUScanFn.apply(lambda_bar, b_seq)
    out_naive = _complex_scan_naive(lambda_bar, b_seq)

    torch.testing.assert_close(out_ext, out_naive, atol=2e-4, rtol=1e-4)

    grad_fn: Callable[[torch.Tensor], torch.Tensor] = (
        lambda out: out.real.sum() + out.imag.sum()
    )
    grad_ext = torch.autograd.grad(grad_fn(out_ext), (lambda_bar, b_seq))
    grad_naive = torch.autograd.grad(grad_fn(out_naive), (lambda_bar, b_seq))
    for lhs, rhs in zip(grad_ext, grad_naive):
        torch.testing.assert_close(lhs, rhs, atol=5e-5, rtol=5e-5)


def test_lru_extension_falls_back_to_python() -> None:
    if not _has_kernel("lru_scan"):
        pytest.skip("compiled LRU extension is unavailable")

    lambda_bar = torch.randn(5, dtype=torch.complex64, requires_grad=True)
    b_seq = torch.randn(7, 3, 5, dtype=torch.complex64, requires_grad=True)

    out_ext = _LRUScanFn.apply(lambda_bar, b_seq)

    with mock.patch("ossm.models._lru_scan.try_run_lru_scan", return_value=None):
        out_python = _LRUScanFn.apply(lambda_bar, b_seq)

    torch.testing.assert_close(out_ext, out_python, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("length,batch,state", [(4, 2, 6), (12, 1, 10)])
def test_s5_scan_matches_naive(length: int, batch: int, state: int) -> None:
    lambda_bar = torch.randn(state, dtype=torch.complex64, requires_grad=True)
    b_seq = torch.randn(length, batch, state, dtype=torch.complex64, requires_grad=True)

    out_ext = _S5ScanFn.apply(lambda_bar, b_seq)
    out_naive = _complex_scan_naive(lambda_bar, b_seq)

    torch.testing.assert_close(out_ext, out_naive, atol=2e-4, rtol=1e-4)

    grad_fn: Callable[[torch.Tensor], torch.Tensor] = (
        lambda out: out.real.sum() + out.imag.sum()
    )
    grad_ext = torch.autograd.grad(grad_fn(out_ext), (lambda_bar, b_seq))
    grad_naive = torch.autograd.grad(grad_fn(out_naive), (lambda_bar, b_seq))
    for lhs, rhs in zip(grad_ext, grad_naive):
        torch.testing.assert_close(lhs, rhs, atol=5e-5, rtol=5e-5)


def test_s5_extension_falls_back_to_python() -> None:
    if not _has_kernel("s5_scan"):
        pytest.skip("compiled S5 extension is unavailable")

    lambda_bar = torch.randn(6, dtype=torch.complex64, requires_grad=True)
    b_seq = torch.randn(5, 2, 6, dtype=torch.complex64, requires_grad=True)

    out_ext = _S5ScanFn.apply(lambda_bar, b_seq)

    with mock.patch("ossm.models._s5_scan.try_run_s5_scan", return_value=None):
        out_python = _S5ScanFn.apply(lambda_bar, b_seq)

    torch.testing.assert_close(out_ext, out_python, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("length,batch,input_size,hidden_size", [(6, 3, 4, 5), (9, 2, 3, 7)])
def test_linear_rnn_scan_matches_naive(
    length: int, batch: int, input_size: int, hidden_size: int
) -> None:
    weight_hh = torch.randn(hidden_size, hidden_size, requires_grad=True)
    weight_xh = torch.randn(hidden_size, input_size, requires_grad=True)
    bias = torch.randn(hidden_size, requires_grad=True)
    inputs = torch.randn(length, batch, input_size, requires_grad=True)
    initial_state = torch.randn(batch, hidden_size, requires_grad=True)

    out_ext = _LinearRNNScanFn.apply(weight_hh, weight_xh, bias, inputs, initial_state)
    out_naive = _linear_rnn_naive(weight_hh, weight_xh, bias, inputs, initial_state)

    torch.testing.assert_close(out_ext, out_naive, atol=2e-4, rtol=1e-4)

    grad_fn: Callable[[torch.Tensor], torch.Tensor] = lambda out: out.sum()
    grad_ext = torch.autograd.grad(
        grad_fn(out_ext), (weight_hh, weight_xh, bias, inputs, initial_state)
    )
    grad_naive = torch.autograd.grad(
        grad_fn(out_naive), (weight_hh, weight_xh, bias, inputs, initial_state)
    )
    for lhs, rhs in zip(grad_ext, grad_naive):
        torch.testing.assert_close(lhs, rhs, atol=2e-4, rtol=1e-4)


def test_linear_rnn_extension_falls_back_to_python() -> None:
    if not _has_kernel("linear_rnn_scan"):
        pytest.skip("compiled linear RNN extension is unavailable")

    weight_hh = torch.randn(4, 4, requires_grad=True)
    weight_xh = torch.randn(4, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    inputs = torch.randn(5, 2, 3, requires_grad=True)
    initial_state = torch.randn(2, 4, requires_grad=True)

    out_ext = _LinearRNNScanFn.apply(weight_hh, weight_xh, bias, inputs, initial_state)

    with mock.patch("ossm.models._rnn_scan.try_run_linear_rnn_scan", return_value=None):
        out_python = _LinearRNNScanFn.apply(weight_hh, weight_xh, bias, inputs, initial_state)

    torch.testing.assert_close(out_ext, out_python, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("model", ["lru", "s5"])
def test_complex_scan_matches_jax(model: str) -> None:
    jax = pytest.importorskip("jax")
    _ = pytest.importorskip("jax.numpy")

    length, batch, state = 12, 3, 5
    rng = np.random.default_rng(0)
    lambda_bar = torch.tensor(
        rng.standard_normal(state) + 1j * rng.standard_normal(state),
        dtype=torch.complex64,
    )
    b_seq = torch.tensor(
        rng.standard_normal((length, batch, state))
        + 1j * rng.standard_normal((length, batch, state)),
        dtype=torch.complex64,
    )

    torch_out = (
        _LRUScanFn.apply(lambda_bar, b_seq)
        if model == "lru"
        else _S5ScanFn.apply(lambda_bar, b_seq)
    )

    lambda_np = np.asarray(lambda_bar.detach().cpu().numpy())
    b_np = np.asarray(b_seq.detach().cpu().numpy())

    def scan_body(state, bu):
        new_state = lambda_np * state + bu
        return new_state, new_state

    _, jax_states = jax.lax.scan(scan_body, np.zeros((batch, state), dtype=np.complex64), b_np)

    torch.testing.assert_close(
        torch_out, torch.from_numpy(np.asarray(jax_states)), atol=5e-6, rtol=5e-6
    )


def test_linear_rnn_matches_jax_reference() -> None:
    jax = pytest.importorskip("jax")
    _ = pytest.importorskip("jax.numpy")

    length, batch, input_size, hidden_size = 15, 4, 6, 7
    rng = np.random.default_rng(0)
    weight_hh = torch.tensor(rng.standard_normal((hidden_size, hidden_size)), dtype=torch.float32)
    weight_xh = torch.tensor(rng.standard_normal((hidden_size, input_size)), dtype=torch.float32)
    bias = torch.tensor(rng.standard_normal(hidden_size), dtype=torch.float32)
    inputs = torch.tensor(
        rng.standard_normal((length, batch, input_size)), dtype=torch.float32
    )
    initial_state = torch.tensor(
        rng.standard_normal((batch, hidden_size)), dtype=torch.float32
    )

    torch_out = _LinearRNNScanFn.apply(weight_hh, weight_xh, bias, inputs, initial_state)

    weight_hh_np = np.asarray(weight_hh.numpy())
    weight_xh_np = np.asarray(weight_xh.numpy())
    bias_np = np.asarray(bias.numpy())
    inputs_np = np.asarray(inputs.numpy())
    init_np = np.asarray(initial_state.numpy())

    def scan_body(state, inp):
        base = inp @ weight_xh_np.T + bias_np
        new_state = state @ weight_hh_np.T + base
        return new_state, new_state

    _, jax_states = jax.lax.scan(scan_body, init_np, inputs_np)

    torch.testing.assert_close(
        torch_out, torch.from_numpy(np.asarray(jax_states)), atol=5e-6, rtol=5e-6
    )
