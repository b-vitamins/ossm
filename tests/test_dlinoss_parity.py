from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from ossm.models._dlinoss_scan import _reference_dlinoss_states
from ossm.models.dlinoss import DampedLinOSSLayer

_TEST_ROOT = Path(__file__).parent
_REFERENCE_PATH = _TEST_ROOT / "dlinoss_reference_cases.json"


@lru_cache(maxsize=1)
def _load_reference_cases() -> dict[str, dict[str, dict[str, object]]]:
    payload = json.loads(_REFERENCE_PATH.read_text())
    cases = payload.get("cases", payload)
    return cases

REFERENCE_CASES = _load_reference_cases()

VARIANTS = tuple(REFERENCE_CASES.keys())
DTYPES = (torch.float32, torch.float64)
DTYPE_KEYS = {torch.float32: "float32", torch.float64: "float64"}
ATOL = {torch.float32: 5e-6, torch.float64: 1e-9}
RTOL = {torch.float32: 5e-6, torch.float64: 1e-9}


def _tensor_from_json(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float64)


def _run_fallback(
    variant: str,
    a_diag: torch.Tensor,
    g_diag: torch.Tensor,
    step: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d_vec: torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, hidden_dim = inputs.shape
    ssm = a_diag.shape[0]

    b_real = b[..., 0]
    b_imag = b[..., 1]
    flat_inputs = inputs.reshape(batch * length, hidden_dim)
    bu_real = flat_inputs @ b_real.transpose(0, 1)
    bu_imag = flat_inputs @ b_imag.transpose(0, 1)
    bu = torch.complex(bu_real, bu_imag).reshape(batch, length, ssm)
    bu_seq = bu.permute(1, 0, 2).contiguous()

    states = _reference_dlinoss_states(variant, a_diag, g_diag, step, bu_seq)
    states_main = states[..., 1].permute(1, 0, 2).contiguous()
    states_aux = states[..., 0].permute(1, 0, 2).contiguous()

    states_flat = states_main.reshape(batch * length, ssm)
    c_real = c[..., 0]
    c_imag = c[..., 1]
    c_real_t = c_real.transpose(0, 1)
    c_imag_t = c_imag.transpose(0, 1)
    projected_real = states_flat.real @ c_real_t - states_flat.imag @ c_imag_t
    projected = projected_real.reshape(batch, length, -1)

    outputs = projected + inputs * d_vec.view(1, 1, -1)
    return outputs, states_main, states_aux


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_dlinoss_matches_reference(variant: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    case = REFERENCE_CASES[variant][DTYPE_KEYS[dtype]]
    device = torch.device("cpu")

    a_diag = _tensor_from_json(case["A_diag"]).to(dtype=dtype, device=device).clone().requires_grad_(True)
    g_diag = _tensor_from_json(case["G_diag"]).to(dtype=dtype, device=device).clone().requires_grad_(True)
    step = _tensor_from_json(case["step"]).to(dtype=dtype, device=device).clone().requires_grad_(True)
    b = _tensor_from_json(case["B"]).to(dtype=dtype, device=device).clone().requires_grad_(True)
    c = _tensor_from_json(case["C"]).to(dtype=dtype, device=device).clone().requires_grad_(True)
    d_vec = _tensor_from_json(case["D"]).to(dtype=dtype, device=device).clone().requires_grad_(True)
    inputs = _tensor_from_json(case["inputs"]).to(dtype=dtype, device=device)

    outputs_expected = _tensor_from_json(case["outputs"]).to(dtype=dtype, device=device)
    states_main_expected = _tensor_from_json(case["states"]["main"]).to(dtype=dtype, device=device)
    states_aux_expected = _tensor_from_json(case["states"]["aux"]).to(dtype=dtype, device=device)
    grads_expected = {
        key: _tensor_from_json(value).to(dtype=dtype, device=device)
        for key, value in case["grads"].items()
    }

    layer = DampedLinOSSLayer(
        ssm_size=int(case["ssm_size"]),
        hidden_dim=int(case["hidden_dim"]),
        variant=variant,
    ).to(dtype=dtype)

    with torch.no_grad():
        layer.A_diag.copy_(a_diag.detach())
        layer.G_diag.copy_(g_diag.detach())
        step_param = torch.logit(step.detach().clamp(1e-6, 1 - 1e-6))
        layer.steps.copy_(step_param.to(dtype=layer.steps.dtype))
        layer.B.copy_(b.detach().to(dtype=layer.B.dtype))
        layer.C.copy_(c.detach().to(dtype=layer.C.dtype))
        layer.D.copy_(d_vec.detach().to(dtype=layer.D.dtype))

    monkeypatch.setenv("OSSM_DLINOSS_DISABLE_KERNEL", "1")
    with torch.no_grad():
        layer_outputs = layer(inputs)
    torch.testing.assert_close(layer_outputs, outputs_expected, atol=ATOL[dtype], rtol=RTOL[dtype])

    outputs, states_main, states_aux = _run_fallback(variant, a_diag, g_diag, step, b, c, d_vec, inputs)

    states_main_realimag = torch.stack((states_main.real, states_main.imag), dim=-1)
    states_aux_realimag = torch.stack((states_aux.real, states_aux.imag), dim=-1)

    torch.testing.assert_close(outputs, outputs_expected, atol=ATOL[dtype], rtol=RTOL[dtype])
    torch.testing.assert_close(states_main_realimag, states_main_expected, atol=ATOL[dtype], rtol=RTOL[dtype])
    torch.testing.assert_close(states_aux_realimag, states_aux_expected, atol=ATOL[dtype], rtol=RTOL[dtype])

    loss = (
        outputs.pow(2).sum()
        + states_main.real.pow(2).sum()
        + states_main.imag.pow(2).sum()
        + states_aux.real.pow(2).sum()
        + states_aux.imag.pow(2).sum()
    )
    loss.backward()

    torch.testing.assert_close(a_diag.grad, grads_expected["A_diag"], atol=ATOL[dtype], rtol=RTOL[dtype])
    torch.testing.assert_close(g_diag.grad, grads_expected["G_diag"], atol=ATOL[dtype], rtol=RTOL[dtype])
    torch.testing.assert_close(step.grad, grads_expected["step"], atol=ATOL[dtype], rtol=RTOL[dtype])
    torch.testing.assert_close(b.grad, grads_expected["B"], atol=ATOL[dtype], rtol=RTOL[dtype])
    torch.testing.assert_close(c.grad, grads_expected["C"], atol=ATOL[dtype], rtol=RTOL[dtype])
    torch.testing.assert_close(d_vec.grad, grads_expected["D"], atol=ATOL[dtype], rtol=RTOL[dtype])
