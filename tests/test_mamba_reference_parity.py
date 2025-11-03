"""Parity tests against goldens generated from mamba-ssm.

Loads tests/mamba_reference_cases.json (produced by scripts/refresh_mamba_goldens.py)
and checks that our in-house _MambaMixer matches forward outputs and gradients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.testing import assert_close

from ossm.models.mambarec import _MambaMixer
import pytest


_GOLDEN_PATH = Path("tests/mamba_reference_cases.json")


def _to_tensor(x: object, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _load_goldens() -> Dict[str, object]:
    if not _GOLDEN_PATH.exists():
        raise RuntimeError(
            f"Golden reference missing at {_GOLDEN_PATH}. Regenerate via scripts/refresh_mamba_goldens.py"
        )
    return json.loads(_GOLDEN_PATH.read_text())


def _build_mixer_from_golden(case: Dict[str, object], device: torch.device) -> _MambaMixer:
    cfg = case["config"]
    mixer = _MambaMixer(
        d_model=int(cfg["d_model"]),
        d_state=int(cfg["d_state"]),
        d_conv=int(cfg["d_conv"]),
        expand=int(cfg["expand"]),
        dt_rank=max(1, int(cfg["d_model"] // 16)),
        conv_bias=True,
        bias=False,
    ).to(device)

    params: Dict[str, list] = case["params"]  # type: ignore[assignment]
    with torch.no_grad():
        mixer.in_proj.weight.copy_(_to_tensor(params["in_proj.weight"], device))
        mixer.conv1d.weight.copy_(_to_tensor(params["conv1d.weight"], device))
        mixer.conv1d.bias.copy_(_to_tensor(params["conv1d.bias"], device))
        mixer.x_proj.weight.copy_(_to_tensor(params["x_proj.weight"], device))
        mixer.dt_proj.weight.copy_(_to_tensor(params["dt_proj.weight"], device))
        mixer.dt_proj.bias.copy_(_to_tensor(params["dt_proj.bias"], device))
        mixer.A_log.copy_(_to_tensor(params["A_log"], device))
        mixer.out_proj.weight.copy_(_to_tensor(params["out_proj.weight"], device))

    return mixer


def test_mamba_mixer_matches_golden_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _load_goldens()
    case = payload["case"]
    device = torch.device("cpu")

    # Force fallback path to avoid kernel/AMP differences
    import ossm.models.mambarec as mambarec
    monkeypatch.setattr(mambarec, "_try_selective_scan", lambda **kwargs: None)

    mixer = _build_mixer_from_golden(case, device)
    mixer.eval()

    inputs = _to_tensor(case["inputs"], device)
    outputs = mixer(inputs)

    expected = _to_tensor(case["outputs"], device)
    assert_close(outputs, expected, atol=1e-5, rtol=1e-5)


def test_mamba_mixer_matches_golden_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _load_goldens()
    case = payload["case"]
    device = torch.device("cpu")

    import ossm.models.mambarec as mambarec
    monkeypatch.setattr(mambarec, "_try_selective_scan", lambda **kwargs: None)

    mixer = _build_mixer_from_golden(case, device)
    mixer.train()

    inputs = _to_tensor(case["inputs"], device).requires_grad_(True)
    outputs = mixer(inputs)
    loss = outputs.square().mean()
    loss.backward()

    grads_expected: Dict[str, list] = case["grads"]  # type: ignore[assignment]

    # Compare input grads
    assert_close(inputs.grad, _to_tensor(grads_expected["inputs"], device), atol=2e-5, rtol=2e-5)

    # Compare parameter grads
    name_map = {
        "in_proj.weight": mixer.in_proj.weight,
        "conv1d.weight": mixer.conv1d.weight,
        "conv1d.bias": mixer.conv1d.bias,
        "x_proj.weight": mixer.x_proj.weight,
        "dt_proj.weight": mixer.dt_proj.weight,
        "dt_proj.bias": mixer.dt_proj.bias,
        "A_log": mixer.A_log,
        "out_proj.weight": mixer.out_proj.weight,
    }
    for key, param in name_map.items():
        assert param.grad is not None, f"Missing grad for {key}"
        assert_close(param.grad, _to_tensor(grads_expected[key], device), atol=2e-5, rtol=2e-5)
