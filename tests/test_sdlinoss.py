"""Tests for Selective D-LinOSS."""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
from torch import Tensor

import ossm.models._sdlinoss_scan as _sdlinoss_scan
from ossm.models.sdlinoss import SelectiveDLinOSSLayer, run_sdlinoss


def seed_all(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_stable_AG_from_rt(r: Tensor, theta: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
    r2 = torch.clamp(r * r, min=1e-8)
    dtc = torch.clamp(dt, min=1e-6)
    A = (r2 - 2.0 * r * torch.cos(theta) + 1.0) / (dtc * dtc * r2)
    G = (1.0 - r2) / (dtc * r2)
    return torch.clamp(A, min=0.0), torch.clamp(G, min=0.0)


def build_M_F_imex1(A: Tensor, G: Tensor, dt: Tensor) -> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
    S = 1.0 + dt * G
    Sinv = 1.0 / torch.clamp(S, min=1e-6)
    M11 = 1.0 - (dt * dt) * Sinv * A
    M12 = Sinv
    M21 = -(dt * dt) * Sinv * A
    M22 = Sinv
    F1 = (dt * dt) * Sinv
    F2 = F1
    return (M11, M12, M21, M22), (F1, F2)


def naive_rollout_imex1(A: Tensor, G: Tensor, dt: Tensor, u: Tensor) -> Tensor:
    L, B, M = u.shape
    w = torch.zeros(B, M, dtype=u.dtype, device=u.device)
    x = torch.zeros(B, M, dtype=u.dtype, device=u.device)
    xs = []
    for t in range(L):
        S = 1.0 + dt[t] * G[t]
        comb = -A[t] * x + u[t]
        w = (w + (dt[t] * dt[t]) * comb) / torch.clamp(S, min=1e-6)
        x = x + w
        xs.append(x)
    return torch.stack(xs, dim=0)


@pytest.mark.parametrize("L,B,M", [(8, 2, 5), (3, 1, 7)])
def test_run_sdlinoss_matches_naive_rollout(L: int, B: int, M: int) -> None:
    seed_all(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_c = torch.complex64

    r = torch.rand(L, B, M, device=device) * 0.79 + 0.20
    th = (torch.rand(L, B, M, device=device) * 2 - 1) * math.pi
    dt = torch.sigmoid(torch.randn(L, B, M, device=device))
    A, G = make_stable_AG_from_rt(r, th, dt)

    u = torch.randn(L, B, M, device=device, dtype=torch.float32)
    u = torch.complex(u, torch.zeros_like(u)).to(dtype=dtype_c)

    out_ref = run_sdlinoss("imex1", A, G, dt, u)
    out_naive = naive_rollout_imex1(A, G, dt, u)
    assert torch.allclose(out_ref.real, out_naive.real, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_ref.imag, out_naive.imag, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("shape_kind", ["M", "LM", "BM", "LBM"])
def test_run_sdlinoss_broadcastability(shape_kind: str) -> None:
    seed_all(0)
    L, B, M = 6, 4, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r = torch.full((L, B, M), 0.9, device=device)
    th = torch.zeros(L, B, M, device=device)
    dt = torch.full((L, B, M), 0.5, device=device)
    A_full, G_full = make_stable_AG_from_rt(r, th, dt)

    if shape_kind == "M":
        A = A_full[0, 0]
        G = G_full[0, 0]
        D = dt[0, 0]
    elif shape_kind == "LM":
        A = A_full[:, 0]
        G = G_full[:, 0]
        D = dt[:, 0]
    elif shape_kind == "BM":
        A = A_full[0]
        G = G_full[0]
        D = dt[0]
    else:
        A, G, D = A_full, G_full, dt

    u = torch.randn(L, B, M, device=device, dtype=torch.float32)
    u = torch.complex(u, torch.zeros_like(u))
    y = run_sdlinoss("imex1", A, G, D, u)
    y_full = run_sdlinoss("imex1", A_full, G_full, dt, u)
    assert torch.allclose(y, y_full, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("shape_kind", ["M", "LM", "BM", "LBM"])
def test_cuda_kernels_accept_broadcast_views(shape_kind: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not _sdlinoss_scan.has_kernels("imex1"):
        pytest.skip("Selective D-LinOSS CUDA kernels unavailable")

    seed_all(42)
    device = torch.device("cuda")
    L, B, M = 7, 3, 5

    base = torch.rand(L, B, M, device=device)
    A_full = base + 0.1
    G_full = base * 0.5
    dt_full = torch.sigmoid(torch.randn(L, B, M, device=device))

    if shape_kind == "M":
        A = A_full[0, 0]
        G = G_full[0, 0]
        D = dt_full[0, 0]
    elif shape_kind == "LM":
        A = A_full[:, 0]
        G = G_full[:, 0]
        D = dt_full[:, 0]
    elif shape_kind == "BM":
        A = A_full[0]
        G = G_full[0]
        D = dt_full[0]
    else:
        A, G, D = A_full, G_full, dt_full

    bu_real = torch.randn(L, B, M, device=device)
    bu_imag = torch.randn(L, B, M, device=device)
    bu = torch.complex(bu_real, bu_imag)

    out_kernel = run_sdlinoss("imex1", A, G, D, bu)
    out_ref = _sdlinoss_scan._fallback_sdlinoss("imex1", A_full, G_full, dt_full, bu)

    torch.testing.assert_close(out_kernel, out_ref, atol=1e-5, rtol=1e-5)


def test_eigenvalue_magnitude_matches_r_and_contraction_law() -> None:
    seed_all(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, B, M = 1, 1, 5

    r = torch.linspace(0.2, 0.98, M, device=device).view(L, B, M)
    th = torch.linspace(-math.pi, math.pi, M, device=device).view(L, B, M)
    dt = torch.full((L, B, M), 0.4, device=device)
    A, G = make_stable_AG_from_rt(r, th, dt)

    (M11, M12, M21, M22), _ = build_M_F_imex1(A, G, dt)
    mat = torch.stack(
        [
            torch.stack([M11, M12], dim=-1),
            torch.stack([M21, M22], dim=-1),
        ],
        dim=-2,
    )
    eigvals = torch.linalg.eigvals(mat)
    mag = eigvals.abs().amax(dim=-1)
    assert torch.allclose(mag.squeeze(), r.squeeze(), atol=5e-4, rtol=1e-4)
    assert torch.allclose(
        mag.squeeze(), (1.0 + dt.squeeze() * G.squeeze()) ** (-0.5), atol=5e-4, rtol=1e-4
    )


def test_memory_monotonicity_with_r() -> None:
    seed_all(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, B, M = 20, 1, 1
    dt = torch.full((L, B, M), 0.5, device=device)

    r_hi = torch.full((L, B, M), 0.98, device=device)
    r_lo = torch.full((L, B, M), 0.70, device=device)
    th = torch.zeros(L, B, M, device=device)

    A_hi, G_hi = make_stable_AG_from_rt(r_hi, th, dt)
    A_lo, G_lo = make_stable_AG_from_rt(r_lo, th, dt)

    u = torch.zeros(L, B, M, device=device, dtype=torch.float32)
    u[0] = 1.0
    u = torch.complex(u, torch.zeros_like(u))

    x_hi = run_sdlinoss("imex1", A_hi, G_hi, dt, u).abs()
    x_lo = run_sdlinoss("imex1", A_lo, G_lo, dt, u).abs()
    assert (x_hi[-1] > x_lo[-1]).all()


def test_run_sdlinoss_zero_input_gives_zero_output() -> None:
    seed_all(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, B, M = 5, 2, 3
    r = torch.full((L, B, M), 0.9, device=device)
    th = torch.zeros(L, B, M, device=device)
    dt = torch.full((L, B, M), 0.5, device=device)
    A, G = make_stable_AG_from_rt(r, th, dt)
    u = torch.zeros(L, B, M, device=device, dtype=torch.complex64)
    y = run_sdlinoss("imex1", A, G, dt, u)
    assert torch.count_nonzero(y).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layer_forward_shapes_and_dtypes(dtype: torch.dtype) -> None:
    seed_all(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is torch.float16 and not torch.cuda.is_available():
        pytest.skip("float16 forward only meaningful on CUDA")

    layer = SelectiveDLinOSSLayer(
        ssm_size=8,
        hidden_dim=16,
        variant="imex1",
        selective_injection=True,
        per_step_dt=True,
    ).to(device=device, dtype=dtype)

    B, L, H = 3, 9, 16
    x = torch.randn(B, L, H, device=device, dtype=dtype)
    y = layer(x)
    assert y.shape == (B, L, H)
    assert y.dtype == dtype
    assert torch.isfinite(y).all()


def test_layer_backward_gradients_exist() -> None:
    seed_all(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = SelectiveDLinOSSLayer(ssm_size=6, hidden_dim=12, per_step_dt=True).to(device)
    x = torch.randn(4, 7, 12, device=device)
    y = layer(x)
    loss = (y**2).mean()
    loss.backward()

    params_to_check = [
        layer.r_head.weight,
        layer.th_head.weight,
        layer.B,
        layer.C,
        layer.D,
    ]
    if layer.dt_head is not None:
        params_to_check.append(layer.dt_head.weight)
    for param in params_to_check:
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_run_under_torch_compile() -> None:
    seed_all(9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, B, M = 10, 2, 4
    r = torch.full((L, B, M), 0.9, device=device)
    th = torch.zeros(L, B, M, device=device)
    dt = torch.full((L, B, M), 0.5, device=device)
    A, G = make_stable_AG_from_rt(r, th, dt)
    u = torch.randn(L, B, M, device=device, dtype=torch.float32)
    u = torch.complex(u, torch.zeros_like(u))

    fn = torch.compile(run_sdlinoss, fullgraph=False)
    y = fn("imex1", A, G, dt, u)
    assert y.shape == (L, B, M)
    assert torch.isfinite(y.real).all()


def test_equivalence_constant_params_vs_time_varying_broadcast() -> None:
    seed_all(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, B, M = 7, 3, 5
    r = torch.full((M,), 0.95, device=device)
    th = torch.linspace(-1.0, 1.0, M, device=device) * math.pi * 0.5
    dt = torch.full((M,), 0.6, device=device)
    A, G = make_stable_AG_from_rt(r, th, dt)

    u = torch.randn(L, B, M, device=device, dtype=torch.float32)
    u = torch.complex(u, torch.zeros_like(u))
    y_const = run_sdlinoss("imex1", A, G, dt, u)
    y_brdc = run_sdlinoss(
        "imex1",
        A.view(1, 1, -1).expand(L, B, M),
        G.view(1, 1, -1).expand(L, B, M),
        dt.view(1, 1, -1).expand(L, B, M),
        u,
    )
    assert torch.allclose(y_const, y_brdc, atol=1e-6, rtol=1e-5)

