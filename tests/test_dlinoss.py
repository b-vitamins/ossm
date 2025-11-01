import pytest
import torch

from ossm.models._dlinoss_scan import _reference_dlinoss_states, run_dlinoss
from ossm.models.dlinoss import DampedLinOSSBackbone, DampedLinOSSLayer


VARIANTS = ("imex1", "imex2", "im", "ex")


def _official_imex1_states(
    a_diag: torch.Tensor, g_diag: torch.Tensor, step: torch.Tensor, bu: torch.Tensor
) -> torch.Tensor:
    """Sequential reference mirroring the official Damped-LinOSS recurrence."""

    length, batch, ssm = bu.shape
    if a_diag.shape != (ssm,) or g_diag.shape != (ssm,) or step.shape != (ssm,):
        raise ValueError("coefficient diagonals must match the state size")

    a_diag_c = a_diag.view(1, -1).to(dtype=bu.dtype, device=bu.device)
    step_c = step.view(1, -1).to(dtype=bu.dtype, device=bu.device)
    denom = (1.0 + step * g_diag).view(1, -1).to(dtype=bu.dtype, device=bu.device)

    z = torch.zeros(batch, ssm, dtype=bu.dtype, device=bu.device)
    x = torch.zeros_like(z)
    traj: list[torch.Tensor] = []

    for t in range(length):
        bu_t = bu[t]
        comb = -a_diag_c * x + bu_t
        z = (z + step_c * comb) / denom
        x = x + step_c * z
        traj.append(torch.stack((z, x), dim=-1))

    return torch.stack(traj, dim=0)


def _reference_dlinoss(layer: DampedLinOSSLayer, inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() != 3:
        raise ValueError("inputs must be (batch, length, hidden_dim)")
    batch, length, hidden_dim = inputs.shape
    if hidden_dim != layer.hidden_dim:
        raise ValueError("hidden dim mismatch")

    device = inputs.device
    dtype = inputs.dtype
    compute_dtype = dtype
    if dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    scan_real_dtype = torch.float32
    scan_complex_dtype = torch.complex64
    if compute_dtype == torch.float64:
        scan_real_dtype = torch.float64
        scan_complex_dtype = torch.complex128

    a_diag, g_diag, step = layer._project_parameters(device=device, dtype=compute_dtype)

    b_real = layer.B[..., 0].to(device=device, dtype=compute_dtype)
    b_imag = layer.B[..., 1].to(device=device, dtype=compute_dtype)
    c_real = layer.C[..., 0].to(device=device, dtype=compute_dtype)
    c_imag = layer.C[..., 1].to(device=device, dtype=compute_dtype)
    d_vec = layer.D.to(device=device, dtype=compute_dtype)

    layer_inputs = inputs if compute_dtype == dtype else inputs.to(dtype=compute_dtype)
    flat_inputs = layer_inputs.reshape(batch * length, hidden_dim)

    bu_real = flat_inputs @ b_real.transpose(0, 1)
    bu_imag = flat_inputs @ b_imag.transpose(0, 1)
    bu = torch.complex(
        bu_real.to(dtype=scan_real_dtype),
        bu_imag.to(dtype=scan_real_dtype),
    ).to(dtype=scan_complex_dtype).reshape(batch, length, layer.ssm_size)

    bu_seq = bu.to(dtype=scan_complex_dtype)
    states = run_dlinoss(
        layer.variant,
        a_diag.to(dtype=scan_real_dtype),
        g_diag.to(dtype=scan_real_dtype),
        step.to(dtype=scan_real_dtype),
        bu_seq.permute(1, 0, 2).contiguous(),
    )

    states = states.permute(1, 0, 2).contiguous()
    states_real = states.real.to(dtype=compute_dtype)
    states_imag = states.imag.to(dtype=compute_dtype)

    c_real_t = c_real.transpose(0, 1).to(dtype=compute_dtype)
    c_imag_t = c_imag.transpose(0, 1).to(dtype=compute_dtype)
    projected_real = states_real.reshape(batch * length, layer.ssm_size) @ c_real_t
    projected_imag = states_imag.reshape(batch * length, layer.ssm_size) @ c_imag_t
    projected = (projected_real - projected_imag).reshape(batch, length, hidden_dim)
    du = layer_inputs * d_vec
    outputs = projected + du
    if compute_dtype == dtype:
        return outputs
    return outputs.to(dtype=dtype)


@pytest.mark.parametrize("variant", VARIANTS)
def test_dlinoss_layer_matches_reference(variant: str) -> None:
    torch.manual_seed(0)
    layer = DampedLinOSSLayer(ssm_size=6, hidden_dim=5, variant=variant)
    inputs = torch.randn(3, 11, 5)

    outputs = layer(inputs)
    reference = _reference_dlinoss(layer, inputs)

    torch.testing.assert_close(outputs, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_dlinoss_layer_zero_length(variant: str) -> None:
    torch.manual_seed(1)
    layer = DampedLinOSSLayer(ssm_size=4, hidden_dim=3, variant=variant)
    inputs = torch.randn(2, 0, 3)

    outputs = layer(inputs)
    assert outputs.shape == (2, 0, 3)
    reference = _reference_dlinoss(layer, inputs)
    torch.testing.assert_close(outputs, reference)


@pytest.mark.parametrize("variant", VARIANTS)
def test_dlinoss_backbone_shapes(variant: str) -> None:
    torch.manual_seed(2)
    backbone = DampedLinOSSBackbone(
        num_blocks=2,
        input_dim=5,
        ssm_size=4,
        hidden_dim=6,
        variant=variant,
    )
    inputs = torch.randn(3, 11, 5)

    output = backbone(inputs)
    assert output.features.shape == (3, 11, 6)
    assert output.pooled.shape == (3, 6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dlinoss_layer_cuda_matches_cpu(variant: str) -> None:
    torch.manual_seed(3)
    layer_cpu = DampedLinOSSLayer(ssm_size=5, hidden_dim=4, variant=variant)
    inputs_cpu = torch.randn(2, 9, 4)

    reference = layer_cpu(inputs_cpu)

    layer_cuda = DampedLinOSSLayer(ssm_size=5, hidden_dim=4, variant=variant).cuda()
    layer_cuda.load_state_dict(layer_cpu.state_dict())
    inputs_cuda = inputs_cpu.cuda()

    outputs_cuda = layer_cuda(inputs_cuda).cpu()
    torch.testing.assert_close(outputs_cuda, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_dlinoss_gradients_flow(variant: str) -> None:
    torch.manual_seed(5)
    layer = DampedLinOSSLayer(ssm_size=4, hidden_dim=3, variant=variant)
    inputs = torch.randn(2, 7, 3, requires_grad=True)

    output = layer(inputs).sum()
    output.backward()

    assert inputs.grad is not None and inputs.grad.abs().sum() > 0
    assert layer.A_diag.grad is not None and layer.A_diag.grad.abs().sum() > 0
    assert layer.G_diag.grad is not None and layer.G_diag.grad.abs().sum() > 0
    assert layer.B.grad is not None and layer.B.grad.abs().sum() > 0
    assert layer.C.grad is not None and layer.C.grad.abs().sum() > 0


def test_dlinoss_imex1_matches_official_recurrence() -> None:
    torch.manual_seed(7)
    length, batch, ssm = 9, 2, 4
    a_diag = torch.rand(ssm)
    g_diag = torch.rand(ssm)
    step = torch.rand(ssm) * 0.5 + 0.1
    bu = torch.randn(length, batch, ssm, dtype=torch.complex64)

    states = _reference_dlinoss_states("imex1", a_diag, g_diag, step, bu)
    official = _official_imex1_states(a_diag, g_diag, step, bu)

    step_scale = step.view(1, 1, -1).to(dtype=states.real.dtype)
    z_kernel = states[..., 0] / step_scale
    torch.testing.assert_close(z_kernel, official[..., 0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(states[..., 1], official[..., 1], rtol=1e-6, atol=1e-7)
