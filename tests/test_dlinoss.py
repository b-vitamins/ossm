import torch
import pytest

from ossm.models.dlinoss import DampedLinOSSBackbone, DampedLinOSSLayer


def _reference_dlinoss(layer: DampedLinOSSLayer, inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() != 3:
        raise ValueError("inputs must be (batch, length, hidden_dim)")
    batch, length, hidden_dim = inputs.shape
    if hidden_dim != layer.hidden_dim:
        raise ValueError("hidden dim mismatch")

    device = inputs.device
    dtype = inputs.dtype
    complex_dtype = (
        torch.complex64 if dtype in {torch.float32, torch.bfloat16} else torch.complex128
    )

    a_raw = layer.A_diag.to(device=device, dtype=dtype)
    g_raw = layer.G_diag.to(device=device, dtype=dtype)
    step_raw = layer.steps.to(device=device, dtype=dtype)

    step = torch.sigmoid(step_raw)
    g_diag = torch.relu(g_raw)
    denom = torch.clamp(step * step, min=1e-6)
    s = step * g_diag
    base = torch.sqrt(torch.clamp(1.0 + s, min=1e-6))
    a_low = (2.0 + s - 2.0 * base) / denom
    a_high = (2.0 + s + 2.0 * base) / denom
    a_diag = a_low + torch.relu(a_raw - a_low) - torch.relu(a_raw - a_high)

    S = 1.0 + step * g_diag
    m11 = 1.0 / S
    m12 = -(step * a_diag) / S
    m21 = step / S
    m22 = 1.0 - (step * step * a_diag) / S
    f1_scale = step / S
    f2_scale = (step * step) / S

    b_complex = torch.view_as_complex(layer.B.contiguous()).to(device=device, dtype=complex_dtype)
    c_complex = torch.view_as_complex(layer.C.contiguous()).to(device=device, dtype=complex_dtype)
    d_vec = layer.D.to(device=device, dtype=dtype)

    inputs_complex = inputs.to(dtype=dtype).to(dtype=complex_dtype)
    bu = torch.einsum("blh,ph->blp", inputs_complex, b_complex)

    state = torch.zeros(batch, layer.ssm_size, 2, dtype=complex_dtype, device=device)
    outputs = []
    for t in range(length):
        f1 = f1_scale * bu[:, t]
        f2 = f2_scale * bu[:, t]
        new0 = m11 * state[..., 0] + m12 * state[..., 1] + f1
        new1 = m21 * state[..., 0] + m22 * state[..., 1] + f2
        state = torch.stack((new0, new1), dim=-1)
        outputs.append(state)

    if not outputs:
        projected = inputs.new_zeros(batch, 0, hidden_dim)
        return projected

    states = torch.stack(outputs, dim=1)[..., 1]
    projected = torch.einsum("blp,hp->blh", states, c_complex).real
    du = inputs * d_vec
    return projected + du


def test_dlinoss_layer_matches_reference() -> None:
    torch.manual_seed(0)
    layer = DampedLinOSSLayer(ssm_size=6, hidden_dim=5)
    inputs = torch.randn(3, 11, 5)

    outputs = layer(inputs)
    reference = _reference_dlinoss(layer, inputs)

    torch.testing.assert_close(outputs, reference, rtol=1e-5, atol=1e-6)


def test_dlinoss_layer_zero_length() -> None:
    torch.manual_seed(1)
    layer = DampedLinOSSLayer(ssm_size=4, hidden_dim=3)
    inputs = torch.randn(2, 0, 3)

    outputs = layer(inputs)
    assert outputs.shape == (2, 0, 3)
    reference = _reference_dlinoss(layer, inputs)
    torch.testing.assert_close(outputs, reference)


def test_dlinoss_backbone_shapes() -> None:
    torch.manual_seed(2)
    backbone = DampedLinOSSBackbone(
        num_blocks=2,
        input_dim=5,
        ssm_size=4,
        hidden_dim=6,
    )
    inputs = torch.randn(3, 11, 5)

    output = backbone(inputs)
    assert output.features.shape == (3, 11, 6)
    assert output.pooled.shape == (3, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dlinoss_layer_cuda_matches_cpu() -> None:
    torch.manual_seed(3)
    layer_cpu = DampedLinOSSLayer(ssm_size=5, hidden_dim=4)
    inputs_cpu = torch.randn(2, 9, 4)

    reference = layer_cpu(inputs_cpu)

    layer_cuda = DampedLinOSSLayer(ssm_size=5, hidden_dim=4).cuda()
    layer_cuda.load_state_dict(layer_cpu.state_dict())
    inputs_cuda = inputs_cpu.cuda()

    outputs_cuda = layer_cuda(inputs_cuda).cpu()
    torch.testing.assert_close(outputs_cuda, reference, rtol=1e-5, atol=1e-6)


def test_dlinoss_gradients_flow() -> None:
    torch.manual_seed(5)
    layer = DampedLinOSSLayer(ssm_size=4, hidden_dim=3)
    inputs = torch.randn(2, 7, 3, requires_grad=True)

    output = layer(inputs).sum()
    output.backward()

    assert inputs.grad is not None and inputs.grad.abs().sum() > 0
    assert layer.A_diag.grad is not None and layer.A_diag.grad.abs().sum() > 0
    assert layer.G_diag.grad is not None and layer.G_diag.grad.abs().sum() > 0
    assert layer.B.grad is not None and layer.B.grad.abs().sum() > 0
    assert layer.C.grad is not None and layer.C.grad.abs().sum() > 0
