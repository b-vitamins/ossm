import pytest
import torch

from ossm.models.linoss import LinOSSBackbone, LinOSSLayer


def _sequential_scan(a_seq: torch.Tensor, b_seq: torch.Tensor) -> torch.Tensor:
    """Reference scan using a naïve sequential loop."""

    length, batch, ssm_size, _ = b_seq.shape
    state = b_seq.new_zeros((batch, ssm_size, 2))
    outputs = []
    for idx in range(length):
        a_t = a_seq[idx]
        state = torch.einsum("spq,bsq->bsp", a_t, state) + b_seq[idx]
        outputs.append(state)
    return torch.stack(outputs)


def _reference_linoss(layer: LinOSSLayer, inputs: torch.Tensor) -> torch.Tensor:
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
    a_diag = torch.relu(layer.A_diag).to(device=device, dtype=dtype)
    step = torch.sigmoid(layer.steps).to(device=device, dtype=dtype)
    b_complex = torch.view_as_complex(layer.B.contiguous()).to(device=device, dtype=complex_dtype)
    c_complex = torch.view_as_complex(layer.C.contiguous()).to(device=device, dtype=complex_dtype)
    d_vec = layer.D.to(device=device, dtype=dtype)

    inputs_complex = inputs.to(dtype=dtype).to(dtype=complex_dtype)
    bu = torch.einsum("blh,ph->blp", inputs_complex, b_complex)

    if layer.discretization == "IM":
        schur = 1.0 / (1.0 + (step**2) * a_diag)
        m11 = 1.0 - (step**2) * a_diag * schur
        m12 = -step * a_diag * schur
        m21 = step * schur
        m22 = schur
        f1 = (m11 * step) * bu
        f2 = (m21 * step) * bu
    else:
        m11 = torch.ones_like(a_diag)
        m12 = -step * a_diag
        m21 = step
        m22 = 1.0 - (step**2) * a_diag
        f1 = bu * step
        f2 = bu * (step**2)

    if length == 0:
        projected = inputs.new_zeros(batch, 0, hidden_dim)
        return projected

    a_matrix = torch.stack(
        (
            torch.stack((m11, m12), dim=-1),
            torch.stack((m21, m22), dim=-1),
        ),
        dim=-2,
    ).to(device=device, dtype=complex_dtype)
    a_elems = a_matrix.unsqueeze(0).expand(length, -1, -1, -1).contiguous()
    b_elems = torch.stack((f1, f2), dim=-1).permute(1, 0, 2, 3).contiguous()

    prefix_b = _sequential_scan(a_elems, b_elems)
    states = prefix_b.permute(1, 0, 2, 3).contiguous()[..., 1]
    projected = torch.einsum("blp,hp->blh", states, c_complex).real
    du = inputs * d_vec
    return projected + du


@pytest.mark.parametrize("discretization", ["IM", "IMEX"])
def test_linoss_layer_matches_reference(discretization: str) -> None:
    torch.manual_seed(0)
    layer = LinOSSLayer(ssm_size=8, hidden_dim=6, discretization=discretization)
    inputs = torch.randn(4, 17, 6)

    outputs = layer(inputs)
    reference = _reference_linoss(layer, inputs)

    torch.testing.assert_close(outputs, reference, rtol=1e-5, atol=1e-6)


def test_linoss_layer_zero_length() -> None:
    torch.manual_seed(1)
    layer = LinOSSLayer(4, 3, "IM")
    inputs = torch.randn(2, 0, 3)

    outputs = layer(inputs)
    assert outputs.shape == (2, 0, 3)
    reference = _reference_linoss(layer, inputs)
    torch.testing.assert_close(outputs, reference)


def test_linoss_backbone_shapes() -> None:
    torch.manual_seed(2)
    backbone = LinOSSBackbone(
        num_blocks=2,
        input_dim=5,
        ssm_size=4,
        hidden_dim=6,
        discretization="IM",
    )
    inputs = torch.randn(3, 11, 5)

    output = backbone(inputs)
    assert output.features.shape == (3, 11, 6)
    assert output.pooled.shape == (3, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("discretization", ["IM", "IMEX"])
def test_linoss_layer_cuda_matches_cpu(discretization: str) -> None:
    torch.manual_seed(3)
    layer_cpu = LinOSSLayer(ssm_size=6, hidden_dim=5, discretization=discretization)
    inputs_cpu = torch.randn(4, 19, 5)

    reference = layer_cpu(inputs_cpu)

    layer_cuda = LinOSSLayer(ssm_size=6, hidden_dim=5, discretization=discretization).cuda()
    layer_cuda.load_state_dict(layer_cpu.state_dict())
    inputs_cuda = inputs_cpu.cuda()

    outputs_cuda = layer_cuda(inputs_cuda).cpu()
    torch.testing.assert_close(outputs_cuda, reference, rtol=1e-5, atol=1e-6)


def test_linoss_gradients_flow_through_extension() -> None:
    torch.manual_seed(5)
    layer = LinOSSLayer(ssm_size=4, hidden_dim=3, discretization="IM")
    inputs = torch.randn(2, 7, 3, requires_grad=True)

    output = layer(inputs).sum()
    output.backward()

    assert inputs.grad is not None and inputs.grad.abs().sum() > 0
    assert layer.A_diag.grad is not None and layer.A_diag.grad.abs().sum() > 0
    assert layer.B.grad is not None and layer.B.grad.abs().sum() > 0
