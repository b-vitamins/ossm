import pytest
import torch

from ossm.models.s5 import S5Backbone, S5Layer


def _reference_scan(lambda_bar: torch.Tensor, bu: torch.Tensor) -> torch.Tensor:
    length, batch, state = bu.shape
    outputs = bu.new_zeros(length, batch, state)
    state_vec = bu.new_zeros(batch, state)
    for idx in range(length):
        state_vec = lambda_bar * state_vec + bu[idx]
        outputs[idx] = state_vec
    return outputs


def _reference_s5(layer: S5Layer, inputs: torch.Tensor) -> torch.Tensor:
    if inputs.dim() != 3:
        raise ValueError("inputs must be (batch, length, hidden_dim)")
    batch, length, hidden = inputs.shape
    if hidden != layer.hidden_dim:
        raise ValueError("hidden dimension mismatch")
    if length == 0:
        return inputs.new_zeros(batch, 0, hidden)

    device = inputs.device
    real_dtype = inputs.dtype
    complex_dtype = (
        torch.complex64 if real_dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.complex128
    )

    lambda_real = layer.lambda_real.detach().to(device=device, dtype=real_dtype)
    lambda_imag = layer.lambda_imag.detach().to(device=device, dtype=real_dtype)
    lambda_complex = torch.complex(lambda_real, lambda_imag).to(dtype=complex_dtype)
    if layer.clip_eigs:
        lambda_complex = torch.complex(torch.clamp(lambda_real, max=-1e-4), lambda_imag).to(dtype=complex_dtype)

    step = layer.step_rescale * torch.exp(layer.log_step.detach().squeeze(-1)).to(device=device, dtype=real_dtype)
    step_complex = step.to(dtype=complex_dtype)

    b_complex = torch.view_as_complex(layer.B.detach().to(device=device, dtype=real_dtype)).to(dtype=complex_dtype)
    c_complex = torch.view_as_complex(layer.C.detach().to(device=device, dtype=real_dtype)).to(dtype=complex_dtype)

    if layer.discretization == "zoh":
        lambda_bar = torch.exp(lambda_complex * step_complex)
        denom = torch.where(lambda_complex == 0, torch.ones_like(lambda_complex), lambda_complex)
        b_bar = ((lambda_bar - 1) / denom).unsqueeze(-1) * b_complex
    else:
        identity = torch.ones_like(lambda_complex)
        bl = 1.0 / (identity - 0.5 * step_complex * lambda_complex)
        lambda_bar = bl * (identity + 0.5 * step_complex * lambda_complex)
        b_bar = (bl * step_complex).unsqueeze(-1) * b_complex

    inputs_complex = inputs.to(dtype=real_dtype).to(dtype=complex_dtype)
    bu = torch.einsum("blh,ph->blp", inputs_complex, b_bar)
    bu_seq = bu.permute(1, 0, 2).contiguous()
    states_seq = _reference_scan(lambda_bar, bu_seq)
    states = states_seq.permute(1, 0, 2)

    projected = torch.einsum("blp,hp->blh", states, c_complex)
    if layer.conj_sym:
        projected = 2.0 * projected.real
    else:
        projected = projected.real

    du = inputs * layer.D.detach().to(device=device, dtype=real_dtype)
    return projected + du


@pytest.mark.parametrize("conj_sym", [False, True])
@pytest.mark.parametrize("discretization", ["zoh", "bilinear"])
def test_s5_layer_matches_reference(conj_sym: bool, discretization: str) -> None:
    torch.manual_seed(0)
    layer = S5Layer(
        ssm_size=8,
        hidden_dim=6,
        blocks=2,
        conj_sym=conj_sym,
        discretization=discretization,
    )
    inputs = torch.randn(3, 11, 6)

    outputs = layer(inputs)
    reference = _reference_s5(layer, inputs)

    torch.testing.assert_close(outputs, reference, rtol=1e-5, atol=1e-6)


def test_s5_layer_zero_length() -> None:
    torch.manual_seed(1)
    layer = S5Layer(ssm_size=4, hidden_dim=3)
    inputs = torch.randn(2, 0, 3)

    outputs = layer(inputs)
    assert outputs.shape == (2, 0, 3)
    reference = _reference_s5(layer, inputs)
    torch.testing.assert_close(outputs, reference)


def test_s5_backbone_shapes() -> None:
    torch.manual_seed(2)
    backbone = S5Backbone(
        num_blocks=2,
        input_dim=5,
        ssm_size=8,
        ssm_blocks=2,
        hidden_dim=6,
    )
    inputs = torch.randn(3, 13, 5)

    output = backbone(inputs)
    assert output.features.shape == (3, 13, 6)
    assert output.pooled.shape == (3, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_s5_layer_cuda_matches_cpu() -> None:
    torch.manual_seed(3)
    layer_cpu = S5Layer(ssm_size=6, hidden_dim=5, blocks=1, conj_sym=False)
    inputs_cpu = torch.randn(4, 19, 5)

    reference = layer_cpu(inputs_cpu)

    layer_cuda = S5Layer(ssm_size=6, hidden_dim=5, blocks=1, conj_sym=False).cuda()
    layer_cuda.load_state_dict(layer_cpu.state_dict())
    inputs_cuda = inputs_cpu.cuda()

    outputs_cuda = layer_cuda(inputs_cuda).cpu()
    torch.testing.assert_close(outputs_cuda, reference, rtol=1e-5, atol=1e-6)
