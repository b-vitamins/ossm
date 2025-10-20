from __future__ import annotations

from typing import Callable

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


REFERENCE_COMPLEX_LAMBDA = torch.tensor(
    [
        -0.0788191556930542 + 0.08510945737361908j,
        -0.26137107610702515 - 0.17000117897987366j,
        -0.8463533520698547 + 0.14797578752040863j,
    ],
    dtype=torch.complex64,
)

REFERENCE_COMPLEX_B = torch.tensor(
    [
        [
            [
                0.1698904186487198 - 0.3900344669818878j,
                -0.40283501148223877 + 0.7124848961830139j,
                -0.0544615313410759 - 0.7215825915336609j,
            ],
            [
                -0.11947012692689896 + 0.6489575505256653j,
                -0.22312743961811066 - 0.258501797914505j,
                -0.46703726053237915 - 0.07807270437479019j,
            ],
        ],
        [
            [
                0.3510580062866211 - 1.111786961555481j,
                0.683469295501709 - 0.8118603825569153j,
                1.3406513929367065 - 0.12377260625362396j,
            ],
            [
                0.967964231967926 - 1.133683204650879j,
                -0.9368863105773926 + 0.12616795301437378j,
                -1.508791208267212 + 0.7441293001174927j,
            ],
        ],
        [
            [
                0.7060761451721191 + 0.1564469188451767j,
                1.2953444719314575 - 0.23888809978961945j,
                0.6226388216018677 + 1.0989559888839722j,
            ],
            [
                0.4430844485759735 - 0.12408819049596786j,
                -0.5797638893127441 - 0.29771891236305237j,
                -0.6802353262901306 + 0.9068470001220703j,
            ],
        ],
        [
            [
                0.620017409324646 + 1.1470290422439575j,
                -1.045062780380249 + 0.8012275695800781j,
                -0.5301902890205383 - 0.7979822158813477j,
            ],
            [
                0.2923958897590637 + 0.2044735550880432j,
                1.5891032218933105 - 0.5682564377784729j,
                -0.1985855996608734 + 0.5442432761192322j,
            ],
        ],
        [
            [
                0.9270536303520203 - 0.15551508963108063j,
                0.15482282638549805 + 0.14461345970630646j,
                0.3638976514339447 + 0.7026970982551575j,
            ],
            [
                -0.1829521507024765 - 0.765479326248169j,
                -0.2791719436645508 + 0.34593626856803894j,
                -0.15331128239631653 - 1.2354317903518677j,
            ],
        ],
        [
            [
                1.2182331085205078 + 0.5471634864807129j,
                0.28611648082733154 - 1.1639872789382935j,
                -0.42078810930252075 - 0.5029067397117615j,
            ],
            [
                0.44050267338752747 - 0.9707740545272827j,
                -0.09055503457784653 - 0.9078081250190735j,
                -0.20514586567878723 + 0.9027916193008423j,
            ],
        ],
    ],
    dtype=torch.complex64,
)

REFERENCE_COMPLEX_STATES = torch.tensor(
    [
        [
            [
                0.1698904186487198 - 0.3900344669818878j,
                -0.40283501148223877 + 0.7124848961830139j,
                -0.0544615313410759 - 0.7215825915336609j,
            ],
            [
                -0.11947012692689896 + 0.6489575505256653j,
                -0.22312743961811066 - 0.258501797914505j,
                -0.46703726053237915 - 0.07807270437479019j,
            ],
        ],
        [
            [
                0.37086302042007446 - 1.0665855407714844j,
                0.9098819494247437 - 0.9296008944511414j,
                1.493521809577942 + 0.4788822531700134j,
            ],
            [
                0.9221483469009399 - 1.195001482963562j,
                -0.9225128889083862 + 0.231664776802063j,
                -1.1019598245620728 + 0.7410961985588074j,
            ],
        ],
        [
            [
                0.7676215767860413 + 0.2720782458782196j,
                0.8994944095611572 - 0.15059831738471985j,
                -0.7122713327407837 + 0.914657473564148j,
            ],
            [
                0.4721074104309082 + 0.04858436435461044j,
                -0.2992624044418335 - 0.20144110918045044j,
                0.14274781942367554 + 0.1165543794631958j,
            ],
        ],
        [
            [
                0.5363577008247375 + 1.1909159421920776j,
                -1.3057664632797241 + 0.6876745223999023j,
                -0.06270423531532288 - 1.677504539489746j,
            ],
            [
                0.25104978680610657 + 0.24082498252391815j,
                1.633076548576355 - 0.4647305905818939j,
                -0.3366479277610779 + 0.466720312833786j,
            ],
        ],
        [
            [
                0.7834201455116272 - 0.20373296737670898j,
                0.6130179166793823 + 0.18685707449913025j,
                0.6651976108551025 + 2.113179922103882j,
            ],
            [
                -0.22323617339134216 - 0.7630942463874817j,
                -0.7850156426429749 + 0.18977846205234528j,
                0.06254851818084717 - 1.680257797241211j,
            ],
        ],
        [
            [
                1.1738241910934448 + 0.6298980116844177j,
                0.15765725076198578 - 1.3170400857925415j,
                -1.2964797019958496 - 2.1929705142974854j,
            ],
            [
                0.5230444669723511 - 0.9296271204948425j,
                0.1468878984451294 - 0.8239571452140808j,
                -0.00944654643535614 + 2.334139108657837j,
            ],
        ],
    ],
    dtype=torch.complex64,
)


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


def test_complex_scans_match_reference() -> None:
    lambda_bar = REFERENCE_COMPLEX_LAMBDA.clone()
    b_seq = REFERENCE_COMPLEX_B.clone()

    lru_out = _LRUScanFn.apply(lambda_bar, b_seq)
    s5_out = _S5ScanFn.apply(lambda_bar, b_seq)
    expected = REFERENCE_COMPLEX_STATES

    torch.testing.assert_close(lru_out, expected, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(s5_out, expected, atol=5e-6, rtol=5e-6)


LINEAR_RNN_REFERENCE = {
    "weight_hh": torch.tensor(
        [
            [-0.8201345205307007, 0.39563116431236267, 0.8989084959030151, -1.3884038925170898],
            [-0.16699601709842682, 0.2851499617099762, -0.641091525554657, -0.8936554193496704],
            [0.9265429973602295, -0.5355122089385986, -1.1597203016281128, -0.4601571559906006],
            [0.7085390686988831, 1.0127553939819336, 0.23039686679840088, 1.090165138244629],
        ]
    ),
    "weight_xh": torch.tensor(
        [
            [0.13817265629768372, -1.6821985244750977, 0.317678302526474],
            [0.13280697166919708, 0.13732409477233887, 0.2405461221933365],
            [1.3954508304595947, 1.3470226526260376, 2.4382081031799316],
            [0.2027582824230194, 2.4505412578582764, 2.025601863861084],
        ]
    ),
    "bias": torch.tensor(
        [1.7791550159454346, -0.9179307222366333, -0.4578188955783844, -0.7244732975959778]
    ),
    "inputs": torch.tensor(
        [
            [
                [-0.1781926155090332, -0.25949886441230774, -0.014487940818071365],
                [-0.3838909864425659, -2.966169834136963, -1.0605549812316895],
            ],
            [
                [-0.3089962899684906, 0.9342882037162781, 1.6243164539337158],
                [0.0015672707231715322, -0.4375407099723816, -2.1085333824157715],
            ],
            [
                [1.1450074911117554, -0.38218432664871216, 0.5460120439529419],
                [0.1485121250152588, -2.257782220840454, 0.40429842472076416],
            ],
            [
                [0.5721989870071411, 0.30781328678131104, -0.12589670717716217],
                [-0.9577822685241699, -1.382198452949524, -0.9632285833358765],
            ],
            [
                [-0.3984588384628296, -0.1731676161289215, 0.7568809390068054],
                [0.9862262606620789, -0.825334370136261, 0.16328637301921844],
            ],
        ]
    ),
    "initial": torch.tensor(
        [
            [-0.7897513508796692, 1.0201793909072876, 0.07951028645038605, -0.19555222988128662],
            [-0.4912847876548767, 0.47959479689598083, 1.2956377267837524, -0.800732433795929],
        ]
    ),
    "states": torch.tensor(
        [
            [
                [3.5807547569274902, -0.4341440796852112, -2.3716354370117188, -1.1471056938171387],
                [9.247945785522461, -1.5275976657867432, -9.421011924743652, -10.656095504760742],
            ],
            [
                [-2.966886520385742, 1.3838403224945068, 11.158416748046875, 5.093075752258301],
                [-0.017022371292114258, 12.097636222839355, 19.029830932617188, -14.849445343017578],
            ],
            [
                [8.693615913391113, -11.501968383789062, -16.817806243896484, 7.099626064300537],
                [48.24941635131836, 3.411905527114868, -24.036502838134766, -4.972250461578369],
            ],
            [
                [-35.3548698425293, -1.1244041919708252, 30.8997859954834, -1.7330818176269531],
                [-49.25832748413086, 11.3018798828125, 67.03691864013672, 20.42658805847168],
            ],
            [
                [60.98904037475586, -13.48987865447998, -66.594970703125, -20.655662536621094],
                [80.12520599365234, -50.643470764160156, -138.6309814453125, 12.041767120361328],
            ],
        ]
    ),
}


def test_linear_rnn_matches_reference_snapshot() -> None:
    ref = LINEAR_RNN_REFERENCE
    torch_out = _LinearRNNScanFn.apply(
        ref["weight_hh"],
        ref["weight_xh"],
        ref["bias"],
        ref["inputs"],
        ref["initial"],
    )

    torch.testing.assert_close(torch_out, ref["states"], atol=5e-6, rtol=5e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kernel_name,fn", [("lru_scan", _LRUScanFn), ("s5_scan", _S5ScanFn)])
def test_complex_scan_cuda_matches_cpu(kernel_name: str, fn) -> None:
    if not _has_kernel(kernel_name):
        pytest.skip("compiled CUDA extension not built")

    device = torch.device("cuda")
    length, batch, state = 5, 4, 6
    torch.manual_seed(0)
    lambda_bar = torch.randn(state, dtype=torch.complex64, device=device)
    b_seq = torch.randn(length, batch, state, dtype=torch.complex64, device=device)

    out_cuda = fn.apply(lambda_bar, b_seq).cpu()
    expected = _complex_scan_naive(lambda_bar.cpu(), b_seq.cpu())
    torch.testing.assert_close(out_cuda, expected, atol=2e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_linear_rnn_cuda_matches_cpu() -> None:
    if not _has_kernel("linear_rnn_scan"):
        pytest.skip("compiled CUDA extension not built")

    torch.manual_seed(0)
    length, batch, input_size, hidden_size = 6, 3, 4, 5
    weight_hh = torch.randn(hidden_size, hidden_size, device="cuda")
    weight_xh = torch.randn(hidden_size, input_size, device="cuda")
    bias = torch.randn(hidden_size, device="cuda")
    inputs = torch.randn(length, batch, input_size, device="cuda")
    initial_state = torch.randn(batch, hidden_size, device="cuda")

    out_cuda = _LinearRNNScanFn.apply(weight_hh, weight_xh, bias, inputs, initial_state).cpu()
    expected = _linear_rnn_naive(
        weight_hh.cpu(), weight_xh.cpu(), bias.cpu(), inputs.cpu(), initial_state.cpu()
    )
    torch.testing.assert_close(out_cuda, expected, atol=2e-4, rtol=1e-4)
