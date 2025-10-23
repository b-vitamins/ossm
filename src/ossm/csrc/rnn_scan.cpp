#include <ATen/TensorUtils.h>
#include <torch/extension.h>

namespace ossm {
namespace {

at::Tensor linear_rnn_scan_cpu(const at::Tensor& weight_hh,
                               const at::Tensor& weight_xh,
                               const at::Tensor& bias,
                               const at::Tensor& inputs,
                               const at::Tensor& initial_state) {
  TORCH_CHECK(weight_hh.device().is_cpu(), "weight_hh must be on CPU");
  TORCH_CHECK(weight_xh.device().is_cpu(), "weight_xh must be on CPU");
  TORCH_CHECK(bias.device().is_cpu(), "bias must be on CPU");
  TORCH_CHECK(inputs.device().is_cpu(), "inputs must be on CPU");
  TORCH_CHECK(initial_state.device().is_cpu(), "initial_state must be on CPU");

  TORCH_CHECK(weight_hh.dim() == 2, "weight_hh must be 2-D");
  TORCH_CHECK(weight_xh.dim() == 2, "weight_xh must be 2-D");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1-D");
  TORCH_CHECK(inputs.dim() == 3, "inputs must be 3-D");
  TORCH_CHECK(initial_state.dim() == 2, "initial_state must be 2-D");

  const auto hidden_size = weight_hh.size(0);
  TORCH_CHECK(weight_hh.size(1) == hidden_size,
              "weight_hh must be square with size hidden_size x hidden_size");
  TORCH_CHECK(weight_xh.size(0) == hidden_size,
              "weight_xh rows must equal hidden_size");
  const auto input_size = weight_xh.size(1);

  TORCH_CHECK(bias.size(0) == hidden_size, "bias must match hidden_size");

  TORCH_CHECK(inputs.size(2) == input_size,
              "inputs last dimension must equal input_size");
  TORCH_CHECK(initial_state.size(1) == hidden_size,
              "initial_state second dimension must equal hidden_size");
  TORCH_CHECK(initial_state.size(0) == inputs.size(1),
              "initial_state batch dimension must match inputs");

  const auto length = inputs.size(0);
  const auto batch = inputs.size(1);

  auto weight_hh_contig = weight_hh.contiguous();
  auto weight_hh_t = weight_hh_contig.transpose(0, 1).contiguous();
  auto weight_xh_t = weight_xh.contiguous().transpose(0, 1).contiguous();
  auto bias_contig = bias.contiguous();
  auto inputs_contig = inputs.contiguous();
  auto inputs_flat = inputs_contig.reshape({length * batch, input_size});
  auto input_proj_flat = at::matmul(inputs_flat, weight_xh_t);
  auto input_proj = input_proj_flat.reshape({length, batch, hidden_size});
  input_proj.add_(bias_contig.view({1, 1, hidden_size}));

  auto state = initial_state.contiguous();
  auto result = at::empty({length, batch, hidden_size}, inputs.options());

  for (int64_t t = 0; t < length; ++t) {
    auto current = input_proj.select(0, t);
    current.addmm_(state, weight_hh_t);
    result.select(0, t).copy_(current);
    state = current;
  }

  return result;
}

#ifdef WITH_CUDA
}  // end anonymous namespace

// Declare CUDA implementation with external linkage in the ossm namespace
// to match the definition in rnn_scan_cuda.cu and avoid undefined symbols.
at::Tensor linear_rnn_scan_cuda(const at::Tensor& weight_hh,
                                const at::Tensor& weight_xh,
                                const at::Tensor& bias,
                                const at::Tensor& inputs,
                                const at::Tensor& initial_state);

namespace {  // reopen anonymous namespace
#endif

}  // namespace

at::Tensor linear_rnn_scan(const at::Tensor& weight_hh,
                           const at::Tensor& weight_xh,
                           const at::Tensor& bias,
                           const at::Tensor& inputs,
                           const at::Tensor& initial_state) {
  if (inputs.size(0) == 0) {
    return at::empty({0, inputs.size(1), weight_hh.size(0)}, inputs.options());
  }

  if (inputs.is_cuda()) {
#ifdef WITH_CUDA
    return linear_rnn_scan_cuda(weight_hh, weight_xh, bias, inputs, initial_state);
#else
    TORCH_CHECK(false, "linear_rnn_scan CUDA extension was not built");
#endif
  }

  return linear_rnn_scan_cpu(weight_hh, weight_xh, bias, inputs, initial_state);
}

}  // namespace ossm
