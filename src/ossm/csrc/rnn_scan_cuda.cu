#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

namespace ossm {
namespace {

at::Tensor linear_rnn_scan_cuda_impl(const at::Tensor& weight_hh,
                                     const at::Tensor& weight_xh,
                                     const at::Tensor& bias,
                                     const at::Tensor& inputs,
                                     const at::Tensor& initial_state) {
  TORCH_CHECK(weight_hh.is_cuda(), "weight_hh must be CUDA");
  TORCH_CHECK(weight_xh.is_cuda(), "weight_xh must be CUDA");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
  TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(initial_state.is_cuda(), "initial_state must be CUDA");

  const auto length = inputs.size(0);
  if (length == 0) {
    return at::empty({0, inputs.size(1), weight_hh.size(0)}, inputs.options());
  }

  const auto batch = inputs.size(1);
  const auto hidden_size = weight_hh.size(0);

  auto weight_hh_t = weight_hh.contiguous().transpose(0, 1);
  auto weight_xh_t = weight_xh.contiguous().transpose(0, 1);
  auto bias_row = bias.contiguous().view({1, bias.size(0)});
  auto inputs_flat = inputs.contiguous().reshape({length * batch, weight_xh_t.size(0)});
  auto projected = at::addmm(bias_row, inputs_flat, weight_xh_t).reshape({length, batch, hidden_size});

  auto outputs = at::empty({length, batch, hidden_size}, inputs.options());
  auto state = initial_state.contiguous();
  auto next = at::empty_like(state);

  for (int64_t t = 0; t < length; ++t) {
    auto proj_t = projected.select(0, t);
    at::addmm_out(next, proj_t, state, weight_hh_t, /*beta=*/1.0, /*alpha=*/1.0);
    outputs.select(0, t).copy_(next);
    state.copy_(next);
  }

  return outputs;
}

}  // namespace

at::Tensor linear_rnn_scan_cuda(const at::Tensor& weight_hh,
                                const at::Tensor& weight_xh,
                                const at::Tensor& bias,
                                const at::Tensor& inputs,
                                const at::Tensor& initial_state) {
  at::cuda::CUDAGuard device_guard(inputs.device());
  return linear_rnn_scan_cuda_impl(weight_hh, weight_xh, bias, inputs, initial_state);
}

}  // namespace ossm
