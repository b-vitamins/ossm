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

  auto weight_hh_contig = weight_hh.contiguous();
  auto weight_xh_t = weight_xh.contiguous().transpose(0, 1).contiguous();
  auto bias_contig = bias.contiguous();
  auto inputs_contig = inputs.contiguous();
  auto inputs_flat = inputs_contig.reshape({length * batch, weight_xh_t.size(0)});
  auto input_proj_flat = at::matmul(inputs_flat, weight_xh_t);
  auto input_proj = input_proj_flat.reshape({length, batch, hidden_size});
  input_proj.add_(bias_contig.view({1, 1, hidden_size}));

  auto state_t = initial_state.transpose(0, 1).contiguous();
  std::vector<at::Tensor> steps;
  steps.reserve(length);

  for (int64_t t = 0; t < length; ++t) {
    auto base_t = input_proj.select(0, t).transpose(0, 1);
    auto next = at::matmul(weight_hh_contig, state_t);
    next.add_(base_t);
    state_t = next;
    steps.push_back(state_t.transpose(0, 1));
  }

  return at::stack(steps, 0);
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
