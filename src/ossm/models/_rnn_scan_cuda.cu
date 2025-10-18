#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void linear_rnn_scan_kernel(const scalar_t* __restrict__ weight_hh,
                                       const scalar_t* __restrict__ weight_xh,
                                       const scalar_t* __restrict__ bias,
                                       const scalar_t* __restrict__ inputs,
                                       const scalar_t* __restrict__ initial_state,
                                       scalar_t* __restrict__ outputs,
                                       int64_t length,
                                       int64_t batch,
                                       int64_t hidden_size,
                                       int64_t input_size) {
  const int64_t batch_index = blockIdx.x;
  if (batch_index >= batch) {
    return;
  }

  extern __shared__ char shared_mem[];
  scalar_t* state = reinterpret_cast<scalar_t*>(shared_mem);
  scalar_t* next = state + hidden_size;

  for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
    state[h] = initial_state[batch_index * hidden_size + h];
  }
  __syncthreads();

  for (int64_t t = 0; t < length; ++t) {
    const scalar_t* input_ptr = inputs + (t * batch + batch_index) * input_size;
    scalar_t* out_ptr = outputs + (t * batch + batch_index) * hidden_size;

    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
      scalar_t acc = bias[h];
      const scalar_t* w_h_row = weight_hh + h * hidden_size;
      const scalar_t* w_x_row = weight_xh + h * input_size;

      for (int64_t k = 0; k < hidden_size; ++k) {
        acc += w_h_row[k] * state[k];
      }
      for (int64_t k = 0; k < input_size; ++k) {
        acc += w_x_row[k] * input_ptr[k];
      }

      next[h] = acc;
      out_ptr[h] = acc;
    }
    __syncthreads();
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
      state[h] = next[h];
    }
    __syncthreads();
  }
}

at::Tensor linear_rnn_scan_cuda_fallback(const at::Tensor& weight_hh,
                                         const at::Tensor& weight_xh,
                                         const at::Tensor& bias,
                                         const at::Tensor& inputs,
                                         const at::Tensor& initial_state) {
  auto weight_hh_t = weight_hh.transpose(0, 1).contiguous();
  auto weight_xh_t = weight_xh.transpose(0, 1).contiguous();
  auto bias_contig = bias.contiguous();
  auto inputs_contig = inputs.contiguous();
  auto state = initial_state.contiguous();

  const auto length = inputs_contig.size(0);
  const auto batch = inputs_contig.size(1);
  const auto hidden_size = weight_hh.size(0);

  auto outputs = at::empty({length, batch, hidden_size}, inputs.options());

  for (int64_t t = 0; t < length; ++t) {
    auto input_t = inputs_contig.select(0, t);
    state = at::addmm(bias_contig, state, weight_hh_t);
    state = state + at::mm(input_t, weight_xh_t);
    outputs.select(0, t).copy_(state);
  }

  return outputs;
}

}  // namespace

at::Tensor linear_rnn_scan_cuda(const at::Tensor& weight_hh,
                                const at::Tensor& weight_xh,
                                const at::Tensor& bias,
                                const at::Tensor& inputs,
                                const at::Tensor& initial_state) {
  TORCH_CHECK(weight_hh.is_cuda(), "weight_hh must be CUDA");
  TORCH_CHECK(weight_xh.is_cuda(), "weight_xh must be CUDA");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
  TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(initial_state.is_cuda(), "initial_state must be CUDA");

  auto weight_hh_contig = weight_hh.contiguous();
  auto weight_xh_contig = weight_xh.contiguous();
  auto bias_contig = bias.contiguous();
  auto inputs_contig = inputs.contiguous();
  auto initial_contig = initial_state.contiguous();

  const auto length = inputs_contig.size(0);
  const auto batch = inputs_contig.size(1);
  const auto hidden_size = weight_hh_contig.size(0);
  const auto input_size = weight_xh_contig.size(1);

  if (length == 0 || batch == 0 || hidden_size == 0) {
    return at::empty({length, batch, hidden_size}, inputs.options());
  }

  const auto props = at::cuda::getCurrentDeviceProperties();
  const size_t scalar_size = weight_hh_contig.element_size();
  const size_t shared_bytes = scalar_size * hidden_size * 2;
  const bool use_shared = shared_bytes <= props->sharedMemPerBlock;

  if (!use_shared) {
    return linear_rnn_scan_cuda_fallback(
        weight_hh_contig, weight_xh_contig, bias_contig, inputs_contig, initial_contig);
  }

  auto outputs = at::empty({length, batch, hidden_size}, inputs.options());

  const int threads = std::min<int>(static_cast<int>(hidden_size), 256);
  const dim3 blocks(batch);

  at::cuda::CUDAGuard device_guard(inputs.device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs_contig.scalar_type(),
                                      "linear_rnn_scan_cuda",
                                      [&] {
                                        linear_rnn_scan_kernel<scalar_t><<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                                            weight_hh_contig.data_ptr<scalar_t>(),
                                            weight_xh_contig.data_ptr<scalar_t>(),
                                            bias_contig.data_ptr<scalar_t>(),
                                            inputs_contig.data_ptr<scalar_t>(),
                                            initial_contig.data_ptr<scalar_t>(),
                                            outputs.data_ptr<scalar_t>(),
                                            length,
                                            batch,
                                            hidden_size,
                                            input_size);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                      });

  return outputs;
}

