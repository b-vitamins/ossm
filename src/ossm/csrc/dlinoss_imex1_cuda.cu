#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>

namespace ossm {
namespace {

template <typename scalar_t>
__device__ inline scalar_t complex_conj(const scalar_t& value) {
  return scalar_t(value.real(), -value.imag());
}

template <typename scalar_t>
__global__ void dlinoss_imex1_forward_kernel(const typename scalar_t::value_type* __restrict__ a_diag,
                                             const typename scalar_t::value_type* __restrict__ g_diag,
                                             const typename scalar_t::value_type* __restrict__ step,
                                             const scalar_t* __restrict__ bu_ptr,
                                             scalar_t* __restrict__ out_ptr,
                                             int64_t length,
                                             int64_t batch,
                                             int64_t ssm) {
  using value_t = typename scalar_t::value_type;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series; idx += blockDim.x * gridDim.x) {
    const int64_t state = idx % ssm;

    const value_t alpha = a_diag[state];
    const value_t gamma = g_diag[state];
    const value_t sigma = step[state];

    const value_t denom = value_t(1) + sigma * gamma;
    const value_t inv = value_t(1) / denom;
    const value_t sigma_inv = sigma * inv;
    const value_t sigma2_inv = sigma * sigma * inv;
    const value_t coeff12 = -alpha * sigma_inv;
    const value_t coeff22 = value_t(1) - alpha * sigma2_inv;

    scalar_t state0 = scalar_t(0, 0);
    scalar_t state1 = scalar_t(0, 0);

    for (int64_t t = 0; t < length; ++t) {
      const int64_t bu_offset = t * series + idx;
      const int64_t out_offset = t * step_stride + idx * 2;

      const scalar_t bu_val = bu_ptr[bu_offset];

      const scalar_t new0 = state0 * inv + state1 * coeff12 + bu_val * sigma_inv;
      const scalar_t new1 = state0 * sigma_inv + state1 * coeff22 + bu_val * sigma2_inv;

      out_ptr[out_offset] = new0;
      out_ptr[out_offset + 1] = new1;

      state0 = new0;
      state1 = new1;
    }
  }
}

template <typename scalar_t>
__global__ void dlinoss_imex1_backward_kernel(const typename scalar_t::value_type* __restrict__ a_diag,
                                              const typename scalar_t::value_type* __restrict__ g_diag,
                                              const typename scalar_t::value_type* __restrict__ step,
                                              const scalar_t* __restrict__ bu_ptr,
                                              const scalar_t* __restrict__ states_ptr,
                                              const scalar_t* __restrict__ grad_out_ptr,
                                              scalar_t* __restrict__ grad_bu_ptr,
                                              typename scalar_t::value_type* __restrict__ grad_a_ptr,
                                              typename scalar_t::value_type* __restrict__ grad_g_ptr,
                                              typename scalar_t::value_type* __restrict__ grad_step_ptr,
                                              int64_t length,
                                              int64_t batch,
                                              int64_t ssm) {
  using value_t = typename scalar_t::value_type;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series; idx += blockDim.x * gridDim.x) {
    const int64_t state = idx % ssm;

    const value_t alpha = a_diag[state];
    const value_t gamma = g_diag[state];
    const value_t sigma = step[state];

    const value_t denom = value_t(1) + sigma * gamma;
    const value_t inv = value_t(1) / denom;
    const value_t sigma_inv = sigma * inv;
    const value_t sigma2_inv = sigma * sigma * inv;
    const value_t coeff12 = -alpha * sigma_inv;
    const value_t coeff22 = value_t(1) - alpha * sigma2_inv;

    scalar_t grad_state0 = scalar_t(0, 0);
    scalar_t grad_state1 = scalar_t(0, 0);

    value_t grad_alpha_local = value_t(0);
    value_t grad_sigma_inv_local = value_t(0);
    value_t grad_sigma2_inv_local = value_t(0);
    value_t grad_inv_local = value_t(0);

    for (int64_t t = length - 1; t >= 0; --t) {
      const int64_t base_offset = t * step_stride + idx * 2;
      const int64_t bu_offset = t * series + idx;

      const scalar_t prev0 = t > 0 ? states_ptr[base_offset - step_stride] : scalar_t(0, 0);
      const scalar_t prev1 = t > 0 ? states_ptr[base_offset - step_stride + 1] : scalar_t(0, 0);
      const scalar_t bu_val = bu_ptr[bu_offset];

      const scalar_t grad_new1 = grad_state1 + grad_out_ptr[bu_offset];
      const scalar_t grad_new0 = grad_state0;

      const scalar_t conj_grad0 = complex_conj(grad_new0);
      const scalar_t conj_grad1 = complex_conj(grad_new1);

      const scalar_t neg_alpha_prev1 = prev1 * (-alpha);
      const scalar_t bu_combined = neg_alpha_prev1 + bu_val;

      grad_sigma_inv_local += (conj_grad0 * bu_combined).real() + (conj_grad1 * prev0).real();
      grad_sigma2_inv_local += (conj_grad1 * bu_combined).real();
      grad_alpha_local += (conj_grad0 * (prev1 * (-sigma_inv))).real() +
                          (conj_grad1 * (prev1 * (-sigma2_inv))).real();
      grad_inv_local += (conj_grad0 * prev0).real();

      grad_bu_ptr[bu_offset] = grad_new0 * sigma_inv + grad_new1 * sigma2_inv;

      const scalar_t grad_prev0 = grad_new0 * inv + grad_new1 * sigma_inv;
      const scalar_t grad_prev1 = grad_new0 * coeff12 + grad_new1 * coeff22;

      grad_state0 = grad_prev0;
      grad_state1 = grad_prev1;
    }

    value_t grad_sigma_local = grad_sigma_inv_local * inv;
    value_t grad_inv_total = grad_inv_local + grad_sigma_inv_local * sigma;

    const value_t grad_sigma_sq = grad_sigma2_inv_local * inv;
    grad_inv_total += grad_sigma2_inv_local * sigma * sigma;
    grad_sigma_local += grad_sigma_sq * value_t(2) * sigma;

    const value_t inv_sq = inv * inv;
    grad_sigma_local += grad_inv_total * (-gamma) * inv_sq;
    const value_t grad_gamma_local = grad_inv_total * (-sigma) * inv_sq;

    atomicAdd(grad_a_ptr + state, grad_alpha_local);
    atomicAdd(grad_step_ptr + state, grad_sigma_local);
    atomicAdd(grad_g_ptr + state, grad_gamma_local);
  }
}

}  // namespace

void dlinoss_imex1_forward_cuda(const at::Tensor& a_diag,
                                const at::Tensor& g_diag,
                                const at::Tensor& step,
                                const at::Tensor& bu,
                                at::Tensor& output) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};

  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  const auto series = batch * ssm;

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(
          (series + threads - 1) / threads,
          at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "dlinoss_imex1_forward_cuda", [&] {
    dlinoss_imex1_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_diag.data_ptr<typename scalar_t::value_type>(),
        g_diag.data_ptr<typename scalar_t::value_type>(),
        step.data_ptr<typename scalar_t::value_type>(),
        bu.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        bu.size(0),
        batch,
        ssm);
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

void dlinoss_imex1_backward_cuda(const at::Tensor& a_diag,
                                 const at::Tensor& g_diag,
                                 const at::Tensor& step,
                                 const at::Tensor& bu,
                                 const at::Tensor& states,
                                 const at::Tensor& grad_output,
                                 at::Tensor& grad_a,
                                 at::Tensor& grad_g,
                                 at::Tensor& grad_step,
                                 at::Tensor& grad_bu) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};

  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  const auto series = batch * ssm;

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(
          (series + threads - 1) / threads,
          at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "dlinoss_imex1_backward_cuda", [&] {
    dlinoss_imex1_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_diag.data_ptr<typename scalar_t::value_type>(),
        g_diag.data_ptr<typename scalar_t::value_type>(),
        step.data_ptr<typename scalar_t::value_type>(),
        bu.data_ptr<scalar_t>(),
        states.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        grad_bu.data_ptr<scalar_t>(),
        grad_a.data_ptr<typename scalar_t::value_type>(),
        grad_g.data_ptr<typename scalar_t::value_type>(),
        grad_step.data_ptr<typename scalar_t::value_type>(),
        bu.size(0),
        batch,
        ssm);
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ossm
