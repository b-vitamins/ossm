#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace ossm {
namespace {

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
    const value_t sigma2_inv = sigma * sigma * inv;
    // The kernel evolves the scaled auxiliary state w = dt * z to avoid the
    // 1/dt conditioning issues present when storing z directly.  This produces
    // the same x trajectory as the official implementation because x_{k+1} =
    // x_k + w_{k+1} with a time-invariant dt per mode.
    const value_t coeff12 = -alpha * sigma2_inv;

    value_t state0_real = value_t(0);
    value_t state0_imag = value_t(0);
    value_t state1_real = value_t(0);
    value_t state1_imag = value_t(0);

    for (int64_t t = 0; t < length; ++t) {
      const int64_t bu_offset = t * series + idx;
      const int64_t out_offset = t * step_stride + idx * 2;

      const scalar_t bu_val = bu_ptr[bu_offset];
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      const value_t bu_scale_real = bu_real * sigma2_inv;
      const value_t bu_scale_imag = bu_imag * sigma2_inv;
      const value_t new0_real = state0_real * inv + state1_real * coeff12 + bu_scale_real;
      const value_t new0_imag = state0_imag * inv + state1_imag * coeff12 + bu_scale_imag;
      const value_t new1_real = new0_real + state1_real;
      const value_t new1_imag = new0_imag + state1_imag;

      out_ptr[out_offset] = scalar_t(new0_real, new0_imag);
      out_ptr[out_offset + 1] = scalar_t(new1_real, new1_imag);

      state0_real = new0_real;
      state0_imag = new0_imag;
      state1_real = new1_real;
      state1_imag = new1_imag;
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
    const value_t sigma2_inv = sigma * sigma * inv;
    const value_t coeff12 = -alpha * sigma2_inv;
    const value_t inv_sq = inv * inv;

    value_t grad_state0_real = value_t(0);
    value_t grad_state0_imag = value_t(0);
    value_t grad_state1_real = value_t(0);
    value_t grad_state1_imag = value_t(0);

    value_t grad_alpha_local = value_t(0);
    value_t grad_sigma2_inv_local = value_t(0);
    value_t grad_inv_local = value_t(0);

    for (int64_t t = length - 1; t >= 0; --t) {
      const int64_t base_offset = t * step_stride + idx * 2;
      const int64_t bu_offset = t * series + idx;

      const scalar_t prev0_val = t > 0 ? states_ptr[base_offset - step_stride] : scalar_t(0, 0);
      const scalar_t prev1_val = t > 0 ? states_ptr[base_offset - step_stride + 1] : scalar_t(0, 0);
      const scalar_t bu_val = bu_ptr[bu_offset];
      const scalar_t grad_out_val = grad_out_ptr[bu_offset];

      const value_t prev0_real = prev0_val.real();
      const value_t prev0_imag = prev0_val.imag();
      const value_t prev1_real = prev1_val.real();
      const value_t prev1_imag = prev1_val.imag();
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();
      const value_t grad_out_real = grad_out_val.real();
      const value_t grad_out_imag = grad_out_val.imag();

      const value_t grad_new1_real = grad_state1_real + grad_out_real;
      const value_t grad_new1_imag = grad_state1_imag + grad_out_imag;
      const value_t grad_new0_real = grad_state0_real;
      const value_t grad_new0_imag = grad_state0_imag;

      const value_t grad_total_real = grad_new0_real + grad_new1_real;
      const value_t grad_total_imag = grad_new0_imag + grad_new1_imag;

      const value_t comb_real = bu_real - alpha * prev1_real;
      const value_t comb_imag = bu_imag - alpha * prev1_imag;

      grad_sigma2_inv_local += grad_total_real * comb_real + grad_total_imag * comb_imag;
      grad_alpha_local += grad_total_real * (-sigma2_inv * prev1_real) +
                          grad_total_imag * (-sigma2_inv * prev1_imag);
      grad_inv_local += grad_total_real * prev0_real + grad_total_imag * prev0_imag;

      const value_t grad_bu_real = grad_total_real * sigma2_inv;
      const value_t grad_bu_imag = grad_total_imag * sigma2_inv;
      grad_bu_ptr[bu_offset] = scalar_t(grad_bu_real, grad_bu_imag);

      const value_t grad_prev0_real = grad_total_real * inv;
      const value_t grad_prev0_imag = grad_total_imag * inv;
      const value_t grad_prev1_real = grad_new1_real + grad_total_real * coeff12;
      const value_t grad_prev1_imag = grad_new1_imag + grad_total_imag * coeff12;

      grad_state0_real = grad_prev0_real;
      grad_state0_imag = grad_prev0_imag;
      grad_state1_real = grad_prev1_real;
      grad_state1_imag = grad_prev1_imag;
    }

    const value_t sigma_sq = sigma * sigma;
    const value_t grad_inv_total = grad_inv_local + grad_sigma2_inv_local * sigma_sq;

    const value_t d_inv_d_sigma = -gamma * inv_sq;
    const value_t d_inv_d_gamma = -sigma * inv_sq;
    const value_t d_sigma2_inv_d_sigma = value_t(2) * sigma * inv + sigma_sq * d_inv_d_sigma;
    const value_t d_sigma2_inv_d_gamma = sigma_sq * d_inv_d_gamma;

    const value_t grad_sigma_local = grad_sigma2_inv_local * d_sigma2_inv_d_sigma +
                                     grad_inv_total * d_inv_d_sigma;
    const value_t grad_gamma_local = grad_sigma2_inv_local * d_sigma2_inv_d_gamma +
                                     grad_inv_total * d_inv_d_gamma;

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
