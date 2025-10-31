#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "dlinoss_common.h"

namespace ossm {
namespace {

constexpr double kDtMin = 1e-6;
constexpr double kDtMax = 1.0;
constexpr double kClampMin = 1e-6;

template <typename value_t>
__device__ inline value_t clamp_step(value_t raw) {
  const value_t lower = static_cast<value_t>(kDtMin);
  const value_t upper = static_cast<value_t>(kDtMax);
  return raw < lower ? lower : (raw > upper ? upper : raw);
}

template <typename value_t>
__device__ inline value_t clamp_stability(value_t raw) {
  const value_t lower = static_cast<value_t>(kClampMin);
  return raw < lower ? lower : raw;
}

template <typename scalar_t>
__global__ void sdlinoss_im_forward_kernel(
    Strided3<typename scalar_t::value_type> A,
    Strided3<typename scalar_t::value_type> G,
    Strided3<typename scalar_t::value_type> step,
    Strided3<scalar_t> bu,
    scalar_t* __restrict__ out_ptr,
    int64_t length,
    int64_t batch,
    int64_t ssm) {
  using value_t = typename scalar_t::value_type;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series; idx += blockDim.x * gridDim.x) {
    const int64_t b = idx / ssm;
    const int64_t m = idx % ssm;
    value_t z_real = value_t(0);
    value_t z_imag = value_t(0);
    value_t x_real = value_t(0);
    value_t x_imag = value_t(0);

    for (int64_t t = 0; t < length; ++t) {
      const int64_t offset = t * series + idx;
      const value_t a_t = A.load(t, b, m);
      const value_t g_t = G.load(t, b, m);
      const value_t step_raw = step.load(t, b, m);
      const value_t dt = clamp_step(step_raw);
      const value_t dt2 = dt * dt;

      const value_t S_raw = value_t(1) + dt * g_t + dt2 * a_t;
      const value_t S = clamp_stability(S_raw);
      const value_t inv_S = value_t(1) / S;

      const scalar_t bu_val = bu.load(t, b, m);
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      const value_t tmp_real = z_real + dt * (-a_t * x_real + bu_real);
      const value_t tmp_imag = z_imag + dt * (-a_t * x_imag + bu_imag);

      const value_t new_z_real = tmp_real * inv_S;
      const value_t new_z_imag = tmp_imag * inv_S;
      const value_t new_x_real = x_real + dt * new_z_real;
      const value_t new_x_imag = x_imag + dt * new_z_imag;

      const int64_t out_offset = t * step_stride + idx * 2;
      out_ptr[out_offset] = scalar_t(new_z_real, new_z_imag);
      out_ptr[out_offset + 1] = scalar_t(new_x_real, new_x_imag);

      z_real = new_z_real;
      z_imag = new_z_imag;
      x_real = new_x_real;
      x_imag = new_x_imag;
    }
  }
}

template <typename scalar_t>
__global__ void sdlinoss_im_backward_kernel(
    Strided3<typename scalar_t::value_type> A,
    Strided3<typename scalar_t::value_type> G,
    Strided3<typename scalar_t::value_type> step,
    Strided3<scalar_t> bu,
    const scalar_t* __restrict__ states_ptr,
    const scalar_t* __restrict__ grad_out_ptr,
    scalar_t* __restrict__ grad_bu_ptr,
    GradStrided3<typename scalar_t::value_type> grad_A,
    GradStrided3<typename scalar_t::value_type> grad_G,
    GradStrided3<typename scalar_t::value_type> grad_step,
    int64_t length,
    int64_t batch,
    int64_t ssm) {
  using value_t = typename scalar_t::value_type;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series; idx += blockDim.x * gridDim.x) {
    const int64_t b = idx / ssm;
    const int64_t m = idx % ssm;
    value_t grad_z_next_real = value_t(0);
    value_t grad_z_next_imag = value_t(0);
    value_t grad_x_next_real = value_t(0);
    value_t grad_x_next_imag = value_t(0);

    for (int64_t t = length - 1; t >= 0; --t) {
      const int64_t offset = t * series + idx;
      const int64_t out_offset = t * step_stride + idx * 2;

      const scalar_t state_z = states_ptr[out_offset];
      const scalar_t state_x = states_ptr[out_offset + 1];
      const value_t z_new_real = state_z.real();
      const value_t z_new_imag = state_z.imag();
      const value_t x_new_real = state_x.real();
      const value_t x_new_imag = state_x.imag();

      value_t z_prev_real = value_t(0);
      value_t z_prev_imag = value_t(0);
      value_t x_prev_real = value_t(0);
      value_t x_prev_imag = value_t(0);
      if (t > 0) {
        const int64_t prev_offset = out_offset - step_stride;
        const scalar_t prev_z = states_ptr[prev_offset];
        const scalar_t prev_x = states_ptr[prev_offset + 1];
        z_prev_real = prev_z.real();
        z_prev_imag = prev_z.imag();
        x_prev_real = prev_x.real();
        x_prev_imag = prev_x.imag();
      }

      const scalar_t grad_out = grad_out_ptr[offset];
      value_t grad_x_new_real = grad_out.real() + grad_x_next_real;
      value_t grad_x_new_imag = grad_out.imag() + grad_x_next_imag;
      value_t grad_z_new_real = grad_z_next_real;
      value_t grad_z_new_imag = grad_z_next_imag;

      const value_t a_t = A.load(t, b, m);
      const value_t g_t = G.load(t, b, m);
      const value_t step_raw = step.load(t, b, m);
      const value_t dt = clamp_step(step_raw);
      const value_t step_mask = (step_raw > value_t(kDtMin) && step_raw < value_t(kDtMax)) ? value_t(1) : value_t(0);
      const value_t dt2 = dt * dt;

      const scalar_t bu_val = bu.load(t, b, m);
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      value_t grad_step_local = value_t(0);
      value_t grad_A_local = value_t(0);
      value_t grad_G_local = value_t(0);

      grad_step_local += grad_x_new_real * z_new_real + grad_x_new_imag * z_new_imag;
      grad_z_new_real += dt * grad_x_new_real;
      grad_z_new_imag += dt * grad_x_new_imag;
      value_t grad_x_prev_real = grad_x_new_real;
      value_t grad_x_prev_imag = grad_x_new_imag;

      const value_t S_raw = value_t(1) + dt * g_t + dt2 * a_t;
      const value_t S = clamp_stability(S_raw);
      const value_t clamp_mask = S_raw > value_t(kClampMin) ? value_t(1) : value_t(0);
      const value_t inv_S = value_t(1) / S;
      const value_t inv_S_sq = inv_S * inv_S;

      const value_t comb_real = -a_t * x_prev_real + bu_real;
      const value_t comb_imag = -a_t * x_prev_imag + bu_imag;

      const value_t tmp_real = z_prev_real + dt * comb_real;
      const value_t tmp_imag = z_prev_imag + dt * comb_imag;

      const value_t grad_temp_real = grad_z_new_real * inv_S;
      const value_t grad_temp_imag = grad_z_new_imag * inv_S;

      const value_t grad_S = -(grad_z_new_real * tmp_real + grad_z_new_imag * tmp_imag) * inv_S_sq;
      const value_t grad_S_raw = grad_S * clamp_mask;

      grad_step_local += grad_S_raw * (g_t + value_t(2) * dt * a_t);
      grad_A_local += grad_S_raw * dt2;
      grad_G_local += grad_S_raw * dt;

      grad_step_local += grad_temp_real * comb_real + grad_temp_imag * comb_imag;
      grad_A_local += grad_temp_real * (-dt * x_prev_real) + grad_temp_imag * (-dt * x_prev_imag);

      grad_x_prev_real += grad_temp_real * (-dt * a_t);
      grad_x_prev_imag += grad_temp_imag * (-dt * a_t);

      const value_t grad_z_prev_real = grad_temp_real;
      const value_t grad_z_prev_imag = grad_temp_imag;

      const value_t grad_bu_real = grad_temp_real * dt;
      const value_t grad_bu_imag = grad_temp_imag * dt;

      grad_A.store(t, b, m, grad_A_local);
      grad_G.store(t, b, m, grad_G_local);
      grad_step.store(t, b, m, grad_step_local * step_mask);
      grad_bu_ptr[offset] = scalar_t(grad_bu_real, grad_bu_imag);

      grad_x_next_real = grad_x_prev_real;
      grad_x_next_imag = grad_x_prev_imag;
      grad_z_next_real = grad_z_prev_real;
      grad_z_next_imag = grad_z_prev_imag;
    }
  }
}

}  // namespace

void sdlinoss_im_forward_cuda(const at::Tensor& A,
                              const at::Tensor& G,
                              const at::Tensor& step,
                              const at::Tensor& bu,
                              at::Tensor& output) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};

  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  const auto series = batch * ssm;

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1, std::min<int64_t>((series + threads - 1) / threads, at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_im_forward_cuda", [&] {
    const auto length = bu.size(0);
    const auto A_strided = make_strided3<typename scalar_t::value_type>(A, length, batch, ssm);
    const auto G_strided = make_strided3<typename scalar_t::value_type>(G, length, batch, ssm);
    const auto step_strided = make_strided3<typename scalar_t::value_type>(step, length, batch, ssm);
    const auto bu_strided = make_strided3<scalar_t>(bu, length, batch, ssm);
    sdlinoss_im_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        A_strided,
        G_strided,
        step_strided,
        bu_strided,
        output.data_ptr<scalar_t>(),
        length,
        batch,
        ssm);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sdlinoss_im_backward_cuda(const at::Tensor& A,
                               const at::Tensor& G,
                               const at::Tensor& step,
                               const at::Tensor& bu,
                               const at::Tensor& states,
                               const at::Tensor& grad_output,
                               at::Tensor& grad_A,
                               at::Tensor& grad_G,
                               at::Tensor& grad_step,
                               at::Tensor& grad_bu) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};

  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  const auto series = batch * ssm;

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1, std::min<int64_t>((series + threads - 1) / threads, at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_im_backward_cuda", [&] {
    const auto length = bu.size(0);
    const auto A_strided = make_strided3<typename scalar_t::value_type>(A, length, batch, ssm);
    const auto G_strided = make_strided3<typename scalar_t::value_type>(G, length, batch, ssm);
    const auto step_strided = make_strided3<typename scalar_t::value_type>(step, length, batch, ssm);
    const auto bu_strided = make_strided3<scalar_t>(bu, length, batch, ssm);
    auto grad_A_strided = make_grad_strided3<typename scalar_t::value_type>(grad_A, length, batch, ssm);
    auto grad_G_strided = make_grad_strided3<typename scalar_t::value_type>(grad_G, length, batch, ssm);
    auto grad_step_strided = make_grad_strided3<typename scalar_t::value_type>(grad_step, length, batch, ssm);
    sdlinoss_im_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        A_strided,
        G_strided,
        step_strided,
        bu_strided,
        states.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        grad_bu.data_ptr<scalar_t>(),
        grad_A_strided,
        grad_G_strided,
        grad_step_strided,
        length,
        batch,
        ssm);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ossm

