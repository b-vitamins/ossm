#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "sdlinoss_common.cuh"

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
struct Imex1TransitionBuilder {
  using value_t = typename scalar_t::value_type;

  __device__ static Transition2x2<value_t> build(value_t a, value_t g, value_t step, scalar_t bu) {
    const value_t dt = clamp_step(step);
    const value_t S = clamp_stability(value_t(1) + dt * g);
    const value_t inv_S = value_t(1) / S;
    const value_t dt_inv_S = dt * inv_S;
    const value_t adt = a * dt;

    Transition2x2<value_t> out;
    out.m00 = inv_S;
    out.m01 = -adt * inv_S;
    out.m10 = dt_inv_S;
    out.m11 = value_t(1) - adt * dt_inv_S;

    const value_t bu_real = bu.real();
    const value_t bu_imag = bu.imag();
    const value_t f0_scale = dt_inv_S;
    const value_t f1_scale = dt * dt_inv_S;
    out.f0_real = f0_scale * bu_real;
    out.f0_imag = f0_scale * bu_imag;
    out.f1_real = f1_scale * bu_real;
    out.f1_imag = f1_scale * bu_imag;
    return out;
  }
};

template <typename scalar_t>
__global__ void sdlinoss_imex1_backward_kernel(
    StridedView3<typename scalar_t::value_type> A,
    StridedView3<typename scalar_t::value_type> G,
    StridedView3<typename scalar_t::value_type> step,
    StridedView3<scalar_t> bu_view,
    const scalar_t* __restrict__ states_ptr,
    const scalar_t* __restrict__ grad_out_ptr,
    scalar_t* __restrict__ grad_bu_ptr,
    typename scalar_t::value_type* __restrict__ grad_A_ptr,
    typename scalar_t::value_type* __restrict__ grad_G_ptr,
    typename scalar_t::value_type* __restrict__ grad_step_ptr,
    int64_t length,
    int64_t batch,
    int64_t ssm) {
  using value_t = typename scalar_t::value_type;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  if (length == 0) {
    return;
  }

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

      const scalar_t bu_val = bu_view.load(t, b, m);
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

      const value_t S_raw = value_t(1) + dt * g_t;
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

      grad_step_local += grad_S_raw * g_t;
      grad_G_local += grad_S_raw * dt;

      grad_step_local += grad_temp_real * comb_real + grad_temp_imag * comb_imag;
      grad_A_local += grad_temp_real * (-dt * x_prev_real) + grad_temp_imag * (-dt * x_prev_imag);

      grad_x_prev_real += grad_temp_real * (-dt * a_t);
      grad_x_prev_imag += grad_temp_imag * (-dt * a_t);

      const value_t grad_z_prev_real = grad_temp_real;
      const value_t grad_z_prev_imag = grad_temp_imag;

      const value_t grad_bu_real = grad_temp_real * dt;
      const value_t grad_bu_imag = grad_temp_imag * dt;

      grad_A_ptr[offset] = grad_A_local;
      grad_G_ptr[offset] = grad_G_local;
      grad_step_ptr[offset] = grad_step_local * step_mask;
      grad_bu_ptr[offset] = scalar_t(grad_bu_real, grad_bu_imag);

      grad_x_next_real = grad_x_prev_real;
      grad_x_next_imag = grad_x_prev_imag;
      grad_z_next_real = grad_z_prev_real;
      grad_z_next_imag = grad_z_prev_imag;
    }
  }
}

}  // namespace

void sdlinoss_imex1_forward_cuda(const at::Tensor& A,
                                 const at::Tensor& G,
                                 const at::Tensor& step,
                                 const at::Tensor& bu,
                                 at::Tensor& output) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  if (length == 0) {
    return;
  }

  const int64_t num_tiles = (length + kSdlinossTileSteps - 1) / kSdlinossTileSteps;
  auto real_options = A.options();
  auto local_prefixes = at::empty({length, batch, ssm, 8}, real_options);
  auto tile_prefixes = at::empty({num_tiles, batch, ssm, 8}, real_options);
  auto tile_scans = at::empty({num_tiles, batch, ssm, 8}, real_options);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_imex1_forward_cuda", [&] {
    const auto A_view = make_strided_view3<typename scalar_t::value_type>(A);
    const auto G_view = make_strided_view3<typename scalar_t::value_type>(G);
    const auto step_view = make_strided_view3<typename scalar_t::value_type>(step);
    const auto bu_view = make_strided_view3<scalar_t>(bu);
    using value_t = typename scalar_t::value_type;
    auto* local_ptr = reinterpret_cast<Transition2x2<value_t>*>(local_prefixes.data_ptr<value_t>());
    auto* tile_prefix_ptr = reinterpret_cast<Transition2x2<value_t>*>(tile_prefixes.data_ptr<value_t>());
    auto* tile_scan_ptr = reinterpret_cast<Transition2x2<value_t>*>(tile_scans.data_ptr<value_t>());

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 tile_block(kSdlinossTileSteps);
    const dim3 tile_grid(static_cast<unsigned int>(ssm),
                         static_cast<unsigned int>(batch),
                         static_cast<unsigned int>(num_tiles));

    sdlinoss_forward_tile_kernel<scalar_t, Imex1TransitionBuilder<scalar_t>, kSdlinossTileSteps>
        <<<tile_grid, tile_block, 0, stream>>>(
            A_view,
            G_view,
            step_view,
            bu_view,
            local_ptr,
            tile_prefix_ptr,
            length,
            batch,
            ssm,
            num_tiles);

    const dim3 prefix_grid(static_cast<unsigned int>(ssm), static_cast<unsigned int>(batch));
    sdlinoss_forward_prefix_kernel<value_t, kSdlinossTileSteps><<<prefix_grid, tile_block, 0, stream>>>(
        tile_prefix_ptr,
        tile_scan_ptr,
        num_tiles,
        batch,
        ssm);

    sdlinoss_forward_apply_kernel<scalar_t, kSdlinossTileSteps><<<tile_grid, tile_block, 0, stream>>>(
        local_ptr,
        tile_scan_ptr,
        output.data_ptr<scalar_t>(),
        length,
        batch,
        ssm,
        num_tiles);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sdlinoss_imex1_backward_cuda(const at::Tensor& A,
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

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_imex1_backward_cuda", [&] {
    const auto A_view = make_strided_view3<typename scalar_t::value_type>(A);
    const auto G_view = make_strided_view3<typename scalar_t::value_type>(G);
    const auto step_view = make_strided_view3<typename scalar_t::value_type>(step);
    const auto bu_view = make_strided_view3<scalar_t>(bu);
    sdlinoss_imex1_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        A_view,
        G_view,
        step_view,
        bu_view,
        states.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        grad_bu.data_ptr<scalar_t>(),
        grad_A.data_ptr<typename scalar_t::value_type>(),
        grad_G.data_ptr<typename scalar_t::value_type>(),
        grad_step.data_ptr<typename scalar_t::value_type>(),
        bu.size(0),
        batch,
        ssm);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ossm

