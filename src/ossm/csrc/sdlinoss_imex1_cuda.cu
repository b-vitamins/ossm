#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>

#include "dlinoss_common.h"

namespace ossm {
constexpr int kTile = 1024;
constexpr int kThreads = kTile;

using cub::BlockScan;

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
struct ComplexTraits;
template <>
struct ComplexTraits<c10::complex<float>> {
  using real_t = float;
  using vec2_t = float2;
};
template <>
struct ComplexTraits<c10::complex<double>> {
  using real_t = double;
  using vec2_t = double2;
};

template <typename scalar_t>
__global__ void sdlinoss_imex1_forward_tile_kernel(
    const typename ComplexTraits<scalar_t>::real_t* __restrict__ A,
    const typename ComplexTraits<scalar_t>::real_t* __restrict__ G,
    const typename ComplexTraits<scalar_t>::real_t* __restrict__ step,
    const scalar_t* __restrict__ bu_ptr,
    scalar_t* __restrict__ tmp_states,
    Pair2x2<typename ComplexTraits<scalar_t>::real_t, scalar_t>* __restrict__ tile_summ,
    int64_t length,
    int64_t batch,
    int64_t ssm) {
  using real_t = typename ComplexTraits<scalar_t>::real_t;

  const int64_t series_n = batch * ssm;
  const int64_t step_stride = series_n * 2;

  const int series_idx = blockIdx.x;
  const int tile_id = blockIdx.y;
  const int t_in_tile = threadIdx.x;
  const int t0 = tile_id * kTile;
  const int t = t0 + t_in_tile;

  if (series_idx >= series_n) {
    return;
  }

  __shared__ Pair2x2<real_t, scalar_t> sh_pairs[kTile];
  __shared__ Pair2x2<real_t, scalar_t> sh_scan[kTile];

  Pair2x2<real_t, scalar_t> p = pair_identity<real_t, scalar_t>();

  if (t < length) {
    const int64_t off = static_cast<int64_t>(t) * series_n + series_idx;

    const real_t a = A[off];
    const real_t g = G[off];
    const real_t dt = clamp_step<real_t>(step[off]);
    const scalar_t bu = bu_ptr[off];

    p = build_pair_imex1<real_t, scalar_t>(a, g, dt, bu);
  }
  sh_pairs[t_in_tile] = p;
  __syncthreads();

  Pair2x2<real_t, scalar_t> acc = sh_pairs[t_in_tile];
#pragma unroll
  for (int offset = 1; offset < kTile; offset <<= 1) {
    Pair2x2<real_t, scalar_t> other;
    if (t_in_tile >= offset) {
      other = sh_pairs[t_in_tile - offset];
    } else {
      other = pair_identity<real_t, scalar_t>();
    }
    __syncthreads();
    if (t_in_tile >= offset) {
      acc = pair_combine(other, acc);
    }
    sh_pairs[t_in_tile] = acc;
    __syncthreads();
  }
  sh_scan[t_in_tile] = acc;
  __syncthreads();

  if (t < length) {
    const Pair2x2<real_t, scalar_t> s = sh_scan[t_in_tile];
    const int64_t out_off = static_cast<int64_t>(t) * step_stride + series_idx * 2;
    tmp_states[out_off + 0] = s.f1;
    tmp_states[out_off + 1] = s.f2;
  }

  if (t_in_tile == kTile - 1) {
    const int64_t ntile = (length + kTile - 1) / kTile;
    const int64_t idx_tile = static_cast<int64_t>(series_idx) * ntile + tile_id;
    int64_t span64 = length - t0;
    if (span64 <= 0) {
      span64 = 1;
    }
    if (span64 > kTile) {
      span64 = kTile;
    }
    const int last = static_cast<int>(span64 - 1);
    tile_summ[idx_tile] = sh_scan[last];
  }
}

template <typename scalar_t>
__global__ void sdlinoss_imex1_forward_kernel(
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
    value_t w_real = value_t(0);
    value_t w_imag = value_t(0);
    value_t x_real = value_t(0);
    value_t x_imag = value_t(0);

    for (int64_t t = 0; t < length; ++t) {
      const int64_t offset = t * series + idx;
      const value_t a_t = A.load(t, b, m);
      const value_t g_t = G.load(t, b, m);
      const value_t step_raw = step.load(t, b, m);
      const value_t dt = clamp_step(step_raw);

      const value_t S_raw = value_t(1) + dt * g_t;
      const value_t S = clamp_stability(S_raw);
      const value_t inv_S = value_t(1) / S;

      const scalar_t bu_val = bu.load(t, b, m);
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      const value_t comb_real = -a_t * x_real + bu_real;
      const value_t comb_imag = -a_t * x_imag + bu_imag;

      const value_t tmpw_real = w_real + (dt * dt) * comb_real;
      const value_t tmpw_imag = w_imag + (dt * dt) * comb_imag;

      const value_t new_w_real = tmpw_real * inv_S;
      const value_t new_w_imag = tmpw_imag * inv_S;
      const value_t new_x_real = x_real + new_w_real;
      const value_t new_x_imag = x_imag + new_w_imag;

      const int64_t out_offset = t * step_stride + idx * 2;
      out_ptr[out_offset] = scalar_t(new_w_real, new_w_imag);
      out_ptr[out_offset + 1] = scalar_t(new_x_real, new_x_imag);

      w_real = new_w_real;
      w_imag = new_w_imag;
      x_real = new_x_real;
      x_imag = new_x_imag;
    }
  }
}

template <typename scalar_t>
__global__ void sdlinoss_imex1_backward_kernel(
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
    value_t grad_w_next_real = value_t(0);
    value_t grad_w_next_imag = value_t(0);
    value_t grad_x_next_real = value_t(0);
    value_t grad_x_next_imag = value_t(0);

    for (int64_t t = length - 1; t >= 0; --t) {
      const int64_t offset = t * series + idx;
      const int64_t out_offset = t * step_stride + idx * 2;

      const scalar_t state_w = states_ptr[out_offset];
      const scalar_t state_x = states_ptr[out_offset + 1];
      const value_t w_new_real = state_w.real();
      const value_t w_new_imag = state_w.imag();
      const value_t x_new_real = state_x.real();
      const value_t x_new_imag = state_x.imag();

      value_t w_prev_real = value_t(0);
      value_t w_prev_imag = value_t(0);
      value_t x_prev_real = value_t(0);
      value_t x_prev_imag = value_t(0);
      if (t > 0) {
        const int64_t prev_offset = out_offset - step_stride;
        const scalar_t prev_w = states_ptr[prev_offset];
        const scalar_t prev_x = states_ptr[prev_offset + 1];
        w_prev_real = prev_w.real();
        w_prev_imag = prev_w.imag();
        x_prev_real = prev_x.real();
        x_prev_imag = prev_x.imag();
      }

      const scalar_t grad_out = grad_out_ptr[offset];
      value_t grad_x_new_real = grad_out.real() + grad_x_next_real;
      value_t grad_x_new_imag = grad_out.imag() + grad_x_next_imag;
      value_t grad_w_new_real = grad_w_next_real;
      value_t grad_w_new_imag = grad_w_next_imag;

      const value_t a_t = A.load(t, b, m);
      const value_t g_t = G.load(t, b, m);
      const value_t step_raw = step.load(t, b, m);
      const value_t dt = clamp_step(step_raw);
      const value_t step_mask = (step_raw > value_t(kDtMin) && step_raw < value_t(kDtMax)) ? value_t(1) : value_t(0);

      const scalar_t bu_val = bu.load(t, b, m);
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      value_t grad_step_local = value_t(0);
      value_t grad_A_local = value_t(0);
      value_t grad_G_local = value_t(0);

      grad_w_new_real += grad_x_new_real;
      grad_w_new_imag += grad_x_new_imag;
      value_t grad_x_prev_real = grad_x_new_real;
      value_t grad_x_prev_imag = grad_x_new_imag;

      const value_t S_raw = value_t(1) + dt * g_t;
      const value_t S = clamp_stability(S_raw);
      const value_t clamp_mask = S_raw > value_t(kClampMin) ? value_t(1) : value_t(0);
      const value_t inv_S = value_t(1) / S;
      const value_t inv_S_sq = inv_S * inv_S;

      const value_t comb_real = -a_t * x_prev_real + bu_real;
      const value_t comb_imag = -a_t * x_prev_imag + bu_imag;

      const value_t tmpw_real = w_prev_real + (dt * dt) * comb_real;
      const value_t tmpw_imag = w_prev_imag + (dt * dt) * comb_imag;

      const value_t grad_tmpw_real = grad_w_new_real * inv_S;
      const value_t grad_tmpw_imag = grad_w_new_imag * inv_S;

      const value_t grad_S = -(grad_w_new_real * tmpw_real + grad_w_new_imag * tmpw_imag) * inv_S_sq;
      const value_t grad_S_raw = grad_S * clamp_mask;

      grad_step_local += grad_S_raw * g_t;
      grad_G_local += grad_S_raw * dt;

      grad_step_local += grad_tmpw_real * (value_t(2) * dt * comb_real) +
                         grad_tmpw_imag * (value_t(2) * dt * comb_imag);
      grad_A_local += grad_tmpw_real * (-(dt * dt) * x_prev_real) +
                      grad_tmpw_imag * (-(dt * dt) * x_prev_imag);

      grad_x_prev_real += grad_tmpw_real * (-(dt * dt) * a_t);
      grad_x_prev_imag += grad_tmpw_imag * (-(dt * dt) * a_t);

      const value_t grad_w_prev_real = grad_tmpw_real;
      const value_t grad_w_prev_imag = grad_tmpw_imag;

      const value_t grad_bu_real = grad_tmpw_real * (dt * dt);
      const value_t grad_bu_imag = grad_tmpw_imag * (dt * dt);

      grad_A.store(t, b, m, grad_A_local);
      grad_G.store(t, b, m, grad_G_local);
      grad_step.store(t, b, m, grad_step_local * step_mask);
      grad_bu_ptr[offset] = scalar_t(grad_bu_real, grad_bu_imag);

      grad_x_next_real = grad_x_prev_real;
      grad_x_next_imag = grad_x_prev_imag;
      grad_w_next_real = grad_w_prev_real;
      grad_w_next_imag = grad_w_prev_imag;
    }
  }
}

}  // namespace

void sdlinoss_imex1_forward_scan_cuda(const at::Tensor& A,
                                      const at::Tensor& G,
                                      const at::Tensor& step,
                                      const at::Tensor& bu,
                                      at::Tensor& tmp_states,
                                      at::Tensor& tile_summ) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};
  const auto series = bu.size(1) * bu.size(2);
  const auto ntiles = (bu.size(0) + kTile - 1) / kTile;

  if (bu.size(0) == 0 || series == 0 || ntiles == 0) {
    return;
  }

  dim3 grid(series, ntiles, 1);
  dim3 block(kThreads, 1, 1);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_imex1_forward_tile", [&] {
    using real_t = typename ComplexTraits<scalar_t>::real_t;
    sdlinoss_imex1_forward_tile_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<real_t>(),
        G.data_ptr<real_t>(),
        step.data_ptr<real_t>(),
        bu.data_ptr<scalar_t>(),
        tmp_states.data_ptr<scalar_t>(),
        reinterpret_cast<Pair2x2<real_t, scalar_t>*>(tile_summ.data_ptr()),
        bu.size(0),
        bu.size(1),
        bu.size(2));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sdlinoss_imex1_forward_cuda(const at::Tensor& A,
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

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_imex1_forward_cuda", [&] {
    const auto length = bu.size(0);
    const auto A_strided = make_strided3<typename scalar_t::value_type>(A, length, batch, ssm);
    const auto G_strided = make_strided3<typename scalar_t::value_type>(G, length, batch, ssm);
    const auto step_strided = make_strided3<typename scalar_t::value_type>(step, length, batch, ssm);
    const auto bu_strided = make_strided3<scalar_t>(bu, length, batch, ssm);
    sdlinoss_imex1_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    const auto length = bu.size(0);
    const auto A_strided = make_strided3<typename scalar_t::value_type>(A, length, batch, ssm);
    const auto G_strided = make_strided3<typename scalar_t::value_type>(G, length, batch, ssm);
    const auto step_strided = make_strided3<typename scalar_t::value_type>(step, length, batch, ssm);
    const auto bu_strided = make_strided3<scalar_t>(bu, length, batch, ssm);
    auto grad_A_strided = make_grad_strided3<typename scalar_t::value_type>(grad_A, length, batch, ssm);
    auto grad_G_strided = make_grad_strided3<typename scalar_t::value_type>(grad_G, length, batch, ssm);
    auto grad_step_strided = make_grad_strided3<typename scalar_t::value_type>(grad_step, length, batch, ssm);
    sdlinoss_imex1_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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

