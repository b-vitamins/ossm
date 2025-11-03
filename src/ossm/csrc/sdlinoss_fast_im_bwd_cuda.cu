#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cmath>

#include "dlinoss_common.h"
#include "sdlinoss_fast_common.h"
#include "sdlinoss_fast_dispatch.h"

#ifndef OSSM_FAST_UNROLL
#define OSSM_FAST_UNROLL 2
#endif

__device__ __forceinline__ float _fma_val(float a, float b, float c) {
  return ::fmaf(a, b, c);
}

__device__ __forceinline__ double _fma_val(double a, double b, double c) {
  return ::fma(a, b, c);
}

namespace ossm {
namespace {

template <bool VL, bool VB, bool VM, typename T>
__device__ __forceinline__ T bload(const T* __restrict__ base,
                                   int64_t t,
                                   int64_t b,
                                   int64_t m,
                                   int64_t sL,
                                   int64_t sB,
                                   int64_t sM) {
  const int64_t offset = (VL ? t * sL : int64_t(0)) + (VB ? b * sB : int64_t(0)) +
                         (VM ? m * sM : int64_t(0));
  return base[offset];
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
__global__ void im_backward_kernel(
    const typename scalar_t::value_type* __restrict__ A,
    int64_t AsL,
    int64_t AsB,
    int64_t AsM,
    const typename scalar_t::value_type* __restrict__ G,
    int64_t GsL,
    int64_t GsB,
    int64_t GsM,
    const typename scalar_t::value_type* __restrict__ step,
    int64_t SsL,
    int64_t SsB,
    int64_t SsM,
    const scalar_t* __restrict__ bu,
    int64_t BusL,
    int64_t BusB,
    int64_t BusM,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_bu,
    GradStrided3<typename scalar_t::value_type> grad_A,
    GradStrided3<typename scalar_t::value_type> grad_G,
    GradStrided3<typename scalar_t::value_type> grad_step,
    int64_t L,
    int64_t series,
    int64_t Mdim) {
  using value_t = typename scalar_t::value_type;

  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= series) {
    return;
  }

  const int64_t b = idx / Mdim;
  const int64_t m = idx % Mdim;
  const int64_t num_tiles = (L + TILE - 1) / TILE;

  const value_t dt_min = static_cast<value_t>(kDtMin);
  const value_t dt_max = static_cast<value_t>(kDtMax);
  const value_t clamp_min = static_cast<value_t>(kClampMin);

  value_t grad_w_next_real = value_t(0);
  value_t grad_w_next_imag = value_t(0);
  value_t grad_x_next_real = value_t(0);
  value_t grad_x_next_imag = value_t(0);

  for (int64_t tile = num_tiles - 1; tile >= 0; --tile) {
    const int64_t start = tile * TILE;
    if (start >= L) {
      continue;
    }
    const int64_t tile_len = std::min<int64_t>(TILE, L - start);

    for (int64_t k = tile_len - 1; k >= 0; --k) {
      const int64_t t = start + k;
      const int64_t offset_series = t * series + idx;
      const int64_t offset_state = offset_series * 2;

      const scalar_t state_w = states[offset_state];
      const scalar_t state_x = states[offset_state + 1];
      (void)state_w;
      (void)state_x;

      const scalar_t prev_w =
          (t > 0) ? states[(offset_series - series) * 2] : scalar_t(0);
      const scalar_t prev_x =
          (t > 0) ? states[(offset_series - series) * 2 + 1] : scalar_t(0);

      const value_t w_prev_real = prev_w.real();
      const value_t w_prev_imag = prev_w.imag();
      const value_t x_prev_real = prev_x.real();
      const value_t x_prev_imag = prev_x.imag();

      const scalar_t grad_out_val = grad_out[offset_series];
      value_t grad_x_new_real = grad_out_val.real() + grad_x_next_real;
      value_t grad_x_new_imag = grad_out_val.imag() + grad_x_next_imag;
      value_t grad_w_new_real = grad_w_next_real + grad_x_new_real;
      value_t grad_w_new_imag = grad_w_next_imag + grad_x_new_imag;

      const value_t A_val =
          bload<VL, VB, VM>(A, t, b, m, AsL, AsB, AsM);
      const value_t G_val =
          bload<VL, VB, VM>(G, t, b, m, GsL, GsB, GsM);
      const value_t step_raw =
          bload<VL, VB, VM>(step, t, b, m, SsL, SsB, SsM);
      const value_t dt = step_raw < dt_min ? dt_min : (step_raw > dt_max ? dt_max : step_raw);
      const value_t step_mask =
          (step_raw > dt_min && step_raw < dt_max) ? value_t(1) : value_t(0);
      const value_t dt2 = dt * dt;

      const scalar_t bu_val =
          bload<VL, VB, VM>(bu, t, b, m, BusL, BusB, BusM);
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      const value_t S_raw = value_t(1) + dt * G_val + dt2 * A_val;
      const value_t S = S_raw < clamp_min ? clamp_min : S_raw;
      const value_t clamp_mask = S_raw > clamp_min ? value_t(1) : value_t(0);
      const value_t inv_S = value_t(1) / S;
      const value_t inv_S_sq = inv_S * inv_S;

      const value_t comb_real = -A_val * x_prev_real + bu_real;
      const value_t comb_imag = -A_val * x_prev_imag + bu_imag;

      const value_t tmpw_real = _fma_val(dt2, comb_real, w_prev_real);
      const value_t tmpw_imag = _fma_val(dt2, comb_imag, w_prev_imag);

      const value_t grad_tmpw_real = grad_w_new_real * inv_S;
      const value_t grad_tmpw_imag = grad_w_new_imag * inv_S;

      const value_t grad_S_inner =
          _fma_val(grad_w_new_imag, tmpw_imag, grad_w_new_real * tmpw_real);
      const value_t grad_S = -grad_S_inner * inv_S_sq;
      const value_t grad_S_raw = grad_S * clamp_mask;

      value_t grad_step_local = value_t(0);
      value_t grad_A_local = value_t(0);
      value_t grad_G_local = value_t(0);

      const value_t step_grad_term = _fma_val(value_t(2) * dt, A_val, G_val);
      grad_step_local = _fma_val(grad_S_raw, step_grad_term, grad_step_local);
      grad_A_local = _fma_val(grad_S_raw, dt2, grad_A_local);
      grad_G_local = _fma_val(grad_S_raw, dt, grad_G_local);

      const value_t tmp_step_real = value_t(2) * dt * comb_real;
      const value_t tmp_step_imag = value_t(2) * dt * comb_imag;
      grad_step_local = _fma_val(grad_tmpw_real, tmp_step_real, grad_step_local);
      grad_step_local = _fma_val(grad_tmpw_imag, tmp_step_imag, grad_step_local);
      const value_t grad_A_tmp_real = -(dt2) * x_prev_real;
      const value_t grad_A_tmp_imag = -(dt2) * x_prev_imag;
      grad_A_local = _fma_val(grad_tmpw_real, grad_A_tmp_real, grad_A_local);
      grad_A_local = _fma_val(grad_tmpw_imag, grad_A_tmp_imag, grad_A_local);

      value_t grad_x_prev_real = grad_x_new_real;
      value_t grad_x_prev_imag = grad_x_new_imag;
      grad_x_prev_real =
          _fma_val(-(dt2) * A_val, grad_tmpw_real, grad_x_prev_real);
      grad_x_prev_imag =
          _fma_val(-(dt2) * A_val, grad_tmpw_imag, grad_x_prev_imag);

      const value_t grad_w_prev_real = grad_tmpw_real;
      const value_t grad_w_prev_imag = grad_tmpw_imag;

      const value_t grad_bu_real = grad_tmpw_real * dt2;
      const value_t grad_bu_imag = grad_tmpw_imag * dt2;

      grad_A.store(t, b, m, grad_A_local);
      grad_G.store(t, b, m, grad_G_local);
      grad_step.store(t, b, m, grad_step_local * step_mask);
      grad_bu[offset_series] = scalar_t(grad_bu_real, grad_bu_imag);

      grad_w_next_real = grad_w_prev_real;
      grad_w_next_imag = grad_w_prev_imag;
      grad_x_next_real = grad_x_prev_real;
      grad_x_next_imag = grad_x_prev_imag;
    }
  }
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
void launch_backward_kernel(const Strides3& A_stride,
                            const Strides3& G_stride,
                            const Strides3& step_stride,
                            const Strides3& bu_stride,
                            const typename scalar_t::value_type* A,
                            const typename scalar_t::value_type* G,
                            const typename scalar_t::value_type* step,
                            const scalar_t* bu,
                            const scalar_t* states,
                            const scalar_t* grad_out,
                            scalar_t* grad_bu,
                            GradStrided3<typename scalar_t::value_type> grad_A,
                            GradStrided3<typename scalar_t::value_type> grad_G,
                            GradStrided3<typename scalar_t::value_type> grad_step,
                            int64_t L,
                            int64_t series,
                            int64_t Mdim,
                            dim3 grid,
                            dim3 block,
                            cudaStream_t stream) {
  im_backward_kernel<scalar_t, VL, VB, VM, TILE><<<grid, block, 0, stream>>>(
      A,
      A_stride.sL,
      A_stride.sB,
      A_stride.sM,
      G,
      G_stride.sL,
      G_stride.sB,
      G_stride.sM,
      step,
      step_stride.sL,
      step_stride.sB,
      step_stride.sM,
      bu,
      bu_stride.sL,
      bu_stride.sB,
      bu_stride.sM,
      states,
      grad_out,
      grad_bu,
      grad_A,
      grad_G,
      grad_step,
      L,
      series,
      Mdim);
}

template <typename scalar_t, int TILE>
void dispatch_backward(int vary_mask,
                       const Strides3& A_stride,
                       const Strides3& G_stride,
                       const Strides3& step_stride,
                       const Strides3& bu_stride,
                       const at::Tensor& A,
                       const at::Tensor& G,
                       const at::Tensor& step,
                       const at::Tensor& bu,
                       const at::Tensor& states,
                       const at::Tensor& grad_out,
                       at::Tensor& grad_A,
                       at::Tensor& grad_G,
                       at::Tensor& grad_step,
                       at::Tensor& grad_bu,
                       int64_t length,
                       int64_t batch,
                       int64_t ssm,
                       cudaStream_t stream) {
  const int64_t series = batch * ssm;
  const dim3 block(256);
  const dim3 grid((series + block.x - 1) / block.x);

  using value_t = typename scalar_t::value_type;

  const value_t* A_ptr = A.data_ptr<value_t>();
  const value_t* G_ptr = G.data_ptr<value_t>();
  const value_t* step_ptr = step.data_ptr<value_t>();
  const scalar_t* bu_ptr = bu.data_ptr<scalar_t>();
  const scalar_t* states_ptr = states.data_ptr<scalar_t>();
  const scalar_t* grad_out_ptr = grad_out.data_ptr<scalar_t>();
  scalar_t* grad_bu_ptr = grad_bu.data_ptr<scalar_t>();

  auto grad_A_desc = make_grad_strided3<value_t>(grad_A, length, batch, ssm);
  auto grad_G_desc = make_grad_strided3<value_t>(grad_G, length, batch, ssm);
  auto grad_step_desc = make_grad_strided3<value_t>(grad_step, length, batch, ssm);

  switch (vary_mask) {
    case 0:
      launch_backward_kernel<scalar_t, false, false, false, TILE>(A_stride,
                                                                  G_stride,
                                                                  step_stride,
                                                                  bu_stride,
                                                                  A_ptr,
                                                                  G_ptr,
                                                                  step_ptr,
                                                                  bu_ptr,
                                                                  states_ptr,
                                                                  grad_out_ptr,
                                                                  grad_bu_ptr,
                                                                  grad_A_desc,
                                                                  grad_G_desc,
                                                                  grad_step_desc,
                                                                  length,
                                                                  series,
                                                                  ssm,
                                                                  grid,
                                                                  block,
                                                                  stream);
      break;
    case 1:
      launch_backward_kernel<scalar_t, true, false, false, TILE>(A_stride,
                                                                 G_stride,
                                                                 step_stride,
                                                                 bu_stride,
                                                                 A_ptr,
                                                                 G_ptr,
                                                                 step_ptr,
                                                                 bu_ptr,
                                                                 states_ptr,
                                                                 grad_out_ptr,
                                                                 grad_bu_ptr,
                                                                 grad_A_desc,
                                                                 grad_G_desc,
                                                                 grad_step_desc,
                                                                 length,
                                                                 series,
                                                                 ssm,
                                                                 grid,
                                                                 block,
                                                                 stream);
      break;
    case 2:
      launch_backward_kernel<scalar_t, false, true, false, TILE>(A_stride,
                                                                 G_stride,
                                                                 step_stride,
                                                                 bu_stride,
                                                                 A_ptr,
                                                                 G_ptr,
                                                                 step_ptr,
                                                                 bu_ptr,
                                                                 states_ptr,
                                                                 grad_out_ptr,
                                                                 grad_bu_ptr,
                                                                 grad_A_desc,
                                                                 grad_G_desc,
                                                                 grad_step_desc,
                                                                 length,
                                                                 series,
                                                                 ssm,
                                                                 grid,
                                                                 block,
                                                                 stream);
      break;
    case 3:
      launch_backward_kernel<scalar_t, true, true, false, TILE>(A_stride,
                                                                G_stride,
                                                                step_stride,
                                                                bu_stride,
                                                                A_ptr,
                                                                G_ptr,
                                                                step_ptr,
                                                                bu_ptr,
                                                                states_ptr,
                                                                grad_out_ptr,
                                                                grad_bu_ptr,
                                                                grad_A_desc,
                                                                grad_G_desc,
                                                                grad_step_desc,
                                                                length,
                                                                series,
                                                                ssm,
                                                                grid,
                                                                block,
                                                                stream);
      break;
    case 4:
      launch_backward_kernel<scalar_t, false, false, true, TILE>(A_stride,
                                                                 G_stride,
                                                                 step_stride,
                                                                 bu_stride,
                                                                 A_ptr,
                                                                 G_ptr,
                                                                 step_ptr,
                                                                 bu_ptr,
                                                                 states_ptr,
                                                                 grad_out_ptr,
                                                                 grad_bu_ptr,
                                                                 grad_A_desc,
                                                                 grad_G_desc,
                                                                 grad_step_desc,
                                                                 length,
                                                                 series,
                                                                 ssm,
                                                                 grid,
                                                                 block,
                                                                 stream);
      break;
    case 5:
      launch_backward_kernel<scalar_t, true, false, true, TILE>(A_stride,
                                                                G_stride,
                                                                step_stride,
                                                                bu_stride,
                                                                A_ptr,
                                                                G_ptr,
                                                                step_ptr,
                                                                bu_ptr,
                                                                states_ptr,
                                                                grad_out_ptr,
                                                                grad_bu_ptr,
                                                                grad_A_desc,
                                                                grad_G_desc,
                                                                grad_step_desc,
                                                                length,
                                                                series,
                                                                ssm,
                                                                grid,
                                                                block,
                                                                stream);
      break;
    case 6:
      launch_backward_kernel<scalar_t, false, true, true, TILE>(A_stride,
                                                                G_stride,
                                                                step_stride,
                                                                bu_stride,
                                                                A_ptr,
                                                                G_ptr,
                                                                step_ptr,
                                                                bu_ptr,
                                                                states_ptr,
                                                                grad_out_ptr,
                                                                grad_bu_ptr,
                                                                grad_A_desc,
                                                                grad_G_desc,
                                                                grad_step_desc,
                                                                length,
                                                                series,
                                                                ssm,
                                                                grid,
                                                                block,
                                                                stream);
      break;
    default:
      launch_backward_kernel<scalar_t, true, true, true, TILE>(A_stride,
                                                               G_stride,
                                                               step_stride,
                                                               bu_stride,
                                                               A_ptr,
                                                               G_ptr,
                                                               step_ptr,
                                                               bu_ptr,
                                                               states_ptr,
                                                               grad_out_ptr,
                                                               grad_bu_ptr,
                                                               grad_A_desc,
                                                               grad_G_desc,
                                                               grad_step_desc,
                                                               length,
                                                               series,
                                                               ssm,
                                                               grid,
                                                               block,
                                                               stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void run_backward_dispatch(int tile,
                           int vary_mask,
                           const Strides3& A_stride,
                           const Strides3& G_stride,
                           const Strides3& step_stride,
                           const Strides3& bu_stride,
                           const at::Tensor& A,
                           const at::Tensor& G,
                           const at::Tensor& step,
                           const at::Tensor& bu,
                           const at::Tensor& states,
                           const at::Tensor& grad_out,
                           at::Tensor& grad_A,
                           at::Tensor& grad_G,
                           at::Tensor& grad_step,
                           at::Tensor& grad_bu,
                           int64_t length,
                           int64_t batch,
                           int64_t ssm,
                           cudaStream_t stream) {
  switch (tile) {
    case 64:
      dispatch_backward<scalar_t, 64>(vary_mask,
                                      A_stride,
                                      G_stride,
                                      step_stride,
                                      bu_stride,
                                      A,
                                      G,
                                      step,
                                      bu,
                                      states,
                                      grad_out,
                                      grad_A,
                                      grad_G,
                                      grad_step,
                                      grad_bu,
                                      length,
                                      batch,
                                      ssm,
                                      stream);
      break;
    case 256:
      dispatch_backward<scalar_t, 256>(vary_mask,
                                       A_stride,
                                       G_stride,
                                       step_stride,
                                       bu_stride,
                                       A,
                                       G,
                                       step,
                                       bu,
                                       states,
                                       grad_out,
                                       grad_A,
                                       grad_G,
                                       grad_step,
                                       grad_bu,
                                       length,
                                       batch,
                                       ssm,
                                       stream);
      break;
    default:
      dispatch_backward<scalar_t, 128>(vary_mask,
                                       A_stride,
                                       G_stride,
                                       step_stride,
                                       bu_stride,
                                       A,
                                       G,
                                       step,
                                       bu,
                                       states,
                                       grad_out,
                                       grad_A,
                                       grad_G,
                                       grad_step,
                                       grad_bu,
                                       length,
                                       batch,
                                       ssm,
                                       stream);
      break;
  }
}

}  // namespace

void sdlinoss_fast_im_backward_cuda_complex64(int tile,
                                              int vary_mask,
                                              const Strides3& A_stride,
                                              const Strides3& G_stride,
                                              const Strides3& step_stride,
                                              const Strides3& bu_stride,
                                              const at::Tensor& A,
                                              const at::Tensor& G,
                                              const at::Tensor& step,
                                              const at::Tensor& bu,
                                              const at::Tensor& states,
                                              const at::Tensor& grad_out,
                                              at::Tensor& grad_A,
                                              at::Tensor& grad_G,
                                              at::Tensor& grad_step,
                                              at::Tensor& grad_bu,
                                              int64_t length,
                                              int64_t batch,
                                              int64_t ssm,
                                              cudaStream_t stream) {
  run_backward_dispatch<c10::complex<float>>(tile,
                                             vary_mask,
                                             A_stride,
                                             G_stride,
                                             step_stride,
                                             bu_stride,
                                             A,
                                             G,
                                             step,
                                             bu,
                                             states,
                                             grad_out,
                                             grad_A,
                                             grad_G,
                                             grad_step,
                                             grad_bu,
                                             length,
                                             batch,
                                             ssm,
                                             stream);
}

void sdlinoss_fast_im_backward_cuda_complex128(int tile,
                                               int vary_mask,
                                               const Strides3& A_stride,
                                               const Strides3& G_stride,
                                               const Strides3& step_stride,
                                               const Strides3& bu_stride,
                                               const at::Tensor& A,
                                               const at::Tensor& G,
                                               const at::Tensor& step,
                                               const at::Tensor& bu,
                                               const at::Tensor& states,
                                               const at::Tensor& grad_out,
                                               at::Tensor& grad_A,
                                               at::Tensor& grad_G,
                                               at::Tensor& grad_step,
                                               at::Tensor& grad_bu,
                                               int64_t length,
                                               int64_t batch,
                                               int64_t ssm,
                                               cudaStream_t stream) {
  run_backward_dispatch<c10::complex<double>>(tile,
                                              vary_mask,
                                              A_stride,
                                              G_stride,
                                              step_stride,
                                              bu_stride,
                                              A,
                                              G,
                                              step,
                                              bu,
                                              states,
                                              grad_out,
                                              grad_A,
                                              grad_G,
                                              grad_step,
                                              grad_bu,
                                              length,
                                              batch,
                                              ssm,
                                              stream);
}

}  // namespace ossm

