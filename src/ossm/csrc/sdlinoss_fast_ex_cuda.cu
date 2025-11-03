#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cmath>

#include "sdlinoss_fast_common.h"
#include "sdlinoss_fast_dispatch.h"

#ifndef OSSM_FAST_UNROLL
#define OSSM_FAST_UNROLL 2
#endif

#if !defined(OSSM_FAST_PREFETCH) && defined(OSSM_SDLINOSS_FAST_PREFETCH)
#define OSSM_FAST_PREFETCH OSSM_SDLINOSS_FAST_PREFETCH
#endif
#if defined(OSSM_FAST_PREFETCH) && OSSM_FAST_PREFETCH
#define OSSM_PREFETCH 1
#else
#define OSSM_PREFETCH 0
#endif

#ifndef OSSM_STRINGIFY_HELPER
#define OSSM_STRINGIFY_HELPER(x) #x
#define OSSM_STRINGIFY(x) OSSM_STRINGIFY_HELPER(x)
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

#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
template <int BYTES>
__device__ __forceinline__ void cp_async_ca(void* dst, const void* src) {
  static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                "cp.async size must be 4, 8, or 16 bytes");
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :
                   : "r"(dst), "l"(src), "n"(BYTES));
}
#endif

template <typename scalar_t>
size_t shared_bytes_prefetch_for_block(int threads) {
#if defined(OSSM_FAST_PREFETCH)
  const int warps = std::max(1, threads >> 5);
  const size_t per_warp =
      32 * sizeof(typename scalar_t::value_type) * 3 + 32 * sizeof(scalar_t);
  return static_cast<size_t>(warps) * per_warp;
#else
  (void)threads;
  return 0;
#endif
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
__global__ void ex_build_tile_summaries_kernel(
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
    typename scalar_t::value_type* __restrict__ M00,
    typename scalar_t::value_type* __restrict__ M01,
    typename scalar_t::value_type* __restrict__ M10,
    typename scalar_t::value_type* __restrict__ M11,
    scalar_t* __restrict__ F0,
    scalar_t* __restrict__ F1,
    int64_t L,
    int64_t series,
    int64_t Mdim,
    int64_t num_tiles) {
  using value_t = typename scalar_t::value_type;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= series) {
    return;
  }

  const int64_t b = idx / Mdim;
  const int64_t m = idx % Mdim;

  for (int64_t tile = 0; tile < num_tiles; ++tile) {
    const int64_t start = tile * TILE;
    if (start >= L) {
      break;
    }
    const int64_t tile_len = std::min<int64_t>(TILE, L - start);

    FastPair<value_t> P{};
    P.m00 = value_t(1);
    P.m01 = value_t(0);
    P.m10 = value_t(0);
    P.m11 = value_t(1);
    P.f0 = scalar_t(0);
    P.f1 = scalar_t(0);

    #pragma unroll OSSM_FAST_UNROLL
    for (int64_t k = 0; k < tile_len; ++k) {
      const int64_t t = start + k;
      const value_t A_val =
          bload<VL, VB, VM>(A, t, b, m, AsL, AsB, AsM);
      const value_t G_val =
          bload<VL, VB, VM>(G, t, b, m, GsL, GsB, GsM);
      const value_t dt =
          bload<VL, VB, VM>(step, t, b, m, SsL, SsB, SsM);

      value_t alpha, beta, gamma;
      step_coeff_ex<value_t>(A_val, G_val, dt, alpha, beta, gamma);

      FastPair<value_t> Q{};
      Q.m00 = alpha;
      Q.m01 = beta;
      Q.m10 = alpha;
      Q.m11 = value_t(1) + beta;
      const scalar_t add =
          bload<VL, VB, VM>(bu, t, b, m, BusL, BusB, BusM) * gamma;
      Q.f0 = add;
      Q.f1 = add;

      P = combine(Q, P);
    }

    const int64_t out_offset = tile * series + idx;
    M00[out_offset] = P.m00;
    M01[out_offset] = P.m01;
    M10[out_offset] = P.m10;
    M11[out_offset] = P.m11;
    F0[out_offset] = P.f0;
    F1[out_offset] = P.f1;
  }
}

template <typename scalar_t, int TILE>
__global__ void ex_prefix_tiles_per_series_kernel(
    const typename scalar_t::value_type* __restrict__ M00,
    const typename scalar_t::value_type* __restrict__ M01,
    const typename scalar_t::value_type* __restrict__ M10,
    const typename scalar_t::value_type* __restrict__ M11,
    const scalar_t* __restrict__ F0,
    const scalar_t* __restrict__ F1,
    scalar_t* __restrict__ S0_w,
    scalar_t* __restrict__ S0_x,
    int64_t series,
    int64_t num_tiles) {
  using value_t = typename scalar_t::value_type;
  const int64_t idx = blockIdx.x;
  if (idx >= series) {
    return;
  }

  FastPair<value_t> prefix{};
  prefix.m00 = value_t(1);
  prefix.m01 = value_t(0);
  prefix.m10 = value_t(0);
  prefix.m11 = value_t(1);
  prefix.f0 = scalar_t(0);
  prefix.f1 = scalar_t(0);

  for (int64_t tile = 0; tile < num_tiles; ++tile) {
    const int64_t offset = tile * series + idx;
    S0_w[offset] = prefix.f0;
    S0_x[offset] = prefix.f1;

    FastPair<value_t> current{};
    current.m00 = M00[offset];
    current.m01 = M01[offset];
    current.m10 = M10[offset];
    current.m11 = M11[offset];
    current.f0 = F0[offset];
    current.f1 = F1[offset];

    prefix = combine(current, prefix);
  }
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
__global__ void ex_expand_tiles_write_states_kernel(
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
    const scalar_t* __restrict__ S0_w,
    const scalar_t* __restrict__ S0_x,
    scalar_t* __restrict__ out,
    int64_t L,
    int64_t series,
    int64_t Mdim,
    int64_t num_tiles) {
  using value_t = typename scalar_t::value_type;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= series) {
    return;
  }

  const int64_t b = idx / Mdim;
  const int64_t m = idx % Mdim;

#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  extern __shared__ __align__(16) unsigned char smem_u8[];
  constexpr int BYTES_VAL = sizeof(value_t);
  constexpr int BYTES_BU = sizeof(scalar_t);
  constexpr int BYTES_PER_WARP =
      32 * BYTES_VAL * 3 + 32 * BYTES_BU;
  unsigned char* base = smem_u8 + warp * BYTES_PER_WARP;
  value_t* shA = reinterpret_cast<value_t*>(base + 0);
  value_t* shG = reinterpret_cast<value_t*>(base + 32 * BYTES_VAL);
  value_t* shS = reinterpret_cast<value_t*>(base + 64 * BYTES_VAL);
  scalar_t* shBU =
      reinterpret_cast<scalar_t*>(base + 96 * BYTES_VAL);
#endif

  for (int64_t tile = 0; tile < num_tiles; ++tile) {
    const int64_t start = tile * TILE;
    if (start >= L) {
      break;
    }
    const int64_t tile_len = std::min<int64_t>(TILE, L - start);

    const int64_t base_offset = tile * series + idx;
    scalar_t w = S0_w[base_offset];
    scalar_t x = S0_x[base_offset];

    scalar_t* out_ptr = out + (start * series + idx) * 2;

#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
    if (tile_len > 0) {
      const int64_t t0 = start;
      const int64_t oA = (VL ? t0 * AsL : int64_t(0)) +
                         (VB ? b * AsB : int64_t(0)) +
                         (VM ? m * AsM : int64_t(0));
      const int64_t oG = (VL ? t0 * GsL : int64_t(0)) +
                         (VB ? b * GsB : int64_t(0)) +
                         (VM ? m * GsM : int64_t(0));
      const int64_t oS = (VL ? t0 * SsL : int64_t(0)) +
                         (VB ? b * SsB : int64_t(0)) +
                         (VM ? m * SsM : int64_t(0));
      const int64_t oBU = (VL ? t0 * BusL : int64_t(0)) +
                          (VB ? b * BusB : int64_t(0)) +
                          (VM ? m * BusM : int64_t(0));

      cp_async_ca<sizeof(value_t)>(&shA[lane], &A[oA]);
      cp_async_ca<sizeof(value_t)>(&shG[lane], &G[oG]);
      cp_async_ca<sizeof(value_t)>(&shS[lane], &step[oS]);
      cp_async_ca<sizeof(scalar_t)>(&shBU[lane], &bu[oBU]);
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 0;\n");
      __syncwarp();
    }
#endif

    #pragma unroll OSSM_FAST_UNROLL
    for (int64_t k = 0; k < tile_len; ++k) {
      const int64_t t = start + k;
      value_t A_val;
      value_t G_val;
      value_t dt;
      scalar_t bu_val;
#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
      A_val = shA[lane];
      G_val = shG[lane];
      dt = shS[lane];
      bu_val = shBU[lane];
#else
      A_val = bload<VL, VB, VM>(A, t, b, m, AsL, AsB, AsM);
      G_val = bload<VL, VB, VM>(G, t, b, m, GsL, GsB, GsM);
      dt = bload<VL, VB, VM>(step, t, b, m, SsL, SsB, SsM);
      bu_val = bload<VL, VB, VM>(bu, t, b, m, BusL, BusB, BusM);
#endif

      value_t alpha, beta, gamma;
      step_coeff_ex<value_t>(A_val, G_val, dt, alpha, beta, gamma);

      const value_t w_real = w.real();
      const value_t w_imag = w.imag();
      const value_t x_real = x.real();
      const value_t x_imag = x.imag();
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      const value_t tmp_real = _fma_val(beta, x_real, alpha * w_real);
      const value_t tmp_imag = _fma_val(beta, x_imag, alpha * w_imag);
      const value_t w_new_real = _fma_val(gamma, bu_real, tmp_real);
      const value_t w_new_imag = _fma_val(gamma, bu_imag, tmp_imag);

      const scalar_t w_new(w_new_real, w_new_imag);
      const scalar_t x_new(x_real + w_new_real, x_imag + w_new_imag);

      out_ptr[0] = w_new;
      out_ptr[1] = x_new;

      w = w_new;
      x = x_new;
      out_ptr += series * 2;

#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
      if (k + 1 < tile_len) {
        const int64_t t_next = t + 1;
        const int64_t oA_next = (VL ? t_next * AsL : int64_t(0)) +
                                 (VB ? b * AsB : int64_t(0)) +
                                 (VM ? m * AsM : int64_t(0));
        const int64_t oG_next = (VL ? t_next * GsL : int64_t(0)) +
                                 (VB ? b * GsB : int64_t(0)) +
                                 (VM ? m * GsM : int64_t(0));
        const int64_t oS_next = (VL ? t_next * SsL : int64_t(0)) +
                                 (VB ? b * SsB : int64_t(0)) +
                                 (VM ? m * SsM : int64_t(0));
        const int64_t oBU_next = (VL ? t_next * BusL : int64_t(0)) +
                                  (VB ? b * BusB : int64_t(0)) +
                                  (VM ? m * BusM : int64_t(0));

        cp_async_ca<sizeof(value_t)>(&shA[lane], &A[oA_next]);
        cp_async_ca<sizeof(value_t)>(&shG[lane], &G[oG_next]);
        cp_async_ca<sizeof(value_t)>(&shS[lane], &step[oS_next]);
        cp_async_ca<sizeof(scalar_t)>(&shBU[lane], &bu[oBU_next]);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncwarp();
      }
#endif
    }
  }
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
__global__ void ex_expand_tiles_write_x_kernel(
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
    const scalar_t* __restrict__ S0_w,
    const scalar_t* __restrict__ S0_x,
    scalar_t* __restrict__ out_x,
    int64_t L,
    int64_t series,
    int64_t Mdim,
    int64_t num_tiles) {
  using value_t = typename scalar_t::value_type;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= series) {
    return;
  }

  const int64_t b = idx / Mdim;
  const int64_t m = idx % Mdim;

#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  extern __shared__ __align__(16) unsigned char smem_u8[];
  constexpr int BYTES_VAL = sizeof(value_t);
  constexpr int BYTES_BU = sizeof(scalar_t);
  constexpr int BYTES_PER_WARP = 32 * BYTES_VAL * 3 + 32 * BYTES_BU;
  unsigned char* base = smem_u8 + warp * BYTES_PER_WARP;
  value_t* shA = reinterpret_cast<value_t*>(base + 0);
  value_t* shG = reinterpret_cast<value_t*>(base + 32 * BYTES_VAL);
  value_t* shS = reinterpret_cast<value_t*>(base + 64 * BYTES_VAL);
  scalar_t* shBU = reinterpret_cast<scalar_t*>(base + 96 * BYTES_VAL);
#endif

  for (int64_t tile = 0; tile < num_tiles; ++tile) {
    const int64_t start = tile * TILE;
    if (start >= L) {
      break;
    }
    const int64_t tile_len = std::min<int64_t>(TILE, L - start);

    const int64_t base_offset = tile * series + idx;
    scalar_t w = S0_w[base_offset];
    scalar_t x = S0_x[base_offset];
    scalar_t* out_ptr = out_x + start * series + idx;

#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
    if (tile_len > 0) {
      const int64_t t0 = start;
      const int64_t oA = (VL ? t0 * AsL : int64_t(0)) +
                         (VB ? b * AsB : int64_t(0)) +
                         (VM ? m * AsM : int64_t(0));
      const int64_t oG = (VL ? t0 * GsL : int64_t(0)) +
                         (VB ? b * GsB : int64_t(0)) +
                         (VM ? m * GsM : int64_t(0));
      const int64_t oS = (VL ? t0 * SsL : int64_t(0)) +
                         (VB ? b * SsB : int64_t(0)) +
                         (VM ? m * SsM : int64_t(0));
      const int64_t oBU = (VL ? t0 * BusL : int64_t(0)) +
                          (VB ? b * BusB : int64_t(0)) +
                          (VM ? m * BusM : int64_t(0));

      cp_async_ca<sizeof(value_t)>(&shA[lane], &A[oA]);
      cp_async_ca<sizeof(value_t)>(&shG[lane], &G[oG]);
      cp_async_ca<sizeof(value_t)>(&shS[lane], &step[oS]);
      cp_async_ca<sizeof(scalar_t)>(&shBU[lane], &bu[oBU]);
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 0;\n");
      __syncwarp();
    }
#endif

    #pragma unroll OSSM_FAST_UNROLL
    for (int64_t k = 0; k < tile_len; ++k) {
      const int64_t t = start + k;
#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
      value_t A_val = shA[lane];
      value_t G_val = shG[lane];
      value_t dt = shS[lane];
      scalar_t bu_val = shBU[lane];
#else
      const value_t A_val =
          bload<VL, VB, VM>(A, t, b, m, AsL, AsB, AsM);
      const value_t G_val =
          bload<VL, VB, VM>(G, t, b, m, GsL, GsB, GsM);
      const value_t dt =
          bload<VL, VB, VM>(step, t, b, m, SsL, SsB, SsM);
      const scalar_t bu_val =
          bload<VL, VB, VM>(bu, t, b, m, BusL, BusB, BusM);
#endif

      value_t alpha, beta, gamma;
      step_coeff_ex<value_t>(A_val, G_val, dt, alpha, beta, gamma);

      const value_t w_real = w.real();
      const value_t w_imag = w.imag();
      const value_t x_real = x.real();
      const value_t x_imag = x.imag();
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      const value_t tmp_real = _fma_val(beta, x_real, alpha * w_real);
      const value_t tmp_imag = _fma_val(beta, x_imag, alpha * w_imag);
      const value_t w_new_real = _fma_val(gamma, bu_real, tmp_real);
      const value_t w_new_imag = _fma_val(gamma, bu_imag, tmp_imag);

      const scalar_t w_new(w_new_real, w_new_imag);
      const scalar_t x_new(x_real + w_new_real, x_imag + w_new_imag);

      out_ptr[0] = x_new;

      w = w_new;
      x = x_new;
      out_ptr += series;

#if __CUDA_ARCH__ >= 800 && OSSM_PREFETCH
      if (k + 1 < tile_len) {
        const int64_t t_next = t + 1;
        const int64_t oA_next = (VL ? t_next * AsL : int64_t(0)) +
                                 (VB ? b * AsB : int64_t(0)) +
                                 (VM ? m * AsM : int64_t(0));
        const int64_t oG_next = (VL ? t_next * GsL : int64_t(0)) +
                                 (VB ? b * GsB : int64_t(0)) +
                                 (VM ? m * GsM : int64_t(0));
        const int64_t oS_next = (VL ? t_next * SsL : int64_t(0)) +
                                 (VB ? b * SsB : int64_t(0)) +
                                 (VM ? m * SsM : int64_t(0));
        const int64_t oBU_next = (VL ? t_next * BusL : int64_t(0)) +
                                  (VB ? b * BusB : int64_t(0)) +
                                  (VM ? m * BusM : int64_t(0));

        cp_async_ca<sizeof(value_t)>(&shA[lane], &A[oA_next]);
        cp_async_ca<sizeof(value_t)>(&shG[lane], &G[oG_next]);
        cp_async_ca<sizeof(value_t)>(&shS[lane], &step[oS_next]);
        cp_async_ca<sizeof(scalar_t)>(&shBU[lane], &bu[oBU_next]);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncwarp();
      }
#endif
    }
  }
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
void launch_build_kernel(const Strides3& A_stride,
                         const Strides3& G_stride,
                         const Strides3& step_stride,
                         const Strides3& bu_stride,
                         const typename scalar_t::value_type* A,
                         const typename scalar_t::value_type* G,
                         const typename scalar_t::value_type* step,
                         const scalar_t* bu,
                         typename scalar_t::value_type* M00,
                         typename scalar_t::value_type* M01,
                         typename scalar_t::value_type* M10,
                         typename scalar_t::value_type* M11,
                         scalar_t* F0,
                         scalar_t* F1,
                         int64_t L,
                         int64_t series,
                         int64_t Mdim,
                         int64_t num_tiles,
                         dim3 grid,
                         dim3 block,
                         cudaStream_t stream) {
  ex_build_tile_summaries_kernel<scalar_t, VL, VB, VM, TILE>
      <<<grid, block, 0, stream>>>(A,
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
                                   M00,
                                   M01,
                                   M10,
                                   M11,
                                   F0,
                                   F1,
                                   L,
                                   series,
                                   Mdim,
                                   num_tiles);
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
void launch_expand_kernel(const Strides3& A_stride,
                          const Strides3& G_stride,
                          const Strides3& step_stride,
                          const Strides3& bu_stride,
                          const typename scalar_t::value_type* A,
                          const typename scalar_t::value_type* G,
                          const typename scalar_t::value_type* step,
                          const scalar_t* bu,
                          const scalar_t* S0_w,
                          const scalar_t* S0_x,
                          scalar_t* states,
                          int64_t L,
                          int64_t series,
                          int64_t Mdim,
                          int64_t num_tiles,
                          dim3 grid,
                          dim3 block,
                          cudaStream_t stream) {
  size_t shared_bytes =
      shared_bytes_prefetch_for_block<scalar_t>(static_cast<int>(block.x));
  if (shared_bytes > static_cast<size_t>(48 * 1024)) {
    shared_bytes = 0;
  }

  ex_expand_tiles_write_states_kernel<scalar_t, VL, VB, VM, TILE>
      <<<grid, block, shared_bytes, stream>>>(A,
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
                                   S0_w,
                                   S0_x,
                                   states,
                                   L,
                                   series,
                                   Mdim,
                                  num_tiles);
}

template <typename scalar_t, bool VL, bool VB, bool VM, int TILE>
void launch_expand_x_kernel(const Strides3& A_stride,
                            const Strides3& G_stride,
                            const Strides3& step_stride,
                            const Strides3& bu_stride,
                            const typename scalar_t::value_type* A,
                            const typename scalar_t::value_type* G,
                            const typename scalar_t::value_type* step,
                            const scalar_t* bu,
                            const scalar_t* S0_w,
                            const scalar_t* S0_x,
                            scalar_t* out_x,
                            int64_t L,
                            int64_t series,
                            int64_t Mdim,
                            int64_t num_tiles,
                            dim3 grid,
                            dim3 block,
                            cudaStream_t stream) {
  size_t shared_bytes =
      shared_bytes_prefetch_for_block<scalar_t>(static_cast<int>(block.x));
  if (shared_bytes > static_cast<size_t>(48 * 1024)) {
    shared_bytes = 0;
  }

  ex_expand_tiles_write_x_kernel<scalar_t, VL, VB, VM, TILE>
      <<<grid, block, shared_bytes, stream>>>(A,
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
                                   S0_w,
                                   S0_x,
                                   out_x,
                                   L,
                                   series,
                                   Mdim,
                                   num_tiles);
}

template <typename scalar_t, int TILE>
void launch_prefix_kernel(const typename scalar_t::value_type* M00,
                          const typename scalar_t::value_type* M01,
                          const typename scalar_t::value_type* M10,
                          const typename scalar_t::value_type* M11,
                          const scalar_t* F0,
                          const scalar_t* F1,
                          scalar_t* S0_w,
                          scalar_t* S0_x,
                          int64_t series,
                          int64_t num_tiles,
                          cudaStream_t stream) {
  dim3 grid(series);
  dim3 block(1);
  ex_prefix_tiles_per_series_kernel<scalar_t, TILE>
      <<<grid, block, 0, stream>>>(M00, M01, M10, M11, F0, F1, S0_w, S0_x, series, num_tiles);
}

template <typename scalar_t, int TILE>
void dispatch_all(int vary_mask,
                  const Strides3& A_stride,
                  const Strides3& G_stride,
                  const Strides3& step_stride,
                  const Strides3& bu_stride,
                  const at::Tensor& A,
                  const at::Tensor& G,
                  const at::Tensor& step,
                  const at::Tensor& bu,
                  const at::Tensor& M00,
                  const at::Tensor& M01,
                  const at::Tensor& M10,
                  const at::Tensor& M11,
                  const at::Tensor& F0,
                  const at::Tensor& F1,
                  const at::Tensor& S0_w,
                  const at::Tensor& S0_x,
                  const at::Tensor& states,
                  int64_t length,
                  int64_t series,
                  int64_t ssm,
                  int64_t num_tiles,
                  cudaStream_t stream) {
  const dim3 block(256);
  const dim3 grid((series + block.x - 1) / block.x);

  using value_t = typename scalar_t::value_type;
  const value_t* A_ptr = A.data_ptr<value_t>();
  const value_t* G_ptr = G.data_ptr<value_t>();
  const value_t* step_ptr = step.data_ptr<value_t>();
  const scalar_t* bu_ptr = bu.data_ptr<scalar_t>();
  value_t* M00_ptr = M00.data_ptr<value_t>();
  value_t* M01_ptr = M01.data_ptr<value_t>();
  value_t* M10_ptr = M10.data_ptr<value_t>();
  value_t* M11_ptr = M11.data_ptr<value_t>();
  scalar_t* F0_ptr = F0.data_ptr<scalar_t>();
  scalar_t* F1_ptr = F1.data_ptr<scalar_t>();
  scalar_t* S0_w_ptr = S0_w.data_ptr<scalar_t>();
  scalar_t* S0_x_ptr = S0_x.data_ptr<scalar_t>();
  scalar_t* states_ptr = states.data_ptr<scalar_t>();

  switch (vary_mask) {
    case 0:
      launch_build_kernel<scalar_t, false, false, false, TILE>(A_stride,
                                                               G_stride,
                                                               step_stride,
                                                               bu_stride,
                                                               A_ptr,
                                                               G_ptr,
                                                               step_ptr,
                                                               bu_ptr,
                                                               M00_ptr,
                                                               M01_ptr,
                                                               M10_ptr,
                                                               M11_ptr,
                                                               F0_ptr,
                                                               F1_ptr,
                                                               length,
                                                               series,
                                                               ssm,
                                                               num_tiles,
                                                               grid,
                                                               block,
                                                               stream);
      break;
    case 1:
      launch_build_kernel<scalar_t, true, false, false, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              M00_ptr,
                                                              M01_ptr,
                                                              M10_ptr,
                                                              M11_ptr,
                                                              F0_ptr,
                                                              F1_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 2:
      launch_build_kernel<scalar_t, false, true, false, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              M00_ptr,
                                                              M01_ptr,
                                                              M10_ptr,
                                                              M11_ptr,
                                                              F0_ptr,
                                                              F1_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 3:
      launch_build_kernel<scalar_t, true, true, false, TILE>(A_stride,
                                                             G_stride,
                                                             step_stride,
                                                             bu_stride,
                                                             A_ptr,
                                                             G_ptr,
                                                             step_ptr,
                                                             bu_ptr,
                                                             M00_ptr,
                                                             M01_ptr,
                                                             M10_ptr,
                                                             M11_ptr,
                                                             F0_ptr,
                                                             F1_ptr,
                                                             length,
                                                             series,
                                                             ssm,
                                                             num_tiles,
                                                             grid,
                                                             block,
                                                             stream);
      break;
    case 4:
      launch_build_kernel<scalar_t, false, false, true, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              M00_ptr,
                                                              M01_ptr,
                                                              M10_ptr,
                                                              M11_ptr,
                                                              F0_ptr,
                                                              F1_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 5:
      launch_build_kernel<scalar_t, true, false, true, TILE>(A_stride,
                                                             G_stride,
                                                             step_stride,
                                                             bu_stride,
                                                             A_ptr,
                                                             G_ptr,
                                                             step_ptr,
                                                             bu_ptr,
                                                             M00_ptr,
                                                             M01_ptr,
                                                             M10_ptr,
                                                             M11_ptr,
                                                             F0_ptr,
                                                             F1_ptr,
                                                             length,
                                                             series,
                                                             ssm,
                                                             num_tiles,
                                                             grid,
                                                             block,
                                                             stream);
      break;
    case 6:
      launch_build_kernel<scalar_t, false, true, true, TILE>(A_stride,
                                                             G_stride,
                                                             step_stride,
                                                             bu_stride,
                                                             A_ptr,
                                                             G_ptr,
                                                             step_ptr,
                                                             bu_ptr,
                                                             M00_ptr,
                                                             M01_ptr,
                                                             M10_ptr,
                                                             M11_ptr,
                                                             F0_ptr,
                                                             F1_ptr,
                                                             length,
                                                             series,
                                                             ssm,
                                                             num_tiles,
                                                             grid,
                                                             block,
                                                             stream);
      break;
    default:
      launch_build_kernel<scalar_t, true, true, true, TILE>(A_stride,
                                                            G_stride,
                                                            step_stride,
                                                            bu_stride,
                                                            A_ptr,
                                                            G_ptr,
                                                            step_ptr,
                                                            bu_ptr,
                                                            M00_ptr,
                                                            M01_ptr,
                                                            M10_ptr,
                                                            M11_ptr,
                                                            F0_ptr,
                                                            F1_ptr,
                                                            length,
                                                            series,
                                                            ssm,
                                                            num_tiles,
                                                            grid,
                                                            block,
                                                            stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_prefix_kernel<scalar_t, TILE>(M00_ptr,
                                       M01_ptr,
                                       M10_ptr,
                                       M11_ptr,
                                       F0_ptr,
                                       F1_ptr,
                                       S0_w_ptr,
                                       S0_x_ptr,
                                       series,
                                       num_tiles,
                                       stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (vary_mask) {
    case 0:
      launch_expand_kernel<scalar_t, false, false, false, TILE>(A_stride,
                                                                G_stride,
                                                                step_stride,
                                                                bu_stride,
                                                                A_ptr,
                                                                G_ptr,
                                                                step_ptr,
                                                                bu_ptr,
                                                                S0_w_ptr,
                                                                S0_x_ptr,
                                                                states_ptr,
                                                                length,
                                                                series,
                                                                ssm,
                                                                num_tiles,
                                                                grid,
                                                                block,
                                                                stream);
      break;
    case 1:
      launch_expand_kernel<scalar_t, true, false, false, TILE>(A_stride,
                                                               G_stride,
                                                               step_stride,
                                                               bu_stride,
                                                               A_ptr,
                                                               G_ptr,
                                                               step_ptr,
                                                               bu_ptr,
                                                               S0_w_ptr,
                                                               S0_x_ptr,
                                                               states_ptr,
                                                               length,
                                                               series,
                                                               ssm,
                                                               num_tiles,
                                                               grid,
                                                               block,
                                                               stream);
      break;
    case 2:
      launch_expand_kernel<scalar_t, false, true, false, TILE>(A_stride,
                                                               G_stride,
                                                               step_stride,
                                                               bu_stride,
                                                               A_ptr,
                                                               G_ptr,
                                                               step_ptr,
                                                               bu_ptr,
                                                               S0_w_ptr,
                                                               S0_x_ptr,
                                                               states_ptr,
                                                               length,
                                                               series,
                                                               ssm,
                                                               num_tiles,
                                                               grid,
                                                               block,
                                                               stream);
      break;
    case 3:
      launch_expand_kernel<scalar_t, true, true, false, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              S0_w_ptr,
                                                              S0_x_ptr,
                                                              states_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 4:
      launch_expand_kernel<scalar_t, false, false, true, TILE>(A_stride,
                                                               G_stride,
                                                               step_stride,
                                                               bu_stride,
                                                               A_ptr,
                                                               G_ptr,
                                                               step_ptr,
                                                               bu_ptr,
                                                               S0_w_ptr,
                                                               S0_x_ptr,
                                                               states_ptr,
                                                               length,
                                                               series,
                                                               ssm,
                                                               num_tiles,
                                                               grid,
                                                               block,
                                                               stream);
      break;
    case 5:
      launch_expand_kernel<scalar_t, true, false, true, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              S0_w_ptr,
                                                              S0_x_ptr,
                                                              states_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 6:
      launch_expand_kernel<scalar_t, false, true, true, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              S0_w_ptr,
                                                              S0_x_ptr,
                                                              states_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    default:
      launch_expand_kernel<scalar_t, true, true, true, TILE>(A_stride,
                                                             G_stride,
                                                             step_stride,
                                                             bu_stride,
                                                             A_ptr,
                                                             G_ptr,
                                                             step_ptr,
                                                             bu_ptr,
                                                             S0_w_ptr,
                                                             S0_x_ptr,
                                                             states_ptr,
                                                             length,
                                                             series,
                                                             ssm,
                                                             num_tiles,
                                                             grid,
                                                             block,
                                                             stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, int TILE>
void dispatch_all_xonly(int vary_mask,
                        const Strides3& A_stride,
                        const Strides3& G_stride,
                        const Strides3& step_stride,
                        const Strides3& bu_stride,
                        const at::Tensor& A,
                        const at::Tensor& G,
                        const at::Tensor& step,
                        const at::Tensor& bu,
                        const at::Tensor& M00,
                        const at::Tensor& M01,
                        const at::Tensor& M10,
                        const at::Tensor& M11,
                        const at::Tensor& F0,
                        const at::Tensor& F1,
                        const at::Tensor& S0_w,
                        const at::Tensor& S0_x,
                        const at::Tensor& x_only,
                        int64_t length,
                        int64_t series,
                        int64_t ssm,
                        int64_t num_tiles,
                        cudaStream_t stream) {
  const dim3 block(256);
  const dim3 grid((series + block.x - 1) / block.x);

  using value_t = typename scalar_t::value_type;
  const value_t* A_ptr = A.data_ptr<value_t>();
  const value_t* G_ptr = G.data_ptr<value_t>();
  const value_t* step_ptr = step.data_ptr<value_t>();
  const scalar_t* bu_ptr = bu.data_ptr<scalar_t>();
  value_t* M00_ptr = M00.data_ptr<value_t>();
  value_t* M01_ptr = M01.data_ptr<value_t>();
  value_t* M10_ptr = M10.data_ptr<value_t>();
  value_t* M11_ptr = M11.data_ptr<value_t>();
  scalar_t* F0_ptr = F0.data_ptr<scalar_t>();
  scalar_t* F1_ptr = F1.data_ptr<scalar_t>();
  scalar_t* S0_w_ptr = S0_w.data_ptr<scalar_t>();
  scalar_t* S0_x_ptr = S0_x.data_ptr<scalar_t>();
  scalar_t* x_ptr = x_only.data_ptr<scalar_t>();

  switch (vary_mask) {
    case 0:
      launch_build_kernel<scalar_t, false, false, false, TILE>(A_stride,
                                                               G_stride,
                                                               step_stride,
                                                               bu_stride,
                                                               A_ptr,
                                                               G_ptr,
                                                               step_ptr,
                                                               bu_ptr,
                                                               M00_ptr,
                                                               M01_ptr,
                                                               M10_ptr,
                                                               M11_ptr,
                                                               F0_ptr,
                                                               F1_ptr,
                                                               length,
                                                               series,
                                                               ssm,
                                                               num_tiles,
                                                               grid,
                                                               block,
                                                               stream);
      break;
    case 1:
      launch_build_kernel<scalar_t, true, false, false, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              M00_ptr,
                                                              M01_ptr,
                                                              M10_ptr,
                                                              M11_ptr,
                                                              F0_ptr,
                                                              F1_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 2:
      launch_build_kernel<scalar_t, false, true, false, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              M00_ptr,
                                                              M01_ptr,
                                                              M10_ptr,
                                                              M11_ptr,
                                                              F0_ptr,
                                                              F1_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 3:
      launch_build_kernel<scalar_t, true, true, false, TILE>(A_stride,
                                                             G_stride,
                                                             step_stride,
                                                             bu_stride,
                                                             A_ptr,
                                                             G_ptr,
                                                             step_ptr,
                                                             bu_ptr,
                                                             M00_ptr,
                                                             M01_ptr,
                                                             M10_ptr,
                                                             M11_ptr,
                                                             F0_ptr,
                                                             F1_ptr,
                                                             length,
                                                             series,
                                                             ssm,
                                                             num_tiles,
                                                             grid,
                                                             block,
                                                             stream);
      break;
    case 4:
      launch_build_kernel<scalar_t, false, false, true, TILE>(A_stride,
                                                              G_stride,
                                                              step_stride,
                                                              bu_stride,
                                                              A_ptr,
                                                              G_ptr,
                                                              step_ptr,
                                                              bu_ptr,
                                                              M00_ptr,
                                                              M01_ptr,
                                                              M10_ptr,
                                                              M11_ptr,
                                                              F0_ptr,
                                                              F1_ptr,
                                                              length,
                                                              series,
                                                              ssm,
                                                              num_tiles,
                                                              grid,
                                                              block,
                                                              stream);
      break;
    case 5:
      launch_build_kernel<scalar_t, true, false, true, TILE>(A_stride,
                                                             G_stride,
                                                             step_stride,
                                                             bu_stride,
                                                             A_ptr,
                                                             G_ptr,
                                                             step_ptr,
                                                             bu_ptr,
                                                             M00_ptr,
                                                             M01_ptr,
                                                             M10_ptr,
                                                             M11_ptr,
                                                             F0_ptr,
                                                             F1_ptr,
                                                             length,
                                                             series,
                                                             ssm,
                                                             num_tiles,
                                                             grid,
                                                             block,
                                                             stream);
      break;
    case 6:
      launch_build_kernel<scalar_t, false, true, true, TILE>(A_stride,
                                                             G_stride,
                                                             step_stride,
                                                             bu_stride,
                                                             A_ptr,
                                                             G_ptr,
                                                             step_ptr,
                                                             bu_ptr,
                                                             M00_ptr,
                                                             M01_ptr,
                                                             M10_ptr,
                                                             M11_ptr,
                                                             F0_ptr,
                                                             F1_ptr,
                                                             length,
                                                             series,
                                                             ssm,
                                                             num_tiles,
                                                             grid,
                                                             block,
                                                             stream);
      break;
    default:
      launch_build_kernel<scalar_t, true, true, true, TILE>(A_stride,
                                                            G_stride,
                                                            step_stride,
                                                            bu_stride,
                                                            A_ptr,
                                                            G_ptr,
                                                            step_ptr,
                                                            bu_ptr,
                                                            M00_ptr,
                                                            M01_ptr,
                                                            M10_ptr,
                                                            M11_ptr,
                                                            F0_ptr,
                                                            F1_ptr,
                                                            length,
                                                            series,
                                                            ssm,
                                                            num_tiles,
                                                            grid,
                                                            block,
                                                            stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_prefix_kernel<scalar_t, TILE>(M00_ptr,
                                       M01_ptr,
                                       M10_ptr,
                                       M11_ptr,
                                       F0_ptr,
                                       F1_ptr,
                                       S0_w_ptr,
                                       S0_x_ptr,
                                       series,
                                       num_tiles,
                                       stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  switch (vary_mask) {
    case 0:
      launch_expand_x_kernel<scalar_t, false, false, false, TILE>(A_stride,
                                                                  G_stride,
                                                                  step_stride,
                                                                  bu_stride,
                                                                  A_ptr,
                                                                  G_ptr,
                                                                  step_ptr,
                                                                  bu_ptr,
                                                                  S0_w_ptr,
                                                                  S0_x_ptr,
                                                                  x_ptr,
                                                                  length,
                                                                  series,
                                                                  ssm,
                                                                  num_tiles,
                                                                  grid,
                                                                  block,
                                                                  stream);
      break;
    case 1:
      launch_expand_x_kernel<scalar_t, true, false, false, TILE>(A_stride,
                                                                 G_stride,
                                                                 step_stride,
                                                                 bu_stride,
                                                                 A_ptr,
                                                                 G_ptr,
                                                                 step_ptr,
                                                                 bu_ptr,
                                                                 S0_w_ptr,
                                                                 S0_x_ptr,
                                                                 x_ptr,
                                                                 length,
                                                                 series,
                                                                 ssm,
                                                                 num_tiles,
                                                                 grid,
                                                                 block,
                                                                 stream);
      break;
    case 2:
      launch_expand_x_kernel<scalar_t, false, true, false, TILE>(A_stride,
                                                                 G_stride,
                                                                 step_stride,
                                                                 bu_stride,
                                                                 A_ptr,
                                                                 G_ptr,
                                                                 step_ptr,
                                                                 bu_ptr,
                                                                 S0_w_ptr,
                                                                 S0_x_ptr,
                                                                 x_ptr,
                                                                 length,
                                                                 series,
                                                                 ssm,
                                                                 num_tiles,
                                                                 grid,
                                                                 block,
                                                                 stream);
      break;
    case 3:
      launch_expand_x_kernel<scalar_t, true, true, false, TILE>(A_stride,
                                                                G_stride,
                                                                step_stride,
                                                                bu_stride,
                                                                A_ptr,
                                                                G_ptr,
                                                                step_ptr,
                                                                bu_ptr,
                                                                S0_w_ptr,
                                                                S0_x_ptr,
                                                                x_ptr,
                                                                length,
                                                                series,
                                                                ssm,
                                                                num_tiles,
                                                                grid,
                                                                block,
                                                                stream);
      break;
    case 4:
      launch_expand_x_kernel<scalar_t, false, false, true, TILE>(A_stride,
                                                                 G_stride,
                                                                 step_stride,
                                                                 bu_stride,
                                                                 A_ptr,
                                                                 G_ptr,
                                                                 step_ptr,
                                                                 bu_ptr,
                                                                 S0_w_ptr,
                                                                 S0_x_ptr,
                                                                 x_ptr,
                                                                 length,
                                                                 series,
                                                                 ssm,
                                                                 num_tiles,
                                                                 grid,
                                                                 block,
                                                                 stream);
      break;
    case 5:
      launch_expand_x_kernel<scalar_t, true, false, true, TILE>(A_stride,
                                                                G_stride,
                                                                step_stride,
                                                                bu_stride,
                                                                A_ptr,
                                                                G_ptr,
                                                                step_ptr,
                                                                bu_ptr,
                                                                S0_w_ptr,
                                                                S0_x_ptr,
                                                                x_ptr,
                                                                length,
                                                                series,
                                                                ssm,
                                                                num_tiles,
                                                                grid,
                                                                block,
                                                                stream);
      break;
    case 6:
      launch_expand_x_kernel<scalar_t, false, true, true, TILE>(A_stride,
                                                                G_stride,
                                                                step_stride,
                                                                bu_stride,
                                                                A_ptr,
                                                                G_ptr,
                                                                step_ptr,
                                                                bu_ptr,
                                                                S0_w_ptr,
                                                                S0_x_ptr,
                                                                x_ptr,
                                                                length,
                                                                series,
                                                                ssm,
                                                                num_tiles,
                                                                grid,
                                                                block,
                                                                stream);
      break;
    default:
      launch_expand_x_kernel<scalar_t, true, true, true, TILE>(A_stride,
                                                               G_stride,
                                                               step_stride,
                                                               bu_stride,
                                                               A_ptr,
                                                               G_ptr,
                                                               step_ptr,
                                                               bu_ptr,
                                                               S0_w_ptr,
                                                               S0_x_ptr,
                                                               x_ptr,
                                                               length,
                                                               series,
                                                               ssm,
                                                               num_tiles,
                                                               grid,
                                                               block,
                                                               stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void sdlinoss_fast_ex_forward_cuda_complex64(int tile,
                                             int vary_mask,
                                             const Strides3& A_stride,
                                             const Strides3& G_stride,
                                             const Strides3& step_stride,
                                             const Strides3& bu_stride,
                                             const at::Tensor& A,
                                             const at::Tensor& G,
                                             const at::Tensor& step,
                                             const at::Tensor& bu,
                                             const at::Tensor& M00,
                                             const at::Tensor& M01,
                                             const at::Tensor& M10,
                                             const at::Tensor& M11,
                                             const at::Tensor& F0,
                                             const at::Tensor& F1,
                                             const at::Tensor& S0_w,
                                             const at::Tensor& S0_x,
                                             const at::Tensor& states,
                                             int64_t length,
                                             int64_t series,
                                             int64_t ssm,
                                             int64_t num_tiles,
                                             cudaStream_t stream) {
  switch (tile) {
    case 64:
      dispatch_all<c10::complex<float>, 64>(vary_mask,
                                            A_stride,
                                            G_stride,
                                            step_stride,
                                            bu_stride,
                                            A,
                                            G,
                                            step,
                                            bu,
                                            M00,
                                            M01,
                                            M10,
                                            M11,
                                            F0,
                                            F1,
                                            S0_w,
                                            S0_x,
                                            states,
                                            length,
                                            series,
                                            ssm,
                                            num_tiles,
                                            stream);
      break;
    case 256:
      dispatch_all<c10::complex<float>, 256>(vary_mask,
                                             A_stride,
                                             G_stride,
                                             step_stride,
                                             bu_stride,
                                             A,
                                             G,
                                             step,
                                             bu,
                                             M00,
                                             M01,
                                             M10,
                                             M11,
                                             F0,
                                             F1,
                                             S0_w,
                                             S0_x,
                                             states,
                                             length,
                                             series,
                                             ssm,
                                             num_tiles,
                                             stream);
      break;
    default:
      dispatch_all<c10::complex<float>, 128>(vary_mask,
                                             A_stride,
                                             G_stride,
                                             step_stride,
                                             bu_stride,
                                             A,
                                             G,
                                             step,
                                             bu,
                                             M00,
                                             M01,
                                             M10,
                                             M11,
                                             F0,
                                             F1,
                                             S0_w,
                                             S0_x,
                                             states,
                                             length,
                                             series,
                                             ssm,
                                            num_tiles,
                                            stream);
      break;
  }
}

void sdlinoss_fast_ex_forward_xonly_cuda_complex64(int tile,
                                                   int vary_mask,
                                                   const Strides3& A_stride,
                                                   const Strides3& G_stride,
                                                   const Strides3& step_stride,
                                                   const Strides3& bu_stride,
                                                   const at::Tensor& A,
                                                   const at::Tensor& G,
                                                   const at::Tensor& step,
                                                   const at::Tensor& bu,
                                                   const at::Tensor& M00,
                                                   const at::Tensor& M01,
                                                   const at::Tensor& M10,
                                                   const at::Tensor& M11,
                                                   const at::Tensor& F0,
                                                   const at::Tensor& F1,
                                                   const at::Tensor& S0_w,
                                                   const at::Tensor& S0_x,
                                                   const at::Tensor& x_only,
                                                   int64_t length,
                                                   int64_t series,
                                                   int64_t ssm,
                                                   int64_t num_tiles,
                                                   cudaStream_t stream) {
  switch (tile) {
    case 64:
      dispatch_all_xonly<c10::complex<float>, 64>(vary_mask,
                                                  A_stride,
                                                  G_stride,
                                                  step_stride,
                                                  bu_stride,
                                                  A,
                                                  G,
                                                  step,
                                                  bu,
                                                  M00,
                                                  M01,
                                                  M10,
                                                  M11,
                                                  F0,
                                                  F1,
                                                  S0_w,
                                                  S0_x,
                                                  x_only,
                                                  length,
                                                  series,
                                                  ssm,
                                                  num_tiles,
                                                  stream);
      break;
    case 256:
      dispatch_all_xonly<c10::complex<float>, 256>(vary_mask,
                                                   A_stride,
                                                   G_stride,
                                                   step_stride,
                                                   bu_stride,
                                                   A,
                                                   G,
                                                   step,
                                                   bu,
                                                   M00,
                                                   M01,
                                                   M10,
                                                   M11,
                                                   F0,
                                                   F1,
                                                   S0_w,
                                                   S0_x,
                                                   x_only,
                                                   length,
                                                   series,
                                                   ssm,
                                                   num_tiles,
                                                   stream);
      break;
    default:
      dispatch_all_xonly<c10::complex<float>, 128>(vary_mask,
                                                   A_stride,
                                                   G_stride,
                                                   step_stride,
                                                   bu_stride,
                                                   A,
                                                   G,
                                                   step,
                                                   bu,
                                                   M00,
                                                   M01,
                                                   M10,
                                                   M11,
                                                   F0,
                                                   F1,
                                                   S0_w,
                                                   S0_x,
                                                   x_only,
                                                   length,
                                                   series,
                                                   ssm,
                                                   num_tiles,
                                                   stream);
      break;
  }
}

void sdlinoss_fast_ex_forward_cuda_complex128(int tile,
                                              int vary_mask,
                                              const Strides3& A_stride,
                                              const Strides3& G_stride,
                                              const Strides3& step_stride,
                                              const Strides3& bu_stride,
                                              const at::Tensor& A,
                                              const at::Tensor& G,
                                              const at::Tensor& step,
                                              const at::Tensor& bu,
                                              const at::Tensor& M00,
                                              const at::Tensor& M01,
                                              const at::Tensor& M10,
                                              const at::Tensor& M11,
                                              const at::Tensor& F0,
                                              const at::Tensor& F1,
                                              const at::Tensor& S0_w,
                                              const at::Tensor& S0_x,
                                              const at::Tensor& states,
                                              int64_t length,
                                              int64_t series,
                                              int64_t ssm,
                                              int64_t num_tiles,
                                              cudaStream_t stream) {
  switch (tile) {
    case 64:
      dispatch_all<c10::complex<double>, 64>(vary_mask,
                                             A_stride,
                                             G_stride,
                                             step_stride,
                                             bu_stride,
                                             A,
                                             G,
                                             step,
                                             bu,
                                             M00,
                                             M01,
                                             M10,
                                             M11,
                                             F0,
                                             F1,
                                             S0_w,
                                             S0_x,
                                             states,
                                             length,
                                             series,
                                             ssm,
                                             num_tiles,
                                             stream);
      break;
    case 256:
      dispatch_all<c10::complex<double>, 256>(vary_mask,
                                              A_stride,
                                              G_stride,
                                              step_stride,
                                              bu_stride,
                                              A,
                                              G,
                                              step,
                                              bu,
                                              M00,
                                              M01,
                                              M10,
                                              M11,
                                              F0,
                                              F1,
                                              S0_w,
                                              S0_x,
                                              states,
                                              length,
                                              series,
                                              ssm,
                                              num_tiles,
                                              stream);
      break;
    default:
      dispatch_all<c10::complex<double>, 128>(vary_mask,
                                              A_stride,
                                              G_stride,
                                              step_stride,
                                              bu_stride,
                                              A,
                                              G,
                                              step,
                                              bu,
                                              M00,
                                              M01,
                                              M10,
                                              M11,
                                              F0,
                                              F1,
                                              S0_w,
                                              S0_x,
                                              states,
                                              length,
                                              series,
                                              ssm,
                                            num_tiles,
                                            stream);
      break;
  }
}

void sdlinoss_fast_ex_forward_xonly_cuda_complex128(int tile,
                                                    int vary_mask,
                                                    const Strides3& A_stride,
                                                    const Strides3& G_stride,
                                                    const Strides3& step_stride,
                                                    const Strides3& bu_stride,
                                                    const at::Tensor& A,
                                                    const at::Tensor& G,
                                                    const at::Tensor& step,
                                                    const at::Tensor& bu,
                                                    const at::Tensor& M00,
                                                    const at::Tensor& M01,
                                                    const at::Tensor& M10,
                                                    const at::Tensor& M11,
                                                    const at::Tensor& F0,
                                                    const at::Tensor& F1,
                                                    const at::Tensor& S0_w,
                                                    const at::Tensor& S0_x,
                                                    const at::Tensor& x_only,
                                                    int64_t length,
                                                    int64_t series,
                                                    int64_t ssm,
                                                    int64_t num_tiles,
                                                    cudaStream_t stream) {
  switch (tile) {
    case 64:
      dispatch_all_xonly<c10::complex<double>, 64>(vary_mask,
                                                   A_stride,
                                                   G_stride,
                                                   step_stride,
                                                   bu_stride,
                                                   A,
                                                   G,
                                                   step,
                                                   bu,
                                                   M00,
                                                   M01,
                                                   M10,
                                                   M11,
                                                   F0,
                                                   F1,
                                                   S0_w,
                                                   S0_x,
                                                   x_only,
                                                   length,
                                                   series,
                                                   ssm,
                                                   num_tiles,
                                                   stream);
      break;
    case 256:
      dispatch_all_xonly<c10::complex<double>, 256>(vary_mask,
                                                    A_stride,
                                                    G_stride,
                                                    step_stride,
                                                    bu_stride,
                                                    A,
                                                    G,
                                                    step,
                                                    bu,
                                                    M00,
                                                    M01,
                                                    M10,
                                                    M11,
                                                    F0,
                                                    F1,
                                                    S0_w,
                                                    S0_x,
                                                    x_only,
                                                    length,
                                                    series,
                                                    ssm,
                                                    num_tiles,
                                                    stream);
      break;
    default:
      dispatch_all_xonly<c10::complex<double>, 128>(vary_mask,
                                                    A_stride,
                                                    G_stride,
                                                    step_stride,
                                                    bu_stride,
                                                    A,
                                                    G,
                                                    step,
                                                    bu,
                                                    M00,
                                                    M01,
                                                    M10,
                                                    M11,
                                                    F0,
                                                    F1,
                                                    S0_w,
                                                    S0_x,
                                                    x_only,
                                                    length,
                                                    series,
                                                    ssm,
                                                    num_tiles,
                                                    stream);
      break;
  }
}

}  // namespace ossm
