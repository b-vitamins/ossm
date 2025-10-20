#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>

namespace ossm {
namespace {

template <typename value_t>
__device__ inline value_t fused_madd(value_t a, value_t b, value_t c) {
  return a * b + c;
}

template <>
__device__ inline float fused_madd<float>(float a, float b, float c) {
#if __CUDA_ARCH__ >= 200
  return fmaf(a, b, c);
#else
  return a * b + c;
#endif
}

template <>
__device__ inline double fused_madd<double>(double a, double b, double c) {
#if __CUDA_ARCH__ >= 200
  return fma(a, b, c);
#else
  return a * b + c;
#endif
}

template <typename scalar_t>
__global__ void linoss_scan_cuda_kernel(const scalar_t* __restrict__ m11,
                                        const scalar_t* __restrict__ m12,
                                        const scalar_t* __restrict__ m21,
                                        const scalar_t* __restrict__ m22,
                                        const scalar_t* __restrict__ b_ptr,
                                        scalar_t* __restrict__ out_ptr,
                                        int64_t length,
                                        int64_t batch,
                                        int64_t ssm) {
  using value_t = typename scalar_t::value_type;

  const value_t* m11_vals = reinterpret_cast<const value_t*>(m11);
  const value_t* m12_vals = reinterpret_cast<const value_t*>(m12);
  const value_t* m21_vals = reinterpret_cast<const value_t*>(m21);
  const value_t* m22_vals = reinterpret_cast<const value_t*>(m22);
  const value_t* b_vals = reinterpret_cast<const value_t*>(b_ptr);
  value_t* out_vals = reinterpret_cast<value_t*>(out_ptr);

  const int64_t complex_stride = 2;  // real + imag
  const int64_t components = 2;      // two complex components per state
  const int64_t scalars_per_state = complex_stride * components;  // 4 value_t entries

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * scalars_per_state;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series;
       idx += blockDim.x * gridDim.x) {
    const int64_t state_index = idx % ssm;
    const int64_t coeff_offset = state_index * complex_stride;

    const value_t a11_r = m11_vals[coeff_offset];
    const value_t a11_i = m11_vals[coeff_offset + 1];
    const value_t a12_r = m12_vals[coeff_offset];
    const value_t a12_i = m12_vals[coeff_offset + 1];
    const value_t a21_r = m21_vals[coeff_offset];
    const value_t a21_i = m21_vals[coeff_offset + 1];
    const value_t a22_r = m22_vals[coeff_offset];
    const value_t a22_i = m22_vals[coeff_offset + 1];

    value_t state0_r = value_t(0);
    value_t state0_i = value_t(0);
    value_t state1_r = value_t(0);
    value_t state1_i = value_t(0);

    const int64_t base = idx * scalars_per_state;

    for (int64_t t = 0; t < length; ++t) {
      const int64_t offset = t * step_stride + base;

      const value_t b0_r = __ldg(b_vals + offset);
      const value_t b0_i = __ldg(b_vals + offset + 1);
      const value_t b1_r = __ldg(b_vals + offset + 2);
      const value_t b1_i = __ldg(b_vals + offset + 3);

      const value_t a11_s0_r = fused_madd(a11_r, state0_r, -a11_i * state0_i);
      const value_t a11_s0_i = fused_madd(a11_r, state0_i, a11_i * state0_r);
      const value_t a12_s1_r = fused_madd(a12_r, state1_r, -a12_i * state1_i);
      const value_t a12_s1_i = fused_madd(a12_r, state1_i, a12_i * state1_r);

      const value_t a21_s0_r = fused_madd(a21_r, state0_r, -a21_i * state0_i);
      const value_t a21_s0_i = fused_madd(a21_r, state0_i, a21_i * state0_r);
      const value_t a22_s1_r = fused_madd(a22_r, state1_r, -a22_i * state1_i);
      const value_t a22_s1_i = fused_madd(a22_r, state1_i, a22_i * state1_r);

      const value_t new0_r = (a11_s0_r + a12_s1_r) + b0_r;
      const value_t new0_i = (a11_s0_i + a12_s1_i) + b0_i;
      const value_t new1_r = (a21_s0_r + a22_s1_r) + b1_r;
      const value_t new1_i = (a21_s0_i + a22_s1_i) + b1_i;

      out_vals[offset] = new0_r;
      out_vals[offset + 1] = new0_i;
      out_vals[offset + 2] = new1_r;
      out_vals[offset + 3] = new1_i;

      state0_r = new0_r;
      state0_i = new0_i;
      state1_r = new1_r;
      state1_i = new1_i;
    }
  }
}

}  // namespace

void linoss_scan_cuda(const at::Tensor& m11,
                      const at::Tensor& m12,
                      const at::Tensor& m21,
                      const at::Tensor& m22,
                      const at::Tensor& b_seq,
                      at::Tensor& output) {
  c10::cuda::OptionalCUDAGuard device_guard{b_seq.device()};

  const auto length = b_seq.size(0);
  const auto batch = b_seq.size(1);
  const auto ssm = b_seq.size(2);
  const auto series = batch * ssm;

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(
          (series + threads - 1) / threads,
          at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  AT_DISPATCH_COMPLEX_TYPES(b_seq.scalar_type(), "linoss_scan_cuda", [&] {
    linoss_scan_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        m11.data_ptr<scalar_t>(),
        m12.data_ptr<scalar_t>(),
        m21.data_ptr<scalar_t>(),
        m22.data_ptr<scalar_t>(),
        b_seq.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        length,
        batch,
        ssm);
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ossm
