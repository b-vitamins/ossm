#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <algorithm>

namespace ossm {
namespace {

template <typename scalar_t>
__device__ inline scalar_t fused_madd(scalar_t a, scalar_t b, scalar_t c) {
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
__global__ void lru_scan_cuda_kernel(const scalar_t* __restrict__ lambda_real,
                                     const scalar_t* __restrict__ lambda_imag,
                                     const scalar_t* __restrict__ b_seq,
                                     scalar_t* __restrict__ out,
                                     int64_t length,
                                     int64_t batch,
                                     int64_t state_size) {
  const int64_t series = batch * state_size;
  const int64_t stride = series * 2;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series;
       idx += blockDim.x * gridDim.x) {
    const int64_t state_index = idx % state_size;
    const scalar_t lam_r = lambda_real[state_index];
    const scalar_t lam_i = lambda_imag[state_index];

    scalar_t state_r = scalar_t(0);
    scalar_t state_i = scalar_t(0);

    const scalar_t* b_ptr = b_seq + idx * 2;
    scalar_t* out_ptr = out + idx * 2;

    for (int64_t t = 0; t < length; ++t) {
      const scalar_t b_r = __ldg(b_ptr);
      const scalar_t b_i = __ldg(b_ptr + 1);

      const scalar_t real_term = fused_madd(lam_r, state_r, scalar_t(-lam_i) * state_i);
      const scalar_t imag_term = fused_madd(lam_r, state_i, lam_i * state_r);

      state_r = real_term + b_r;
      state_i = imag_term + b_i;

      out_ptr[0] = state_r;
      out_ptr[1] = state_i;

      b_ptr += stride;
      out_ptr += stride;
    }
  }
}

}  // namespace

at::Tensor lru_scan_cuda(const at::Tensor& lambda_real,
                         const at::Tensor& lambda_imag,
                         const at::Tensor& b_seq) {
  TORCH_CHECK(lambda_real.is_cuda(), "lambda_real must be CUDA tensor");
  TORCH_CHECK(lambda_imag.is_cuda(), "lambda_imag must be CUDA tensor");
  TORCH_CHECK(b_seq.is_cuda(), "b_seq must be CUDA tensor");

  const auto length = b_seq.size(0);
  const auto batch = b_seq.size(1);
  const auto state_size = b_seq.size(2);
  const auto series = batch * state_size;

  auto result = at::empty_like(b_seq);

  if (length == 0 || batch == 0 || state_size == 0) {
    return result;
  }

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(
          (series + threads - 1) / threads,
          at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  AT_DISPATCH_FLOATING_TYPES(lambda_real.scalar_type(), "lru_scan_cuda", [&] {
    lru_scan_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        lambda_real.data_ptr<scalar_t>(),
        lambda_imag.data_ptr<scalar_t>(),
        b_seq.data_ptr<scalar_t>(),
        result.data_ptr<scalar_t>(),
        length,
        batch,
        state_size);
  });

  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

}  // namespace ossm
