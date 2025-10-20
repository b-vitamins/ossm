#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

namespace ossm {
namespace {

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
  const int64_t series = batch * ssm;
  const int64_t stride_t = series * 2;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series;
       idx += blockDim.x * gridDim.x) {
    const int64_t state_index = idx % ssm;

    const scalar_t a11 = m11[state_index];
    const scalar_t a12 = m12[state_index];
    const scalar_t a21 = m21[state_index];
    const scalar_t a22 = m22[state_index];

    scalar_t state0 = scalar_t(0);
    scalar_t state1 = scalar_t(0);

    const int64_t base = idx * 2;

    for (int64_t t = 0; t < length; ++t) {
      const int64_t offset = t * stride_t + base;
      const scalar_t b0 = b_ptr[offset];
      const scalar_t b1 = b_ptr[offset + 1];

      const scalar_t new0 = a11 * state0 + a12 * state1 + b0;
      const scalar_t new1 = a21 * state0 + a22 * state1 + b1;

      out_ptr[offset] = new0;
      out_ptr[offset + 1] = new1;

      state0 = new0;
      state1 = new1;
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
  at::cuda::CUDAGuard device_guard(b_seq.device());

  const auto length = b_seq.size(0);
  const auto batch = b_seq.size(1);
  const auto ssm = b_seq.size(2);
  const auto series = batch * ssm;

  const dim3 threads(256);
  const dim3 blocks((series + threads.x - 1) / threads.x);

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
