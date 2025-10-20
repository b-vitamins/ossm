#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

namespace ossm {
namespace {

template <typename scalar_t>
__global__ void s5_scan_cuda_kernel(const scalar_t* lambda_real,
                                    const scalar_t* lambda_imag,
                                    const scalar_t* b_seq,
                                    scalar_t* out,
                                    int64_t length,
                                    int64_t batch,
                                    int64_t state_size) {
  using complex_t = c10::complex<scalar_t>;
  const int64_t series = batch * state_size;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= series) {
    return;
  }

  const int64_t state_index = idx % state_size;
  const complex_t lambda(lambda_real[state_index], lambda_imag[state_index]);

  complex_t state = complex_t(0, 0);
  const complex_t* b_step = reinterpret_cast<const complex_t*>(b_seq) + idx;
  complex_t* out_step = reinterpret_cast<complex_t*>(out) + idx;

  for (int64_t t = 0; t < length; ++t) {
    state = lambda * state + *b_step;
    *out_step = state;
    b_step += series;
    out_step += series;
  }
}

}  // namespace

at::Tensor s5_scan_cuda(const at::Tensor& lambda_real,
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

  const dim3 threads(256);
  const dim3 blocks((series + threads.x - 1) / threads.x);

  AT_DISPATCH_FLOATING_TYPES(lambda_real.scalar_type(), "s5_scan_cuda", [&] {
    s5_scan_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
