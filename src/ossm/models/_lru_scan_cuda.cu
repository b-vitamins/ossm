#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/complex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void lru_scan_kernel(const scalar_t* __restrict__ lambda_real,
                                const scalar_t* __restrict__ lambda_imag,
                                const scalar_t* __restrict__ b_seq,
                                scalar_t* __restrict__ out,
                                int64_t length,
                                int64_t batch,
                                int64_t state_size) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = batch * state_size;

  if (index >= total) {
    return;
  }

  const int64_t state_index = index % state_size;
  using complex_t = c10::complex<scalar_t>;
  static_assert(sizeof(complex_t) == sizeof(scalar_t) * 2,
                "Complex representation must occupy two scalars");

  const complex_t lambda(lambda_real[state_index], lambda_imag[state_index]);

  complex_t state = complex_t(0, 0);
  const complex_t* b_ptr = reinterpret_cast<const complex_t*>(b_seq) + index;
  complex_t* out_ptr = reinterpret_cast<complex_t*>(out) + index;

  for (int64_t t = 0; t < length; ++t) {
    state = lambda * state + *b_ptr;
    *out_ptr = state;
    b_ptr += total;
    out_ptr += total;
  }
}

at::Tensor lru_scan_cuda(const at::Tensor& lambda_real,
                         const at::Tensor& lambda_imag,
                         const at::Tensor& b_seq) {
  TORCH_CHECK(lambda_real.device().is_cuda(), "lambda_real must be on CUDA");
  TORCH_CHECK(lambda_imag.device().is_cuda(), "lambda_imag must be on CUDA");
  TORCH_CHECK(b_seq.device().is_cuda(), "b_seq must be on CUDA");

  TORCH_CHECK(lambda_real.dim() == 1, "lambda_real must be 1-D");
  TORCH_CHECK(lambda_imag.dim() == 1, "lambda_imag must be 1-D");
  TORCH_CHECK(b_seq.dim() == 4 && b_seq.size(-1) == 2,
              "b_seq must have shape (length, batch, state, 2)");

  const auto length = b_seq.size(0);
  const auto batch = b_seq.size(1);
  const auto state_size = b_seq.size(2);

  TORCH_CHECK(lambda_real.size(0) == state_size,
              "lambda_real must match state dimension");
  TORCH_CHECK(lambda_imag.size(0) == state_size,
              "lambda_imag must match state dimension");

  auto lambda_real_contig = lambda_real.contiguous();
  auto lambda_imag_contig = lambda_imag.contiguous();
  auto b_seq_contig = b_seq.contiguous();

  auto result = at::empty_like(b_seq_contig);

  if (length == 0 || batch == 0 || state_size == 0) {
    return result;
  }

  const int64_t threads = 256;
  const int64_t total = batch * state_size;
  const int64_t blocks = (total + threads - 1) / threads;

  at::cuda::CUDAGuard device_guard(lambda_real.device());

  AT_DISPATCH_FLOATING_TYPES(lambda_real_contig.scalar_type(), "lru_scan_cuda", [&] {
    lru_scan_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        lambda_real_contig.data_ptr<scalar_t>(),
        lambda_imag_contig.data_ptr<scalar_t>(),
        b_seq_contig.data_ptr<scalar_t>(),
        result.data_ptr<scalar_t>(),
        length,
        batch,
        state_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return result;
}

