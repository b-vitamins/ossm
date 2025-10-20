#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

namespace ossm {
namespace {

template <typename scalar_t>
void lru_scan_cpu_kernel(const scalar_t* lambda_real,
                         const scalar_t* lambda_imag,
                         const scalar_t* b_seq,
                         scalar_t* out,
                         int64_t length,
                         int64_t batch,
                         int64_t state_size) {
  if (length == 0 || batch == 0 || state_size == 0) {
    return;
  }

  using complex_t = c10::complex<scalar_t>;
  static_assert(sizeof(complex_t) == sizeof(scalar_t) * 2,
                "Complex representation must occupy two scalars");

  const auto total_series = batch * state_size;
  const auto* b_complex = reinterpret_cast<const complex_t*>(b_seq);
  auto* out_complex = reinterpret_cast<complex_t*>(out);

  at::parallel_for(0, total_series, 0, [&](int64_t start, int64_t end) {
    for (int64_t index = start; index < end; ++index) {
      const int64_t state_index = index % state_size;
      const complex_t lambda(lambda_real[state_index], lambda_imag[state_index]);

      complex_t state = complex_t(0, 0);
      const complex_t* b_step = b_complex + index;
      complex_t* out_step = out_complex + index;

      for (int64_t t = 0; t < length; ++t) {
        state = lambda * state + *b_step;
        *out_step = state;
        b_step += total_series;
        out_step += total_series;
      }
    }
  });
}

at::Tensor lru_scan_cpu(const at::Tensor& lambda_real,
                        const at::Tensor& lambda_imag,
                        const at::Tensor& b_seq) {
  TORCH_CHECK(lambda_real.device().is_cpu(), "lambda_real must be on CPU");
  TORCH_CHECK(lambda_imag.device().is_cpu(), "lambda_imag must be on CPU");
  TORCH_CHECK(b_seq.device().is_cpu(), "b_seq must be on CPU");

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

  AT_DISPATCH_FLOATING_TYPES(lambda_real_contig.scalar_type(), "lru_scan_cpu", [&] {
    lru_scan_cpu_kernel<scalar_t>(lambda_real_contig.data_ptr<scalar_t>(),
                                  lambda_imag_contig.data_ptr<scalar_t>(),
                                  b_seq_contig.data_ptr<scalar_t>(),
                                  result.data_ptr<scalar_t>(),
                                  length,
                                  batch,
                                  state_size);
  });

  return result;
}

#ifdef WITH_CUDA
at::Tensor lru_scan_cuda(const at::Tensor& lambda_real,
                         const at::Tensor& lambda_imag,
                         const at::Tensor& b_seq);
#endif

}  // namespace

at::Tensor lru_scan(const at::Tensor& lambda_real,
                    const at::Tensor& lambda_imag,
                    const at::Tensor& b_seq) {
  if (b_seq.size(0) == 0) {
    return at::empty_like(b_seq);
  }

  TORCH_CHECK(b_seq.dim() == 4,
              "b_seq must have shape (length, batch, state, 2), got ",
              b_seq.sizes());

  if (b_seq.is_cuda()) {
#ifdef WITH_CUDA
    return lru_scan_cuda(lambda_real, lambda_imag, b_seq);
#else
    TORCH_CHECK(false, "lru_scan CUDA extension was not built");
#endif
  }

  return lru_scan_cpu(lambda_real, lambda_imag, b_seq);
}

}  // namespace ossm
