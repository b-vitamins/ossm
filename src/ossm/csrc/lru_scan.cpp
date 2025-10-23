#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <c10/util/complex.h>
#include <torch/extension.h>
#include <vector>

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
  const auto time_stride = total_series;
  const auto batch_stride = state_size;

  std::vector<complex_t> lambda_values(state_size);
  for (int64_t s = 0; s < state_size; ++s) {
    lambda_values[s] = complex_t(lambda_real[s], lambda_imag[s]);
  }

  const auto* b_complex = reinterpret_cast<const complex_t*>(b_seq);
  auto* out_complex = reinterpret_cast<complex_t*>(out);

  at::parallel_for(0, batch, 0, [&](int64_t batch_start, int64_t batch_end) {
    const int64_t local_batch = batch_end - batch_start;
    if (local_batch == 0) {
      return;
    }

    std::vector<complex_t> state(local_batch * state_size, complex_t(0, 0));
    const complex_t* lambda_ptr = lambda_values.data();

    for (int64_t t = 0; t < length; ++t) {
      const int64_t base_offset = t * time_stride + batch_start * batch_stride;
      const complex_t* b_step = b_complex + base_offset;
      complex_t* out_step = out_complex + base_offset;
      complex_t* state_ptr = state.data();

      for (int64_t b = 0; b < local_batch; ++b) {
        const complex_t* b_row = b_step + b * batch_stride;
        complex_t* out_row = out_step + b * batch_stride;
        complex_t* state_row = state_ptr + b * batch_stride;

        for (int64_t s = 0; s < state_size; ++s) {
          const complex_t updated = lambda_ptr[s] * state_row[s] + b_row[s];
          state_row[s] = updated;
          out_row[s] = updated;
        }
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
}  // end anonymous namespace

// Declare CUDA implementation with external linkage in the ossm namespace
// to match the definition in lru_scan_cuda.cu and avoid undefined symbols.
at::Tensor lru_scan_cuda(const at::Tensor& lambda_real,
                         const at::Tensor& lambda_imag,
                         const at::Tensor& b_seq);

namespace {  // reopen anonymous namespace
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
