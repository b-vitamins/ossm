#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include <vector>

namespace ossm {
namespace {

template <typename scalar_t>
void linoss_scan_cpu_kernel(const scalar_t* __restrict__ m11,
                            const scalar_t* __restrict__ m12,
                            const scalar_t* __restrict__ m21,
                            const scalar_t* __restrict__ m22,
                            const scalar_t* __restrict__ b_ptr,
                            scalar_t* __restrict__ out_ptr,
                            int64_t length,
                            int64_t batch,
                            int64_t ssm) {
  if (length == 0) {
    return;
  }

  const int64_t series = batch * ssm;
  const int64_t stride_t = series * 2;
  const int64_t stride_batch = ssm * 2;

  std::vector<scalar_t> state0(series, scalar_t(0));
  std::vector<scalar_t> state1(series, scalar_t(0));

  scalar_t* state0_ptr = state0.data();
  scalar_t* state1_ptr = state1.data();

  for (int64_t t = 0; t < length; ++t) {
    const scalar_t* time_b = b_ptr + t * stride_t;
    scalar_t* time_out = out_ptr + t * stride_t;

    at::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
      for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
        const scalar_t* b_row = time_b + batch_idx * stride_batch;
        scalar_t* out_row = time_out + batch_idx * stride_batch;

        scalar_t* row_state0 = state0_ptr + batch_idx * ssm;
        scalar_t* row_state1 = state1_ptr + batch_idx * ssm;

        for (int64_t state_idx = 0; state_idx < ssm; ++state_idx) {
          const scalar_t coeff11 = m11[state_idx];
          const scalar_t coeff12 = m12[state_idx];
          const scalar_t coeff21 = m21[state_idx];
          const scalar_t coeff22 = m22[state_idx];

          const int64_t base = state_idx * 2;

          const scalar_t prev0 = row_state0[state_idx];
          const scalar_t prev1 = row_state1[state_idx];

          const scalar_t b0 = b_row[base];
          const scalar_t b1 = b_row[base + 1];

          const scalar_t new0 = coeff11 * prev0 + coeff12 * prev1 + b0;
          const scalar_t new1 = coeff21 * prev0 + coeff22 * prev1 + b1;

          out_row[base] = new0;
          out_row[base + 1] = new1;

          row_state0[state_idx] = new0;
          row_state1[state_idx] = new1;
        }
      }
    });
  }
}

void linoss_scan_cpu(const at::Tensor& m11,
                     const at::Tensor& m12,
                     const at::Tensor& m21,
                     const at::Tensor& m22,
                     const at::Tensor& b_seq,
                     at::Tensor& output) {
  const auto length = b_seq.size(0);
  const auto batch = b_seq.size(1);
  const auto ssm = b_seq.size(2);

  AT_DISPATCH_COMPLEX_TYPES(b_seq.scalar_type(), "linoss_scan_cpu", [&] {
    linoss_scan_cpu_kernel<scalar_t>(m11.data_ptr<scalar_t>(),
                                     m12.data_ptr<scalar_t>(),
                                     m21.data_ptr<scalar_t>(),
                                     m22.data_ptr<scalar_t>(),
                                     b_seq.data_ptr<scalar_t>(),
                                     output.data_ptr<scalar_t>(),
                                     length,
                                     batch,
                                     ssm);
  });
}

#ifdef WITH_CUDA
void linoss_scan_cuda(const at::Tensor& m11,
                      const at::Tensor& m12,
                      const at::Tensor& m21,
                      const at::Tensor& m22,
                      const at::Tensor& b_seq,
                      at::Tensor& output);
#endif

}  // namespace

torch::Tensor linoss_scan(const at::Tensor& m11,
                           const at::Tensor& m12,
                           const at::Tensor& m21,
                           const at::Tensor& m22,
                           const at::Tensor& b_seq) {
  auto output = at::empty_like(b_seq);
  const auto length = b_seq.size(0);
  if (length == 0) {
    return output;
  }

  TORCH_CHECK(b_seq.dim() == 4,
              "b_seq must have shape (length, batch, ssm_size, 2), got ",
              b_seq.sizes());

  TORCH_CHECK(m11.is_contiguous() && m12.is_contiguous() && m21.is_contiguous() &&
                  m22.is_contiguous(),
              "Coefficient tensors must be contiguous");
  TORCH_CHECK(b_seq.is_contiguous(), "b_seq must be contiguous");

  if (b_seq.is_cuda()) {
#ifdef WITH_CUDA
    linoss_scan_cuda(m11, m12, m21, m22, b_seq, output);
#else
    TORCH_CHECK(false, "linoss_scan CUDA extension was not built");
#endif
  } else {
    linoss_scan_cpu(m11, m12, m21, m22, b_seq, output);
  }

  return output;
}

}  // namespace ossm
