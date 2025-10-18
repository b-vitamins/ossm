#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>
#include <vector>

namespace {

template <typename scalar_t>
void linoss_scan_cpu_kernel(
    const scalar_t* __restrict__ m11,
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

  at::parallel_for(0, series, 0, [&](int64_t begin, int64_t end) {
    const int64_t chunk = end - begin;
    std::vector<scalar_t> state0(chunk, scalar_t(0));
    std::vector<scalar_t> state1(chunk, scalar_t(0));
    std::vector<scalar_t> a11_vals(chunk);
    std::vector<scalar_t> a12_vals(chunk);
    std::vector<scalar_t> a21_vals(chunk);
    std::vector<scalar_t> a22_vals(chunk);

    for (int64_t offset = 0; offset < chunk; ++offset) {
      const int64_t idx = begin + offset;
      const int64_t state_index = idx % ssm;
      a11_vals[offset] = m11[state_index];
      a12_vals[offset] = m12[state_index];
      a21_vals[offset] = m21[state_index];
      a22_vals[offset] = m22[state_index];
    }

    for (int64_t t = 0; t < length; ++t) {
      const scalar_t* b_step = b_ptr + t * stride_t + begin * 2;
      scalar_t* out_step = out_ptr + t * stride_t + begin * 2;

      for (int64_t offset = 0; offset < chunk; ++offset) {
        const scalar_t prev0 = state0[offset];
        const scalar_t prev1 = state1[offset];

        const scalar_t b0 = b_step[offset * 2];
        const scalar_t b1 = b_step[offset * 2 + 1];

        const scalar_t new0 = a11_vals[offset] * prev0 + a12_vals[offset] * prev1 + b0;
        const scalar_t new1 = a21_vals[offset] * prev0 + a22_vals[offset] * prev1 + b1;

        out_step[offset * 2] = new0;
        out_step[offset * 2 + 1] = new1;

        state0[offset] = new0;
        state1[offset] = new1;
      }
    }
  });
}

void linoss_scan_cpu(
    const at::Tensor& m11,
    const at::Tensor& m12,
    const at::Tensor& m21,
    const at::Tensor& m22,
    const at::Tensor& b_seq,
    at::Tensor& output) {
  const auto length = b_seq.size(0);
  const auto batch = b_seq.size(1);
  const auto ssm = b_seq.size(2);

  AT_DISPATCH_COMPLEX_TYPES(b_seq.scalar_type(), "linoss_scan_cpu", [&] {
    linoss_scan_cpu_kernel<scalar_t>(
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
}

#ifdef WITH_CUDA
void linoss_scan_cuda(
    const at::Tensor& m11,
    const at::Tensor& m12,
    const at::Tensor& m21,
    const at::Tensor& m22,
    const at::Tensor& b_seq,
    at::Tensor& output);
#endif

}  // namespace

torch::Tensor linoss_scan(
    const at::Tensor& m11,
    const at::Tensor& m12,
    const at::Tensor& m21,
    const at::Tensor& m22,
    const at::Tensor& b_seq) {
  auto output = at::empty_like(b_seq);
  const auto length = b_seq.size(0);
  if (length == 0) {
    return output;
  }

  TORCH_CHECK(
      b_seq.dim() == 4,
      "b_seq must have shape (length, batch, ssm_size, 2), got ",
      b_seq.sizes());

  TORCH_CHECK(
      m11.is_contiguous() && m12.is_contiguous() && m21.is_contiguous() &&
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linoss_scan", &linoss_scan, "LinOSS associative scan kernel");
}

