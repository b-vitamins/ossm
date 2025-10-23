#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include <vector>

namespace ossm {
namespace {

template <typename scalar_t>
struct ComplexTraits {
  using value_t = typename scalar_t::value_type;
};

template <typename scalar_t>
void dlinoss_imex1_forward_cpu_kernel(const typename ComplexTraits<scalar_t>::value_t* __restrict__ a_diag,
                                      const typename ComplexTraits<scalar_t>::value_t* __restrict__ g_diag,
                                      const typename ComplexTraits<scalar_t>::value_t* __restrict__ step,
                                      const scalar_t* __restrict__ bu_ptr,
                                      scalar_t* __restrict__ out_ptr,
                                      int64_t length,
                                      int64_t batch,
                                      int64_t ssm) {
  using value_t = typename ComplexTraits<scalar_t>::value_t;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  at::parallel_for(0, ssm, 1, [&](int64_t begin, int64_t end) {
    for (int64_t state = begin; state < end; ++state) {
      const value_t alpha = a_diag[state];
      const value_t gamma = g_diag[state];
      const value_t sigma = step[state];

      const value_t denom = value_t(1) + sigma * gamma;
      const value_t inv = value_t(1) / denom;
      const value_t sigma_inv = sigma * inv;
      const value_t sigma2_inv = sigma * sigma * inv;
      const value_t coeff12 = -alpha * sigma_inv;
      const value_t coeff22 = value_t(1) - alpha * sigma2_inv;

      for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const int64_t series_idx = batch_idx * ssm + state;

        value_t state0_real = value_t(0);
        value_t state0_imag = value_t(0);
        value_t state1_real = value_t(0);
        value_t state1_imag = value_t(0);

        for (int64_t t = 0; t < length; ++t) {
          const int64_t bu_offset = t * series + series_idx;
          const int64_t out_offset = t * step_stride + series_idx * 2;

          const scalar_t bu_val = bu_ptr[bu_offset];
          const value_t bu_real = bu_val.real();
          const value_t bu_imag = bu_val.imag();

          const value_t new0_real = state0_real * inv + state1_real * coeff12 + bu_real * sigma_inv;
          const value_t new0_imag = state0_imag * inv + state1_imag * coeff12 + bu_imag * sigma_inv;
          const value_t new1_real = state0_real * sigma_inv + state1_real * coeff22 + bu_real * sigma2_inv;
          const value_t new1_imag = state0_imag * sigma_inv + state1_imag * coeff22 + bu_imag * sigma2_inv;

          out_ptr[out_offset] = scalar_t(new0_real, new0_imag);
          out_ptr[out_offset + 1] = scalar_t(new1_real, new1_imag);

          state0_real = new0_real;
          state0_imag = new0_imag;
          state1_real = new1_real;
          state1_imag = new1_imag;
        }
      }
    }
  });
}

template <typename scalar_t>
void dlinoss_imex1_backward_cpu_kernel(const typename ComplexTraits<scalar_t>::value_t* __restrict__ a_diag,
                                       const typename ComplexTraits<scalar_t>::value_t* __restrict__ g_diag,
                                       const typename ComplexTraits<scalar_t>::value_t* __restrict__ step,
                                       const scalar_t* __restrict__ bu_ptr,
                                       const scalar_t* __restrict__ states_ptr,
                                       const scalar_t* __restrict__ grad_out_ptr,
                                       scalar_t* __restrict__ grad_bu_ptr,
                                       typename ComplexTraits<scalar_t>::value_t* __restrict__ grad_a_ptr,
                                       typename ComplexTraits<scalar_t>::value_t* __restrict__ grad_g_ptr,
                                       typename ComplexTraits<scalar_t>::value_t* __restrict__ grad_step_ptr,
                                       int64_t length,
                                       int64_t batch,
                                       int64_t ssm) {
  using value_t = typename ComplexTraits<scalar_t>::value_t;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  at::parallel_for(0, ssm, 1, [&](int64_t begin, int64_t end) {
    for (int64_t state = begin; state < end; ++state) {
      const value_t alpha = a_diag[state];
      const value_t gamma = g_diag[state];
      const value_t sigma = step[state];

      const value_t denom = value_t(1) + sigma * gamma;
      const value_t inv = value_t(1) / denom;
      const value_t sigma_inv = sigma * inv;
      const value_t sigma2_inv = sigma * sigma * inv;
      const value_t coeff12 = -alpha * sigma_inv;
      const value_t coeff22 = value_t(1) - alpha * sigma2_inv;

      value_t grad_alpha_state = value_t(0);
      value_t grad_sigma_inv_state = value_t(0);
      value_t grad_sigma2_inv_state = value_t(0);
      value_t grad_inv_state = value_t(0);

      for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const int64_t series_idx = batch_idx * ssm + state;

        value_t grad_state0_real = value_t(0);
        value_t grad_state0_imag = value_t(0);
        value_t grad_state1_real = value_t(0);
        value_t grad_state1_imag = value_t(0);

        for (int64_t t = length - 1; t >= 0; --t) {
          const int64_t base_offset = t * step_stride + series_idx * 2;
          const int64_t bu_offset = t * series + series_idx;

          const scalar_t prev0_val = t > 0 ? states_ptr[base_offset - step_stride] : scalar_t(0, 0);
          const scalar_t prev1_val = t > 0 ? states_ptr[base_offset - step_stride + 1] : scalar_t(0, 0);
          const scalar_t bu_val = bu_ptr[bu_offset];
          const scalar_t grad_out_val = grad_out_ptr[bu_offset];

          const value_t prev0_real = prev0_val.real();
          const value_t prev0_imag = prev0_val.imag();
          const value_t prev1_real = prev1_val.real();
          const value_t prev1_imag = prev1_val.imag();
          const value_t bu_real = bu_val.real();
          const value_t bu_imag = bu_val.imag();

          const value_t grad_out_real = grad_out_val.real();
          const value_t grad_out_imag = grad_out_val.imag();

          const value_t grad_new1_real = grad_state1_real + grad_out_real;
          const value_t grad_new1_imag = grad_state1_imag + grad_out_imag;
          const value_t grad_new0_real = grad_state0_real;
          const value_t grad_new0_imag = grad_state0_imag;

          const value_t neg_alpha_prev1_real = -alpha * prev1_real;
          const value_t neg_alpha_prev1_imag = -alpha * prev1_imag;
          const value_t bu_combined_real = neg_alpha_prev1_real + bu_real;
          const value_t bu_combined_imag = neg_alpha_prev1_imag + bu_imag;

          grad_sigma_inv_state += grad_new0_real * bu_combined_real + grad_new0_imag * bu_combined_imag;
          grad_sigma_inv_state += grad_new1_real * prev0_real + grad_new1_imag * prev0_imag;
          grad_sigma2_inv_state += grad_new1_real * bu_combined_real + grad_new1_imag * bu_combined_imag;
          grad_alpha_state += grad_new0_real * (-sigma_inv * prev1_real) +
                              grad_new0_imag * (-sigma_inv * prev1_imag);
          grad_alpha_state += grad_new1_real * (-sigma2_inv * prev1_real) +
                              grad_new1_imag * (-sigma2_inv * prev1_imag);
          grad_inv_state += grad_new0_real * prev0_real + grad_new0_imag * prev0_imag;

          const value_t grad_bu_real = grad_new0_real * sigma_inv + grad_new1_real * sigma2_inv;
          const value_t grad_bu_imag = grad_new0_imag * sigma_inv + grad_new1_imag * sigma2_inv;
          grad_bu_ptr[bu_offset] = scalar_t(grad_bu_real, grad_bu_imag);

          const value_t grad_prev0_real = grad_new0_real * inv + grad_new1_real * sigma_inv;
          const value_t grad_prev0_imag = grad_new0_imag * inv + grad_new1_imag * sigma_inv;
          const value_t grad_prev1_real = grad_new0_real * coeff12 + grad_new1_real * coeff22;
          const value_t grad_prev1_imag = grad_new0_imag * coeff12 + grad_new1_imag * coeff22;

          grad_state0_real = grad_prev0_real;
          grad_state0_imag = grad_prev0_imag;
          grad_state1_real = grad_prev1_real;
          grad_state1_imag = grad_prev1_imag;
        }
      }

      value_t grad_sigma_state = grad_sigma_inv_state * inv;
      value_t grad_inv_total = grad_inv_state + grad_sigma_inv_state * sigma;

      const value_t grad_sigma_sq = grad_sigma2_inv_state * inv;
      grad_inv_total += grad_sigma2_inv_state * sigma * sigma;
      grad_sigma_state += grad_sigma_sq * value_t(2) * sigma;

      const value_t inv_sq = inv * inv;
      grad_sigma_state += grad_inv_total * (-gamma) * inv_sq;
      const value_t grad_gamma_state = grad_inv_total * (-sigma) * inv_sq;

      grad_a_ptr[state] = grad_alpha_state;
      grad_step_ptr[state] = grad_sigma_state;
      grad_g_ptr[state] = grad_gamma_state;
    }
  });
}

void dlinoss_imex1_forward_cpu(const at::Tensor& a_diag,
                                const at::Tensor& g_diag,
                                const at::Tensor& step,
                                const at::Tensor& bu,
                                at::Tensor& output) {
  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "dlinoss_imex1_forward_cpu", [&] {
    dlinoss_imex1_forward_cpu_kernel<scalar_t>(a_diag.data_ptr<typename scalar_t::value_type>(),
                                               g_diag.data_ptr<typename scalar_t::value_type>(),
                                               step.data_ptr<typename scalar_t::value_type>(),
                                               bu.data_ptr<scalar_t>(),
                                               output.data_ptr<scalar_t>(),
                                               length,
                                               batch,
                                               ssm);
  });
}

void dlinoss_imex1_backward_cpu(const at::Tensor& a_diag,
                                 const at::Tensor& g_diag,
                                 const at::Tensor& step,
                                 const at::Tensor& bu,
                                 const at::Tensor& states,
                                 const at::Tensor& grad_output,
                                 at::Tensor& grad_a,
                                 at::Tensor& grad_g,
                                 at::Tensor& grad_step,
                                 at::Tensor& grad_bu) {
  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "dlinoss_imex1_backward_cpu", [&] {
    dlinoss_imex1_backward_cpu_kernel<scalar_t>(a_diag.data_ptr<typename scalar_t::value_type>(),
                                                g_diag.data_ptr<typename scalar_t::value_type>(),
                                                step.data_ptr<typename scalar_t::value_type>(),
                                                bu.data_ptr<scalar_t>(),
                                                states.data_ptr<scalar_t>(),
                                                grad_output.data_ptr<scalar_t>(),
                                                grad_bu.data_ptr<scalar_t>(),
                                                grad_a.data_ptr<typename scalar_t::value_type>(),
                                                grad_g.data_ptr<typename scalar_t::value_type>(),
                                                grad_step.data_ptr<typename scalar_t::value_type>(),
                                                length,
                                                batch,
                                                ssm);
  });
}

#ifdef WITH_CUDA
}  // end anonymous namespace

// Declare CUDA implementations with external linkage in the ossm namespace to
// match the definitions in dlinoss_imex1_cuda.cu and avoid undefined symbols at
// import time.
void dlinoss_imex1_forward_cuda(const at::Tensor& a_diag,
                                const at::Tensor& g_diag,
                                const at::Tensor& step,
                                const at::Tensor& bu,
                                at::Tensor& output);

void dlinoss_imex1_backward_cuda(const at::Tensor& a_diag,
                                 const at::Tensor& g_diag,
                                 const at::Tensor& step,
                                 const at::Tensor& bu,
                                 const at::Tensor& states,
                                 const at::Tensor& grad_output,
                                 at::Tensor& grad_a,
                                 at::Tensor& grad_g,
                                 at::Tensor& grad_step,
                                 at::Tensor& grad_bu);

namespace {  // reopen anonymous namespace for the remaining helpers
#endif

}  // namespace

torch::Tensor dlinoss_imex1_forward(const at::Tensor& a_diag,
                                    const at::Tensor& g_diag,
                                    const at::Tensor& step,
                                    const at::Tensor& bu) {
  TORCH_CHECK(a_diag.dim() == 1 && g_diag.dim() == 1 && step.dim() == 1,
              "Coefficient diagonals must be 1-D tensors");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (length, batch, ssm_size)");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  TORCH_CHECK(a_diag.size(0) == ssm && g_diag.size(0) == ssm && step.size(0) == ssm,
              "Coefficient diagonals must match state size");

  auto output = at::empty({length, batch, ssm, 2}, bu.options());
  if (length == 0) {
    return output;
  }

  TORCH_CHECK(a_diag.is_contiguous() && g_diag.is_contiguous() && step.is_contiguous(),
              "Coefficient diagonals must be contiguous");
  TORCH_CHECK(bu.is_contiguous(), "bu must be contiguous");

  if (bu.is_cuda()) {
#ifdef WITH_CUDA
    dlinoss_imex1_forward_cuda(a_diag, g_diag, step, bu, output);
#else
    TORCH_CHECK(false, "dlinoss_imex1 CUDA extension was not built");
#endif
  } else {
    dlinoss_imex1_forward_cpu(a_diag, g_diag, step, bu, output);
  }

  return output;
}

std::vector<at::Tensor> dlinoss_imex1_backward(const at::Tensor& a_diag,
                                               const at::Tensor& g_diag,
                                               const at::Tensor& step,
                                               const at::Tensor& bu,
                                               const at::Tensor& states,
                                               const at::Tensor& grad_output) {
  TORCH_CHECK(states.dim() == 4 && states.size(3) == 2,
              "states must have shape (length, batch, ssm_size, 2)");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (length, batch, ssm_size)");
  TORCH_CHECK(grad_output.sizes() == bu.sizes(),
              "grad_output must match bu shape");

  auto grad_a = at::zeros_like(a_diag);
  auto grad_g = at::zeros_like(g_diag);
  auto grad_step = at::zeros_like(step);
  auto grad_bu = at::empty_like(bu);

  const auto length = bu.size(0);
  if (length == 0) {
    return {grad_a, grad_g, grad_step, grad_bu};
  }

  TORCH_CHECK(states.is_contiguous() && grad_output.is_contiguous(),
              "states and grad_output must be contiguous");

  if (bu.is_cuda()) {
#ifdef WITH_CUDA
    dlinoss_imex1_backward_cuda(a_diag, g_diag, step, bu, states, grad_output, grad_a, grad_g, grad_step, grad_bu);
#else
    TORCH_CHECK(false, "dlinoss_imex1 CUDA extension was not built");
#endif
  } else {
    dlinoss_imex1_backward_cpu(a_diag, g_diag, step, bu, states, grad_output, grad_a, grad_g, grad_step, grad_bu);
  }

  return {grad_a, grad_g, grad_step, grad_bu};
}

}  // namespace ossm
