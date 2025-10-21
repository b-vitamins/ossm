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

        scalar_t state0 = scalar_t(0, 0);
        scalar_t state1 = scalar_t(0, 0);

        for (int64_t t = 0; t < length; ++t) {
          const int64_t bu_offset = t * series + series_idx;
          const int64_t out_offset = t * step_stride + series_idx * 2;

          const scalar_t bu_val = bu_ptr[bu_offset];

          const scalar_t new0 = state0 * inv + state1 * coeff12 + bu_val * sigma_inv;
          const scalar_t new1 = state0 * sigma_inv + state1 * coeff22 + bu_val * sigma2_inv;

          out_ptr[out_offset] = new0;
          out_ptr[out_offset + 1] = new1;

          state0 = new0;
          state1 = new1;
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

        scalar_t grad_state0 = scalar_t(0, 0);
        scalar_t grad_state1 = scalar_t(0, 0);

        for (int64_t t = length - 1; t >= 0; --t) {
          const int64_t base_offset = t * step_stride + series_idx * 2;
          const int64_t bu_offset = t * series + series_idx;

          const scalar_t prev0 = t > 0 ? states_ptr[base_offset - step_stride] : scalar_t(0, 0);
          const scalar_t prev1 = t > 0 ? states_ptr[base_offset - step_stride + 1] : scalar_t(0, 0);
          const scalar_t bu_val = bu_ptr[bu_offset];

          const scalar_t grad_new1 = grad_state1 + grad_out_ptr[bu_offset];
          const scalar_t grad_new0 = grad_state0;

          const scalar_t conj_grad0(grad_new0.real(), -grad_new0.imag());
          const scalar_t conj_grad1(grad_new1.real(), -grad_new1.imag());

          const scalar_t neg_alpha_prev1 = prev1 * (-alpha);
          const scalar_t bu_combined = neg_alpha_prev1 + bu_val;

          grad_sigma_inv_state += (conj_grad0 * bu_combined).real() + (conj_grad1 * prev0).real();
          grad_sigma2_inv_state += (conj_grad1 * bu_combined).real();
          grad_alpha_state += (conj_grad0 * (prev1 * (-sigma_inv))).real() +
                              (conj_grad1 * (prev1 * (-sigma2_inv))).real();
          grad_inv_state += (conj_grad0 * prev0).real();

          grad_bu_ptr[bu_offset] = grad_new0 * sigma_inv + grad_new1 * sigma2_inv;

          const scalar_t grad_prev0 = grad_new0 * inv + grad_new1 * sigma_inv;
          const scalar_t grad_prev1 = grad_new0 * coeff12 + grad_new1 * coeff22;

          grad_state0 = grad_prev0;
          grad_state1 = grad_prev1;
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
