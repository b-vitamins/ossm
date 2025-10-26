#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include <vector>

#include "dlinoss_common.h"

namespace ossm {
namespace {

template <typename scalar_t>
void dlinoss_imex2_forward_cpu_kernel(const typename ComplexTraits<scalar_t>::value_t* __restrict__ a_diag,
                                      const typename ComplexTraits<scalar_t>::value_t* __restrict__ g_diag,
                                      const typename ComplexTraits<scalar_t>::value_t* __restrict__ step,
                                      const scalar_t* __restrict__ bu_ptr,
                                      scalar_t* __restrict__ out_ptr,
                                      int64_t length,
                                      int64_t batch,
                                      int64_t ssm) {
  using traits = ComplexTraits<scalar_t>;
  using value_t = typename traits::value_t;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  std::vector<value_t> sigma(ssm);
  std::vector<value_t> sigma_sq(ssm);
  std::vector<value_t> m11(ssm);
  std::vector<value_t> m12(ssm);
  std::vector<value_t> m21(ssm);
  std::vector<value_t> m22(ssm);

  for (int64_t state = 0; state < ssm; ++state) {
    const value_t alpha = a_diag[state];
    const value_t gamma = g_diag[state];
    const value_t sigma_val = step[state];
    const value_t sigma_sq_val = sigma_val * sigma_val;

    sigma[state] = sigma_val;
    sigma_sq[state] = sigma_sq_val;
    m11[state] = value_t(1) - sigma_val * gamma;
    m12[state] = -sigma_val * alpha;
    m21[state] = sigma_val - sigma_sq_val * gamma;
    m22[state] = value_t(1) - sigma_sq_val * alpha;
  }

  at::parallel_for(0, ssm, 1, [&](int64_t begin, int64_t end) {
    for (int64_t state = begin; state < end; ++state) {
      const value_t sigma_val = sigma[state];
      const value_t sigma_sq_val = sigma_sq[state];
      const value_t m11_val = m11[state];
      const value_t m12_val = m12[state];
      const value_t m21_val = m21[state];
      const value_t m22_val = m22[state];

      for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const int64_t series_idx = batch_idx * ssm + state;

        value_t state0_real = value_t(0);
        value_t state0_imag = value_t(0);
        value_t state1_real = value_t(0);
        value_t state1_imag = value_t(0);

        const scalar_t* bu_series = bu_ptr + series_idx;
        scalar_t* out_series = out_ptr + series_idx * 2;

        for (int64_t t = 0; t < length; ++t) {
          const scalar_t bu_val = *bu_series;
          const value_t bu_real = bu_val.real();
          const value_t bu_imag = bu_val.imag();

          const value_t new0_real = state0_real * m11_val + state1_real * m12_val + bu_real * sigma_val;
          const value_t new0_imag = state0_imag * m11_val + state1_imag * m12_val + bu_imag * sigma_val;
          const value_t new1_real = state0_real * m21_val + state1_real * m22_val + bu_real * sigma_sq_val;
          const value_t new1_imag = state0_imag * m21_val + state1_imag * m22_val + bu_imag * sigma_sq_val;

          out_series[0] = scalar_t(new0_real, new0_imag);
          out_series[1] = scalar_t(new1_real, new1_imag);

          state0_real = new0_real;
          state0_imag = new0_imag;
          state1_real = new1_real;
          state1_imag = new1_imag;

          bu_series += series;
          out_series += step_stride;
        }
      }
    }
  });
}

template <typename scalar_t>
void dlinoss_imex2_backward_cpu_kernel(const typename ComplexTraits<scalar_t>::value_t* __restrict__ a_diag,
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
  using traits = ComplexTraits<scalar_t>;
  using value_t = typename traits::value_t;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  std::vector<value_t> sigma(ssm);
  std::vector<value_t> sigma_sq(ssm);
  std::vector<value_t> m11(ssm);
  std::vector<value_t> m12(ssm);
  std::vector<value_t> m21(ssm);
  std::vector<value_t> m22(ssm);
  std::vector<value_t> d_sigma_prev0(ssm);
  std::vector<value_t> d_sigma_prev1(ssm);
  std::vector<value_t> d_sigma_bu(ssm);

  for (int64_t state = 0; state < ssm; ++state) {
    const value_t alpha = a_diag[state];
    const value_t gamma = g_diag[state];
    const value_t sigma_val = step[state];
    const value_t sigma_sq_val = sigma_val * sigma_val;

    sigma[state] = sigma_val;
    sigma_sq[state] = sigma_sq_val;
    m11[state] = value_t(1) - sigma_val * gamma;
    m12[state] = -sigma_val * alpha;
    m21[state] = sigma_val - sigma_sq_val * gamma;
    m22[state] = value_t(1) - sigma_sq_val * alpha;
    d_sigma_prev0[state] = value_t(1) - value_t(2) * sigma_val * gamma;
    d_sigma_prev1[state] = -value_t(2) * sigma_val * alpha;
    d_sigma_bu[state] = value_t(2) * sigma_val;
  }

  at::parallel_for(0, ssm, 1, [&](int64_t begin, int64_t end) {
    for (int64_t state = begin; state < end; ++state) {
      const value_t alpha = a_diag[state];
      const value_t gamma = g_diag[state];
      const value_t sigma_val = sigma[state];
      const value_t sigma_sq_val = sigma_sq[state];
      const value_t m11_val = m11[state];
      const value_t m12_val = m12[state];
      const value_t m21_val = m21[state];
      const value_t m22_val = m22[state];
      const value_t d_sigma_prev0_val = d_sigma_prev0[state];
      const value_t d_sigma_prev1_val = d_sigma_prev1[state];
      const value_t d_sigma_bu_val = d_sigma_bu[state];

      value_t grad_alpha_state = value_t(0);
      value_t grad_gamma_state = value_t(0);
      value_t grad_sigma_state = value_t(0);

      for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const int64_t series_idx = batch_idx * ssm + state;

        value_t grad_state0_real = value_t(0);
        value_t grad_state0_imag = value_t(0);
        value_t grad_state1_real = value_t(0);
        value_t grad_state1_imag = value_t(0);

        const scalar_t* bu_series = bu_ptr + (length - 1) * series + series_idx;
        const scalar_t* grad_out_series = grad_out_ptr + (length - 1) * series + series_idx;
        scalar_t* grad_bu_series = grad_bu_ptr + (length - 1) * series + series_idx;
        const scalar_t* prev_states =
            length > 1 ? states_ptr + (length - 2) * step_stride + series_idx * 2 : nullptr;

        for (int64_t t = length - 1; t >= 0; --t) {
          const scalar_t grad_out_val = *grad_out_series;
          const scalar_t bu_val = *bu_series;

          const bool has_prev = t > 0;
          value_t prev0_real = value_t(0);
          value_t prev0_imag = value_t(0);
          value_t prev1_real = value_t(0);
          value_t prev1_imag = value_t(0);
          if (has_prev) {
            const scalar_t prev0_val = prev_states[0];
            const scalar_t prev1_val = prev_states[1];
            prev0_real = prev0_val.real();
            prev0_imag = prev0_val.imag();
            prev1_real = prev1_val.real();
            prev1_imag = prev1_val.imag();
          }

          const value_t grad_out_real = grad_out_val.real();
          const value_t grad_out_imag = grad_out_val.imag();
          const value_t bu_real = bu_val.real();
          const value_t bu_imag = bu_val.imag();

          const value_t grad_new1_real = grad_state1_real + grad_out_real;
          const value_t grad_new1_imag = grad_state1_imag + grad_out_imag;
          const value_t grad_new0_real = grad_state0_real;
          const value_t grad_new0_imag = grad_state0_imag;

          grad_alpha_state += (-sigma_val) * (grad_new0_real * prev1_real + grad_new0_imag * prev1_imag);
          grad_alpha_state += (-sigma_sq_val) * (grad_new1_real * prev1_real + grad_new1_imag * prev1_imag);

          grad_gamma_state += (-sigma_val) * (grad_new0_real * prev0_real + grad_new0_imag * prev0_imag);
          grad_gamma_state += (-sigma_sq_val) * (grad_new1_real * prev0_real + grad_new1_imag * prev0_imag);

          const value_t sigma_term_real = prev0_real * (-gamma) + prev1_real * (-alpha) + bu_real;
          const value_t sigma_term_imag = prev0_imag * (-gamma) + prev1_imag * (-alpha) + bu_imag;
          grad_sigma_state += grad_new0_real * sigma_term_real + grad_new0_imag * sigma_term_imag;

          const value_t sigma_grad_term_real = prev0_real * d_sigma_prev0_val +
                                               prev1_real * d_sigma_prev1_val +
                                               bu_real * d_sigma_bu_val;
          const value_t sigma_grad_term_imag = prev0_imag * d_sigma_prev0_val +
                                               prev1_imag * d_sigma_prev1_val +
                                               bu_imag * d_sigma_bu_val;
          grad_sigma_state +=
              grad_new1_real * sigma_grad_term_real + grad_new1_imag * sigma_grad_term_imag;

          const value_t grad_bu_real = grad_new0_real * sigma_val + grad_new1_real * sigma_sq_val;
          const value_t grad_bu_imag = grad_new0_imag * sigma_val + grad_new1_imag * sigma_sq_val;
          *grad_bu_series = scalar_t(grad_bu_real, grad_bu_imag);

          const value_t next_state0_real = grad_new0_real * m11_val + grad_new1_real * m21_val;
          const value_t next_state0_imag = grad_new0_imag * m11_val + grad_new1_imag * m21_val;
          const value_t next_state1_real = grad_new0_real * m12_val + grad_new1_real * m22_val;
          const value_t next_state1_imag = grad_new0_imag * m12_val + grad_new1_imag * m22_val;

          grad_state0_real = next_state0_real;
          grad_state0_imag = next_state0_imag;
          grad_state1_real = next_state1_real;
          grad_state1_imag = next_state1_imag;

          if (has_prev) {
            prev_states -= step_stride;
          }

          bu_series -= series;
          grad_out_series -= series;
          grad_bu_series -= series;
        }
      }

      grad_a_ptr[state] = grad_alpha_state;
      grad_step_ptr[state] = grad_sigma_state;
      grad_g_ptr[state] = grad_gamma_state;
    }
  });
}

void dlinoss_imex2_forward_cpu(const at::Tensor& a_diag,
                                const at::Tensor& g_diag,
                                const at::Tensor& step,
                                const at::Tensor& bu,
                                at::Tensor& output) {
  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "dlinoss_imex2_forward_cpu", [&] {
    dlinoss_imex2_forward_cpu_kernel<scalar_t>(a_diag.data_ptr<typename scalar_t::value_type>(),
                                               g_diag.data_ptr<typename scalar_t::value_type>(),
                                               step.data_ptr<typename scalar_t::value_type>(),
                                               bu.data_ptr<scalar_t>(),
                                               output.data_ptr<scalar_t>(),
                                               length,
                                               batch,
                                               ssm);
  });
}

void dlinoss_imex2_backward_cpu(const at::Tensor& a_diag,
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

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "dlinoss_imex2_backward_cpu", [&] {
    dlinoss_imex2_backward_cpu_kernel<scalar_t>(a_diag.data_ptr<typename scalar_t::value_type>(),
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

void dlinoss_imex2_forward_cuda(const at::Tensor& a_diag,
                                const at::Tensor& g_diag,
                                const at::Tensor& step,
                                const at::Tensor& bu,
                                at::Tensor& output);

void dlinoss_imex2_backward_cuda(const at::Tensor& a_diag,
                                 const at::Tensor& g_diag,
                                 const at::Tensor& step,
                                 const at::Tensor& bu,
                                 const at::Tensor& states,
                                 const at::Tensor& grad_output,
                                 at::Tensor& grad_a,
                                 at::Tensor& grad_g,
                                 at::Tensor& grad_step,
                                 at::Tensor& grad_bu);

namespace {
#endif

}  // namespace

torch::Tensor dlinoss_imex2_forward(const at::Tensor& a_diag,
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
    dlinoss_imex2_forward_cuda(a_diag, g_diag, step, bu, output);
#else
    TORCH_CHECK(false, "dlinoss_imex2 CUDA extension was not built");
#endif
  } else {
    dlinoss_imex2_forward_cpu(a_diag, g_diag, step, bu, output);
  }

  return output;
}

std::vector<at::Tensor> dlinoss_imex2_backward(const at::Tensor& a_diag,
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
    dlinoss_imex2_backward_cuda(a_diag, g_diag, step, bu, states, grad_output, grad_a, grad_g, grad_step, grad_bu);
#else
    TORCH_CHECK(false, "dlinoss_imex2 CUDA extension was not built");
#endif
  } else {
    dlinoss_imex2_backward_cpu(a_diag, g_diag, step, bu, states, grad_output, grad_a, grad_g, grad_step, grad_bu);
  }

  return {grad_a, grad_g, grad_step, grad_bu};
}

}  // namespace ossm
