#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <torch/extension.h>

#include "dlinoss_common.h"
#include "sdlinoss_cpu_utils.h"

namespace ossm {
namespace {

constexpr double kDtMin = 1e-6;
constexpr double kDtMax = 1.0;

template <typename scalar_t>
inline typename ComplexTraits<scalar_t>::value_t clamp_step(
    typename ComplexTraits<scalar_t>::value_t raw) {
  using value_t = typename ComplexTraits<scalar_t>::value_t;
  if (std::isnan(raw)) {
    return value_t(kDtMin);
  }
  return std::min(std::max(raw, value_t(kDtMin)), value_t(kDtMax));
}

template <typename scalar_t>
void sdlinoss_ex_forward_cpu_kernel(
    const typename ComplexTraits<scalar_t>::value_t* __restrict__ A,
    const typename ComplexTraits<scalar_t>::value_t* __restrict__ G,
    const typename ComplexTraits<scalar_t>::value_t* __restrict__ step,
    const scalar_t* __restrict__ bu_ptr,
    scalar_t* __restrict__ out_ptr,
    int64_t length,
    int64_t batch,
    int64_t ssm) {
  using traits = ComplexTraits<scalar_t>;
  using value_t = typename traits::value_t;

  if (length == 0) {
    return;
  }

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  at::parallel_for(0, series, 1, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      value_t w_real = value_t(0);
      value_t w_imag = value_t(0);
      value_t x_real = value_t(0);
      value_t x_imag = value_t(0);

      for (int64_t t = 0; t < length; ++t) {
        const int64_t offset = t * series + idx;
        value_t a_t = A[offset];
        value_t g_t = G[offset];
        if (std::isnan(a_t)) {
          a_t = value_t(0);
        }
        if (std::isnan(g_t)) {
          g_t = value_t(0);
        }
        const value_t step_raw = step[offset];
        const value_t dt = clamp_step<scalar_t>(step_raw);

        const scalar_t bu_val = bu_ptr[offset];
        const value_t bu_real = bu_val.real();
        const value_t bu_imag = bu_val.imag();
        const value_t dt2 = dt * dt;
        const value_t alpha = value_t(1) - dt * g_t;

        const value_t comb_real = -a_t * x_real + bu_real;
        const value_t comb_imag = -a_t * x_imag + bu_imag;

        const value_t new_w_real = std::fma(dt2, comb_real, alpha * w_real);
        const value_t new_w_imag = std::fma(dt2, comb_imag, alpha * w_imag);
        const value_t new_x_real = x_real + new_w_real;
        const value_t new_x_imag = x_imag + new_w_imag;

        const int64_t out_offset = t * step_stride + idx * 2;
        out_ptr[out_offset] = scalar_t(new_w_real, new_w_imag);
        out_ptr[out_offset + 1] = scalar_t(new_x_real, new_x_imag);

        w_real = new_w_real;
        w_imag = new_w_imag;
        x_real = new_x_real;
        x_imag = new_x_imag;
      }
    }
  });
}

template <typename scalar_t>
void sdlinoss_ex_backward_cpu_kernel(
    const typename ComplexTraits<scalar_t>::value_t* __restrict__ A,
    const typename ComplexTraits<scalar_t>::value_t* __restrict__ G,
    const typename ComplexTraits<scalar_t>::value_t* __restrict__ step,
    const scalar_t* __restrict__ bu_ptr,
    const scalar_t* __restrict__ states_ptr,
    const scalar_t* __restrict__ grad_out_ptr,
    scalar_t* __restrict__ grad_bu_ptr,
    typename ComplexTraits<scalar_t>::value_t* __restrict__ grad_A_ptr,
    typename ComplexTraits<scalar_t>::value_t* __restrict__ grad_G_ptr,
    typename ComplexTraits<scalar_t>::value_t* __restrict__ grad_step_ptr,
    int64_t length,
    int64_t batch,
    int64_t ssm,
    bool allow_parallel) {
  using traits = ComplexTraits<scalar_t>;
  using value_t = typename traits::value_t;

  if (length == 0) {
    return;
  }

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  auto loop_body = [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      value_t grad_w_next_real = value_t(0);
      value_t grad_w_next_imag = value_t(0);
      value_t grad_x_next_real = value_t(0);
      value_t grad_x_next_imag = value_t(0);

      for (int64_t t = length - 1; t >= 0; --t) {
        const int64_t offset = t * series + idx;
        const int64_t out_offset = t * step_stride + idx * 2;

        const scalar_t state_w = states_ptr[out_offset];
        const scalar_t state_x = states_ptr[out_offset + 1];
        const value_t w_new_real = state_w.real();
        const value_t w_new_imag = state_w.imag();
        const value_t x_new_real = state_x.real();
        const value_t x_new_imag = state_x.imag();

        value_t w_prev_real = value_t(0);
        value_t w_prev_imag = value_t(0);
        value_t x_prev_real = value_t(0);
        value_t x_prev_imag = value_t(0);
        if (t > 0) {
          const int64_t prev_offset = out_offset - step_stride;
          const scalar_t prev_w = states_ptr[prev_offset];
          const scalar_t prev_x = states_ptr[prev_offset + 1];
          w_prev_real = prev_w.real();
          w_prev_imag = prev_w.imag();
          x_prev_real = prev_x.real();
          x_prev_imag = prev_x.imag();
        }

        const scalar_t grad_out = grad_out_ptr[offset];
        value_t grad_x_new_real = grad_out.real() + grad_x_next_real;
        value_t grad_x_new_imag = grad_out.imag() + grad_x_next_imag;
        value_t grad_w_new_real = grad_w_next_real;
        value_t grad_w_new_imag = grad_w_next_imag;

        value_t a_t = A[offset];
        value_t g_t = G[offset];
        if (std::isnan(a_t)) {
          a_t = value_t(0);
        }
        if (std::isnan(g_t)) {
          g_t = value_t(0);
        }
        const value_t step_raw = step[offset];
        const value_t dt = clamp_step<scalar_t>(step_raw);
        const value_t step_mask = (!std::isnan(step_raw) && step_raw > value_t(kDtMin) && step_raw < value_t(kDtMax))
                                      ? value_t(1)
                                      : value_t(0);
        const value_t dt2 = dt * dt;
        const value_t alpha = value_t(1) - dt * g_t;

        const scalar_t bu_val = bu_ptr[offset];
        const value_t bu_real = bu_val.real();
        const value_t bu_imag = bu_val.imag();

        value_t grad_step_local = value_t(0);
        value_t grad_A_local = value_t(0);
        value_t grad_G_local = value_t(0);

        grad_w_new_real += grad_x_new_real;
        grad_w_new_imag += grad_x_new_imag;
        value_t grad_x_prev_real = grad_x_new_real;
        value_t grad_x_prev_imag = grad_x_new_imag;

        const value_t comb_real = -a_t * x_prev_real + bu_real;
        const value_t comb_imag = -a_t * x_prev_imag + bu_imag;

        const value_t grad_w_prev_real = grad_w_new_real * alpha;
        const value_t grad_w_prev_imag = grad_w_new_imag * alpha;

        grad_step_local += grad_w_new_real * (-g_t * w_prev_real + value_t(2) * dt * comb_real) +
                           grad_w_new_imag * (-g_t * w_prev_imag + value_t(2) * dt * comb_imag);
        grad_G_local += grad_w_new_real * (-dt * w_prev_real) + grad_w_new_imag * (-dt * w_prev_imag);
        grad_A_local += grad_w_new_real * (-(dt2) * x_prev_real) +
                        grad_w_new_imag * (-(dt2) * x_prev_imag);

        grad_x_prev_real += grad_w_new_real * (-(dt2) * a_t);
        grad_x_prev_imag += grad_w_new_imag * (-(dt2) * a_t);

        const value_t grad_bu_real = grad_w_new_real * dt2;
        const value_t grad_bu_imag = grad_w_new_imag * dt2;

        const value_t step_grad = grad_step_local * step_mask;

        grad_A_ptr[offset] = grad_A_local;
        grad_G_ptr[offset] = grad_G_local;
        grad_step_ptr[offset] = step_grad;

        grad_bu_ptr[offset] = scalar_t(grad_bu_real, grad_bu_imag);

        grad_w_next_real = grad_w_prev_real;
        grad_w_next_imag = grad_w_prev_imag;
        grad_x_next_real = grad_x_prev_real;
        grad_x_next_imag = grad_x_prev_imag;
      }
    }
  };

  if (allow_parallel && series > 0) {
    at::parallel_for(0, series, 1, loop_body);
  } else {
    loop_body(0, series);
  }
}

#ifdef WITH_CUDA
}  // end anonymous namespace

void sdlinoss_ex_forward_cuda(const at::Tensor& A,
                              const at::Tensor& G,
                              const at::Tensor& step,
                              const at::Tensor& bu,
                              at::Tensor& output);

void sdlinoss_ex_backward_cuda(const at::Tensor& A,
                               const at::Tensor& G,
                               const at::Tensor& step,
                               const at::Tensor& bu,
                               const at::Tensor& states,
                               const at::Tensor& grad_output,
                               at::Tensor& grad_A,
                               at::Tensor& grad_G,
                               at::Tensor& grad_step,
                               at::Tensor& grad_bu);

namespace {  // reopen anonymous namespace for any additional helpers
#endif

}  // namespace

void sdlinoss_ex_forward_cpu(const at::Tensor& A,
                             const at::Tensor& G,
                             const at::Tensor& step,
                             const at::Tensor& bu,
                             at::Tensor& output) {
  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_ex_forward_cpu", [&] {
    sdlinoss_ex_forward_cpu_kernel<scalar_t>(
        A.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        G.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        step.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        bu.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        length,
        batch,
        ssm);
  });
}

void sdlinoss_ex_backward_cpu(const at::Tensor& A,
                              const at::Tensor& G,
                              const at::Tensor& step,
                              const at::Tensor& bu,
                              const at::Tensor& states,
                              const at::Tensor& grad_output,
                              at::Tensor& grad_A,
                              at::Tensor& grad_G,
                              at::Tensor& grad_step,
                              at::Tensor& grad_bu,
                              bool allow_parallel) {
  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_ex_backward_cpu", [&] {
    sdlinoss_ex_backward_cpu_kernel<scalar_t>(
        A.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        G.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        step.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        bu.data_ptr<scalar_t>(),
        states.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        grad_bu.data_ptr<scalar_t>(),
        grad_A.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        grad_G.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        grad_step.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
        length,
        batch,
        ssm,
        allow_parallel);
  });
}

torch::Tensor sdlinoss_ex_forward(const at::Tensor& A,
                                  const at::Tensor& G,
                                  const at::Tensor& step,
                                  const at::Tensor& bu) {
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (length, batch, ssm_size)");
  TORCH_CHECK(A.device() == bu.device() && G.device() == bu.device() && step.device() == bu.device(),
              "All tensors must live on the same device");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  auto output = at::empty({length, batch, ssm, 2}, bu.options());
  if (length == 0) {
    return output;
  }

  if (bu.is_cuda()) {
#ifdef WITH_CUDA
    validate_strided3_dims(A);
    validate_strided3_dims(G);
    validate_strided3_dims(step);
    sdlinoss_ex_forward_cuda(A, G, step, bu, output);
#else
    TORCH_CHECK(false, "sdlinoss_ex CUDA extension was not built");
#endif
  } else {
    TORCH_CHECK(bu.is_contiguous(), "bu must be contiguous");
    const auto A_view = normalize_param_view_cpu(A, "A", length, batch, ssm);
    const auto G_view = normalize_param_view_cpu(G, "G", length, batch, ssm);
    const auto step_view = normalize_param_view_cpu(step, "step", length, batch, ssm);
    const auto A_broadcast = materialize_param_cpu(A_view, length, batch, ssm);
    const auto G_broadcast = materialize_param_cpu(G_view, length, batch, ssm);
    const auto step_broadcast = materialize_param_cpu(step_view, length, batch, ssm);
    sdlinoss_ex_forward_cpu(A_broadcast, G_broadcast, step_broadcast, bu, output);
  }

  return output;
}
std::vector<at::Tensor> sdlinoss_ex_backward(const at::Tensor& A,
                                             const at::Tensor& G,
                                             const at::Tensor& step,
                                             const at::Tensor& bu,
                                             const at::Tensor& states,
                                             const at::Tensor& grad_output) {
  TORCH_CHECK(states.dim() == 4 && states.size(3) == 2,
              "states must have shape (length, batch, ssm_size, 2)");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (length, batch, ssm_size)");
  TORCH_CHECK(grad_output.sizes() == bu.sizes(), "grad_output must match bu shape");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  TORCH_CHECK(
      A.device() == bu.device() && G.device() == bu.device() && step.device() == bu.device(),
      "All tensors must live on the same device");

  at::Tensor grad_A;
  at::Tensor grad_G;
  at::Tensor grad_step;
  at::Tensor grad_bu;

  if (bu.is_cuda()) {
#ifdef WITH_CUDA
    validate_strided3_dims(A);
    validate_strided3_dims(G);
    validate_strided3_dims(step);
    TORCH_CHECK(states.is_contiguous() && grad_output.is_contiguous(),
                "states and grad_output must be contiguous on CUDA");
    grad_A = at::zeros_like(A);
    grad_G = at::zeros_like(G);
    grad_step = at::zeros_like(step);
    grad_bu = at::empty_like(bu);
    sdlinoss_ex_backward_cuda(A, G, step, bu, states, grad_output, grad_A, grad_G, grad_step, grad_bu);
#else
    TORCH_CHECK(false, "sdlinoss_ex CUDA extension was not built");
#endif
  } else {
    TORCH_CHECK(bu.is_contiguous() && states.is_contiguous() && grad_output.is_contiguous(),
                "bu, states, and grad_output must be contiguous");
    const auto A_view = normalize_param_view_cpu(A, "A", length, batch, ssm);
    const auto G_view = normalize_param_view_cpu(G, "G", length, batch, ssm);
    const auto step_view = normalize_param_view_cpu(step, "step", length, batch, ssm);

    const auto A_broadcast = materialize_param_cpu(A_view, length, batch, ssm);
    const auto G_broadcast = materialize_param_cpu(G_view, length, batch, ssm);
    const auto step_broadcast = materialize_param_cpu(step_view, length, batch, ssm);

    auto grad_A_buffer = at::zeros({length, batch, ssm}, A.options());
    auto grad_G_buffer = at::zeros({length, batch, ssm}, G.options());
    auto grad_step_buffer = at::zeros({length, batch, ssm}, step.options());
    grad_bu = at::empty_like(bu);

    const bool reduce_A =
        ((A_view.size(0) == 1 && length > 1) || (A_view.size(1) == 1 && batch > 1));
    const bool reduce_G =
        ((G_view.size(0) == 1 && length > 1) || (G_view.size(1) == 1 && batch > 1));
    const bool reduce_step =
        ((step_view.size(0) == 1 && length > 1) || (step_view.size(1) == 1 && batch > 1));
    const bool needs_reduction = reduce_A || reduce_G || reduce_step;

    sdlinoss_ex_backward_cpu(A_broadcast,
                             G_broadcast,
                             step_broadcast,
                             bu,
                             states,
                             grad_output,
                             grad_A_buffer,
                             grad_G_buffer,
                             grad_step_buffer,
                             grad_bu,
                             !needs_reduction);

    grad_A = reduce_broadcast_grad(grad_A_buffer, A_view, length, batch).reshape(A.sizes());
    grad_G = reduce_broadcast_grad(grad_G_buffer, G_view, length, batch).reshape(G.sizes());
    grad_step = reduce_broadcast_grad(grad_step_buffer, step_view, length, batch).reshape(step.sizes());
  }

  return {grad_A, grad_G, grad_step, grad_bu};
}

}  // namespace ossm
