#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include "dlinoss_common.h"

namespace ossm {
namespace {

constexpr double kDtMin = 1e-6;
constexpr double kDtMax = 1.0;
constexpr double kClampMin = 1e-6;

template <typename scalar_t>
inline typename ComplexTraits<scalar_t>::value_t clamp_step(
    typename ComplexTraits<scalar_t>::value_t raw) {
  using value_t = typename ComplexTraits<scalar_t>::value_t;
  const value_t clamped = std::min(std::max(raw, value_t(kDtMin)), value_t(kDtMax));
  return clamped;
}

template <typename scalar_t>
inline typename ComplexTraits<scalar_t>::value_t clamp_stability(
    typename ComplexTraits<scalar_t>::value_t raw) {
  using value_t = typename ComplexTraits<scalar_t>::value_t;
  return std::max(raw, value_t(kClampMin));
}

template <typename scalar_t>
void sdlinoss_imex2_forward_cpu_kernel(
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
      value_t z_real = value_t(0);
      value_t z_imag = value_t(0);
      value_t x_real = value_t(0);
      value_t x_imag = value_t(0);

      for (int64_t t = 0; t < length; ++t) {
        const int64_t offset = t * series + idx;
        const value_t a_t = A[offset];
        const value_t g_t = G[offset];
        const value_t step_raw = step[offset];
        const value_t dt = clamp_step<scalar_t>(step_raw);
        const value_t dt2 = dt * dt;

        const value_t S_raw = value_t(1) + dt2 * a_t;
        const value_t S = clamp_stability<scalar_t>(S_raw);
        const value_t inv_S = value_t(1) / S;

        const scalar_t bu_val = bu_ptr[offset];
        const value_t bu_real = bu_val.real();
        const value_t bu_imag = bu_val.imag();

        const value_t tmp_real = z_real + dt * (-a_t * x_real - g_t * z_real + bu_real);
        const value_t tmp_imag = z_imag + dt * (-a_t * x_imag - g_t * z_imag + bu_imag);

        const value_t new_z_real = tmp_real * inv_S;
        const value_t new_z_imag = tmp_imag * inv_S;
        const value_t new_x_real = x_real + dt * new_z_real;
        const value_t new_x_imag = x_imag + dt * new_z_imag;

        const int64_t out_offset = t * step_stride + idx * 2;
        out_ptr[out_offset] = scalar_t(new_z_real, new_z_imag);
        out_ptr[out_offset + 1] = scalar_t(new_x_real, new_x_imag);

        z_real = new_z_real;
        z_imag = new_z_imag;
        x_real = new_x_real;
        x_imag = new_x_imag;
      }
    }
  });
}

template <typename scalar_t>
void sdlinoss_imex2_backward_cpu_kernel(
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
      value_t grad_z_next_real = value_t(0);
      value_t grad_z_next_imag = value_t(0);
      value_t grad_x_next_real = value_t(0);
      value_t grad_x_next_imag = value_t(0);

      for (int64_t t = length - 1; t >= 0; --t) {
        const int64_t offset = t * series + idx;
        const int64_t out_offset = t * step_stride + idx * 2;

        const scalar_t state_z = states_ptr[out_offset];
        const scalar_t state_x = states_ptr[out_offset + 1];
        const value_t z_new_real = state_z.real();
        const value_t z_new_imag = state_z.imag();
        const value_t x_new_real = state_x.real();
        const value_t x_new_imag = state_x.imag();

        value_t z_prev_real = value_t(0);
        value_t z_prev_imag = value_t(0);
        value_t x_prev_real = value_t(0);
        value_t x_prev_imag = value_t(0);
        if (t > 0) {
          const int64_t prev_offset = out_offset - step_stride;
          const scalar_t prev_z = states_ptr[prev_offset];
          const scalar_t prev_x = states_ptr[prev_offset + 1];
          z_prev_real = prev_z.real();
          z_prev_imag = prev_z.imag();
          x_prev_real = prev_x.real();
          x_prev_imag = prev_x.imag();
        }

        const scalar_t grad_out = grad_out_ptr[offset];
        const value_t grad_out_real = grad_out.real();
        const value_t grad_out_imag = grad_out.imag();

        value_t grad_x_new_real = grad_out_real + grad_x_next_real;
        value_t grad_x_new_imag = grad_out_imag + grad_x_next_imag;
        value_t grad_z_new_real = grad_z_next_real;
        value_t grad_z_new_imag = grad_z_next_imag;

        const value_t a_t = A[offset];
        const value_t g_t = G[offset];
        const value_t step_raw = step[offset];
        const value_t dt = clamp_step<scalar_t>(step_raw);
        const value_t step_mask = (step_raw > value_t(kDtMin) && step_raw < value_t(kDtMax))
                                      ? value_t(1)
                                      : value_t(0);
        const value_t dt2 = dt * dt;

        const scalar_t bu_val = bu_ptr[offset];
        const value_t bu_real = bu_val.real();
        const value_t bu_imag = bu_val.imag();

        value_t grad_step_local = value_t(0);
        value_t grad_A_local = value_t(0);
        value_t grad_G_local = value_t(0);

        grad_step_local += grad_x_new_real * z_new_real + grad_x_new_imag * z_new_imag;
        grad_z_new_real += dt * grad_x_new_real;
        grad_z_new_imag += dt * grad_x_new_imag;
        value_t grad_x_prev_real = grad_x_new_real;
        value_t grad_x_prev_imag = grad_x_new_imag;

        const value_t S_raw = value_t(1) + dt2 * a_t;
        const value_t S = clamp_stability<scalar_t>(S_raw);
        const value_t clamp_mask = S_raw > value_t(kClampMin) ? value_t(1) : value_t(0);
        const value_t inv_S = value_t(1) / S;
        const value_t inv_S_sq = inv_S * inv_S;

        const value_t comb_real = -a_t * x_prev_real - g_t * z_prev_real + bu_real;
        const value_t comb_imag = -a_t * x_prev_imag - g_t * z_prev_imag + bu_imag;

        const value_t tmp_real = z_prev_real + dt * comb_real;
        const value_t tmp_imag = z_prev_imag + dt * comb_imag;

        const value_t grad_temp_real = grad_z_new_real * inv_S;
        const value_t grad_temp_imag = grad_z_new_imag * inv_S;

        const value_t grad_S = -(grad_z_new_real * tmp_real + grad_z_new_imag * tmp_imag) * inv_S_sq;
        const value_t grad_S_raw = grad_S * clamp_mask;

        grad_step_local += grad_S_raw * (value_t(2) * dt * a_t);
        grad_A_local += grad_S_raw * dt2;

        grad_step_local += grad_temp_real * comb_real + grad_temp_imag * comb_imag;
        grad_A_local += grad_temp_real * (-dt * x_prev_real) + grad_temp_imag * (-dt * x_prev_imag);
        grad_G_local += grad_temp_real * (-dt * z_prev_real) + grad_temp_imag * (-dt * z_prev_imag);

        grad_x_prev_real += grad_temp_real * (-dt * a_t);
        grad_x_prev_imag += grad_temp_imag * (-dt * a_t);

        const value_t grad_z_prev_factor = value_t(1) - dt * g_t;
        const value_t grad_z_prev_real = grad_temp_real * grad_z_prev_factor;
        const value_t grad_z_prev_imag = grad_temp_imag * grad_z_prev_factor;

        const value_t grad_bu_real = grad_temp_real * dt;
        const value_t grad_bu_imag = grad_temp_imag * dt;

        grad_A_ptr[offset] = grad_A_local;
        grad_G_ptr[offset] = grad_G_local;
        grad_step_ptr[offset] = grad_step_local * step_mask;
        grad_bu_ptr[offset] = scalar_t(grad_bu_real, grad_bu_imag);

        grad_x_next_real = grad_x_prev_real;
        grad_x_next_imag = grad_x_prev_imag;
        grad_z_next_real = grad_z_prev_real;
        grad_z_next_imag = grad_z_prev_imag;
      }
    }
  });
}

#ifdef WITH_CUDA
}  // end anonymous namespace

// Declare CUDA implementations with external linkage in the ossm namespace to
// match the definitions in sdlinoss_imex2_cuda.cu and avoid undefined symbols at
// import time.
void sdlinoss_imex2_forward_cuda(const at::Tensor& A,
                                 const at::Tensor& G,
                                 const at::Tensor& step,
                                 const at::Tensor& bu,
                                 at::Tensor& output);

void sdlinoss_imex2_backward_cuda(const at::Tensor& A,
                                  const at::Tensor& G,
                                  const at::Tensor& step,
                                  const at::Tensor& bu,
                                  const at::Tensor& states,
                                  const at::Tensor& grad_output,
                                  at::Tensor& grad_A,
                                  at::Tensor& grad_G,
                                  at::Tensor& grad_step,
                                  at::Tensor& grad_bu);

namespace {  // reopen anonymous namespace for the remaining helpers
#endif

}  // namespace

void sdlinoss_imex2_forward_cpu(const at::Tensor& A,
                                const at::Tensor& G,
                                const at::Tensor& step,
                                const at::Tensor& bu,
                                at::Tensor& output) {
  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_imex2_forward_cpu", [&] {
    sdlinoss_imex2_forward_cpu_kernel<scalar_t>(A.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
                                                G.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
                                                step.data_ptr<typename ComplexTraits<scalar_t>::value_t>(),
                                                bu.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(),
                                                length,
                                                batch,
                                                ssm);
  });
}

void sdlinoss_imex2_backward_cpu(const at::Tensor& A,
                                 const at::Tensor& G,
                                 const at::Tensor& step,
                                 const at::Tensor& bu,
                                 const at::Tensor& states,
                                 const at::Tensor& grad_output,
                                 at::Tensor& grad_A,
                                 at::Tensor& grad_G,
                                 at::Tensor& grad_step,
                                 at::Tensor& grad_bu) {
  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_imex2_backward_cpu", [&] {
    sdlinoss_imex2_backward_cpu_kernel<scalar_t>(
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
        ssm);
  });
}

torch::Tensor sdlinoss_imex2_forward(const at::Tensor& A,
                                     const at::Tensor& G,
                                     const at::Tensor& step,
                                     const at::Tensor& bu) {
  TORCH_CHECK(A.dim() == 3 && G.dim() == 3 && step.dim() == 3,
              "A, G, and step must have shape (length, batch, ssm_size)");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (length, batch, ssm_size)");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);

  TORCH_CHECK(A.sizes() == bu.sizes() && G.sizes() == bu.sizes() && step.sizes() == bu.sizes(),
              "A, G, and step must match bu shape");

  auto output = at::empty({length, batch, ssm, 2}, bu.options());
  if (length == 0) {
    return output;
  }

  TORCH_CHECK(A.is_contiguous() && G.is_contiguous() && step.is_contiguous(),
              "A, G, and step must be contiguous");
  TORCH_CHECK(bu.is_contiguous(), "bu must be contiguous");

  if (bu.is_cuda()) {
#ifdef WITH_CUDA
    sdlinoss_imex2_forward_cuda(A, G, step, bu, output);
#else
    TORCH_CHECK(false, "sdlinoss_imex2 CUDA extension was not built");
#endif
  } else {
    sdlinoss_imex2_forward_cpu(A, G, step, bu, output);
  }

  return output;
}

std::vector<at::Tensor> sdlinoss_imex2_backward(const at::Tensor& A,
                                                const at::Tensor& G,
                                                const at::Tensor& step,
                                                const at::Tensor& bu,
                                                const at::Tensor& states,
                                                const at::Tensor& grad_output) {
  TORCH_CHECK(states.dim() == 4 && states.size(3) == 2,
              "states must have shape (length, batch, ssm_size, 2)");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (length, batch, ssm_size)");
  TORCH_CHECK(A.sizes() == bu.sizes() && G.sizes() == bu.sizes() && step.sizes() == bu.sizes(),
              "A, G, and step must match bu shape");
  TORCH_CHECK(grad_output.sizes() == bu.sizes(), "grad_output must match bu shape");

  auto grad_A = at::zeros_like(A);
  auto grad_G = at::zeros_like(G);
  auto grad_step = at::zeros_like(step);
  auto grad_bu = at::empty_like(bu);

  TORCH_CHECK(A.is_contiguous() && G.is_contiguous() && step.is_contiguous(),
              "A, G, and step must be contiguous");
  TORCH_CHECK(bu.is_contiguous() && states.is_contiguous() && grad_output.is_contiguous(),
              "bu, states, and grad_output must be contiguous");

  if (bu.is_cuda()) {
#ifdef WITH_CUDA
    sdlinoss_imex2_backward_cuda(A, G, step, bu, states, grad_output, grad_A, grad_G, grad_step, grad_bu);
#else
    TORCH_CHECK(false, "sdlinoss_imex2 CUDA extension was not built");
#endif
  } else {
    sdlinoss_imex2_backward_cpu(A, G, step, bu, states, grad_output, grad_A, grad_G, grad_step, grad_bu);
  }

  return {grad_A, grad_G, grad_step, grad_bu};
}

}  // namespace ossm
