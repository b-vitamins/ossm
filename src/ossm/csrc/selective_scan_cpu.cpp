#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Optional.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cstddef>

namespace ossm {
namespace {

void _check_inputs(const at::Tensor& inputs,
                   const at::Tensor& dt,
                   const at::Tensor& A,
                   const at::Tensor& B,
                   const at::Tensor& C,
                   const c10::optional<at::Tensor>& gate) {
  TORCH_CHECK(inputs.device().is_cpu(), "inputs must be on CPU");
  TORCH_CHECK(dt.device().is_cpu() && A.device().is_cpu() && B.device().is_cpu() &&
                  C.device().is_cpu(),
              "All tensors must be on CPU");
  TORCH_CHECK(
      inputs.scalar_type() == at::kFloat && dt.scalar_type() == at::kFloat &&
          A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat &&
          C.scalar_type() == at::kFloat,
      "selective_scan_cpu only supports float32 tensors");
  if (gate.has_value()) {
    TORCH_CHECK(gate->device().is_cpu(), "gate must be on CPU");
    TORCH_CHECK(gate->scalar_type() == at::kFloat, "gate must be float32");
  }

  TORCH_CHECK(inputs.dim() == 3 && dt.dim() == 3 && A.dim() == 2 && B.dim() == 3 &&
                  C.dim() == 3,
              "Unexpected tensor ranks for selective_scan_cpu");

  const auto batch = inputs.size(0);
  const auto channels = inputs.size(1);
  const auto length = inputs.size(2);

  TORCH_CHECK(dt.size(0) == batch && dt.size(1) == channels && dt.size(2) == length,
              "dt shape mismatch");
  TORCH_CHECK(B.size(0) == batch && B.size(1) == length,
              "B must have shape (batch, length, state)");
  TORCH_CHECK(C.size(0) == batch && C.size(1) == length,
              "C must have shape (batch, length, state)");
  TORCH_CHECK(A.size(0) == channels, "A must have shape (channels, state)");
  TORCH_CHECK(B.size(2) == A.size(1) && C.size(2) == A.size(1),
              "State dimension mismatch between A, B, and C");
  if (gate.has_value()) {
    TORCH_CHECK(gate->sizes() == inputs.sizes(),
                "Gate must have shape (batch, channels, length)");
  }
}

}  // namespace

namespace {

inline float _sigmoid(float x) {
  return 1.0f / (1.0f + static_cast<float>(std::exp(-x)));
}

inline float _silu(float x) {
  const float sigma = _sigmoid(x);
  return x * sigma;
}

inline float _dsilu(float x) {
  const float sigma = _sigmoid(x);
  return sigma * (1.0f + x * (1.0f - sigma));
}

}  // namespace

at::Tensor selective_scan_cpu(const at::Tensor& inputs,
                              const at::Tensor& dt,
                              const at::Tensor& A,
                              const at::Tensor& B,
                              const at::Tensor& C,
                              const c10::optional<at::Tensor>& gate) {
  _check_inputs(inputs, dt, A, B, C, gate);

  const auto batch = inputs.size(0);
  const auto channels = inputs.size(1);
  const auto length = inputs.size(2);
  const auto state_dim = A.size(1);

  if (length == 0) {
    return at::zeros_like(inputs);
  }

  auto x = inputs.contiguous();
  auto dt_contig = dt.contiguous();
  auto A_contig = A.contiguous();
  auto B_contig = B.contiguous();
  auto C_contig = C.contiguous();
  at::Tensor gate_contig;
  if (gate.has_value()) {
    gate_contig = gate->contiguous();
  }

  auto output = at::empty_like(x);

  const float* x_ptr = x.data_ptr<float>();
  const float* dt_ptr = dt_contig.data_ptr<float>();
  const float* A_ptr = A_contig.data_ptr<float>();
  const float* B_ptr = B_contig.data_ptr<float>();
  const float* C_ptr = C_contig.data_ptr<float>();
  const float* gate_ptr = gate.has_value() ? gate_contig.data_ptr<float>() : nullptr;
  float* out_ptr = output.data_ptr<float>();

  const auto num_threads = at::get_num_threads();
  at::Tensor state_buffer = at::zeros({num_threads, state_dim}, x.options());
  float* state_base = state_buffer.data_ptr<float>();

  at::parallel_for(0, batch * channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bc = begin; bc < end; ++bc) {
      const int thread_id = at::get_thread_num();
      float* state = state_base + static_cast<int64_t>(thread_id) * state_dim;
      std::fill_n(state, state_dim, 0.0f);

      const int64_t b = bc / channels;
      const int64_t c = bc % channels;

      const float* A_row = A_ptr + c * state_dim;

      const float* x_lane = x_ptr + (static_cast<int64_t>(b) * channels + c) * length;
      const float* dt_lane = dt_ptr + (static_cast<int64_t>(b) * channels + c) * length;
      const float* gate_lane = gate_ptr ? gate_ptr + (static_cast<int64_t>(b) * channels + c) * length : nullptr;
      float* out_lane = out_ptr + (static_cast<int64_t>(b) * channels + c) * length;

      for (int64_t t = 0; t < length; ++t) {
        const float dt_scalar = dt_lane[t];
        const float input_scalar = x_lane[t];
        const float* B_step =
            B_ptr + (static_cast<int64_t>(b) * length + t) * state_dim;
        const float* C_step =
            C_ptr + (static_cast<int64_t>(b) * length + t) * state_dim;

        float accumulator = 0.0f;

#pragma omp simd reduction(+ : accumulator)
        for (int64_t n = 0; n < state_dim; ++n) {
          const float a = A_row[n];
          const float delta = dt_scalar * a;
          const float A_bar = static_cast<float>(std::exp(delta));
          const float updated = std::fmaf(dt_scalar * B_step[n], input_scalar, A_bar * state[n]);
          state[n] = updated;
          accumulator += updated * C_step[n];
        }

        float value = accumulator;
        if (gate_lane != nullptr) {
          const float g = gate_lane[t];
          value *= _silu(g);
        }
        out_lane[t] = value;
      }
    }
  });

  return output;
}

std::vector<at::Tensor> selective_scan_cpu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& inputs,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& gate) {
  _check_inputs(inputs, dt, A, B, C, gate);
  TORCH_CHECK(
      grad_output.sizes() == inputs.sizes(),
      "grad_output must match the shape of inputs");

  const auto batch = inputs.size(0);
  const auto channels = inputs.size(1);
  const auto length = inputs.size(2);
  const auto state_dim = A.size(1);

  if (length == 0) {
    auto zeros = at::zeros_like(inputs);
    auto zeros_dt = at::zeros_like(dt);
    auto zeros_A = at::zeros_like(A);
    auto zeros_B = at::zeros_like(B);
    auto zeros_C = at::zeros_like(C);
    auto zeros_gate = gate.has_value() ? at::zeros_like(*gate) : at::zeros_like(inputs);
    return {zeros, zeros_dt, zeros_A, zeros_B, zeros_C, zeros_gate};
  }

  auto grad_y = grad_output.contiguous();
  auto x = inputs.contiguous();
  auto dt_contig = dt.contiguous();
  auto A_contig = A.contiguous();
  auto B_contig = B.contiguous();
  auto C_contig = C.contiguous();
  at::Tensor gate_contig;
  if (gate.has_value()) {
    gate_contig = gate->contiguous();
  }

  auto grad_inputs = at::empty_like(x);
  auto grad_dt = at::empty_like(dt_contig);
  auto grad_A = at::zeros_like(A_contig);
  auto grad_B = at::zeros_like(B_contig);
  auto grad_C = at::zeros_like(C_contig);
  auto grad_gate = gate.has_value() ? at::empty_like(gate_contig)
                                    : at::zeros_like(x);

  const float* x_ptr = x.data_ptr<float>();
  const float* dt_ptr = dt_contig.data_ptr<float>();
  const float* A_ptr = A_contig.data_ptr<float>();
  const float* B_ptr = B_contig.data_ptr<float>();
  const float* C_ptr = C_contig.data_ptr<float>();
  const float* grad_y_ptr = grad_y.data_ptr<float>();
  const float* gate_ptr = gate.has_value() ? gate_contig.data_ptr<float>() : nullptr;

  float* grad_x_ptr = grad_inputs.data_ptr<float>();
  float* grad_dt_ptr = grad_dt.data_ptr<float>();
  float* grad_A_ptr = grad_A.data_ptr<float>();
  float* grad_B_ptr = grad_B.data_ptr<float>();
  float* grad_C_ptr = grad_C.data_ptr<float>();
  float* grad_gate_ptr = grad_gate.data_ptr<float>();

  at::parallel_for(0, batch * channels, 0, [&](int64_t begin, int64_t end) {
    std::vector<float> A_row(static_cast<std::size_t>(state_dim));
    std::vector<float> lane_state(static_cast<std::size_t>(length) * state_dim);
    std::vector<float> grad_state(static_cast<std::size_t>(state_dim));

    for (int64_t bc = begin; bc < end; ++bc) {
      const int64_t b = bc / channels;
      const int64_t c = bc % channels;

      const float* x_lane =
          x_ptr + (static_cast<int64_t>(b) * channels + c) * length;
      const float* dt_lane =
          dt_ptr + (static_cast<int64_t>(b) * channels + c) * length;
      const float* grad_y_lane =
          grad_y_ptr + (static_cast<int64_t>(b) * channels + c) * length;
      const float* gate_lane = gate_ptr
                                   ? gate_ptr + (static_cast<int64_t>(b) * channels + c) * length
                                   : nullptr;

      float* grad_x_lane =
          grad_x_ptr + (static_cast<int64_t>(b) * channels + c) * length;
      float* grad_dt_lane =
          grad_dt_ptr + (static_cast<int64_t>(b) * channels + c) * length;
      float* grad_gate_lane = gate_ptr
                                  ? grad_gate_ptr +
                                        (static_cast<int64_t>(b) * channels + c) * length
                                  : nullptr;

      const float* A_row_ptr = A_ptr + c * state_dim;

      for (int64_t n = 0; n < state_dim; ++n) {
        const float a = A_row_ptr[n];
        A_row[static_cast<std::size_t>(n)] = a;
      }

      std::fill(grad_state.begin(), grad_state.end(), 0.0f);

      for (int64_t t = 0; t < length; ++t) {
        const float dt_scalar = dt_lane[t];
        const float input_scalar = x_lane[t];
        const float* B_step =
            B_ptr + (static_cast<int64_t>(b) * length + t) * state_dim;

        std::size_t base_index = static_cast<std::size_t>(t) * state_dim;
#pragma omp simd
        for (int64_t n = 0; n < state_dim; ++n) {
          const float a = A_row[static_cast<std::size_t>(n)];
          const float prev = (t == 0) ? 0.0f
                                      : lane_state[base_index - state_dim + n];
          const float delta = dt_scalar * a;
          const float A_bar = static_cast<float>(std::exp(delta));
          const float updated =
              std::fmaf(dt_scalar * B_step[n], input_scalar, A_bar * prev);
          lane_state[base_index + static_cast<std::size_t>(n)] = updated;
        }
      }

      for (int64_t t = length - 1; t >= 0; --t) {
        std::size_t base_index = static_cast<std::size_t>(t) * state_dim;
        const float* C_step =
            C_ptr + (static_cast<int64_t>(b) * length + t) * state_dim;
        const float* B_step =
            B_ptr + (static_cast<int64_t>(b) * length + t) * state_dim;

        float y_t = 0.0f;
#pragma omp simd reduction(+ : y_t)
        for (int64_t n = 0; n < state_dim; ++n) {
          y_t += lane_state[base_index + static_cast<std::size_t>(n)] * C_step[n];
        }

        const float grad_y_scalar = grad_y_lane[t];
        float grad_y_gated = grad_y_scalar;
        float grad_gate_scalar = 0.0f;
        if (gate_lane != nullptr) {
          const float gate_value = gate_lane[t];
          grad_y_gated = grad_y_scalar * _silu(gate_value);
          grad_gate_scalar = grad_y_scalar * y_t * _dsilu(gate_value);
          grad_gate_lane[t] = grad_gate_scalar;
        }

        float grad_input = 0.0f;
        float grad_dt_scalar = 0.0f;

        for (int64_t n = 0; n < state_dim; ++n) {
          const float a = A_row[static_cast<std::size_t>(n)];
          const float state_t = lane_state[base_index + static_cast<std::size_t>(n)];
          const float state_prev =
              (t == 0) ? 0.0f
                       : lane_state[base_index - state_dim + static_cast<std::size_t>(n)];

          const float dt_scalar = dt_lane[t];
          const float input_scalar = x_lane[t];

          const float delta = dt_scalar * a;
          const float A_bar = static_cast<float>(std::exp(delta));

          float grad_state_accum =
              grad_state[static_cast<std::size_t>(n)] + grad_y_gated * C_step[n];

          grad_state[static_cast<std::size_t>(n)] = grad_state_accum * A_bar;

          const float grad_C = grad_y_gated * state_t;
#pragma omp atomic
          grad_C_ptr[(static_cast<int64_t>(b) * length + t) * state_dim + n] += grad_C;

          const float grad_B = grad_state_accum * dt_scalar * input_scalar;
#pragma omp atomic
          grad_B_ptr[(static_cast<int64_t>(b) * length + t) * state_dim + n] += grad_B;

          grad_input += grad_state_accum * dt_scalar * B_step[n];

          const float grad_A_update = grad_state_accum * (dt_scalar * A_bar * state_prev);

#pragma omp atomic
          grad_A_ptr[c * state_dim + n] += grad_A_update;
          grad_dt_scalar +=
              grad_state_accum * (a * A_bar * state_prev + B_step[n] * input_scalar);
        }

        grad_x_lane[t] = grad_input;
        grad_dt_lane[t] = grad_dt_scalar;
      }
    }
  });

  if (!gate.has_value()) {
    grad_gate.zero_();
  }

  return {grad_inputs, grad_dt, grad_A, grad_B, grad_C, grad_gate};
}

}  // namespace ossm
