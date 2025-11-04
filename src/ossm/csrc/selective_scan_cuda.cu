/******************************************************************************
 * OSSM selective scan CUDA wrapper built atop the mamba-ssm kernels.
 *
 * Portions of this file incorporate code from the mamba repository
 * (https://github.com/state-spaces/mamba) which is licensed under the
 * Apache License, Version 2.0. The relevant headers and device kernels
 * are vendored under ``src/ossm/csrc/mamba`` without functional changes
 * beyond namespace adjustments.
 ******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "mamba/selective_scan.h"
#include "mamba/selective_scan_bwd_kernel.cuh"
#include "mamba/selective_scan_common.h"
#include "mamba/selective_scan_fwd_kernel.cuh"

namespace ossm {
namespace {

constexpr int kMaxStateDim = MAX_DSTATE;

int kernel_chunk_span(int64_t length) {
#ifdef USE_ROCM
  if (length <= 256) {
    return 256;
  }
  if (length <= 512) {
    return 512;
  }
  if (length <= 1024) {
    return 1024;
  }
  return 2048;
#else
  if (length <= 128) {
    return 128;
  }
  if (length <= 256) {
    return 256;
  }
  if (length <= 512) {
    return 512;
  }
  if (length <= 1024) {
    return 1024;
  }
  return 2048;
#endif
}

void check_inputs(const at::Tensor& inputs,
                  const at::Tensor& dt,
                  const at::Tensor& A,
                  const at::Tensor& B,
                  const at::Tensor& C,
                  const c10::optional<at::Tensor>& gate) {
  TORCH_CHECK(inputs.device().is_cuda(), "inputs must be CUDA tensor");
  TORCH_CHECK(inputs.dtype() == at::kFloat,
              "selective_scan_cuda requires float32 inputs");
  TORCH_CHECK(inputs.dim() == 3, "inputs must have shape (batch, channels, length)");

  TORCH_CHECK(dt.device() == inputs.device(), "dt must be on the same device as inputs");
  TORCH_CHECK(dt.dtype() == at::kFloat, "dt must be float32");
  TORCH_CHECK(dt.sizes() == inputs.sizes(), "dt shape mismatch");

  TORCH_CHECK(A.device() == inputs.device(), "A must be on the same device as inputs");
  TORCH_CHECK(A.dtype() == at::kFloat, "A must be float32");
  TORCH_CHECK(A.dim() == 2, "A must have shape (channels, state)");
  TORCH_CHECK(A.size(0) == inputs.size(1), "A channels mismatch");
  TORCH_CHECK(A.size(1) > 0, "state dimension must be positive");
  TORCH_CHECK(A.size(1) <= kMaxStateDim,
              "state dimension exceeds supported maximum (", kMaxStateDim, ")");

  TORCH_CHECK(B.device() == inputs.device(), "B must be on the same device as inputs");
  TORCH_CHECK(B.dtype() == at::kFloat, "B must be float32");
  TORCH_CHECK(B.dim() == 3, "B must have shape (batch, length, state)");
  TORCH_CHECK(B.size(0) == inputs.size(0) && B.size(1) == inputs.size(2) &&
                  B.size(2) == A.size(1),
              "B shape mismatch");

  TORCH_CHECK(C.device() == inputs.device(), "C must be on the same device as inputs");
  TORCH_CHECK(C.dtype() == at::kFloat, "C must be float32");
  TORCH_CHECK(C.dim() == 3, "C must have shape (batch, length, state)");
  TORCH_CHECK(C.sizes() == B.sizes(), "C shape mismatch");

  if (gate.has_value()) {
    const at::Tensor& gate_tensor = gate.value();
    TORCH_CHECK(gate_tensor.device() == inputs.device(),
                "gate must be on the same device as inputs");
    TORCH_CHECK(gate_tensor.dtype() == at::kFloat, "gate must be float32");
    TORCH_CHECK(gate_tensor.sizes() == inputs.sizes(), "gate shape mismatch");
  }
}

at::Tensor permute_bc_for_kernel(const at::Tensor& tensor) {
  // Input layout: (batch, length, state)
  // Kernel expects: (batch, groups, state, length) with groups==1.
  return tensor.permute({0, 2, 1}).contiguous().unsqueeze(1).contiguous();
}

int compute_num_chunks(int64_t length) {
  const int span = kernel_chunk_span(length);
  return static_cast<int>((length + span - 1) / span);
}

void populate_fwd_params(SSMParamsBase& params,
                         const at::Tensor& inputs,
                         const at::Tensor& dt,
                         const at::Tensor& A,
                         const at::Tensor& B_kernel,
                         const at::Tensor& C_kernel,
                         const at::Tensor& raw_out,
                         const at::Tensor& outputs,
                         const c10::optional<at::Tensor>& gate,
                         const at::Tensor& scan_intermediates) {
  params.batch = static_cast<int>(inputs.size(0));
  params.dim = static_cast<int>(inputs.size(1));
  params.seqlen = static_cast<int>(inputs.size(2));
  params.dstate = static_cast<int>(A.size(1));
  params.n_groups = 1;
  params.n_chunks = compute_num_chunks(params.seqlen);
  params.dim_ngroups_ratio = params.dim;

  params.is_variable_B = true;
  params.is_variable_C = true;
  params.delta_softplus = false;

  params.A_ptr = A.data_ptr();
  params.B_ptr = B_kernel.data_ptr();
  params.C_ptr = C_kernel.data_ptr();
  params.D_ptr = nullptr;
  params.delta_bias_ptr = nullptr;
  params.u_ptr = inputs.data_ptr();
  params.delta_ptr = dt.data_ptr();
  params.out_ptr = raw_out.data_ptr();
  params.x_ptr = scan_intermediates.data_ptr();
  params.z_ptr = gate.has_value() ? gate->data_ptr() : nullptr;
  params.out_z_ptr = gate.has_value() ? outputs.data_ptr() : nullptr;

  params.A_d_stride = static_cast<SSMParamsBase::index_t>(A.stride(0));
  params.A_dstate_stride = static_cast<SSMParamsBase::index_t>(A.stride(1));

  params.B_batch_stride = static_cast<SSMParamsBase::index_t>(B_kernel.stride(0));
  params.B_group_stride = static_cast<SSMParamsBase::index_t>(B_kernel.stride(1));
  params.B_d_stride = 0;  // unused when is_variable_B == true
  params.B_dstate_stride = static_cast<SSMParamsBase::index_t>(B_kernel.stride(2));

  params.C_batch_stride = static_cast<SSMParamsBase::index_t>(C_kernel.stride(0));
  params.C_group_stride = static_cast<SSMParamsBase::index_t>(C_kernel.stride(1));
  params.C_d_stride = 0;  // unused when is_variable_C == true
  params.C_dstate_stride = static_cast<SSMParamsBase::index_t>(C_kernel.stride(2));

  params.u_batch_stride = static_cast<SSMParamsBase::index_t>(inputs.stride(0));
  params.u_d_stride = static_cast<SSMParamsBase::index_t>(inputs.stride(1));

  params.delta_batch_stride = static_cast<SSMParamsBase::index_t>(dt.stride(0));
  params.delta_d_stride = static_cast<SSMParamsBase::index_t>(dt.stride(1));

  params.out_batch_stride = static_cast<SSMParamsBase::index_t>(raw_out.stride(0));
  params.out_d_stride = static_cast<SSMParamsBase::index_t>(raw_out.stride(1));

  params.z_batch_stride = gate.has_value()
                              ? static_cast<SSMParamsBase::index_t>(gate->stride(0))
                              : 0;
  params.z_d_stride = gate.has_value()
                          ? static_cast<SSMParamsBase::index_t>(gate->stride(1))
                          : 0;

  params.out_z_batch_stride = gate.has_value()
                                  ? static_cast<SSMParamsBase::index_t>(outputs.stride(0))
                                  : 0;
  params.out_z_d_stride = gate.has_value()
                              ? static_cast<SSMParamsBase::index_t>(outputs.stride(1))
                              : 0;
}

void populate_bwd_params(SSMParamsBwd& params,
                         const at::Tensor& inputs,
                         const at::Tensor& dt,
                         const at::Tensor& A,
                         const at::Tensor& B_kernel,
                         const at::Tensor& C_kernel,
                         const c10::optional<at::Tensor>& gate,
                         const at::Tensor& raw_out,
                         const at::Tensor& grad_output,
                         const at::Tensor& chunk_states,
                         const at::Tensor& grad_inputs,
                         const at::Tensor& grad_dt,
                         const at::Tensor& grad_A,
                         const at::Tensor& grad_B_kernel,
                         const at::Tensor& grad_C_kernel,
                         const at::Tensor& grad_gate) {
  populate_fwd_params(params,
                      inputs,
                      dt,
                      A,
                      B_kernel,
                      C_kernel,
                      raw_out,
                      raw_out,
                      gate,
                      chunk_states);

  if (gate.has_value()) {
    params.out_z_ptr = nullptr;
  }

  params.dout_ptr = grad_output.data_ptr();
  params.du_ptr = grad_inputs.data_ptr();
  params.ddelta_ptr = grad_dt.data_ptr();
  params.dA_ptr = grad_A.data_ptr();
  params.dB_ptr = grad_B_kernel.data_ptr();
  params.dC_ptr = grad_C_kernel.data_ptr();
  params.dD_ptr = nullptr;
  params.ddelta_bias_ptr = nullptr;
  params.dz_ptr = gate.has_value() ? grad_gate.data_ptr() : nullptr;

  params.dout_batch_stride = static_cast<SSMParamsBwd::index_t>(grad_output.stride(0));
  params.dout_d_stride = static_cast<SSMParamsBwd::index_t>(grad_output.stride(1));
  params.du_batch_stride = static_cast<SSMParamsBwd::index_t>(grad_inputs.stride(0));
  params.du_d_stride = static_cast<SSMParamsBwd::index_t>(grad_inputs.stride(1));
  params.ddelta_batch_stride = static_cast<SSMParamsBwd::index_t>(grad_dt.stride(0));
  params.ddelta_d_stride = static_cast<SSMParamsBwd::index_t>(grad_dt.stride(1));
  params.dA_d_stride = static_cast<SSMParamsBwd::index_t>(grad_A.stride(0));
  params.dA_dstate_stride = static_cast<SSMParamsBwd::index_t>(grad_A.stride(1));
  params.dB_batch_stride = static_cast<SSMParamsBwd::index_t>(grad_B_kernel.stride(0));
  params.dB_group_stride = static_cast<SSMParamsBwd::index_t>(grad_B_kernel.stride(1));
  params.dB_d_stride = 0;
  params.dB_dstate_stride = static_cast<SSMParamsBwd::index_t>(grad_B_kernel.stride(2));
  params.dC_batch_stride = static_cast<SSMParamsBwd::index_t>(grad_C_kernel.stride(0));
  params.dC_group_stride = static_cast<SSMParamsBwd::index_t>(grad_C_kernel.stride(1));
  params.dC_d_stride = 0;
  params.dC_dstate_stride = static_cast<SSMParamsBwd::index_t>(grad_C_kernel.stride(2));
  params.dz_batch_stride = gate.has_value()
                               ? static_cast<SSMParamsBwd::index_t>(grad_gate.stride(0))
                               : 0;
  params.dz_d_stride = gate.has_value()
                           ? static_cast<SSMParamsBwd::index_t>(grad_gate.stride(1))
                           : 0;
}

}  // namespace

std::vector<at::Tensor> selective_scan_cuda_forward(const at::Tensor& inputs,
                                                    const at::Tensor& dt,
                                                    const at::Tensor& A,
                                                    const at::Tensor& B,
                                                    const at::Tensor& C,
                                                    const c10::optional<at::Tensor>& gate,
                                                    int64_t /*chunk_length*/) {
  check_inputs(inputs, dt, A, B, C, gate);

  const int64_t length = inputs.size(2);
  if (length == 0) {
    auto empty = at::zeros_like(inputs);
    auto empty_states = at::zeros({inputs.size(0), inputs.size(1), 0, A.size(1) * 2},
                                  inputs.options());
    auto raw = at::zeros_like(inputs);
    return {empty, empty_states, raw};
  }

  c10::cuda::CUDAGuard device_guard(inputs.device());

  auto inputs_c = inputs.contiguous();
  auto dt_c = dt.contiguous();
  auto A_c = A.contiguous();
  auto B_perm = permute_bc_for_kernel(B);
  auto C_perm = permute_bc_for_kernel(C);
  c10::optional<at::Tensor> gate_c;
  if (gate.has_value()) {
    gate_c = gate->contiguous();
  }

  auto raw_outputs = at::empty_like(inputs_c);
  at::Tensor outputs = gate_c.has_value() ? at::empty_like(inputs_c) : raw_outputs;

  const auto n_chunks = compute_num_chunks(length);
  auto scan_intermediates = at::empty({inputs.size(0), inputs.size(1), n_chunks, A.size(1) * 2},
                                      inputs.options());

  SSMParamsBase params{};
  populate_fwd_params(params,
                      inputs_c,
                      dt_c,
                      A_c,
                      B_perm,
                      C_perm,
                      raw_outputs,
                      outputs,
                      gate_c,
                      scan_intermediates);

  auto stream = at::cuda::getCurrentCUDAStream();
  selective_scan_fwd_cuda<float, float>(params, stream.stream());

  if (!gate_c.has_value()) {
    outputs = raw_outputs;
  }

  return {outputs, scan_intermediates, raw_outputs};
}

std::vector<at::Tensor> selective_scan_cuda_backward(const at::Tensor& grad_output,
                                                     const at::Tensor& inputs,
                                                     const at::Tensor& dt,
                                                     const at::Tensor& A,
                                                     const at::Tensor& B,
                                                     const at::Tensor& C,
                                                     const c10::optional<at::Tensor>& gate,
                                                     const at::Tensor& chunk_states,
                                                     const at::Tensor& raw_outputs,
                                                     int64_t /*chunk_length*/) {
  check_inputs(inputs, dt, A, B, C, gate);
  TORCH_CHECK(grad_output.device() == inputs.device(),
              "grad_output must be on the same device as inputs");
  TORCH_CHECK(grad_output.dtype() == at::kFloat, "grad_output must be float32");
  TORCH_CHECK(grad_output.sizes() == inputs.sizes(), "grad_output shape mismatch");

  TORCH_CHECK(chunk_states.device() == inputs.device(),
              "chunk_states must be on the same device as inputs");
  TORCH_CHECK(chunk_states.dtype() == at::kFloat, "chunk_states must be float32");
  TORCH_CHECK(chunk_states.dim() == 4,
              "chunk_states must have shape (batch, channels, num_chunks, state * 2)");

  TORCH_CHECK(raw_outputs.device() == inputs.device(),
              "raw_outputs must be on the same device as inputs");
  TORCH_CHECK(raw_outputs.dtype() == at::kFloat, "raw_outputs must be float32");
  TORCH_CHECK(raw_outputs.sizes() == inputs.sizes(), "raw_outputs shape mismatch");

  c10::cuda::CUDAGuard device_guard(inputs.device());

  auto grad_out_c = grad_output.contiguous();
  auto inputs_c = inputs.contiguous();
  auto dt_c = dt.contiguous();
  auto A_c = A.contiguous();
  auto B_perm = permute_bc_for_kernel(B);
  auto C_perm = permute_bc_for_kernel(C);
  auto raw_out_c = raw_outputs.contiguous();
  auto chunk_c = chunk_states.contiguous();

  c10::optional<at::Tensor> gate_c;
  if (gate.has_value()) {
    gate_c = gate->contiguous();
  }

  auto grad_inputs = at::empty_like(inputs_c);
  auto grad_dt = at::empty_like(dt_c);
  auto grad_A = at::zeros_like(A_c);
  auto grad_B_perm = at::zeros_like(B_perm);
  auto grad_C_perm = at::zeros_like(C_perm);
  at::Tensor grad_gate = gate_c.has_value() ? at::zeros_like(*gate_c) : at::empty({0}, inputs.options());

  SSMParamsBwd params{};
  populate_bwd_params(params,
                      inputs_c,
                      dt_c,
                      A_c,
                      B_perm,
                      C_perm,
                      gate_c,
                      raw_out_c,
                      grad_out_c,
                      chunk_c,
                      grad_inputs,
                      grad_dt,
                      grad_A,
                      grad_B_perm,
                      grad_C_perm,
                      grad_gate);

  params.x_ptr = chunk_c.data_ptr();

  auto stream = at::cuda::getCurrentCUDAStream();
  selective_scan_bwd_cuda<float, float>(params, stream.stream());

  auto grad_B = grad_B_perm.squeeze(1).permute({0, 2, 1}).contiguous();
  auto grad_C = grad_C_perm.squeeze(1).permute({0, 2, 1}).contiguous();

  if (!gate_c.has_value()) {
    grad_gate = at::zeros_like(inputs_c);
  }

  return {grad_inputs, grad_dt, grad_A, grad_B, grad_C, grad_gate};
}

}  // namespace ossm
