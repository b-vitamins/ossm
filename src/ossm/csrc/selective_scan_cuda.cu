#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace ossm {
namespace {

constexpr int kBlockThreads = 256;
constexpr int kMaxStride = 8;  // supports state dim up to 2048 for 256 threads

__device__ __forceinline__ float _sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float _silu(float x) {
  const float sigma = _sigmoid(x);
  return x * sigma;
}

__device__ __forceinline__ float _dsilu(float x) {
  const float sigma = _sigmoid(x);
  return sigma * (1.0f + x * (1.0f - sigma));
}

template <int Threads>
__device__ __forceinline__ float _block_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }

  __shared__ float warp_partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_partial[warp] = value;
  }
  __syncthreads();

  float block_value = 0.0f;
  if (threadIdx.x < Threads / 32) {
    block_value = warp_partial[lane];
  }
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_value += __shfl_down_sync(0xffffffff, block_value, offset);
    }
  }
  return __shfl_sync(0xffffffff, block_value, 0);
}

int _num_chunks(int length, int chunk) {
  return (length + chunk - 1) / chunk;
}

template <int Threads>
__global__ void selective_scan_forward_kernel(const float* __restrict__ inputs,
                                              const float* __restrict__ dt,
                                              const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              const float* __restrict__ C,
                                              const float* __restrict__ gate,
                                              float* __restrict__ outputs,
                                              float* __restrict__ chunk_states,
                                              int batch,
                                              int channels,
                                              int length,
                                              int state_dim,
                                              int chunk,
                                              int num_chunks) {
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  if (b >= batch || c >= channels) {
    return;
  }

  const std::size_t lane_offset = (static_cast<std::size_t>(b) * channels + c) * length;
  const float* inputs_lane = inputs + lane_offset;
  const float* dt_lane = dt + lane_offset;
  const float* gate_lane = gate ? gate + lane_offset : nullptr;
  float* output_lane = outputs + lane_offset;

  const float* A_row = A + static_cast<std::size_t>(c) * state_dim;
  float* chunk_lane = chunk_states +
                      ((static_cast<std::size_t>(b) * channels + c) * num_chunks) * state_dim;

  extern __shared__ float shared[];
  float* shared_A = shared;
  float* shared_invA = shared_A + state_dim;
  float* state = shared_invA + state_dim;

  for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
    const float a = A_row[idx];
    shared_A[idx] = a;
    shared_invA[idx] = 1.0f / a;
    state[idx] = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < length; ++t) {
    const float dt_scalar = dt_lane[t];
    const float input_scalar = inputs_lane[t];

    float partial = 0.0f;
    for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
      const float a = shared_A[idx];
      const float inv_a = shared_invA[idx];
      float lane_state = state[idx];

      const float* B_step = B + (static_cast<std::size_t>(b) * length + t) * state_dim;
      const float* C_step = C + (static_cast<std::size_t>(b) * length + t) * state_dim;

      const float delta = dt_scalar * a;
      const float A_bar = __expf(delta);
      const float phi = __expm1f(delta) * inv_a;

      lane_state = fmaf(A_bar, lane_state, phi * B_step[idx] * input_scalar);
      state[idx] = lane_state;

      partial = fmaf(lane_state, C_step[idx], partial);
    }

    const float y_value = _block_sum<Threads>(partial);
    if (threadIdx.x == 0) {
      float value = y_value;
      if (gate_lane) {
        value *= _silu(gate_lane[t]);
      }
      output_lane[t] = value;
    }
    __syncthreads();

    if ((t % chunk == chunk - 1) || (t == length - 1)) {
      const int chunk_index = t / chunk;
      float* chunk_ptr = chunk_lane + static_cast<std::size_t>(chunk_index) * state_dim;
      for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
        chunk_ptr[idx] = state[idx];
      }
    }
    __syncthreads();
  }
}

template <int Threads>
__global__ void selective_scan_backward_kernel(const float* __restrict__ grad_output,
                                               const float* __restrict__ inputs,
                                               const float* __restrict__ dt,
                                               const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               const float* __restrict__ C,
                                               const float* __restrict__ gate,
                                               const float* __restrict__ chunk_states,
                                               float* __restrict__ grad_inputs,
                                               float* __restrict__ grad_dt,
                                               float* __restrict__ grad_A,
                                               float* __restrict__ grad_B,
                                               float* __restrict__ grad_C,
                                               float* __restrict__ grad_gate,
                                               int batch,
                                               int channels,
                                               int length,
                                               int state_dim,
                                               int chunk,
                                               int num_chunks) {
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  if (b >= batch || c >= channels) {
    return;
  }

  const std::size_t lane_offset = (static_cast<std::size_t>(b) * channels + c) * length;
  const float* inputs_lane = inputs + lane_offset;
  const float* dt_lane = dt + lane_offset;
  const float* gate_lane = gate ? gate + lane_offset : nullptr;
  const float* grad_output_lane = grad_output + lane_offset;

  float* grad_inputs_lane = grad_inputs + lane_offset;
  float* grad_dt_lane = grad_dt + lane_offset;
  float* grad_gate_lane = grad_gate + lane_offset;

  const float* A_row = A + static_cast<std::size_t>(c) * state_dim;
  const float* chunk_lane = chunk_states +
                            ((static_cast<std::size_t>(b) * channels + c) * num_chunks) * state_dim;

  extern __shared__ float shared[];
  float* shared_A = shared;
  float* shared_invA = shared_A + state_dim;
  float* shared_dA = shared_invA + state_dim;
  float* shared_states = shared_dA + state_dim;

  for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
    const float a = A_row[idx];
    shared_A[idx] = a;
    shared_invA[idx] = 1.0f / a;
  }
  __syncthreads();

  const int stride = (state_dim + Threads - 1) / Threads;
  if (stride > kMaxStride) {
    return;
  }
  float grad_state_local[kMaxStride];
  for (int i = 0; i < stride; ++i) {
    grad_state_local[i] = 0.0f;
  }

  for (int chunk_index = num_chunks - 1; chunk_index >= 0; --chunk_index) {
    for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
      shared_dA[idx] = 0.0f;
    }
    __syncthreads();

    const int start = chunk_index * chunk;
    const int max_length = min(chunk, length - start);

    for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
      float state_prev = (chunk_index == 0) ? 0.0f
                                           : chunk_lane[(static_cast<std::size_t>(chunk_index - 1) * state_dim) + idx];
      float* state_ptr = shared_states + idx;
      for (int step = 0; step < max_length; ++step) {
        const int t = start + step;
        const float dt_scalar = dt_lane[t];
        const float input_scalar = inputs_lane[t];
        const float a = shared_A[idx];
        const float inv_a = shared_invA[idx];
        const float* B_step = B + (static_cast<std::size_t>(b) * length + t) * state_dim;

        const float delta = dt_scalar * a;
        const float A_bar = __expf(delta);
        const float phi = __expm1f(delta) * inv_a;

        state_prev = fmaf(A_bar, state_prev, phi * B_step[idx] * input_scalar);
        state_ptr[step * state_dim] = state_prev;
      }
    }
    __syncthreads();

    for (int step = max_length - 1; step >= 0; --step) {
      const int t = start + step;
      const float* C_step = C + (static_cast<std::size_t>(b) * length + t) * state_dim;
      const float* B_step = B + (static_cast<std::size_t>(b) * length + t) * state_dim;

      float partial = 0.0f;
      for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
        const float state_t = shared_states[step * state_dim + idx];
        partial = fmaf(state_t, C_step[idx], partial);
      }

      const float y_value = _block_sum<Threads>(partial);
      __shared__ float grad_y_scalar;
      __shared__ float grad_gate_scalar;
      if (threadIdx.x == 0) {
        const float grad_out = grad_output_lane[t];
        if (gate_lane) {
          const float gate_value = gate_lane[t];
          const float silu_value = _silu(gate_value);
          grad_y_scalar = grad_out * silu_value;
          grad_gate_scalar = grad_out * y_value * _dsilu(gate_value);
          grad_gate_lane[t] = grad_gate_scalar;
        } else {
          grad_y_scalar = grad_out;
          grad_gate_scalar = 0.0f;
        }
      }
      __syncthreads();

      float grad_input = 0.0f;
      float grad_dt_value = 0.0f;

      for (int stride_index = 0, idx = threadIdx.x; idx < state_dim;
           ++stride_index, idx += Threads) {
        const float a = shared_A[idx];
        const float inv_a = shared_invA[idx];
        const float state_t = shared_states[step * state_dim + idx];
        const float state_prev = (step == 0)
                                     ? ((chunk_index == 0)
                                            ? 0.0f
                                            : chunk_lane[(static_cast<std::size_t>(chunk_index - 1) * state_dim) + idx])
                                     : shared_states[(step - 1) * state_dim + idx];

        float grad_state = grad_state_local[stride_index] + grad_y_scalar * C_step[idx];

        const float dt_scalar = dt_lane[t];
        const float input_scalar = inputs_lane[t];

        const float delta = dt_scalar * a;
        const float A_bar = __expf(delta);
        const float phi = __expm1f(delta) * inv_a;

        atomicAdd(grad_C + (static_cast<std::size_t>(b) * length + t) * state_dim + idx,
                  grad_y_scalar * state_t);

        atomicAdd(grad_B + (static_cast<std::size_t>(b) * length + t) * state_dim + idx,
                  grad_state * phi * input_scalar);

        grad_input += grad_state * phi * B_step[idx];

        const float grad_E = grad_state * state_prev;
        const float grad_phi = grad_state * B_step[idx] * input_scalar;
        const float grad_delta = grad_E * A_bar + grad_phi * (A_bar * inv_a);

        shared_dA[idx] += grad_delta * dt_scalar - grad_phi * phi * inv_a;
        grad_dt_value += grad_delta * a;

        grad_state_local[stride_index] = grad_state * A_bar;
      }

      const float grad_input_sum = _block_sum<Threads>(grad_input);
      const float grad_dt_sum = _block_sum<Threads>(grad_dt_value);
      if (threadIdx.x == 0) {
        grad_inputs_lane[t] = grad_input_sum;
        grad_dt_lane[t] = grad_dt_sum;
        if (!gate_lane) {
          grad_gate_lane[t] = 0.0f;
        }
      }
      __syncthreads();
    }

    for (int idx = threadIdx.x; idx < state_dim; idx += Threads) {
      atomicAdd(grad_A + static_cast<std::size_t>(c) * state_dim + idx, shared_dA[idx]);
    }
    __syncthreads();
  }
}

}  // namespace

std::vector<at::Tensor> selective_scan_cuda_forward(const at::Tensor& inputs,
                                                    const at::Tensor& dt,
                                                    const at::Tensor& A,
                                                    const at::Tensor& B,
                                                    const at::Tensor& C,
                                                    const c10::optional<at::Tensor>& gate,
                                                    int64_t chunk_length) {
  TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(dt.is_cuda() && A.is_cuda() && B.is_cuda() && C.is_cuda(),
              "All inputs must be CUDA tensors");
  TORCH_CHECK(inputs.scalar_type() == at::kFloat,
              "selective_scan_cuda only supports float32 tensors");
  TORCH_CHECK(dt.sizes() == inputs.sizes(), "dt shape mismatch");
  TORCH_CHECK(inputs.dim() == 3, "inputs must have shape (batch, channels, length)");
  TORCH_CHECK(A.dim() == 2, "A must have shape (channels, state)");
  TORCH_CHECK(B.dim() == 3 && C.dim() == 3, "B and C must have shape (batch, length, state)");
  TORCH_CHECK(B.size(0) == inputs.size(0) && B.size(1) == inputs.size(2),
              "B shape mismatch");
  TORCH_CHECK(C.sizes() == B.sizes(), "C shape mismatch");
  TORCH_CHECK(A.size(0) == inputs.size(1), "A channels mismatch");

  if (gate.has_value()) {
    TORCH_CHECK(gate->is_cuda(), "gate must be CUDA tensor");
    TORCH_CHECK(gate->sizes() == inputs.sizes(), "gate shape mismatch");
  }

  const int batch = static_cast<int>(inputs.size(0));
  const int channels = static_cast<int>(inputs.size(1));
  const int length = static_cast<int>(inputs.size(2));
  const int state_dim = static_cast<int>(A.size(1));
  TORCH_CHECK(state_dim <= kBlockThreads * kMaxStride,
              "state dimension exceeds supported maximum for selective_scan_cuda");

  if (length == 0) {
    auto empty = at::zeros_like(inputs);
    auto empty_states = at::zeros({inputs.size(0), inputs.size(1), 0, A.size(1)}, inputs.options());
    return {empty, empty_states};
  }

  TORCH_CHECK(chunk_length > 0, "chunk_length must be positive");
  const int chunk = static_cast<int>(chunk_length);
  const int num_chunks = _num_chunks(length, chunk);

  auto inputs_contig = inputs.contiguous();
  auto dt_contig = dt.contiguous();
  auto A_contig = A.contiguous();
  auto B_contig = B.contiguous();
  auto C_contig = C.contiguous();
  at::Tensor gate_contig;
  if (gate.has_value()) {
    gate_contig = gate->contiguous();
  }

  auto outputs = at::empty_like(inputs_contig);
  auto chunk_states = at::empty(
      {inputs.size(0), inputs.size(1), num_chunks, A.size(1)}, inputs.options());

  dim3 grid(batch, channels);
  const std::size_t shared_bytes = static_cast<std::size_t>(state_dim * 3) * sizeof(float);

  auto stream = at::cuda::getCurrentCUDAStream();
  selective_scan_forward_kernel<kBlockThreads><<<grid, kBlockThreads, shared_bytes, stream>>>(
      inputs_contig.data_ptr<float>(),
      dt_contig.data_ptr<float>(),
      A_contig.data_ptr<float>(),
      B_contig.data_ptr<float>(),
      C_contig.data_ptr<float>(),
      gate_contig.defined() ? gate_contig.data_ptr<float>() : nullptr,
      outputs.data_ptr<float>(),
      chunk_states.data_ptr<float>(),
      batch,
      channels,
      length,
      state_dim,
      chunk,
      num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {outputs, chunk_states};
}

std::vector<at::Tensor> selective_scan_cuda_backward(const at::Tensor& grad_output,
                                                     const at::Tensor& inputs,
                                                     const at::Tensor& dt,
                                                     const at::Tensor& A,
                                                     const at::Tensor& B,
                                                     const at::Tensor& C,
                                                     const c10::optional<at::Tensor>& gate,
                                                     const at::Tensor& chunk_states,
                                                     int64_t chunk_length) {
  TORCH_CHECK(inputs.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(dt.is_cuda() && A.is_cuda() && B.is_cuda() && C.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA tensor");
  TORCH_CHECK(chunk_states.is_cuda(), "chunk_states must be CUDA tensor");
  TORCH_CHECK(inputs.scalar_type() == at::kFloat, "float32 only");
  TORCH_CHECK(dt.sizes() == inputs.sizes(), "dt shape mismatch");
  TORCH_CHECK(grad_output.sizes() == inputs.sizes(), "grad_output shape mismatch");
  TORCH_CHECK(chunk_length > 0, "chunk_length must be positive");

  if (gate.has_value()) {
    TORCH_CHECK(gate->is_cuda(), "gate must be CUDA tensor");
    TORCH_CHECK(gate->sizes() == inputs.sizes(), "gate shape mismatch");
  }

  const int batch = static_cast<int>(inputs.size(0));
  const int channels = static_cast<int>(inputs.size(1));
  const int length = static_cast<int>(inputs.size(2));
  const int state_dim = static_cast<int>(A.size(1));
  TORCH_CHECK(state_dim <= kBlockThreads * kMaxStride,
              "state dimension exceeds supported maximum for selective_scan_cuda");
  const int chunk = static_cast<int>(chunk_length);
  const int num_chunks = _num_chunks(length, chunk);

  TORCH_CHECK(chunk_states.sizes() == at::IntArrayRef({inputs.size(0), inputs.size(1), num_chunks, A.size(1)}),
              "chunk_states shape mismatch");

  auto grad_output_contig = grad_output.contiguous();
  auto inputs_contig = inputs.contiguous();
  auto dt_contig = dt.contiguous();
  auto A_contig = A.contiguous();
  auto B_contig = B.contiguous();
  auto C_contig = C.contiguous();
  auto chunk_contig = chunk_states.contiguous();
  at::Tensor gate_contig;
  if (gate.has_value()) {
    gate_contig = gate->contiguous();
  }

  auto grad_inputs = at::empty_like(inputs_contig);
  auto grad_dt = at::empty_like(dt_contig);
  auto grad_A = at::zeros_like(A_contig);
  auto grad_B = at::zeros_like(B_contig);
  auto grad_C = at::zeros_like(C_contig);
  auto grad_gate = at::zeros_like(inputs_contig);

  dim3 grid(batch, channels);
  const std::size_t shared_bytes =
      static_cast<std::size_t>((3 + chunk) * state_dim) * sizeof(float);

  auto stream = at::cuda::getCurrentCUDAStream();
  selective_scan_backward_kernel<kBlockThreads><<<grid, kBlockThreads, shared_bytes, stream>>>(
      grad_output_contig.data_ptr<float>(),
      inputs_contig.data_ptr<float>(),
      dt_contig.data_ptr<float>(),
      A_contig.data_ptr<float>(),
      B_contig.data_ptr<float>(),
      C_contig.data_ptr<float>(),
      gate_contig.defined() ? gate_contig.data_ptr<float>() : nullptr,
      chunk_contig.data_ptr<float>(),
      grad_inputs.data_ptr<float>(),
      grad_dt.data_ptr<float>(),
      grad_A.data_ptr<float>(),
      grad_B.data_ptr<float>(),
      grad_C.data_ptr<float>(),
      grad_gate.data_ptr<float>(),
      batch,
      channels,
      length,
      state_dim,
      chunk,
      num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {grad_inputs, grad_dt, grad_A, grad_B, grad_C, grad_gate};
}

}  // namespace ossm

