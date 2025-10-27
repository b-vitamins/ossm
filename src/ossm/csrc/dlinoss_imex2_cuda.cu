#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace ossm {
namespace {

template <typename scalar_t>
__global__ void dlinoss_imex2_forward_kernel(
    const typename scalar_t::value_type *__restrict__ a_diag,
    const typename scalar_t::value_type *__restrict__ g_diag,
    const typename scalar_t::value_type *__restrict__ step,
    const scalar_t *__restrict__ bu_ptr, scalar_t *__restrict__ out_ptr,
    int64_t length, int64_t batch, int64_t ssm) {
  using value_t = typename scalar_t::value_type;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < series;
       idx += blockDim.x * gridDim.x) {
    const int64_t state = idx % ssm;

    const value_t alpha = a_diag[state];
    const value_t gamma = g_diag[state];
    const value_t sigma = step[state];
    const value_t sigma_sq = sigma * sigma;

    const value_t m11 = value_t(1) - sigma * gamma;
    const value_t m12 = -sigma * alpha;
    const value_t m21 = sigma - sigma_sq * gamma;
    const value_t m22 = value_t(1) - sigma_sq * alpha;

    value_t state0_real = value_t(0);
    value_t state0_imag = value_t(0);
    value_t state1_real = value_t(0);
    value_t state1_imag = value_t(0);

    for (int64_t t = 0; t < length; ++t) {
      const int64_t bu_offset = t * series + idx;
      const int64_t out_offset = t * step_stride + idx * 2;

      const scalar_t bu_val = bu_ptr[bu_offset];
      const value_t bu_real = bu_val.real();
      const value_t bu_imag = bu_val.imag();

      const value_t new0_real =
          state0_real * m11 + state1_real * m12 + bu_real * sigma;
      const value_t new0_imag =
          state0_imag * m11 + state1_imag * m12 + bu_imag * sigma;
      const value_t new1_real =
          state0_real * m21 + state1_real * m22 + bu_real * sigma_sq;
      const value_t new1_imag =
          state0_imag * m21 + state1_imag * m22 + bu_imag * sigma_sq;

      out_ptr[out_offset] = scalar_t(new0_real, new0_imag);
      out_ptr[out_offset + 1] = scalar_t(new1_real, new1_imag);

      state0_real = new0_real;
      state0_imag = new0_imag;
      state1_real = new1_real;
      state1_imag = new1_imag;
    }
  }
}

template <typename scalar_t>
__global__ void dlinoss_imex2_backward_kernel(
    const typename scalar_t::value_type *__restrict__ a_diag,
    const typename scalar_t::value_type *__restrict__ g_diag,
    const typename scalar_t::value_type *__restrict__ step,
    const scalar_t *__restrict__ bu_ptr,
    const scalar_t *__restrict__ states_ptr,
    const scalar_t *__restrict__ grad_out_ptr,
    scalar_t *__restrict__ grad_bu_ptr,
    typename scalar_t::value_type *__restrict__ grad_a_ptr,
    typename scalar_t::value_type *__restrict__ grad_g_ptr,
    typename scalar_t::value_type *__restrict__ grad_step_ptr, int64_t length,
    int64_t batch, int64_t ssm) {
  using value_t = typename scalar_t::value_type;
  using acc_t = at::opmath_t<value_t>;

  const int64_t series = batch * ssm;
  const int64_t step_stride = series * 2;

  extern __shared__ __align__(sizeof(acc_t)) unsigned char shared_buffer[];
  acc_t *shared_grad_alpha = reinterpret_cast<acc_t *>(shared_buffer);
  acc_t *shared_grad_gamma = shared_grad_alpha + blockDim.x;
  acc_t *shared_grad_sigma = shared_grad_gamma + blockDim.x;

  for (int64_t state = blockIdx.x; state < ssm; state += gridDim.x) {
    const acc_t alpha = static_cast<acc_t>(a_diag[state]);
    const acc_t gamma = static_cast<acc_t>(g_diag[state]);
    const acc_t sigma = static_cast<acc_t>(step[state]);
    const acc_t sigma_sq = sigma * sigma;

    const acc_t m11 = acc_t(1) - sigma * gamma;
    const acc_t m12 = -sigma * alpha;
    const acc_t m21 = sigma - sigma_sq * gamma;
    const acc_t m22 = acc_t(1) - sigma_sq * alpha;

    const acc_t d_sigma_prev0 = acc_t(1) - acc_t(2) * sigma * gamma;
    const acc_t d_sigma_prev1 = -acc_t(2) * sigma * alpha;
    const acc_t d_sigma_bu = acc_t(2) * sigma;

    acc_t grad_alpha_local = acc_t(0);
    acc_t grad_gamma_local = acc_t(0);
    acc_t grad_sigma_local = acc_t(0);

    for (int64_t batch_idx = threadIdx.x; batch_idx < batch;
         batch_idx += blockDim.x) {
      const int64_t series_idx = batch_idx * ssm + state;

      acc_t grad_state0_real = acc_t(0);
      acc_t grad_state0_imag = acc_t(0);
      acc_t grad_state1_real = acc_t(0);
      acc_t grad_state1_imag = acc_t(0);

      const scalar_t *bu_series = bu_ptr + (length - 1) * series + series_idx;
      const scalar_t *grad_out_series =
          grad_out_ptr + (length - 1) * series + series_idx;
      scalar_t *grad_bu_series =
          grad_bu_ptr + (length - 1) * series + series_idx;
      const scalar_t *prev_states =
          length > 1 ? states_ptr + (length - 2) * step_stride + series_idx * 2
                     : nullptr;

      for (int64_t t = length - 1; t >= 0; --t) {
        const scalar_t grad_out_val = *grad_out_series;
        const scalar_t bu_val = *bu_series;

        acc_t prev0_real = acc_t(0);
        acc_t prev0_imag = acc_t(0);
        acc_t prev1_real = acc_t(0);
        acc_t prev1_imag = acc_t(0);
        if (t > 0 && prev_states != nullptr) {
          const scalar_t prev0_val = prev_states[0];
          const scalar_t prev1_val = prev_states[1];
          prev0_real = static_cast<acc_t>(prev0_val.real());
          prev0_imag = static_cast<acc_t>(prev0_val.imag());
          prev1_real = static_cast<acc_t>(prev1_val.real());
          prev1_imag = static_cast<acc_t>(prev1_val.imag());
        }

        const acc_t grad_out_real = static_cast<acc_t>(grad_out_val.real());
        const acc_t grad_out_imag = static_cast<acc_t>(grad_out_val.imag());
        const acc_t bu_real = static_cast<acc_t>(bu_val.real());
        const acc_t bu_imag = static_cast<acc_t>(bu_val.imag());

        const acc_t grad_new1_real = grad_state1_real + grad_out_real;
        const acc_t grad_new1_imag = grad_state1_imag + grad_out_imag;
        const acc_t grad_new0_real = grad_state0_real;
        const acc_t grad_new0_imag = grad_state0_imag;

        grad_alpha_local += (-sigma) * (grad_new0_real * prev1_real +
                                        grad_new0_imag * prev1_imag);
        grad_alpha_local += (-sigma_sq) * (grad_new1_real * prev1_real +
                                           grad_new1_imag * prev1_imag);

        grad_gamma_local += (-sigma) * (grad_new0_real * prev0_real +
                                        grad_new0_imag * prev0_imag);
        grad_gamma_local += (-sigma_sq) * (grad_new1_real * prev0_real +
                                           grad_new1_imag * prev0_imag);

        const acc_t sigma_term_real =
            prev0_real * (-gamma) + prev1_real * (-alpha) + bu_real;
        const acc_t sigma_term_imag =
            prev0_imag * (-gamma) + prev1_imag * (-alpha) + bu_imag;
        grad_sigma_local +=
            grad_new0_real * sigma_term_real + grad_new0_imag * sigma_term_imag;

        const acc_t sigma_grad_term_real = prev0_real * d_sigma_prev0 +
                                           prev1_real * d_sigma_prev1 +
                                           bu_real * d_sigma_bu;
        const acc_t sigma_grad_term_imag = prev0_imag * d_sigma_prev0 +
                                           prev1_imag * d_sigma_prev1 +
                                           bu_imag * d_sigma_bu;
        grad_sigma_local += grad_new1_real * sigma_grad_term_real +
                            grad_new1_imag * sigma_grad_term_imag;

        const acc_t grad_bu_real =
            grad_new0_real * sigma + grad_new1_real * sigma_sq;
        const acc_t grad_bu_imag =
            grad_new0_imag * sigma + grad_new1_imag * sigma_sq;
        *grad_bu_series = scalar_t(static_cast<value_t>(grad_bu_real),
                                   static_cast<value_t>(grad_bu_imag));

        const acc_t next_state0_real =
            grad_new0_real * m11 + grad_new1_real * m21;
        const acc_t next_state0_imag =
            grad_new0_imag * m11 + grad_new1_imag * m21;
        const acc_t next_state1_real =
            grad_new0_real * m12 + grad_new1_real * m22;
        const acc_t next_state1_imag =
            grad_new0_imag * m12 + grad_new1_imag * m22;

        grad_state0_real = next_state0_real;
        grad_state0_imag = next_state0_imag;
        grad_state1_real = next_state1_real;
        grad_state1_imag = next_state1_imag;

        if (t > 0 && prev_states != nullptr) {
          prev_states -= step_stride;
        }
        bu_series -= series;
        grad_out_series -= series;
        grad_bu_series -= series;
      }
    }

    shared_grad_alpha[threadIdx.x] = grad_alpha_local;
    shared_grad_gamma[threadIdx.x] = grad_gamma_local;
    shared_grad_sigma[threadIdx.x] = grad_sigma_local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        shared_grad_alpha[threadIdx.x] +=
            shared_grad_alpha[threadIdx.x + offset];
        shared_grad_gamma[threadIdx.x] +=
            shared_grad_gamma[threadIdx.x + offset];
        shared_grad_sigma[threadIdx.x] +=
            shared_grad_sigma[threadIdx.x + offset];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      grad_a_ptr[state] = static_cast<value_t>(shared_grad_alpha[0]);
      grad_g_ptr[state] = static_cast<value_t>(shared_grad_gamma[0]);
      grad_step_ptr[state] = static_cast<value_t>(shared_grad_sigma[0]);
    }

    __syncthreads();
  }
}

} // namespace

void dlinoss_imex2_forward_cuda(const at::Tensor &a_diag,
                                const at::Tensor &g_diag,
                                const at::Tensor &step, const at::Tensor &bu,
                                at::Tensor &output) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};

  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  const auto series = batch * ssm;

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1, std::min<int64_t>(
             (series + threads - 1) / threads,
             at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  AT_DISPATCH_COMPLEX_TYPES(
      bu.scalar_type(), "dlinoss_imex2_forward_cuda", [&] {
        dlinoss_imex2_forward_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                a_diag.data_ptr<typename scalar_t::value_type>(),
                g_diag.data_ptr<typename scalar_t::value_type>(),
                step.data_ptr<typename scalar_t::value_type>(),
                bu.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                bu.size(0), batch, ssm);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void dlinoss_imex2_backward_cuda(const at::Tensor &a_diag,
                                 const at::Tensor &g_diag,
                                 const at::Tensor &step, const at::Tensor &bu,
                                 const at::Tensor &states,
                                 const at::Tensor &grad_output,
                                 at::Tensor &grad_a, at::Tensor &grad_g,
                                 at::Tensor &grad_step, at::Tensor &grad_bu) {
  c10::cuda::OptionalCUDAGuard device_guard{bu.device()};

  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  const auto series = batch * ssm;

  constexpr int64_t threads = 256;
  const int64_t blocks = std::max<int64_t>(
      1, std::min<int64_t>(
             ssm, at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));

  grad_a.zero_();
  grad_g.zero_();
  grad_step.zero_();

  AT_DISPATCH_COMPLEX_TYPES(
      bu.scalar_type(), "dlinoss_imex2_backward_cuda", [&] {
        const size_t shared_mem_size = static_cast<size_t>(threads) * 3 *
                                       sizeof(at::opmath_t<typename scalar_t::value_type>);
        dlinoss_imex2_backward_kernel<scalar_t>
            <<<blocks, threads, shared_mem_size,
               at::cuda::getCurrentCUDAStream()>>>(
                a_diag.data_ptr<typename scalar_t::value_type>(),
                g_diag.data_ptr<typename scalar_t::value_type>(),
                step.data_ptr<typename scalar_t::value_type>(),
                bu.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(), grad_bu.data_ptr<scalar_t>(),
                grad_a.data_ptr<typename scalar_t::value_type>(),
                grad_g.data_ptr<typename scalar_t::value_type>(),
                grad_step.data_ptr<typename scalar_t::value_type>(), bu.size(0),
                batch, ssm);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace ossm
