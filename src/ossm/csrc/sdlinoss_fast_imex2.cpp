#include <ATen/ATen.h>
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif
#include <c10/util/Exception.h>

#include <cstdlib>
#include <tuple>
#include <type_traits>

#include "dlinoss_common.h"
#include "sdlinoss_fast_common.h"
#include "sdlinoss_fast_dispatch.h"

namespace ossm {
namespace {

constexpr int64_t kDefaultTile = 128;

int64_t parse_tile_env() {
  const char* env = std::getenv("OSSM_SDLINOSS_FAST_TILE");
  if (env == nullptr) {
    return kDefaultTile;
  }
  char* end_ptr = nullptr;
  long value = std::strtol(env, &end_ptr, 10);
  if (end_ptr == env || value <= 0) {
    return kDefaultTile;
  }
  switch (value) {
    case 64:
      return 64;
    case 256:
      return 256;
    default:
      return kDefaultTile;
  }
}

struct BroadcastStrides {
  Strides3 A;
  Strides3 G;
  Strides3 step;
  Strides3 bu;

  [[nodiscard]] int mask() const {
    const bool vary_L = (A.sL != 0) || (G.sL != 0) || (step.sL != 0) || (bu.sL != 0);
    const bool vary_B = (A.sB != 0) || (G.sB != 0) || (step.sB != 0) || (bu.sB != 0);
    const bool vary_M = (A.sM != 0) || (G.sM != 0) || (step.sM != 0) || (bu.sM != 0);
    return static_cast<int>(vary_L) | (static_cast<int>(vary_B) << 1) |
           (static_cast<int>(vary_M) << 2);
  }
};

#ifdef WITH_CUDA

void sdlinoss_fast_imex2_forward_cuda_complex64(int tile,
                                                int vary_mask,
                                                const Strides3& A_stride,
                                                const Strides3& G_stride,
                                                const Strides3& step_stride,
                                                const Strides3& bu_stride,
                                                const at::Tensor& A,
                                                const at::Tensor& G,
                                                const at::Tensor& step,
                                                const at::Tensor& bu,
                                                const at::Tensor& M00,
                                                const at::Tensor& M01,
                                                const at::Tensor& M10,
                                                const at::Tensor& M11,
                                                const at::Tensor& F0,
                                                const at::Tensor& F1,
                                                const at::Tensor& S0_w,
                                                const at::Tensor& S0_x,
                                                const at::Tensor& states,
                                                int64_t length,
                                                int64_t series,
                                                int64_t ssm,
                                                int64_t num_tiles,
                                                cudaStream_t stream);

void sdlinoss_fast_imex2_forward_cuda_complex128(int tile,
                                                 int vary_mask,
                                                 const Strides3& A_stride,
                                                 const Strides3& G_stride,
                                                 const Strides3& step_stride,
                                                 const Strides3& bu_stride,
                                                 const at::Tensor& A,
                                                 const at::Tensor& G,
                                                 const at::Tensor& step,
                                                 const at::Tensor& bu,
                                                 const at::Tensor& M00,
                                                 const at::Tensor& M01,
                                                 const at::Tensor& M10,
                                                 const at::Tensor& M11,
                                                 const at::Tensor& F0,
                                                 const at::Tensor& F1,
                                                 const at::Tensor& S0_w,
                                                 const at::Tensor& S0_x,
                                                 const at::Tensor& states,
                                                 int64_t length,
                                                 int64_t series,
                                                 int64_t ssm,
                                                 int64_t num_tiles,
                                                 cudaStream_t stream);

void sdlinoss_fast_imex2_forward_xonly_cuda_complex64(int tile,
                                                      int vary_mask,
                                                      const Strides3& A_stride,
                                                      const Strides3& G_stride,
                                                      const Strides3& step_stride,
                                                      const Strides3& bu_stride,
                                                      const at::Tensor& A,
                                                      const at::Tensor& G,
                                                      const at::Tensor& step,
                                                      const at::Tensor& bu,
                                                      const at::Tensor& M00,
                                                      const at::Tensor& M01,
                                                      const at::Tensor& M10,
                                                      const at::Tensor& M11,
                                                      const at::Tensor& F0,
                                                      const at::Tensor& F1,
                                                      const at::Tensor& S0_w,
                                                      const at::Tensor& S0_x,
                                                      const at::Tensor& x_only,
                                                      int64_t length,
                                                      int64_t series,
                                                      int64_t ssm,
                                                      int64_t num_tiles,
                                                      cudaStream_t stream);

void sdlinoss_fast_imex2_forward_xonly_cuda_complex128(int tile,
                                                       int vary_mask,
                                                       const Strides3& A_stride,
                                                       const Strides3& G_stride,
                                                       const Strides3& step_stride,
                                                       const Strides3& bu_stride,
                                                       const at::Tensor& A,
                                                       const at::Tensor& G,
                                                       const at::Tensor& step,
                                                       const at::Tensor& bu,
                                                       const at::Tensor& M00,
                                                       const at::Tensor& M01,
                                                       const at::Tensor& M10,
                                                       const at::Tensor& M11,
                                                       const at::Tensor& F0,
                                                       const at::Tensor& F1,
                                                       const at::Tensor& S0_w,
                                                       const at::Tensor& S0_x,
                                                       const at::Tensor& x_only,
                                                       int64_t length,
                                                       int64_t series,
                                                       int64_t ssm,
                                                       int64_t num_tiles,
                                                       cudaStream_t stream);

void sdlinoss_fast_imex2_backward_cuda_complex64(int tile,
                                                 int vary_mask,
                                                 const Strides3& A_stride,
                                                 const Strides3& G_stride,
                                                 const Strides3& step_stride,
                                                 const Strides3& bu_stride,
                                                 const at::Tensor& A,
                                                 const at::Tensor& G,
                                                 const at::Tensor& step,
                                                 const at::Tensor& bu,
                                                 const at::Tensor& states,
                                                 const at::Tensor& grad_out,
                                                 at::Tensor& grad_A,
                                                 at::Tensor& grad_G,
                                                 at::Tensor& grad_step,
                                                 at::Tensor& grad_bu,
                                                 int64_t length,
                                                 int64_t batch,
                                                 int64_t ssm,
                                                 cudaStream_t stream);

void sdlinoss_fast_imex2_backward_cuda_complex128(int tile,
                                                  int vary_mask,
                                                  const Strides3& A_stride,
                                                  const Strides3& G_stride,
                                                  const Strides3& step_stride,
                                                  const Strides3& bu_stride,
                                                  const at::Tensor& A,
                                                  const at::Tensor& G,
                                                  const at::Tensor& step,
                                                  const at::Tensor& bu,
                                                  const at::Tensor& states,
                                                  const at::Tensor& grad_out,
                                                  at::Tensor& grad_A,
                                                  at::Tensor& grad_G,
                                                  at::Tensor& grad_step,
                                                  at::Tensor& grad_bu,
                                                  int64_t length,
                                                  int64_t batch,
                                                  int64_t ssm,
                                                  cudaStream_t stream);

void sdlinoss_fast_imex2_backward_xonly_cuda_complex64(int tile,
                                                       int vary_mask,
                                                       const Strides3& A_stride,
                                                       const Strides3& G_stride,
                                                       const Strides3& step_stride,
                                                       const Strides3& bu_stride,
                                                       const at::Tensor& A,
                                                       const at::Tensor& G,
                                                       const at::Tensor& step,
                                                       const at::Tensor& bu,
                                                       const at::Tensor& x_only,
                                                       const at::Tensor& grad_out,
                                                       at::Tensor& grad_A,
                                                       at::Tensor& grad_G,
                                                       at::Tensor& grad_step,
                                                       at::Tensor& grad_bu,
                                                       int64_t length,
                                                       int64_t batch,
                                                       int64_t ssm,
                                                       cudaStream_t stream);

void sdlinoss_fast_imex2_backward_xonly_cuda_complex128(int tile,
                                                        int vary_mask,
                                                        const Strides3& A_stride,
                                                        const Strides3& G_stride,
                                                        const Strides3& step_stride,
                                                        const Strides3& bu_stride,
                                                        const at::Tensor& A,
                                                        const at::Tensor& G,
                                                        const at::Tensor& step,
                                                        const at::Tensor& bu,
                                                        const at::Tensor& x_only,
                                                        const at::Tensor& grad_out,
                                                        at::Tensor& grad_A,
                                                        at::Tensor& grad_G,
                                                        at::Tensor& grad_step,
                                                        at::Tensor& grad_bu,
                                                        int64_t length,
                                                        int64_t batch,
                                                        int64_t ssm,
                                                        cudaStream_t stream);

#endif  // WITH_CUDA

}  // namespace

at::Tensor sdlinoss_fast_imex2_forward(const at::Tensor& A,
                                       const at::Tensor& G,
                                       const at::Tensor& step,
                                       const at::Tensor& bu) {
#ifndef WITH_CUDA
  TORCH_CHECK(false, "sdlinoss_fast_imex2_forward requires CUDA support.");
#else
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor.");
  TORCH_CHECK(G.is_cuda(), "G must be a CUDA tensor.");
  TORCH_CHECK(step.is_cuda(), "step must be a CUDA tensor.");
  TORCH_CHECK(bu.is_cuda(), "bu must be a CUDA tensor.");

  TORCH_CHECK(!A.is_complex(), "A must be real-valued.");
  TORCH_CHECK(!G.is_complex(), "G must be real-valued.");
  TORCH_CHECK(!step.is_complex(), "step must be real-valued.");
  TORCH_CHECK(bu.is_complex(), "bu must be complex-valued.");

  TORCH_CHECK(A.dim() == 3, "A must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(G.dim() == 3, "G must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(step.dim() == 3, "step must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (L, B, M).");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  TORCH_CHECK(length >= 0 && batch >= 0 && ssm >= 0, "Invalid shapes for selective D-LinOSS inputs.");
  TORCH_CHECK(length == A.size(0) || A.size(0) == 1,
              "A must broadcast across the length dimension.");
  TORCH_CHECK(length == G.size(0) || G.size(0) == 1,
              "G must broadcast across the length dimension.");
  TORCH_CHECK(length == step.size(0) || step.size(0) == 1,
              "step must broadcast across the length dimension.");
  TORCH_CHECK(batch == A.size(1) || A.size(1) == 1,
              "A must broadcast across the batch dimension.");
  TORCH_CHECK(batch == G.size(1) || G.size(1) == 1,
              "G must broadcast across the batch dimension.");
  TORCH_CHECK(batch == step.size(1) || step.size(1) == 1,
              "step must broadcast across the batch dimension.");
  TORCH_CHECK(ssm == A.size(2) && ssm == G.size(2) && ssm == step.size(2),
              "All parameters must agree on the state dimension.");

  const auto real_dtype = A.scalar_type();
  TORCH_CHECK(real_dtype == G.scalar_type() && real_dtype == step.scalar_type(),
              "A, G, and step must share the same real dtype.");
  TORCH_CHECK((bu.scalar_type() == at::kComplexFloat && real_dtype == at::kFloat) ||
                  (bu.scalar_type() == at::kComplexDouble && real_dtype == at::kDouble),
              "bu dtype must correspond to the real parameters.");

  if (length == 0 || batch == 0 || ssm == 0) {
    return at::empty({length, batch, ssm, 2}, bu.options());
  }

  const int64_t tile_env = parse_tile_env();
  const int64_t tile = tile_env == 64 || tile_env == 256 ? tile_env : kDefaultTile;
  const int64_t num_tiles = (length + tile - 1) / tile;
  const int64_t series = batch * ssm;
  BroadcastStrides strides{
      collapse_strides3(A), collapse_strides3(G), collapse_strides3(step), collapse_strides3(bu)};

  auto options_real = A.options();
  auto options_complex = bu.options();

  auto M00 = at::empty({num_tiles, series}, options_real);
  auto M01 = at::empty({num_tiles, series}, options_real);
  auto M10 = at::empty({num_tiles, series}, options_real);
  auto M11 = at::empty({num_tiles, series}, options_real);
  auto F0 = at::empty({num_tiles, series}, options_complex);
  auto F1 = at::empty({num_tiles, series}, options_complex);

  auto S0_w = at::empty({num_tiles, series}, options_complex);
  auto S0_x = at::empty({num_tiles, series}, options_complex);

  auto states = at::empty({length, batch, ssm, 2}, options_complex);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int vary_mask = strides.mask();
  const auto& A_stride = strides.A;
  const auto& G_stride = strides.G;
  const auto& step_stride = strides.step;
  const auto& bu_stride = strides.bu;

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_fast_imex2_forward", [&] {
    if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
      sdlinoss_fast_imex2_forward_cuda_complex64(tile,
                                                 vary_mask,
                                                 A_stride,
                                                 G_stride,
                                                 step_stride,
                                                 bu_stride,
                                                 A,
                                                 G,
                                                 step,
                                                 bu,
                                                 M00,
                                                 M01,
                                                 M10,
                                                 M11,
                                                 F0,
                                                 F1,
                                                 S0_w,
                                                 S0_x,
                                                 states,
                                                 length,
                                                 series,
                                                 ssm,
                                                 num_tiles,
                                                 stream);
    } else {
      sdlinoss_fast_imex2_forward_cuda_complex128(tile,
                                                  vary_mask,
                                                  A_stride,
                                                  G_stride,
                                                  step_stride,
                                                  bu_stride,
                                                  A,
                                                  G,
                                                  step,
                                                  bu,
                                                  M00,
                                                  M01,
                                                  M10,
                                                  M11,
                                                  F0,
                                                  F1,
                                                  S0_w,
                                                  S0_x,
                                                  states,
                                                  length,
                                                  series,
                                                  ssm,
                                                  num_tiles,
                                                  stream);
    }
  });

  return states;
#endif  // WITH_CUDA
}

at::Tensor sdlinoss_fast_imex2_forward_xonly(const at::Tensor& A,
                                             const at::Tensor& G,
                                             const at::Tensor& step,
                                             const at::Tensor& bu) {
#ifndef WITH_CUDA
  TORCH_CHECK(false, "sdlinoss_fast_imex2_forward_xonly requires CUDA support.");
#else
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor.");
  TORCH_CHECK(G.is_cuda(), "G must be a CUDA tensor.");
  TORCH_CHECK(step.is_cuda(), "step must be a CUDA tensor.");
  TORCH_CHECK(bu.is_cuda(), "bu must be a CUDA tensor.");

  TORCH_CHECK(!A.is_complex(), "A must be real-valued.");
  TORCH_CHECK(!G.is_complex(), "G must be real-valued.");
  TORCH_CHECK(!step.is_complex(), "step must be real-valued.");
  TORCH_CHECK(bu.is_complex(), "bu must be complex-valued.");

  TORCH_CHECK(A.dim() == 3, "A must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(G.dim() == 3, "G must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(step.dim() == 3, "step must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (L, B, M).");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  TORCH_CHECK(length >= 0 && batch >= 0 && ssm >= 0, "Invalid shapes for selective D-LinOSS inputs.");
  TORCH_CHECK(length == A.size(0) || A.size(0) == 1,
              "A must broadcast across the length dimension.");
  TORCH_CHECK(length == G.size(0) || G.size(0) == 1,
              "G must broadcast across the length dimension.");
  TORCH_CHECK(length == step.size(0) || step.size(0) == 1,
              "step must broadcast across the length dimension.");
  TORCH_CHECK(batch == A.size(1) || A.size(1) == 1,
              "A must broadcast across the batch dimension.");
  TORCH_CHECK(batch == G.size(1) || G.size(1) == 1,
              "G must broadcast across the batch dimension.");
  TORCH_CHECK(batch == step.size(1) || step.size(1) == 1,
              "step must broadcast across the batch dimension.");
  TORCH_CHECK(ssm == A.size(2) && ssm == G.size(2) && ssm == step.size(2),
              "All parameters must agree on the state dimension.");

  const auto real_dtype = A.scalar_type();
  TORCH_CHECK(real_dtype == G.scalar_type() && real_dtype == step.scalar_type(),
              "A, G, and step must share the same real dtype.");
  TORCH_CHECK((bu.scalar_type() == at::kComplexFloat && real_dtype == at::kFloat) ||
                  (bu.scalar_type() == at::kComplexDouble && real_dtype == at::kDouble),
              "bu dtype must correspond to the real parameters.");

  if (length == 0 || batch == 0 || ssm == 0) {
    return at::empty({length, batch, ssm}, bu.options());
  }

  const int64_t tile_env = parse_tile_env();
  const int64_t tile = tile_env == 64 || tile_env == 256 ? tile_env : kDefaultTile;
  const int64_t num_tiles = (length + tile - 1) / tile;
  const int64_t series = batch * ssm;
  BroadcastStrides strides{
      collapse_strides3(A), collapse_strides3(G), collapse_strides3(step), collapse_strides3(bu)};

  auto options_real = A.options();
  auto options_complex = bu.options();

  auto M00 = at::empty({num_tiles, series}, options_real);
  auto M01 = at::empty({num_tiles, series}, options_real);
  auto M10 = at::empty({num_tiles, series}, options_real);
  auto M11 = at::empty({num_tiles, series}, options_real);
  auto F0 = at::empty({num_tiles, series}, options_complex);
  auto F1 = at::empty({num_tiles, series}, options_complex);

  auto S0_w = at::empty({num_tiles, series}, options_complex);
  auto S0_x = at::empty({num_tiles, series}, options_complex);

  auto x_only = at::empty({length, batch, ssm}, options_complex);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int vary_mask = strides.mask();
  const auto& A_stride = strides.A;
  const auto& G_stride = strides.G;
  const auto& step_stride = strides.step;
  const auto& bu_stride = strides.bu;

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_fast_imex2_forward_xonly", [&] {
    if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
      sdlinoss_fast_imex2_forward_xonly_cuda_complex64(tile,
                                                       vary_mask,
                                                       A_stride,
                                                       G_stride,
                                                       step_stride,
                                                       bu_stride,
                                                       A,
                                                       G,
                                                       step,
                                                       bu,
                                                       M00,
                                                       M01,
                                                       M10,
                                                       M11,
                                                       F0,
                                                       F1,
                                                       S0_w,
                                                       S0_x,
                                                       x_only,
                                                       length,
                                                       series,
                                                       ssm,
                                                       num_tiles,
                                                       stream);
    } else {
      sdlinoss_fast_imex2_forward_xonly_cuda_complex128(tile,
                                                        vary_mask,
                                                        A_stride,
                                                        G_stride,
                                                        step_stride,
                                                        bu_stride,
                                                        A,
                                                        G,
                                                        step,
                                                        bu,
                                                        M00,
                                                        M01,
                                                        M10,
                                                        M11,
                                                        F0,
                                                        F1,
                                                        S0_w,
                                                        S0_x,
                                                        x_only,
                                                        length,
                                                        series,
                                                        ssm,
                                                        num_tiles,
                                                        stream);
    }
  });

  return x_only;
#endif  // WITH_CUDA
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdlinoss_fast_imex2_backward(
    const at::Tensor& A,
    const at::Tensor& G,
    const at::Tensor& step,
    const at::Tensor& bu,
    const at::Tensor& states,
    const at::Tensor& grad_out) {
#ifndef WITH_CUDA
  TORCH_CHECK(false, "sdlinoss_fast_imex2_backward requires CUDA support.");
#else
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor.");
  TORCH_CHECK(G.is_cuda(), "G must be a CUDA tensor.");
  TORCH_CHECK(step.is_cuda(), "step must be a CUDA tensor.");
  TORCH_CHECK(bu.is_cuda(), "bu must be a CUDA tensor.");
  TORCH_CHECK(states.is_cuda(), "states must be a CUDA tensor.");
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor.");

  TORCH_CHECK(!A.is_complex(), "A must be real-valued.");
  TORCH_CHECK(!G.is_complex(), "G must be real-valued.");
  TORCH_CHECK(!step.is_complex(), "step must be real-valued.");
  TORCH_CHECK(bu.is_complex(), "bu must be complex-valued.");
  TORCH_CHECK(states.is_complex(), "states must be complex-valued.");
  TORCH_CHECK(grad_out.is_complex(), "grad_out must be complex-valued.");

  TORCH_CHECK(A.dim() == 3, "A must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(G.dim() == 3, "G must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(step.dim() == 3, "step must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (L, B, M).");
  TORCH_CHECK(states.dim() == 4 && states.size(3) == 2,
              "states must have shape (L, B, M, 2).");
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must have shape (L, B, M).");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  TORCH_CHECK(length >= 0 && batch >= 0 && ssm >= 0,
              "Invalid shapes for selective D-LinOSS inputs.");

  TORCH_CHECK(length == states.size(0) && batch == states.size(1) && ssm == states.size(2),
              "states shape must align with inputs.");
  TORCH_CHECK(length == grad_out.size(0) && batch == grad_out.size(1) && ssm == grad_out.size(2),
              "grad_out shape must align with inputs.");

  TORCH_CHECK(length == A.size(0) || A.size(0) == 1,
              "A must broadcast across the length dimension.");
  TORCH_CHECK(length == G.size(0) || G.size(0) == 1,
              "G must broadcast across the length dimension.");
  TORCH_CHECK(length == step.size(0) || step.size(0) == 1,
              "step must broadcast across the length dimension.");
  TORCH_CHECK(batch == A.size(1) || A.size(1) == 1,
              "A must broadcast across the batch dimension.");
  TORCH_CHECK(batch == G.size(1) || G.size(1) == 1,
              "G must broadcast across the batch dimension.");
  TORCH_CHECK(batch == step.size(1) || step.size(1) == 1,
              "step must broadcast across the batch dimension.");
  TORCH_CHECK(ssm == A.size(2) && ssm == G.size(2) && ssm == step.size(2),
              "All parameters must agree on the state dimension.");

  const auto real_dtype = A.scalar_type();
  TORCH_CHECK(real_dtype == G.scalar_type() && real_dtype == step.scalar_type(),
              "A, G, and step must share the same real dtype.");
  TORCH_CHECK(bu.scalar_type() == states.scalar_type(),
              "states dtype must match bu dtype.");
  TORCH_CHECK(grad_out.scalar_type() == bu.scalar_type(),
              "grad_out dtype must match bu dtype.");
  TORCH_CHECK((bu.scalar_type() == at::kComplexFloat && real_dtype == at::kFloat) ||
                  (bu.scalar_type() == at::kComplexDouble && real_dtype == at::kDouble),
              "bu dtype must correspond to the real parameters.");

  if (length == 0 || batch == 0 || ssm == 0) {
    auto grad_A = at::zeros_like(A);
    auto grad_G = at::zeros_like(G);
    auto grad_step = at::zeros_like(step);
    auto grad_bu = at::zeros_like(bu);
    return std::make_tuple(grad_A, grad_G, grad_step, grad_bu);
  }

  const int64_t tile_env = parse_tile_env();
  const int64_t tile = tile_env == 64 || tile_env == 256 ? tile_env : kDefaultTile;

  BroadcastStrides strides{
      collapse_strides3(A), collapse_strides3(G), collapse_strides3(step), collapse_strides3(bu)};

  auto grad_A = at::zeros_like(A);
  auto grad_G = at::zeros_like(G);
  auto grad_step = at::zeros_like(step);
  auto grad_bu = at::zeros_like(bu);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int vary_mask = strides.mask();

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_fast_imex2_backward", [&] {
    if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
      sdlinoss_fast_imex2_backward_cuda_complex64(tile,
                                                  vary_mask,
                                                  strides.A,
                                                  strides.G,
                                                  strides.step,
                                                  strides.bu,
                                                  A,
                                                  G,
                                                  step,
                                                  bu,
                                                  states,
                                                  grad_out,
                                                  grad_A,
                                                  grad_G,
                                                  grad_step,
                                                  grad_bu,
                                                  length,
                                                  batch,
                                                  ssm,
                                                  stream);
    } else {
      sdlinoss_fast_imex2_backward_cuda_complex128(tile,
                                                   vary_mask,
                                                   strides.A,
                                                   strides.G,
                                                   strides.step,
                                                   strides.bu,
                                                   A,
                                                   G,
                                                   step,
                                                   bu,
                                                   states,
                                                   grad_out,
                                                   grad_A,
                                                   grad_G,
                                                   grad_step,
                                                   grad_bu,
                                                   length,
                                                   batch,
                                                   ssm,
                                                   stream);
    }
  });

  return std::make_tuple(grad_A, grad_G, grad_step, grad_bu);
#endif  // WITH_CUDA
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdlinoss_fast_imex2_backward_xonly(
    const at::Tensor& A,
    const at::Tensor& G,
    const at::Tensor& step,
    const at::Tensor& bu,
    const at::Tensor& x_only,
    const at::Tensor& grad_out) {
#ifndef WITH_CUDA
  TORCH_CHECK(false, "sdlinoss_fast_imex2_backward_xonly requires CUDA support.");
#else
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor.");
  TORCH_CHECK(G.is_cuda(), "G must be a CUDA tensor.");
  TORCH_CHECK(step.is_cuda(), "step must be a CUDA tensor.");
  TORCH_CHECK(bu.is_cuda(), "bu must be a CUDA tensor.");
  TORCH_CHECK(x_only.is_cuda(), "x_only must be a CUDA tensor.");
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor.");

  TORCH_CHECK(!A.is_complex(), "A must be real-valued.");
  TORCH_CHECK(!G.is_complex(), "G must be real-valued.");
  TORCH_CHECK(!step.is_complex(), "step must be real-valued.");
  TORCH_CHECK(bu.is_complex(), "bu must be complex-valued.");
  TORCH_CHECK(x_only.is_complex(), "x_only must be complex-valued.");
  TORCH_CHECK(grad_out.is_complex(), "grad_out must be complex-valued.");

  TORCH_CHECK(A.dim() == 3, "A must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(G.dim() == 3, "G must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(step.dim() == 3, "step must have shape (L, B, M) after broadcast normalization.");
  TORCH_CHECK(bu.dim() == 3, "bu must have shape (L, B, M).");
  TORCH_CHECK(x_only.dim() == 3, "x_only must have shape (L, B, M).");
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must have shape (L, B, M).");

  const auto length = bu.size(0);
  const auto batch = bu.size(1);
  const auto ssm = bu.size(2);
  TORCH_CHECK(length >= 0 && batch >= 0 && ssm >= 0, "Invalid shapes for selective D-LinOSS inputs.");

  TORCH_CHECK(length == x_only.size(0) && batch == x_only.size(1) && ssm == x_only.size(2),
              "x_only shape must align with inputs.");
  TORCH_CHECK(length == grad_out.size(0) && batch == grad_out.size(1) && ssm == grad_out.size(2),
              "grad_out shape must align with inputs.");

  TORCH_CHECK(length == A.size(0) || A.size(0) == 1,
              "A must broadcast across the length dimension.");
  TORCH_CHECK(length == G.size(0) || G.size(0) == 1,
              "G must broadcast across the length dimension.");
  TORCH_CHECK(length == step.size(0) || step.size(0) == 1,
              "step must broadcast across the length dimension.");
  TORCH_CHECK(batch == A.size(1) || A.size(1) == 1,
              "A must broadcast across the batch dimension.");
  TORCH_CHECK(batch == G.size(1) || G.size(1) == 1,
              "G must broadcast across the batch dimension.");
  TORCH_CHECK(batch == step.size(1) || step.size(1) == 1,
              "step must broadcast across the batch dimension.");
  TORCH_CHECK(ssm == A.size(2) && ssm == G.size(2) && ssm == step.size(2),
              "All parameters must agree on the state dimension.");

  const auto real_dtype = A.scalar_type();
  TORCH_CHECK(real_dtype == G.scalar_type() && real_dtype == step.scalar_type(),
              "A, G, and step must share the same real dtype.");
  TORCH_CHECK(bu.scalar_type() == x_only.scalar_type(), "x_only dtype must match bu dtype.");
  TORCH_CHECK(grad_out.scalar_type() == bu.scalar_type(), "grad_out dtype must match bu dtype.");
  TORCH_CHECK((bu.scalar_type() == at::kComplexFloat && real_dtype == at::kFloat) ||
                  (bu.scalar_type() == at::kComplexDouble && real_dtype == at::kDouble),
              "bu dtype must correspond to the real parameters.");

  if (length == 0 || batch == 0 || ssm == 0) {
    auto grad_A = at::zeros_like(A);
    auto grad_G = at::zeros_like(G);
    auto grad_step = at::zeros_like(step);
    auto grad_bu = at::zeros_like(bu);
    return std::make_tuple(grad_A, grad_G, grad_step, grad_bu);
  }

  const int64_t tile_env = parse_tile_env();
  const int64_t tile = tile_env == 64 || tile_env == 256 ? tile_env : kDefaultTile;

  BroadcastStrides strides{
      collapse_strides3(A), collapse_strides3(G), collapse_strides3(step), collapse_strides3(bu)};

  auto grad_A = at::zeros_like(A);
  auto grad_G = at::zeros_like(G);
  auto grad_step = at::zeros_like(step);
  auto grad_bu = at::zeros_like(bu);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int vary_mask = strides.mask();

  AT_DISPATCH_COMPLEX_TYPES(bu.scalar_type(), "sdlinoss_fast_imex2_backward_xonly", [&] {
    if constexpr (std::is_same_v<scalar_t, c10::complex<float>>) {
      sdlinoss_fast_imex2_backward_xonly_cuda_complex64(tile,
                                                        vary_mask,
                                                        strides.A,
                                                        strides.G,
                                                        strides.step,
                                                        strides.bu,
                                                        A,
                                                        G,
                                                        step,
                                                        bu,
                                                        x_only,
                                                        grad_out,
                                                        grad_A,
                                                        grad_G,
                                                        grad_step,
                                                        grad_bu,
                                                        length,
                                                        batch,
                                                        ssm,
                                                        stream);
    } else {
      sdlinoss_fast_imex2_backward_xonly_cuda_complex128(tile,
                                                         vary_mask,
                                                         strides.A,
                                                         strides.G,
                                                         strides.step,
                                                         strides.bu,
                                                         A,
                                                         G,
                                                         step,
                                                         bu,
                                                         x_only,
                                                         grad_out,
                                                         grad_A,
                                                         grad_G,
                                                         grad_step,
                                                         grad_bu,
                                                         length,
                                                         batch,
                                                         ssm,
                                                         stream);
    }
  });

  return std::make_tuple(grad_A, grad_G, grad_step, grad_bu);
#endif  // WITH_CUDA
}

}  // namespace ossm
