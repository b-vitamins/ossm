#pragma once

#include <ATen/ATen.h>
#include <cub/block/block_scan.cuh>

namespace ossm {

template <typename scalar_t>
struct StridedView3 {
  const scalar_t* ptr;
  int64_t stride_l;
  int64_t stride_b;
  int64_t stride_m;

  __host__ __device__ inline scalar_t load(int64_t t, int64_t b, int64_t m) const {
    const int64_t offset = (stride_l == 0 ? 0 : t * stride_l) +
                           (stride_b == 0 ? 0 : b * stride_b) +
                           (stride_m == 0 ? 0 : m * stride_m);
    return ptr[offset];
  }
};

template <typename scalar_t>
inline StridedView3<scalar_t> make_strided_view3(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 3, "Expected a 3D tensor, got ", tensor.dim(), "D tensor.");
  StridedView3<scalar_t> view;
  view.ptr = tensor.data_ptr<scalar_t>();
  const auto& strides = tensor.strides();
  view.stride_l = strides[0];
  view.stride_b = strides[1];
  view.stride_m = strides[2];
  return view;
}

template <typename scalar_t>
struct StridedView4 {
  const scalar_t* ptr;
  int64_t stride_l;
  int64_t stride_b;
  int64_t stride_m;
  int64_t stride_q;

  __host__ __device__ inline scalar_t load(int64_t t, int64_t b, int64_t m, int64_t q) const {
    const int64_t offset = (stride_l == 0 ? 0 : t * stride_l) +
                           (stride_b == 0 ? 0 : b * stride_b) +
                           (stride_m == 0 ? 0 : m * stride_m) +
                           (stride_q == 0 ? 0 : q * stride_q);
    return ptr[offset];
  }
};

template <typename scalar_t>
inline StridedView4<scalar_t> make_strided_view4(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 4, "Expected a 4D tensor, got ", tensor.dim(), "D tensor.");
  StridedView4<scalar_t> view;
  view.ptr = tensor.data_ptr<scalar_t>();
  const auto& strides = tensor.strides();
  view.stride_l = strides[0];
  view.stride_b = strides[1];
  view.stride_m = strides[2];
  view.stride_q = strides[3];
  return view;
}

template <typename value_t>
struct Transition2x2 {
  value_t m00;
  value_t m01;
  value_t m10;
  value_t m11;
  value_t f0_real;
  value_t f0_imag;
  value_t f1_real;
  value_t f1_imag;

  __host__ __device__ static Transition2x2 identity() {
    Transition2x2 out;
    out.m00 = value_t(1);
    out.m01 = value_t(0);
    out.m10 = value_t(0);
    out.m11 = value_t(1);
    out.f0_real = value_t(0);
    out.f0_imag = value_t(0);
    out.f1_real = value_t(0);
    out.f1_imag = value_t(0);
    return out;
  }
};

template <typename value_t>
__host__ __device__ inline Transition2x2<value_t> compose(
    const Transition2x2<value_t>& rhs,
    const Transition2x2<value_t>& lhs) {
  Transition2x2<value_t> out;
  out.m00 = rhs.m00 * lhs.m00 + rhs.m01 * lhs.m10;
  out.m01 = rhs.m00 * lhs.m01 + rhs.m01 * lhs.m11;
  out.m10 = rhs.m10 * lhs.m00 + rhs.m11 * lhs.m10;
  out.m11 = rhs.m10 * lhs.m01 + rhs.m11 * lhs.m11;
  out.f0_real = rhs.m00 * lhs.f0_real + rhs.m01 * lhs.f1_real + rhs.f0_real;
  out.f0_imag = rhs.m00 * lhs.f0_imag + rhs.m01 * lhs.f1_imag + rhs.f0_imag;
  out.f1_real = rhs.m10 * lhs.f0_real + rhs.m11 * lhs.f1_real + rhs.f1_real;
  out.f1_imag = rhs.m10 * lhs.f0_imag + rhs.m11 * lhs.f1_imag + rhs.f1_imag;
  return out;
}

template <typename value_t>
struct TransitionCombine {
  __host__ __device__ inline Transition2x2<value_t> operator()(const Transition2x2<value_t>& prefix,
                                                               const Transition2x2<value_t>& element) const {
    return compose(element, prefix);
  }
};

template <typename scalar_t, typename Builder, int TileSteps>
__global__ void sdlinoss_forward_tile_kernel(
    StridedView3<typename scalar_t::value_type> A,
    StridedView3<typename scalar_t::value_type> G,
    StridedView3<typename scalar_t::value_type> step,
    StridedView3<scalar_t> bu,
    Transition2x2<typename scalar_t::value_type>* __restrict__ local_prefixes,
    Transition2x2<typename scalar_t::value_type>* __restrict__ tile_prefixes,
    int64_t length,
    int64_t batch,
    int64_t ssm,
    int64_t num_tiles) {
  using value_t = typename scalar_t::value_type;
  const int m = blockIdx.x;
  const int b = blockIdx.y;
  const int tile = blockIdx.z;
  if (m >= ssm || b >= batch || tile >= num_tiles) {
    return;
  }

  const int64_t tile_offset = static_cast<int64_t>(tile) * TileSteps;
  int steps = 0;
  if (tile_offset < length) {
    steps = static_cast<int>(length - tile_offset);
    if (steps > TileSteps) {
      steps = TileSteps;
    }
  }

  Transition2x2<value_t> element = Transition2x2<value_t>::identity();
  if (threadIdx.x < steps) {
    const int64_t t = tile_offset + threadIdx.x;
    const value_t a_val = A.load(t, b, m);
    const value_t g_val = G.load(t, b, m);
    const value_t step_val = step.load(t, b, m);
    const scalar_t bu_val = bu.load(t, b, m);
    element = Builder::build(a_val, g_val, step_val, bu_val);
  }

  using BlockScan = cub::BlockScan<Transition2x2<value_t>, TileSteps>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  Transition2x2<value_t> prefix = element;
  BlockScan(temp_storage).InclusiveScan(element, prefix, TransitionCombine<value_t>());

  if (threadIdx.x < steps) {
    const int64_t index = (((tile_offset + threadIdx.x) * batch) + b) * ssm + m;
    local_prefixes[index] = prefix;
  }

  if (steps > 0 && threadIdx.x == steps - 1) {
    const int64_t tile_index = (((int64_t)tile * batch) + b) * ssm + m;
    tile_prefixes[tile_index] = prefix;
  }
}

template <typename value_t, int TileSteps>
__global__ void sdlinoss_forward_prefix_kernel(
    const Transition2x2<value_t>* __restrict__ tile_prefixes,
    Transition2x2<value_t>* __restrict__ tile_scans,
    int64_t num_tiles,
    int64_t batch,
    int64_t ssm) {
  const int m = blockIdx.x;
  const int b = blockIdx.y;
  if (m >= ssm || b >= batch) {
    return;
  }

  using BlockScan = cub::BlockScan<Transition2x2<value_t>, TileSteps>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ Transition2x2<value_t> carry_shared;

  Transition2x2<value_t> carry = Transition2x2<value_t>::identity();

  for (int64_t tile_base = 0; tile_base < num_tiles; tile_base += TileSteps) {
    const int remaining = static_cast<int>(num_tiles - tile_base);
    const int steps = remaining < TileSteps ? remaining : TileSteps;

    Transition2x2<value_t> element = Transition2x2<value_t>::identity();
    if (threadIdx.x < steps) {
      const int64_t tile_index = tile_base + threadIdx.x;
      const int64_t index = ((tile_index * batch) + b) * ssm + m;
      element = tile_prefixes[index];
    }

    Transition2x2<value_t> prefix = element;
    BlockScan(temp_storage).InclusiveScan(element, prefix, TransitionCombine<value_t>());

    if (threadIdx.x < steps) {
      const Transition2x2<value_t> total = compose(prefix, carry);
      const int64_t tile_index = tile_base + threadIdx.x;
      const int64_t index = ((tile_index * batch) + b) * ssm + m;
      tile_scans[index] = total;
      if (threadIdx.x == steps - 1) {
        carry_shared = total;
      }
    }

    __syncthreads();
    if (threadIdx.x == 0 && steps > 0) {
      carry = carry_shared;
    }
    __syncthreads();
  }
}

template <typename scalar_t, int TileSteps>
__global__ void sdlinoss_forward_apply_kernel(
    const Transition2x2<typename scalar_t::value_type>* __restrict__ local_prefixes,
    const Transition2x2<typename scalar_t::value_type>* __restrict__ tile_scans,
    scalar_t* __restrict__ out_ptr,
    int64_t length,
    int64_t batch,
    int64_t ssm,
    int64_t num_tiles) {
  using value_t = typename scalar_t::value_type;
  const int m = blockIdx.x;
  const int b = blockIdx.y;
  const int tile = blockIdx.z;
  if (m >= ssm || b >= batch || tile >= num_tiles) {
    return;
  }

  const int64_t tile_offset = static_cast<int64_t>(tile) * TileSteps;
  int steps = 0;
  if (tile_offset < length) {
    steps = static_cast<int>(length - tile_offset);
    if (steps > TileSteps) {
      steps = TileSteps;
    }
  }

  Transition2x2<value_t> carry = Transition2x2<value_t>::identity();
  if (tile > 0) {
    const int64_t prev_index = (((int64_t)(tile - 1) * batch) + b) * ssm + m;
    carry = tile_scans[prev_index];
  }

  if (threadIdx.x < steps) {
    const int64_t t = tile_offset + threadIdx.x;
    const int64_t index = ((t * batch) + b) * ssm + m;
    const Transition2x2<value_t> local = local_prefixes[index];
    const Transition2x2<value_t> total = compose(local, carry);
    const int64_t out_index = index * 2;
    out_ptr[out_index] = scalar_t(total.f0_real, total.f0_imag);
    out_ptr[out_index + 1] = scalar_t(total.f1_real, total.f1_imag);
  }
}

constexpr int kSdlinossTileSteps = 128;

}  // namespace ossm

