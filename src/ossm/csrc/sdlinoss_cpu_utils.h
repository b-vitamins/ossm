// Utilities shared by selective D-LinOSS CPU entrypoints for handling
// broadcastable parameter tensors.

#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

namespace ossm {

inline at::Tensor normalize_param_view_cpu(
    const at::Tensor& param,
    const char* name,
    int64_t length,
    int64_t batch,
    int64_t ssm) {
  TORCH_CHECK(
      param.dim() >= 1 && param.dim() <= 3,
      name,
      " must be shaped as (M,), (L,M), (B,M), or (L,B,M); got ",
      param.sizes());

  if (param.dim() == 1) {
    TORCH_CHECK(
        param.size(0) == ssm,
        name,
        " must have size ",
        ssm,
        " along the last dimension; got ",
        param.sizes());
    return param.unsqueeze(0).unsqueeze(0);
  }

  if (param.dim() == 2) {
    TORCH_CHECK(
        param.size(1) == ssm,
        name,
        " must have size ",
        ssm,
        " along the last dimension; got ",
        param.sizes());
    if (param.size(0) == length) {
      return param.unsqueeze(1);
    }
    if (param.size(0) == batch) {
      return param.unsqueeze(0);
    }
  }

  if (param.dim() == 3) {
    TORCH_CHECK(
        param.size(2) == ssm,
        name,
        " must have size ",
        ssm,
        " along the last dimension; got ",
        param.sizes());
    const bool length_ok = param.size(0) == length || param.size(0) == 1;
    const bool batch_ok = param.size(1) == batch || param.size(1) == 1;
    TORCH_CHECK(
        length_ok && batch_ok,
        name,
        " must have shape (L,B,M) with optional singleton L/B axes; got ",
        param.sizes());
    return param;
  }

  TORCH_CHECK(
      false,
      name,
      " must be shaped as (M,), (L,M), (B,M), or (L,B,M); got ",
      param.sizes());
}

inline at::Tensor materialize_param_cpu(
    const at::Tensor& param_view, int64_t length, int64_t batch, int64_t ssm) {
  const bool matches_length = param_view.size(0) == length;
  const bool matches_batch = param_view.size(1) == batch;
  const bool matches_ssm = param_view.size(2) == ssm;

  if (matches_length && matches_batch && matches_ssm) {
    return param_view.is_contiguous() ? param_view : param_view.contiguous();
  }

  return param_view.expand({length, batch, ssm}).contiguous();
}

inline at::Tensor reduce_broadcast_grad(
    const at::Tensor& grad_buffer,
    const at::Tensor& param_view,
    int64_t length,
    int64_t batch) {
  at::Tensor grad = grad_buffer;
  if (param_view.size(0) == 1 && length > 1) {
    grad = grad.sum(0, /*keepdim=*/true);
  }
  if (param_view.size(1) == 1 && batch > 1) {
    grad = grad.sum(1, /*keepdim=*/true);
  }
  return grad;
}

} // namespace ossm

