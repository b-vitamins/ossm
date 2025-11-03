#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace ossm {
enum class VaryL : uint8_t { No = 0, Yes = 1 };
enum class VaryB : uint8_t { No = 0, Yes = 1 };
enum class VaryM : uint8_t { No = 0, Yes = 1 };

struct Strides3 {
  int64_t sL;
  int64_t sB;
  int64_t sM;
};

inline Strides3 collapse_strides3(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() == 3, "Expected a 3D tensor to compute broadcast strides.");
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  Strides3 result{};
  result.sL = sizes[0] == 1 ? 0 : strides[0];
  result.sB = sizes[1] == 1 ? 0 : strides[1];
  result.sM = sizes[2] == 1 ? 0 : strides[2];
  return result;
}
} // namespace ossm

