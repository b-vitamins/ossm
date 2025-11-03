// Shared helpers for the D-LinOSS C++ kernels.

#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>
#include <type_traits>

namespace ossm {

template <typename scalar_t>
struct ComplexTraits {
  using value_t = typename scalar_t::value_type;
  using complex_t = scalar_t;
};

template <typename scalar_t>
inline scalar_t zero_complex() {
  return scalar_t(0, 0);
}

template <typename scalar_t>
struct Strided3 {
  const scalar_t* p = nullptr;
  int64_t sL = 0;
  int64_t sB = 0;
  int64_t sM = 0;
  int64_t nL = 1;
  int64_t nB = 1;
  int64_t nM = 1;

  C10_HOST_DEVICE inline scalar_t load(int64_t t, int64_t b, int64_t m) const {
    const int64_t tt = nL == 1 ? 0 : t;
    const int64_t bb = nB == 1 ? 0 : b;
    const int64_t mm = nM == 1 ? 0 : m;
    return p[tt * sL + bb * sB + mm * sM];
  }
};

template <typename scalar_t>
struct GradStrided3 {
  scalar_t* p = nullptr;
  int64_t sL = 0;
  int64_t sB = 0;
  int64_t sM = 0;
  int64_t nL = 1;
  int64_t nB = 1;
  int64_t nM = 1;
  bool reduce_L = false;
  bool reduce_B = false;
  bool reduce_M = false;

  C10_HOST_DEVICE inline void store(int64_t t, int64_t b, int64_t m, scalar_t value) const {
    const int64_t tt = nL == 1 ? 0 : t;
    const int64_t bb = nB == 1 ? 0 : b;
    const int64_t mm = nM == 1 ? 0 : m;
    scalar_t* addr = p + tt * sL + bb * sB + mm * sM;
#ifdef __CUDA_ARCH__
    const bool reduce_broadcast = (sB == 0 || sM == 0 || reduce_B || reduce_M);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    static_assert(
        !std::is_same_v<scalar_t, c10::complex<double>>,
        "complex128 broadcast reductions require SM60+. Disable complex128 or target a newer GPU.");
#endif
    if (reduce_broadcast) {
      if constexpr (std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>) {
        atomicAdd(addr, value);
      } else {
        using real_t = typename scalar_t::value_type;
        auto real_addr = reinterpret_cast<real_t*>(addr);
        atomicAdd(real_addr + 0, value.real());
        atomicAdd(real_addr + 1, value.imag());
      }
    } else if (reduce_L) {
      *addr += value;
    } else {
      *addr = value;
    }
#else
    const bool reduce_broadcast = (sB == 0 || sM == 0 || reduce_B || reduce_M);
    if (reduce_L || reduce_broadcast) {
      *addr += value;
    } else {
      *addr = value;
    }
#endif
  }
};

inline void validate_strided3_dims(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.dim() >= 1 && tensor.dim() <= 3,
      "Selective D-LinOSS expects parameters with 1, 2, or 3 dimensions; got ",
      tensor.dim());
}

template <typename scalar_t>
inline Strided3<scalar_t> make_strided3(
    const at::Tensor& tensor, int64_t length, int64_t batch, int64_t ssm) {
  validate_strided3_dims(tensor);

  Strided3<scalar_t> result;
  result.p = tensor.data_ptr<scalar_t>();

  const int64_t total_size = length * batch * ssm;
  const int64_t storage_offset = tensor.storage_offset();
  const int64_t storage_size = static_cast<int64_t>(
      tensor.storage().nbytes() / tensor.element_size());
  const bool shares_full_storage =
      tensor.dim() <= 2 && storage_offset == 0 && storage_size >= total_size;
  if (shares_full_storage && tensor.numel() == total_size) {
    result.nL = length;
    result.sL = batch > 0 ? batch * ssm : 0;
    result.nB = batch;
    result.sB = ssm;
    result.nM = ssm;
    result.sM = 1;
    return result;
  }

  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();

  auto assign_axis = [&](int axis, int64_t size, int64_t stride) {
    int64_t* n = nullptr;
    int64_t* s = nullptr;
    if (axis == 0) {
      n = &result.nL;
      s = &result.sL;
    } else if (axis == 1) {
      n = &result.nB;
      s = &result.sB;
    } else {
      n = &result.nM;
      s = &result.sM;
    }
    *n = size;
    *s = size == 1 ? 0 : stride;
  };

  auto check_size = [](int64_t size, int64_t expected) {
    return size == expected || size == 1;
  };

  switch (tensor.dim()) {
    case 1: {
      const int64_t size_m = sizes[0];
      TORCH_CHECK(
          check_size(size_m, ssm),
          "Expected size ",
          ssm,
          " or 1 along M axis but got ",
          size_m);
      assign_axis(2, size_m, strides[0]);
      break;
    }
    case 2: {
      const int64_t size0 = sizes[0];
      const int64_t size1 = sizes[1];
      TORCH_CHECK(
          check_size(size1, ssm),
          "Expected size ",
          ssm,
          " or 1 along last axis but got ",
          size1);
      if (check_size(size0, length)) {
        assign_axis(0, size0, strides[0]);
      } else if (check_size(size0, batch)) {
        assign_axis(1, size0, strides[0]);
      } else {
        TORCH_CHECK(
            false,
            "First dimension must match length or batch (or be 1), got ",
            size0);
      }
      assign_axis(2, size1, strides[1]);
      break;
    }
    case 3: {
      const int64_t size_l = sizes[0];
      const int64_t size_b = sizes[1];
      const int64_t size_m = sizes[2];
      TORCH_CHECK(
          check_size(size_l, length),
          "Expected size ",
          length,
          " or 1 along length axis but got ",
          size_l);
      TORCH_CHECK(
          check_size(size_b, batch),
          "Expected size ",
          batch,
          " or 1 along batch axis but got ",
          size_b);
      TORCH_CHECK(
          check_size(size_m, ssm),
          "Expected size ",
          ssm,
          " or 1 along state axis but got ",
          size_m);
      assign_axis(0, size_l, strides[0]);
      assign_axis(1, size_b, strides[1]);
      assign_axis(2, size_m, strides[2]);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "unreachable");
  }

  return result;
}

template <typename scalar_t>
inline GradStrided3<scalar_t> make_grad_strided3(
    at::Tensor& tensor, int64_t length, int64_t batch, int64_t ssm) {
  GradStrided3<scalar_t> result;
  result.p = tensor.data_ptr<scalar_t>();

  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();

  auto assign_axis = [&](int axis, int64_t size, int64_t stride) {
    int64_t* n = nullptr;
    int64_t* s = nullptr;
    if (axis == 0) {
      n = &result.nL;
      s = &result.sL;
    } else if (axis == 1) {
      n = &result.nB;
      s = &result.sB;
    } else {
      n = &result.nM;
      s = &result.sM;
    }
    *n = size;
    *s = size == 1 ? 0 : stride;
  };

  auto check_size = [](int64_t size, int64_t expected) {
    return size == expected || size == 1;
  };

  switch (tensor.dim()) {
    case 1: {
      const int64_t size_m = sizes[0];
      TORCH_CHECK(
          check_size(size_m, ssm),
          "Expected gradient last dimension to match state size ",
          ssm,
          " or be 1, got ",
          size_m);
      assign_axis(2, size_m, strides[0]);
      break;
    }
    case 2: {
      const int64_t size0 = sizes[0];
      const int64_t size1 = sizes[1];
      TORCH_CHECK(
          check_size(size1, ssm),
          "Expected gradient last dimension to match state size ",
          ssm,
          " or be 1, got ",
          size1);
      if (size0 == length || size0 == 1) {
        assign_axis(0, size0, strides[0]);
      } else {
        TORCH_CHECK(
            size0 == batch,
            "Expected gradient dimension to match length, batch, or be 1; got ",
            size0);
        assign_axis(1, size0, strides[0]);
      }
      assign_axis(2, size1, strides[1]);
      break;
    }
    case 3: {
      assign_axis(0, sizes[0], strides[0]);
      assign_axis(1, sizes[1], strides[1]);
      assign_axis(2, sizes[2], strides[2]);
      break;
    }
    default:
      TORCH_CHECK(false, "Grad tensor must have 1, 2, or 3 dimensions");
  }

  result.reduce_L = (result.nL == 1 && length > 1);
  result.reduce_B = (result.nB == 1 && batch > 1);
  result.reduce_M = (result.nM == 1 && ssm > 1);
  return result;
}

}  // namespace ossm

