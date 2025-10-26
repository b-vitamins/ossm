// Shared helpers for the D-LinOSS C++ kernels.

#pragma once

#include <c10/util/complex.h>

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

}  // namespace ossm

