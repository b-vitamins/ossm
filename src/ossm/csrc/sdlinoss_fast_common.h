#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

#include <cmath>

namespace ossm {
constexpr double kDtMin = 1e-6;
constexpr double kDtMax = 1.0;
constexpr double kClampMin = 1e-6;

template <typename T>
C10_HOST_DEVICE inline T fma_val(T a, T b, T c) {
  return a * b + c;
}

template <>
C10_HOST_DEVICE inline float fma_val<float>(float a, float b, float c) {
  return
#if defined(__CUDA_ARCH__)
      ::fmaf
#else
      std::fma
#endif
      (a, b, c);
}

template <>
C10_HOST_DEVICE inline double fma_val<double>(double a, double b, double c) {
  return
#if defined(__CUDA_ARCH__)
      ::fma
#else
      std::fma
#endif
      (a, b, c);
}

template <typename value_t>
struct FastPair {
  value_t m00, m01, m10, m11;
  c10::complex<value_t> f0, f1;
};

template <typename value_t>
C10_HOST_DEVICE inline FastPair<value_t> combine(
    const FastPair<value_t>& p2,
    const FastPair<value_t>& p1) {
  FastPair<value_t> out;
  out.m00 = fma_val(p2.m01, p1.m10, p2.m00 * p1.m00);
  out.m01 = fma_val(p2.m01, p1.m11, p2.m00 * p1.m01);
  out.m10 = fma_val(p2.m11, p1.m10, p2.m10 * p1.m00);
  out.m11 = fma_val(p2.m11, p1.m11, p2.m10 * p1.m01);
  auto g0 = p2.m00 * p1.f0 + p2.m01 * p1.f1;
  auto g1 = p2.m10 * p1.f0 + p2.m11 * p1.f1;
  out.f0 = g0 + p2.f0;
  out.f1 = g1 + p2.f1;
  return out;
}

template <typename value_t>
C10_HOST_DEVICE inline void step_coeff_ex(
    value_t A,
    value_t G,
    value_t dt,
    value_t& alpha,
    value_t& beta,
    value_t& gamma) {
  dt = dt < kDtMin ? kDtMin : (dt > kDtMax ? kDtMax : dt);
  const value_t dt2 = dt * dt;
  alpha = value_t(1) - dt * G;
  beta = -dt2 * A;
  gamma = dt2;
}

template <typename value_t>
C10_HOST_DEVICE inline void step_coeff_imex1(
    value_t A,
    value_t G,
    value_t dt,
    value_t& alpha,
    value_t& beta,
    value_t& gamma) {
  dt = dt < kDtMin ? kDtMin : (dt > kDtMax ? kDtMax : dt);
  const value_t dt2 = dt * dt;
  value_t S_raw = value_t(1) + dt * G;
  value_t S = S_raw < kClampMin ? kClampMin : S_raw;
  value_t invS = value_t(1) / S;
  alpha = invS;
  beta = -(dt2 * A) * invS;
  gamma = dt2 * invS;
}

template <typename value_t>
C10_HOST_DEVICE inline void step_coeff_imex2(
    value_t A,
    value_t G,
    value_t dt,
    value_t& alpha,
    value_t& beta,
    value_t& gamma) {
  dt = dt < kDtMin ? kDtMin : (dt > kDtMax ? kDtMax : dt);
  const value_t dt2 = dt * dt;
  value_t S_raw = value_t(1) + dt2 * A;
  value_t S = S_raw < kClampMin ? kClampMin : S_raw;
  value_t invS = value_t(1) / S;
  alpha = (value_t(1) - dt * G) * invS;
  beta = -(dt2 * A) * invS;
  gamma = dt2 * invS;
}

template <typename value_t>
C10_HOST_DEVICE inline void step_coeff_im(
    value_t A,
    value_t G,
    value_t dt,
    value_t& alpha,
    value_t& beta,
    value_t& gamma) {
  dt = dt < kDtMin ? kDtMin : (dt > kDtMax ? kDtMax : dt);
  const value_t dt2 = dt * dt;
  value_t S_raw = value_t(1) + dt * G + dt2 * A;
  value_t S = S_raw < kClampMin ? kClampMin : S_raw;
  value_t invS = value_t(1) / S;
  alpha = invS;
  beta = -(dt2 * A) * invS;
  gamma = dt2 * invS;
}
} // namespace ossm

