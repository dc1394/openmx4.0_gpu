#pragma once

#include <openmx/core/pair_bessel.hpp>
#include "openmx_complex.h"
#include <openmx/core/two_center.hpp>

namespace openmx::core {

struct OverlapShape {
  int maximum_angular, multiplicity, points;
  double kmin, kmax;
  std::size_t bessel_cache_limit;
};

/* Invocation-local host storage. C adapters borrow flat views; numerical
   modules use mdspan. Packing all magnetic states of an angular group
   together removes the unused rectangular six-dimensional padding. */
class OverlapWorkspace {
 public:
  explicit OverlapWorkspace(OverlapShape shape);
  OverlapWorkspace(const OverlapWorkspace&) = delete;
  OverlapWorkspace& operator=(const OverlapWorkspace&) = delete;

  const OverlapShape& shape() const noexcept { return shape_; }
  const std::vector<double>& wavevectors() const noexcept { return wavevectors_; }
  auto radial() noexcept { return radial_.view(); }
  auto sums() noexcept { return sums_.view(); }
  auto offsets() noexcept { return offsets_.view(); }
  auto angular() noexcept { return angular_.view(); }
  auto conversion() noexcept { return two_center_.left(); }
  auto radial() const noexcept { return radial_.view(); }
  auto sums() const noexcept { return sums_.view(); }
  auto offsets() const noexcept { return offsets_.view(); }
  auto angular() const noexcept { return angular_.view(); }
  auto conversion() const noexcept { return two_center_.left(); }
  auto two_center() noexcept { return two_center_.buffers(); }
  PairBesselTable sample(double distance, int angular) { return bessel_.sample(distance,angular); }
  PairBesselStatistics statistics() const noexcept { return bessel_.statistics(); }

 private:
  struct Validated {};
  OverlapWorkspace(OverlapShape shape, Validated);
  OverlapShape shape_;
  std::vector<double> wavevectors_;
  Tensor<double,3> radial_;
  Tensor<double,3> sums_;
  Tensor<int,2> offsets_;
  Tensor<dcomplex,3> angular_;
  TwoCenterWorkspace two_center_;
  PairBesselWorkspace bessel_;
};

} // namespace openmx::core
