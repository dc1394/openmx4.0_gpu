#pragma once
#include <openmx/core/pair_bessel.hpp>
#include "openmx_complex.h"
#include <openmx/core/two_center.hpp>
#include <memory>

namespace openmx::core {

struct NonlocalShape { int maximum_angular, multiplicity, projector_maximum_angular; };
struct NonlocalSpecies { std::span<const int> angular; };

class NonlocalProjectors {
 public:
  NonlocalProjectors(int maximum_angular, std::span<const NonlocalSpecies> species);
  std::span<const int> counts() const noexcept { return counts_; }
  std::span<const int> angular() const noexcept { return angular_; }
  std::span<const std::size_t> species_offsets() const noexcept { return species_offsets_; }
  std::span<const std::size_t> offsets() const noexcept { return offsets_; }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t slots() const noexcept { return slots_; }
 private:
  std::vector<int> counts_, angular_;
  std::vector<std::size_t> species_offsets_, offsets_;
  std::size_t columns_ = 1, slots_ = 1;
};

class NonlocalWorkspace {
 public:
  NonlocalWorkspace(NonlocalShape shape, std::span<const NonlocalSpecies> species,
      std::span<const double> wavevectors = {}, std::size_t bessel_cache_limit = 0);
  NonlocalWorkspace(const NonlocalWorkspace&) = delete;
  NonlocalWorkspace& operator=(const NonlocalWorkspace&) = delete;
  const NonlocalShape& shape() const noexcept { return shape_; }
  const NonlocalProjectors& projectors() const noexcept { return projectors_; }
  auto angular() noexcept { return angular_.view(); }
  auto conversion() noexcept { return two_center_.left(); }
  auto sums() noexcept { return sums_.view(); }
  auto radial() noexcept { return radial_.view(); }
  auto nonlocal_radial() noexcept { return nonlocal_radial_.view(); }
  auto angular() const noexcept { return angular_.view(); }
  auto conversion() const noexcept { return two_center_.left(); }
  auto sums() const noexcept { return sums_.view(); }
  auto radial() const noexcept { return radial_.view(); }
  auto nonlocal_radial() const noexcept { return nonlocal_radial_.view(); }
  auto two_center() noexcept { return two_center_.buffers(); }
  PairBesselTable sample(double distance, int angular);
 private:
  NonlocalShape shape_;
  NonlocalProjectors projectors_;
  std::size_t points_;
  Tensor<dcomplex,3> angular_;
  TwoCenterWorkspace two_center_;
  Tensor<double,3> sums_;
  Tensor<double,2> radial_;
  Tensor<double,3> nonlocal_radial_;
  std::unique_ptr<PairBesselWorkspace> bessel_;
};
} // namespace openmx::core
