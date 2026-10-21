#pragma once

#include <openmx/core/tensor.hpp>

namespace openmx::core {

struct VpsShape {
  std::size_t species, spins, projectors, radial_points, reciprocal_points;
};

// Capacities match the C file reader. The projector table is zero-based,
// while transformed projectors reserve legacy index 0 and two extra slots.
class VpsStorage {
 public:
  explicit VpsStorage(VpsShape shape);
  auto log_radii() { return log_radii_.view(); }
  auto radii() { return radii_.view(); }
  auto neutral_potential() { return neutral_potential_.view(); }
  auto hartree_potential() { return hartree_potential_.view(); }
  auto partial_core_density() { return partial_core_density_.view(); }
  auto projectors() { return projectors_.view(); }
  auto energies() { return energies_.view(); }
  auto angular() { return angular_.view(); }
  auto transforms() { return transforms_.view(); }
  auto log_radii() const { return log_radii_.view(); }
  auto radii() const { return radii_.view(); }
  auto neutral_potential() const { return neutral_potential_.view(); }
  auto hartree_potential() const { return hartree_potential_.view(); }
  auto partial_core_density() const { return partial_core_density_.view(); }
  auto projectors() const { return projectors_.view(); }
  auto energies() const { return energies_.view(); }
  auto angular() const { return angular_.view(); }
  auto transforms() const { return transforms_.view(); }

  VpsStorage(const VpsStorage&) = delete;
  VpsStorage& operator=(const VpsStorage&) = delete;
  VpsStorage(VpsStorage&&) = delete;
  VpsStorage& operator=(VpsStorage&&) = delete;

 private:
  struct Validated {};
  VpsStorage(VpsShape shape, Validated);
  Tensor<double,2> log_radii_, radii_, neutral_potential_, hartree_potential_, partial_core_density_;
  Tensor<double,4> projectors_;
  Tensor<double,3> energies_;
  Tensor<int,2> angular_;
  Tensor<double,4> transforms_;
};

struct NeutralProjectorShape {
  std::size_t species, angular_channels, multiplicity, radial_points, wavevectors;
};

// Optional VNA expansion tables, absent when projector expansion is disabled.
// Product-orbital tables have a separate ragged layout and are not owned here.
class NeutralProjectorStorage {
 public:
  explicit NeutralProjectorStorage(NeutralProjectorShape shape);
  auto projectors() { return projectors_.view(); }
  auto energies() { return energies_.view(); }
  auto transforms() { return transforms_.view(); }
  auto potential_transform() { return potential_transform_.view(); }
  auto projectors() const { return projectors_.view(); }
  auto energies() const { return energies_.view(); }
  auto transforms() const { return transforms_.view(); }
  auto potential_transform() const { return potential_transform_.view(); }

  NeutralProjectorStorage(const NeutralProjectorStorage&) = delete;
  NeutralProjectorStorage& operator=(const NeutralProjectorStorage&) = delete;
  NeutralProjectorStorage(NeutralProjectorStorage&&) = delete;
  NeutralProjectorStorage& operator=(NeutralProjectorStorage&&) = delete;

 private:
  struct Validated {};
  NeutralProjectorStorage(NeutralProjectorShape shape, Validated);
  Tensor<double,4> projectors_;
  Tensor<double,3> energies_;
  Tensor<double,4> transforms_;
  Tensor<double,2> potential_transform_;
};

} // namespace openmx::core
