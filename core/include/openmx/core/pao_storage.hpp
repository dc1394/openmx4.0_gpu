#pragma once

#include <openmx/core/tensor.hpp>

namespace openmx::core {

struct PaoShape {
  std::size_t species, angular_channels, multiplicity, radial_points, reciprocal_points;
};

// Calculation-scoped storage, allocated before reading PAO files. Capacities
// include the legacy padding, which is initialized along with the active data.
// The numerical interface exposes mdspan, never a row-pointer ownership tree.
class PaoStorage {
 public:
  explicit PaoStorage(PaoShape shape);
  auto log_radii() { return log_radii_.view(); }
  auto radii() { return radii_.view(); }
  auto density() { return density_.view(); }
  auto density_backup() { return density_backup_.view(); }
  auto orbitals() { return orbitals_.view(); }
  auto transforms() { return transforms_.view(); }
  auto log_radii() const { return log_radii_.view(); }
  auto radii() const { return radii_.view(); }
  auto density() const { return density_.view(); }
  auto density_backup() const { return density_backup_.view(); }
  auto orbitals() const { return orbitals_.view(); }
  auto transforms() const { return transforms_.view(); }

  PaoStorage(const PaoStorage&) = delete;
  PaoStorage& operator=(const PaoStorage&) = delete;
  // Stable addresses are required by borrowed C views for this calculation.
  PaoStorage(PaoStorage&&) = delete;
  PaoStorage& operator=(PaoStorage&&) = delete;

 private:
  struct Validated {};
  PaoStorage(PaoShape shape, Validated);
  Tensor<double,2> log_radii_, radii_, density_, density_backup_;
  Tensor<double,4> orbitals_, transforms_;
};

} // namespace openmx::core
