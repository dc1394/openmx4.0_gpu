#pragma once

#include <openmx/core/tensor.hpp>

namespace openmx::core {

struct OrbitalShape {
  int maximum_angular, multiplicity;
  bool derivatives;
};

// One owner per evaluation or OpenMP worker. Numerical buffers have no row
// pointer trees, and sampling never allocates or changes their addresses.
class OrbitalWorkspace {
 public:
  explicit OrbitalWorkspace(OrbitalShape shape);
  OrbitalWorkspace(const OrbitalWorkspace&) = delete;
  OrbitalWorkspace& operator=(const OrbitalWorkspace&) = delete;
  OrbitalWorkspace(OrbitalWorkspace&&) = delete;
  OrbitalWorkspace& operator=(OrbitalWorkspace&&) = delete;
  const OrbitalShape& shape() const noexcept { return shape_; }
  auto radial() noexcept { return radial_.view(); }
  auto angular() noexcept { return angular_.view(); }
  auto radial() const noexcept { return radial_.view(); }
  auto angular() const noexcept { return angular_.view(); }

 private:
  struct Validated {};
  OrbitalWorkspace(OrbitalShape shape, Validated);
  OrbitalShape shape_;
  Tensor<double,3> radial_, angular_;
};

} // namespace openmx::core
