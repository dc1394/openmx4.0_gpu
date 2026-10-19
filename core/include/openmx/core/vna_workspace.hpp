#pragma once

#include <openmx/core/tensor.hpp>
#include "openmx_complex.h"

namespace openmx::core {

struct VnaShape { int maximum_angular, multiplicity, points; };

// Per-worker host storage for both orientations of the neutral-potential
// pair integral. The numerical adapter fills the Bessel views using the
// original Gauss-quadrature recurrence, independently of GPU runtimes.
class VnaWorkspace {
 public:
  explicit VnaWorkspace(VnaShape shape);
  VnaWorkspace(const VnaWorkspace&) = delete;
  VnaWorkspace& operator=(const VnaWorkspace&) = delete;
  const VnaShape& shape() const noexcept { return shape_; }
  auto angular() noexcept { return angular_.view(); }
  auto conversion() noexcept { return conversion_.view(); }
  auto bessel() noexcept { return bessel_.view(); }
  auto scratch() noexcept { return scratch_.view(); }
  auto angular() const noexcept { return angular_.view(); }
  auto conversion() const noexcept { return conversion_.view(); }
  auto bessel() const noexcept { return bessel_.view(); }
  auto scratch() const noexcept { return scratch_.view(); }
 private:
  VnaShape shape_;
  Tensor<dcomplex,3> angular_, conversion_;
  Tensor<double,3> bessel_;
  Tensor<double,2> scratch_;
};

} // namespace openmx::core
