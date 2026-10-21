#pragma once

#include <openmx/core/vna_preparation.hpp>
#include "openmx_complex.h"

namespace openmx::core {

// Per-worker host storage for the neutral-potential projector integral.
// Only capacities are copied from preparation; no borrowed views are retained.
class DsvnaWorkspace {
 public:
  explicit DsvnaWorkspace(const VnaPreparation& preparation);
  DsvnaWorkspace(const DsvnaWorkspace&) = delete;
  DsvnaWorkspace& operator=(const DsvnaWorkspace&) = delete;
  auto angular() noexcept { return angular_.view(); }
  auto conversion() noexcept { return conversion_.view(); }
  auto sums() noexcept { return sums_.view(); }
  auto radial() noexcept { return radial_.view(); }
  auto bessel() noexcept { return bessel_.view(); }
  auto scratch() noexcept { return scratch_.view(); }
  auto angular() const noexcept { return angular_.view(); }
  auto conversion() const noexcept { return conversion_.view(); }
  auto sums() const noexcept { return sums_.view(); }
  auto radial() const noexcept { return radial_.view(); }
  auto bessel() const noexcept { return bessel_.view(); }
  auto scratch() const noexcept { return scratch_.view(); }
 private:
  Tensor<dcomplex,3> angular_, conversion_;
  Tensor<double,3> sums_, bessel_;
  Tensor<double,2> radial_, scratch_;
};

} // namespace openmx::core
