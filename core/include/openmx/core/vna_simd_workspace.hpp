#pragma once
#include <openmx/core/tensor.hpp>
#include "openmx_core_vna_simd_workspace.h"

namespace openmx::core {

// CPU scratch only: flat owned storage with mdspan views, no device runtime.
class VnaSimdWorkspace {
 public:
  VnaSimdWorkspace(int points, int orders);
  OpenMXVnaSimdBuffers buffers() noexcept;
  int orders() const noexcept { return orders_; }
 private:
  int orders_;
  Tensor<double,1> arguments_;
  Tensor<double,3> bessel_;
  Tensor<double,2> sums_;
  Tensor<dcomplex,3> harmonics_;
};

struct DsvnaSimdShape {
  int points, l0max, l1max, llmax, projectors;
  std::size_t radials, orbitals, components;
};
class DsvnaSimdWorkspace {
 public:
  DsvnaSimdWorkspace(DsvnaSimdShape shape, bool angular);
  OpenMXDsvnaSimdBuffers buffers() noexcept;
  bool angular() const noexcept { return angular_; }
 private:
  DsvnaSimdShape shape_;
  bool angular_;
  int pad_;
  std::size_t cy_stride_, cy_length_;
  Tensor<double,1> arguments_;
  Tensor<double,3> weighted_, accumulators_;
  Tensor<double,2> harmonics_, coefficients_;
  Tensor<double,4> sums_;
};
} // namespace openmx::core
