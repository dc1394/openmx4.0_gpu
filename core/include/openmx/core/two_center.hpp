#pragma once
#include <openmx/core/tensor.hpp>
#include "openmx_core_two_center.h"

namespace openmx::core {

// Host-side numerical module shared by every native GPU backend. The caller
// supplies the OpenMP team; this owner does not initialize a Kokkos runtime.
class TwoCenterWorkspace {
 public:
  TwoCenterWorkspace(std::size_t channels, int left_angular, int right_angular);
  auto left() noexcept { return left_.view(); }
  auto left() const noexcept { return left_.view(); }
  OpenMXTwoCenterBuffers buffers() noexcept;
 private:
  Tensor<dcomplex,3> left_;
  Tensor<double,3> values_, momenta_;
};

void transform_two_center(const OpenMXTwoCenterInput& input,
    const OpenMXTwoCenterBuffers& buffers);
} // namespace openmx::core
