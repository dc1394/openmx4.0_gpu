#pragma once
#include <openmx/core/tensor.hpp>
#include <vector>
#include "openmx_core_vna_host_tables.h"

namespace openmx::core {
class RadialBasisLayout;

struct VnaProjectorPackingShape {
  int species, points, angular, multiplicity;
};
// Immutable packing owners: constructing one consumes explicit views only.
// The objects stay in place while borrowed views are shared with CPU workers.
class VnaHostProjectors {
 public:
  VnaHostProjectors(VnaProjectorPackingShape shape, const OpenMXVnaHostProjectorInput& input);
  VnaHostProjectors(const VnaHostProjectors&) = delete;
  VnaHostProjectors& operator=(const VnaHostProjectors&) = delete;
  OpenMXVnaHostProjectorData view() const noexcept;
 private:
  std::size_t stride_, width_;
  Tensor<double,3> projectors_;
  Tensor<double,4> conversion_;
};
class VnaHostProducts {
 public:
  VnaHostProducts(const RadialBasisLayout& basis, int points, const OpenMXProductRadialView& input);
  VnaHostProducts(const VnaHostProducts&) = delete;
  VnaHostProducts& operator=(const VnaHostProducts&) = delete;
  OpenMXVnaHostProductData view() const noexcept;
 private:
  int points_, orders_, radials_max_ = 0;
  std::vector<int> radials_;
  std::vector<std::size_t> offsets_;
  Tensor<double,3> products_;
};
} // namespace openmx::core
