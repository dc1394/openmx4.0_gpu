#pragma once
#include <openmx/core/tensor.hpp>
#include "openmx_core_core_hole.h"
#include <array>

namespace openmx::core {
class CoreHole {
 public:
  explicit CoreHole(const OpenMXCoreHoleInput& input);
  OpenMXCoreHoleView view() const noexcept;
 private:
  struct Shape {
    int atom,spin,spin_orbit,angular,state;
    std::size_t orbitals,offset;
    bool active;
    double penalty;
  };
  static Shape validate(const OpenMXCoreHoleInput& input);
  explicit CoreHole(Shape shape);
  Shape shape_;
  Tensor<double,2> collinear_;
  std::array<double,2*OPENMX_CH_WEIGHTS> weights_{};
};
void core_hole_trace(const OpenMXCoreHoleView&,const OpenMXCoreHoleTrace&,std::span<dcomplex,9>);
}
