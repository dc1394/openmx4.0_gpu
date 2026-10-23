#pragma once
#include <openmx/core/tensor.hpp>
#include "openmx_core_dispersion.h"
#include <array>

namespace openmx::core {
class DispersionWorkspace {
 public:
  using shape_type=std::array<std::size_t,5>;
  static shape_type dimensions(const OpenMXDispersionShape&);
  explicit DispersionWorkspace(const OpenMXDispersionShape&);
  const shape_type& shape() const noexcept { return shape_; }
  OpenMXDispersionView view() noexcept;
  void clear() noexcept;
  DispersionWorkspace(const DispersionWorkspace&)=delete;
  DispersionWorkspace& operator=(const DispersionWorkspace&)=delete;
 private:
  shape_type shape_;
  Tensor<double,1> coordination_;
  Tensor<double,2> c6_derivative_,energy_derivative_;
  Tensor<double,5> coordination_derivative_;
};
}
