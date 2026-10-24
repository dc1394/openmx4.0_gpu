#pragma once
#include <openmx/core/overwrite_tensor.hpp>
#include "openmx_core_neutral_grid.h"
#include <array>
#include <stdexcept>

namespace openmx::core::compat {
/* Own numeric values once. Native C and Core algorithms borrow the same
   flat layout_right storage without a row-pointer projection. */
class NeutralGridOwner {
 public:
  explicit NeutralGridOwner(const OpenMXNeutralGridShape& shape)
      : values_(checked_shape(shape)) {}
  NeutralGridOwner(const NeutralGridOwner&)=delete;
  NeutralGridOwner& operator=(const NeutralGridOwner&)=delete;
  NeutralGridOwner(NeutralGridOwner&&)=delete;
  NeutralGridOwner& operator=(NeutralGridOwner&&)=delete;
  OpenMXNeutralGridTables views() noexcept {
    OpenMXNeutralGridTables output{};
    output.values=values_.data();
    for(std::size_t axis=0;axis<3;++axis)output.extent[axis]=values_.extent(axis);
    return output;
  }
 private:
  static std::array<std::size_t,3> checked_shape(const OpenMXNeutralGridShape& shape) {
    if(shape.species<=0||shape.capacity<=0)
      throw std::invalid_argument("neutral grid requires positive species and capacity");
    return {5,static_cast<std::size_t>(shape.species),static_cast<std::size_t>(shape.capacity)};
  }
  OverwriteTensor<double,3> values_;
};
} // namespace openmx::core::compat
