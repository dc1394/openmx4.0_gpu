#pragma once
#include <openmx/core/overwrite_tensor.hpp>
#include "openmx_core_neutral_grid.h"
#include <array>
#include <stdexcept>
#include <vector>

namespace openmx::core::compat {
/* Own numeric values once. The pointer projection only adapts existing C
   tables; it owns no cells. Core algorithms can borrow the flat mdspan. */
class NeutralGridOwner {
 public:
  explicit NeutralGridOwner(const OpenMXNeutralGridShape& shape)
      : values_(checked_shape(shape)),rows_(5*values_.extent(1)) {
    auto data=values_.view();
    for(std::size_t component=0;component<5;++component)
      for(std::size_t species=0;species<values_.extent(1);++species)
        rows_[component*values_.extent(1)+species]=&element(data,component,species,0);
  }
  NeutralGridOwner(const NeutralGridOwner&)=delete;
  NeutralGridOwner& operator=(const NeutralGridOwner&)=delete;
  NeutralGridOwner(NeutralGridOwner&&)=delete;
  NeutralGridOwner& operator=(NeutralGridOwner&&)=delete;
  OpenMXNeutralGridTables views() noexcept {
    OpenMXNeutralGridTables output{};
    output.values=values_.data();
    for(std::size_t axis=0;axis<3;++axis)output.extent[axis]=values_.extent(axis);
    output.legacy_rows=rows_.data();
    return output;
  }
 private:
  static std::array<std::size_t,3> checked_shape(const OpenMXNeutralGridShape& shape) {
    if(shape.species<=0||shape.capacity<=0)
      throw std::invalid_argument("neutral grid requires positive species and capacity");
    return {5,static_cast<std::size_t>(shape.species),static_cast<std::size_t>(shape.capacity)};
  }
  OverwriteTensor<double,3> values_;
  std::vector<double*> rows_;
};
} // namespace openmx::core::compat
