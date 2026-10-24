#pragma once
#include <openmx/core/overwrite_tensor.hpp>
#include "openmx_core_neutral_grid_metadata.h"
#include <climits>
#include <stdexcept>

namespace openmx::core::compat {
class NeutralGridMetadataOwner {
 public:
  explicit NeutralGridMetadataOwner(const OpenMXNeutralGridMetadataShape& shape)
      :species_(checked_species(shape)),counts_({species_}),spacing_({species_}) {}
  NeutralGridMetadataOwner(const NeutralGridMetadataOwner&)=delete;
  NeutralGridMetadataOwner& operator=(const NeutralGridMetadataOwner&)=delete;
  NeutralGridMetadataOwner(NeutralGridMetadataOwner&&)=delete;
  NeutralGridMetadataOwner& operator=(NeutralGridMetadataOwner&&)=delete;
  OpenMXNeutralGridMetadataView views() noexcept {
    return {counts_.data(),spacing_.data(),species_};
  }
 private:
  static std::size_t checked_species(const OpenMXNeutralGridMetadataShape& shape) {
    if(shape.species==0||shape.species>static_cast<std::size_t>(INT_MAX))
      throw std::invalid_argument("neutral grid metadata species must fit the native positive int range");
    return shape.species;
  }
  std::size_t species_;
  OverwriteTensor<int,1> counts_;
  OverwriteTensor<double,1> spacing_;
};
} // namespace openmx::core::compat
