#pragma once

#include <openmx/core/product_storage.hpp>
#include "openmx_core_product.h"
#include "openmx_core_reciprocal.h"

namespace openmx::core::compat {

// Six flat pointer tables adapt the ragged layout for unmigrated C consumers.
// They own no numeric values, and have no inactive lower-triangle row storage.
class ProductOwner {
 public:
  ProductOwner(std::span<const ProductSpecies> species, std::size_t wavevectors)
    : storage_(species,wavevectors), species_(species.size()), flat_species_(species.size()) {
    std::size_t angular = 0, orbitals = 0, right_groups = 0, pairs = 0;
    for (std::size_t s = 0; s < species.size(); ++s) {
      const auto& layout = storage_.layout(s);
      flat_species_[s]={layout.orbital_count(),storage_.species_rows()[s],
          layout.orbital_offsets().data(),layout.pair_rows().data()};
      angular += layout.multiplicities().size();
      orbitals += layout.orbital_count();
      right_groups += layout.orbital_count()*layout.multiplicities().size();
      pairs += layout.pair_count();
    }
    left_angular_.resize(angular); left_orbitals_.resize(orbitals);
    right_angular_.resize(right_groups); right_orbitals_.resize(pairs);
    rows_.resize(storage_.values().extent(0));
    std::size_t a = 0, b = 0, c = 0, d = 0, e = 0;
    for (std::size_t s = 0; s < species.size(); ++s) {
      const auto counts = storage_.layout(s).multiplicities();
      species_[s] = left_angular_.data()+a;
      for (std::size_t l1 = 0; l1 < counts.size(); ++l1) {
        left_angular_[a++] = counts[l1] ? left_orbitals_.data()+b : nullptr;
        for (int m1 = 0; m1 < counts[l1]; ++m1) {
          left_orbitals_[b++] = right_angular_.data()+c;
          for (std::size_t l2 = 0; l2 < counts.size(); ++l2) {
            if (l1 > l2 || !counts[l2]) { right_angular_[c++] = nullptr; continue; }
            right_angular_[c++] = right_orbitals_.data()+d;
            for (int m2 = 0; m2 < counts[l2]; ++m2) {
              right_orbitals_[d++] = rows_.data()+e;
              const auto block = storage_.block(s,l1,m1,l2,m2);
              for (std::size_t l = 0; l <= 2*l2; ++l) rows_[e++] = &element(block,l,0);
            }
          }
        }
      }
    }
  }
  OpenMXProductViews views() noexcept { return {species_.data()}; }
  void reciprocal_views(OpenMXReciprocalViews& output) noexcept {
    const auto v=storage_.values();
    output.product_transform={v.data_handle(),species_.size(),v.extent(0),v.extent(1),flat_species_.data()};
    output.products=views();
  }

 private:
  ProductStorage storage_;
  std::vector<double******> species_;
  std::vector<double*****> left_angular_;
  std::vector<double****> left_orbitals_;
  std::vector<double***> right_angular_;
  std::vector<double**> right_orbitals_;
  std::vector<double*> rows_;
  std::vector<OpenMXProductSpeciesView> flat_species_;
};

} // namespace openmx::core::compat
