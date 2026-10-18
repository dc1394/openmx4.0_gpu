#pragma once

#include <openmx/core/pao_storage.hpp>
#include "openmx_core_pao.h"

namespace openmx::core::compat {

// Pointer tables belong solely to the C compatibility boundary. They never
// own numerical rows; PaoStorage owns all values and exposes only mdspan.
class Rows2 {
 public:
  explicit Rows2(Tensor<double,2>::view_type view) : rows_(view.extent(0)) {
    for (std::size_t i = 0; i < rows_.size(); ++i)
      rows_[i] = &element(view,i,0);
  }
  double** data() noexcept { return rows_.data(); }
 private:
  std::vector<double*> rows_;
};

class Rows4 {
 public:
  explicit Rows4(Tensor<double,4>::view_type view)
    : species_(view.extent(0)), angular_(view.extent(0)*view.extent(1)),
      rows_(view.extent(0)*view.extent(1)*view.extent(2)) {
    for (std::size_t s = 0; s < view.extent(0); ++s) {
      species_[s] = angular_.data()+s*view.extent(1);
      for (std::size_t l = 0; l < view.extent(1); ++l) {
        const auto block = s*view.extent(1)+l;
        angular_[block] = rows_.data()+block*view.extent(2);
        for (std::size_t m = 0; m < view.extent(2); ++m)
          rows_[block*view.extent(2)+m] = &element(view,s,l,m,0);
      }
    }
  }
  double**** data() noexcept { return species_.data(); }
 private:
  std::vector<double***> species_;
  std::vector<double**> angular_;
  std::vector<double*> rows_;
};

class PaoOwner {
 public:
  explicit PaoOwner(PaoShape shape)
    : storage_(shape), log_radii_(storage_.log_radii()), radii_(storage_.radii()),
      density_(storage_.density()), density_backup_(storage_.density_backup()),
      orbitals_(storage_.orbitals()), transforms_(storage_.transforms()) {}
  OpenMXPaoViews views() noexcept {
    return {log_radii_.data(),radii_.data(),density_.data(),density_backup_.data(),
            orbitals_.data(),transforms_.data()};
  }
 private:
  PaoStorage storage_;
  Rows2 log_radii_, radii_, density_, density_backup_;
  Rows4 orbitals_, transforms_;
};

} // namespace openmx::core::compat
