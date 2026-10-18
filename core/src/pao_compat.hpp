#pragma once

#include <openmx/core/pao_storage.hpp>
#include "openmx_core_pao.h"
#include "legacy_rows.hpp"

namespace openmx::core::compat {

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
  Rows2<double> log_radii_, radii_, density_, density_backup_;
  Rows4<double> orbitals_, transforms_;
};

} // namespace openmx::core::compat
