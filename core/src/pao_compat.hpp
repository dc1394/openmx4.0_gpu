#pragma once

#include <openmx/core/pao_storage.hpp>
#include "openmx_core_pao.h"
#include "openmx_core_reciprocal.h"
#include "openmx_core_atomic.h"
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
  void atomic_views(OpenMXAtomicViews& output) noexcept {
    const auto x=storage_.log_radii(), r=storage_.radii(), d=storage_.density(), b=storage_.density_backup();
    const auto o=storage_.orbitals();
    output.pao_log_radii={x.data_handle(),{x.extent(0),x.extent(1)}};
    output.pao_radii={r.data_handle(),{r.extent(0),r.extent(1)}};
    output.density={d.data_handle(),{d.extent(0),d.extent(1)}};
    output.density_backup={b.data_handle(),{b.extent(0),b.extent(1)}};
    output.orbitals={o.data_handle(),{o.extent(0),o.extent(1),o.extent(2),o.extent(3)}};
  }
  void reciprocal_views(OpenMXReciprocalViews& output) noexcept {
    const auto v=storage_.transforms();
    output.pao_transform={v.data_handle(),{v.extent(0),v.extent(1),v.extent(2),v.extent(3)}};
    output.pao=views();
  }
 private:
  PaoStorage storage_;
  Rows2<double> log_radii_, radii_, density_, density_backup_;
  Rows4<double> orbitals_, transforms_;
};

} // namespace openmx::core::compat
