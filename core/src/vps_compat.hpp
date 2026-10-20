#pragma once

#include <openmx/core/vps_storage.hpp>
#include "legacy_rows.hpp"
#include "openmx_core_vps.h"
#include "openmx_core_reciprocal.h"
#include "openmx_core_atomic.h"
#include <memory>
#include <optional>

namespace openmx::core::compat {

class NeutralProjectorOwner {
 public:
  explicit NeutralProjectorOwner(NeutralProjectorShape shape)
    : storage_(shape), projectors_(storage_.projectors()), energies_(storage_.energies()),
      transforms_(storage_.transforms()), potential_transform_(storage_.potential_transform()) {}
  void views(OpenMXVpsViews& output) noexcept {
    output.neutral_projectors = projectors_.data();
    output.neutral_energies = energies_.data();
    output.neutral_transforms = transforms_.data();
    output.potential_transform = potential_transform_.data();
  }
  void reciprocal_views(OpenMXReciprocalViews& output) noexcept {
    const auto p=storage_.projectors(), t=storage_.transforms();
    const auto e=storage_.energies();
    const auto c=storage_.potential_transform();
    output.neutral_projectors={p.data_handle(),{p.extent(0),p.extent(1),p.extent(2),p.extent(3)}};
    output.neutral_transform={t.data_handle(),{t.extent(0),t.extent(1),t.extent(2),t.extent(3)}};
    output.neutral_energies={e.data_handle(),{e.extent(0),e.extent(1),e.extent(2)}};
    output.potential_transform={c.data_handle(),{c.extent(0),c.extent(1)}};
  }
 private:
  NeutralProjectorStorage storage_;
  Rows4<double> projectors_;
  Rows3<double> energies_;
  Rows4<double> transforms_;
  Rows2<double> potential_transform_;
};

class VpsOwner {
 public:
  explicit VpsOwner(VpsShape shape, std::optional<NeutralProjectorShape> neutral)
    : storage_(shape), log_radii_(storage_.log_radii()), radii_(storage_.radii()),
      neutral_potential_(storage_.neutral_potential()), hartree_potential_(storage_.hartree_potential()),
      partial_core_density_(storage_.partial_core_density()), projectors_(storage_.projectors()),
      energies_(storage_.energies()), angular_(storage_.angular()), transforms_(storage_.transforms()) {
    if (neutral) neutral_ = std::make_unique<NeutralProjectorOwner>(*neutral);
  }
  void atomic_views(OpenMXAtomicViews& output) noexcept {
    const auto x=storage_.log_radii(), r=storage_.radii(), v=storage_.neutral_potential();
    const auto h=storage_.hartree_potential(), c=storage_.partial_core_density();
    const auto p=storage_.projectors(); const auto e=storage_.energies();
    output.vps_log_radii={x.data_handle(),{x.extent(0),x.extent(1)}};
    output.vps_radii={r.data_handle(),{r.extent(0),r.extent(1)}};
    output.neutral_potential={v.data_handle(),{v.extent(0),v.extent(1)}};
    output.hartree_potential={h.data_handle(),{h.extent(0),h.extent(1)}};
    output.partial_core_density={c.data_handle(),{c.extent(0),c.extent(1)}};
    output.projectors={p.data_handle(),{p.extent(0),p.extent(1),p.extent(2),p.extent(3)}};
    output.projector_energies={e.data_handle(),{e.extent(0),e.extent(1),e.extent(2)}};
  }
  OpenMXVpsViews views() noexcept {
    OpenMXVpsViews output{log_radii_.data(),radii_.data(),neutral_potential_.data(),
      hartree_potential_.data(),partial_core_density_.data(),projectors_.data(),energies_.data(),
      angular_.data(),transforms_.data(),nullptr,nullptr,nullptr,nullptr};
    if (neutral_) neutral_->views(output);
    return output;
  }
  void reciprocal_views(OpenMXReciprocalViews& output) noexcept {
    const auto v=storage_.transforms();
    output.nonlocal_transform={v.data_handle(),{v.extent(0),v.extent(1),v.extent(2),v.extent(3)}};
    output.vps=views();
    if (neutral_) neutral_->reciprocal_views(output);
  }
 private:
  VpsStorage storage_;
  Rows2<double> log_radii_, radii_, neutral_potential_, hartree_potential_, partial_core_density_;
  Rows4<double> projectors_;
  Rows3<double> energies_;
  Rows2<int> angular_;
  Rows4<double> transforms_;
  std::unique_ptr<NeutralProjectorOwner> neutral_;
};

} // namespace openmx::core::compat
