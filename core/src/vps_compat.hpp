#pragma once

#include <openmx/core/vps_storage.hpp>
#include "legacy_rows.hpp"
#include "openmx_core_vps.h"
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
  OpenMXVpsViews views() noexcept {
    OpenMXVpsViews output{log_radii_.data(),radii_.data(),neutral_potential_.data(),
      hartree_potential_.data(),partial_core_density_.data(),projectors_.data(),energies_.data(),
      angular_.data(),transforms_.data(),nullptr,nullptr,nullptr,nullptr};
    if (neutral_) neutral_->views(output);
    return output;
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
