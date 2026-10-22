#pragma once
#include <openmx/core/tensor.hpp>
#include "openmx_core_projector_force.h"
#include <span>
#include <vector>

namespace openmx::core {
class ProjectorForce {
 public:
  ProjectorForce(const OpenMXAtomicViews&,int spin,int spin_orbit);
  OpenMXProjectorForceView view() const noexcept;
  ProjectorForce(const ProjectorForce&) = delete;
  ProjectorForce& operator=(const ProjectorForce&) = delete;
  ProjectorForce(ProjectorForce&&) = delete;
  ProjectorForce& operator=(ProjectorForce&&) = delete;
 private:
  int spin_,spin_orbit_;
  std::vector<OpenMXForceProjectorChannel> channels_;
  std::vector<OpenMXForceProjectorSpecies> species_;
};
void projector_force_trace(const OpenMXProjectorForceView&,std::size_t species,
    const OpenMXProjectorForceRows&,std::span<dcomplex,9>);
void projector_force_block(const OpenMXProjectorForceView&,std::size_t species,
    const OpenMXProjectorForceBlock&);
}
