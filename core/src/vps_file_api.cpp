#include "openmx_core_vps_file.h"
#include <openmx/basis/vps_file.hpp>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" int openmx_core_with_vps_file(const char* path, int metadata_only,
    OpenMXVpsFileConsumer consumer, void* user, char* error, std::size_t capacity)
{
  if (error && capacity) *error = '\0';
  try {
    if (!path || !consumer || (metadata_only != 0 && metadata_only != 1))
      throw std::invalid_argument("invalid VPS read request");
    const auto file = openmx::basis::VpsFile::from_file(path,metadata_only ?
        openmx::basis::VpsReadMode::metadata : openmx::basis::VpsReadMode::full);
    const auto& m = file.metadata();
    std::vector<double> local;
    if (!metadata_only) {
      local.resize(static_cast<std::size_t>(m.mesh_points)+2,0.);
      std::copy(file.local_potential().begin(),file.local_potential().end(),local.begin());
    }
    const OpenMXVpsFileData data{m.atom_species,m.total_electrons,m.valence_electrons,
      m.atomic_number(),m.mesh_points,m.projector_components,m.max_angular,
      m.j_dependent,m.partial_core,m.format == openmx::basis::VpsFormat::legacy,metadata_only,
      file.angular().size(),file.angular().data(),file.projector_energies().data(),
      file.log_radii().data(),file.radii().data(),file.projector_values().data(),
      file.partial_core_density().data(),local.data(),local.size()};
    if (consumer(user,&data,error,capacity)) return 1;
    if (error && capacity && !*error) std::snprintf(error,capacity,"VPS consumer failed");
  } catch (const std::exception& exception) {
    if (error && capacity) std::snprintf(error,capacity,"%s",exception.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error,capacity,"unknown VPS reader error");
  }
  return 0;
}
