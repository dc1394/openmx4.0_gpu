#include "openmx_core_pao_file.h"
#include <openmx/basis/pao_file.hpp>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
template<std::size_t N> void check_storage(const double* values, const std::size_t (&extent)[N])
{
  if (!values) throw std::invalid_argument("missing PAO output storage");
  std::size_t count = 1;
  const auto maximum = std::numeric_limits<std::ptrdiff_t>::max()/sizeof(double);
  for (auto size : extent) {
    if (!size || count > maximum/size) throw std::invalid_argument("invalid PAO output extent");
    count *= size;
  }
}
void check_row(const OpenMXRadialView2& row, std::size_t species, std::size_t points)
{
  check_storage(row.values,row.extent);
  if (species >= row.extent[0] || points > row.extent[1])
    throw std::invalid_argument("PAO output row capacity is insufficient");
}
} // namespace

extern "C" int openmx_core_pao_read_file(const char* path, int metadata_only, int eigen_angular,
    const OpenMXAtomicViews* target, std::size_t species, OpenMXPaoFileInfo* information,
    char* error, std::size_t capacity)
{
  if (error && capacity) *error = '\0';
  try {
    if (!path || !target || !information || (metadata_only != 0 && metadata_only != 1) ||
        eigen_angular < -1 || eigen_angular > 29)
      throw std::invalid_argument("invalid PAO read request");
    const auto& s = target->species;
    if (species >= s.species || !s.atomic_number || !s.pao_mesh || !s.pao_lmax ||
        !s.pao_multiplicity || !s.cutoff)
      throw std::invalid_argument("missing PAO species output metadata");
    const auto file = openmx::basis::PaoFile::from_file(path,metadata_only ?
        openmx::basis::PaoReadMode::metadata : openmx::basis::PaoReadMode::full);
    const auto& m = file.metadata();
    OpenMXPaoFileInfo result{m.atom_species,m.cutoff,m.atomic_number(),m.mesh_points,
                            m.lmax,m.multiplicity,0,0};
    if (eigen_angular >= 0 && eigen_angular <= m.lmax) {
      if (const auto value = file.orbital_energy(eigen_angular,0)) {
        result.eigenvalue_found = 1;
        result.eigenvalue = *value;
      }
    }
    if (!metadata_only) {
      if (!s.basis_lmax || !s.basis || !s.angular_channels ||
          s.species > std::numeric_limits<std::ptrdiff_t>::max()/sizeof(int)/s.angular_channels ||
          s.basis_lmax[species] < 0 || s.basis_lmax[species] > m.lmax ||
          static_cast<std::size_t>(s.basis_lmax[species]) >= s.angular_channels)
        throw std::invalid_argument("PAO angular data is insufficient for selected basis");
      for (int l = 0; l <= s.basis_lmax[species]; ++l) {
        const int requested = s.basis[species*s.angular_channels+l];
        if (requested < 0 || requested > m.multiplicity)
          throw std::invalid_argument("PAO multiplicity is insufficient for selected basis");
      }
      check_row(target->pao_log_radii,species,m.mesh_points);
      check_row(target->pao_radii,species,m.mesh_points);
      check_row(target->density,species,static_cast<std::size_t>(m.mesh_points)+2);
      check_row(target->density_backup,species,static_cast<std::size_t>(m.mesh_points)+2);
      const auto& orbitals = target->orbitals;
      check_storage(orbitals.values,orbitals.extent);
      if (species >= orbitals.extent[0] || static_cast<std::size_t>(m.lmax) >= orbitals.extent[1] ||
          static_cast<std::size_t>(m.multiplicity) > orbitals.extent[2] ||
          static_cast<std::size_t>(m.mesh_points) > orbitals.extent[3])
        throw std::invalid_argument("PAO orbital output capacity is insufficient");

      // All fallible work precedes the commit. No allocation or callbacks
      // occur below; the validated owner remains alive through the copy.
      std::copy(file.log_radii().begin(),file.log_radii().end(),openmx_radial_row2(&target->pao_log_radii,species));
      std::copy(file.radii().begin(),file.radii().end(),openmx_radial_row2(&target->pao_radii,species));
      std::copy(file.density().begin(),file.density().end(),openmx_radial_row2(&target->density,species)+1);
      std::copy(file.density().begin(),file.density().end(),openmx_radial_row2(&target->density_backup,species)+1);
      for (int l = 0; l <= m.lmax; ++l)
        for (int mu = 0; mu < m.multiplicity; ++mu) {
          const auto row = file.orbital(l,mu);
          std::copy(row.begin(),row.end(),openmx_radial_row4(&target->orbitals,species,l,mu));
        }
    }
    s.atomic_number[species] = result.atomic_number;
    s.pao_mesh[species] = m.mesh_points;
    s.pao_lmax[species] = m.lmax;
    s.pao_multiplicity[species] = m.multiplicity;
    s.cutoff[species] = m.cutoff;
    *information = result;
    return 1;
  } catch (const std::exception& exception) {
    if (error && capacity) std::snprintf(error,capacity,"%s",exception.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error,capacity,"unknown PAO reader error");
  }
  return 0;
}
