#include <openmx/core/hvna_layout.hpp>
#include <algorithm>
#include <climits>

namespace openmx::core {

ProjectorArchiveLayout::ProjectorArchiveLayout(int components,
    std::span<const ProjectorAtomBlock> atoms)
{
  if (components < 0 || atoms.size() > INT_MAX-2)
    throw std::invalid_argument("invalid HVNA archive shape");
  for (const auto& atom : atoms) {
    if (atom.orbitals < 0 || atom.neighbors <= 0)
      throw std::invalid_argument("invalid HVNA atom block");
    if (static_cast<std::size_t>(atom.neighbors) > INT_MAX-rows_)
      throw std::length_error("HVNA row index exceeds int");
    rows_ += atom.neighbors;
    const auto limit = std::numeric_limits<std::size_t>::max()/sizeof(float);
    const auto orbitals = static_cast<std::size_t>(atom.orbitals);
    if (components && (orbitals > limit/components || orbitals*components > (limit-elements_)/atom.neighbors))
      throw std::length_error("HVNA archive byte size overflows size_t");
    elements_ += orbitals*components*atom.neighbors;
  }
  bases_.resize(atoms.size()+2);
  offsets_.resize(std::max<std::size_t>(1,rows_));
  std::size_t row = 0, offset = 0;
  for (std::size_t a = 0; a < atoms.size(); ++a) {
    bases_[a+1] = static_cast<int>(row);
    for (int n = 0; n < atoms[a].neighbors; ++n) {
      offsets_[row++] = offset;
      offset += static_cast<std::size_t>(atoms[a].orbitals)*components;
    }
  }
  bases_.back() = static_cast<int>(rows_);
}

HvnaLayout::HvnaLayout(const VnaPreparation& preparation,
    std::span<const ProjectorAtomBlock> local, std::span<const ProjectorAtomBlock> halo,
    std::span<const double> energies)
  : local_(preparation.projector_components(),local),
    halo_(preparation.projector_components(),halo),
    projectors_(preparation,energies)
{}

} // namespace openmx::core
