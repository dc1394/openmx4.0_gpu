#include <openmx/core/hvna_workspace.hpp>
#include <algorithm>

namespace openmx::core {

HvnaProjectors::HvnaProjectors(const VnaPreparation& preparation, std::span<const double> energies)
  : components_(preparation.projector_components()), groups_(preparation.projector_count()),
    orbitals_(preparation.basis().orbitals().begin(),preparation.basis().orbitals().end()),
    lengths_(groups_), energies_({orbitals_.size(),static_cast<std::size_t>(groups_)})
{
  if (energies.size() != energies_.size())
    throw std::invalid_argument("invalid HVNA energy table size");
  std::copy(energies.begin(),energies.end(),energies_.data());
  for (int p = 0; p < groups_; ++p)
    lengths_[p] = 2*element(preparation.projectors(),0,p)+1;
}

HvnaWorkspace::HvnaWorkspace(const HvnaProjectors& projectors)
  : matrix_({static_cast<std::size_t>(*std::max_element(projectors.orbitals().begin(),projectors.orbitals().end())),
             static_cast<std::size_t>(*std::max_element(projectors.orbitals().begin(),projectors.orbitals().end()))})
{}

} // namespace openmx::core
