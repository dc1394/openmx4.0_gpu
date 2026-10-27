#include <openmx/core/vna_preparation.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace openmx::core {
namespace {
std::size_t validated_basis(int angular, int multiplicity, std::span<const SpeciesBasisView> species)
{
  if (angular < 0 || angular > 14 || multiplicity <= 0 || species.empty() ||
      species.size() > static_cast<std::size_t>(INT_MAX/(angular+1)))
    throw std::invalid_argument("invalid radial basis capacities");
  for (const auto& item : species) {
    if (item.maximum_angular < -1 || item.maximum_angular > angular ||
        item.counts.size() != static_cast<std::size_t>(item.maximum_angular+1))
      throw std::invalid_argument("invalid species basis angular range");
    std::int64_t rows = 0, orbitals = 0;
    for (std::size_t l = 0; l < item.counts.size(); ++l) {
      if (item.counts[l] < 0 || item.counts[l] > multiplicity)
        throw std::invalid_argument("invalid species basis multiplicity");
      rows += item.counts[l];
      orbitals += static_cast<std::int64_t>(item.counts[l])*(2*l+1);
    }
    if (rows > INT_MAX || orbitals > INT_MAX)
      throw std::length_error("species basis exceeds C indexing range");
  }
  return species.size();
}

VnaPreparationShape validated(VnaPreparationShape shape)
{
  if (shape.projector_angular < 0 || shape.projector_angular > 14 ||
      shape.projector_multiplicity < 0 || shape.points <= 0)
    throw std::invalid_argument("invalid VNA projector or integration capacities");
  const auto channels = shape.projector_angular+1;
  if (shape.projector_multiplicity > (INT_MAX-2)/channels ||
      shape.projector_multiplicity > INT_MAX/channels/channels)
    throw std::length_error("VNA projector count exceeds C indexing range");
  return shape;
}

int projection_components(VnaPreparationShape shape, const RadialBasisLayout& basis)
{
  const int components = (shape.projector_angular+1)*(shape.projector_angular+1)*shape.projector_multiplicity;
  for (int orbitals : basis.orbitals())
    if (components && orbitals > INT_MAX/components)
      throw std::length_error("VNA orbital/projector block exceeds C indexing range");
  return components;
}
} // namespace

RadialBasisLayout::RadialBasisLayout(int angular, int multiplicity, std::span<const SpeciesBasisView> species)
  : counts_({validated_basis(angular,multiplicity,species),static_cast<std::size_t>(angular+1)}),
    orbital_offsets_({species.size(),static_cast<std::size_t>(angular+2)})
{
  offsets_.push_back(0);
  for (std::size_t s = 0; s < species.size(); ++s) {
    maximum_angular_.push_back(species[s].maximum_angular);
    int orbitals = 0;
    for (int l = 0; l <= angular; ++l) {
      const int count = l <= species[s].maximum_angular ? species[s].counts[l] : 0;
      element(counts_.view(),s,l) = count;
      if (offsets_.back() > std::numeric_limits<std::size_t>::max()-count)
        throw std::length_error("radial basis offsets overflow size_t");
      offsets_.push_back(offsets_.back()+count);
      orbitals += count*(2*l+1); // The complete sum was validated before allocating.
      element(orbital_offsets_.view(),s,l+1) = orbitals;
    }
    orbitals_.push_back(orbitals);
  }
}

VnaPreparation::VnaPreparation(VnaPreparationShape shape, std::span<const SpeciesBasisView> species)
  : shape_(validated(shape)), basis_(shape_.maximum_angular,shape_.multiplicity,species),
    projector_count_((shape_.projector_angular+1)*shape_.projector_multiplicity),
    projector_components_(projection_components(shape_,basis_)),
    projectors_({2,static_cast<std::size_t>(projector_count_)+2}),
    weights_({2,basis_.radial_count(),static_cast<std::size_t>(shape_.points)})
{
  int row = 0;
  for (int l = 0; l <= shape_.projector_angular; ++l)
    for (int radial = 0; radial < shape_.projector_multiplicity; ++radial,++row) {
      element(projectors_.view(),0,row) = l;
      element(projectors_.view(),1,row) = radial;
    }
}

const openmx::basis::GauntTable& VnaPreparation::gaunt_table(bool projector) const
{
  auto& once=projector?projector_gaunt_once_:product_gaunt_once_;
  auto& owner=projector?projector_gaunt_:product_gaunt_;
  std::call_once(once,[this,projector,&owner] {
    int l0max=0;
    for(int angular:basis_.maximum_angular())l0max=std::max(l0max,angular);
    const int l1max=projector?shape_.projector_angular:l0max;
    const int llmax=projector?2*std::max(l0max,shape_.projector_multiplicity?l1max:0):2*l0max;
    auto table=std::make_unique<openmx::basis::GauntTable>(l0max,l1max,llmax);
    owner=std::move(table);
  });
  return *owner;
}

} // namespace openmx::core
