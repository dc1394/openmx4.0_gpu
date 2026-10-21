#include <openmx/core/nonlocal_workspace.hpp>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace openmx::core {
namespace {
NonlocalShape validated(NonlocalShape shape)
{
  if (shape.maximum_angular < 0 || shape.maximum_angular > 14 || shape.multiplicity <= 0 ||
      shape.projector_maximum_angular < 0 || shape.projector_maximum_angular > 14)
    throw std::invalid_argument("invalid nonlocal angular capacities");
  if (shape.multiplicity > INT_MAX/(shape.maximum_angular+1)/(shape.maximum_angular+1))
    throw std::length_error("nonlocal orbital capacity exceeds C indexing range");
  return shape;
}
std::size_t radial_count(NonlocalShape shape)
{
  return static_cast<std::size_t>(shape.maximum_angular+1)*shape.multiplicity;
}
std::size_t grid_points(NonlocalShape shape, const NonlocalProjectors& projectors,
    std::span<const double> grid)
{
  if (grid.size() == 1 || grid.size() > INT_MAX)
    throw std::invalid_argument("nonlocal radial grid needs at least two points");
  if (!grid.empty() && (radial_count(shape) > INT_MAX/grid.size() ||
                        projectors.slots() > INT_MAX/grid.size()))
    throw std::length_error("nonlocal radial offsets exceed C indexing range");
  return grid.size();
}
} // namespace

NonlocalProjectors::NonlocalProjectors(int maximum_angular, std::span<const NonlocalSpecies> species)
{
  if (maximum_angular < 0 || maximum_angular > 14 || species.empty() || species.size() > INT_MAX)
    throw std::invalid_argument("invalid nonlocal projector species");
  for (const auto& item : species) {
    if (item.angular.size() >= INT_MAX)
      throw std::length_error("nonlocal projector count exceeds C indexing range");
    counts_.push_back(static_cast<int>(item.angular.size()));
    species_offsets_.push_back(offsets_.size());
    offsets_.push_back(0); angular_.push_back(0); // Legacy unused projector 0.
    std::size_t column = 0;
    for (int l : item.angular) {
      if (l < 0 || l > maximum_angular)
        throw std::invalid_argument("nonlocal projector angular number exceeds capacity");
      const auto magnetic = static_cast<std::size_t>(2*l+1);
      if (column > INT_MAX-magnetic)
        throw std::length_error("nonlocal magnetic-component count exceeds C indexing range");
      offsets_.push_back(column); angular_.push_back(l);
      column += magnetic;
    }
    columns_ = std::max(columns_,column);
    slots_ = std::max(slots_,item.angular.size()+1);
  }
  species_offsets_.push_back(offsets_.size());
}

NonlocalWorkspace::NonlocalWorkspace(NonlocalShape shape, std::span<const NonlocalSpecies> species,
    std::span<const double> wavevectors, std::size_t bessel_cache_limit)
  : shape_(validated(shape)), projectors_(shape_.projector_maximum_angular,species),
    points_(grid_points(shape_,projectors_,wavevectors)),
    angular_({4,radial_count(shape_)*(shape_.maximum_angular+1),projectors_.columns()}),
    two_center_(4,shape_.maximum_angular,shape_.projector_maximum_angular),
    sums_({2,radial_count(shape_),points_ ? projectors_.slots() : 0}),
    radial_({radial_count(shape_),points_}),
    nonlocal_radial_({2,projectors_.slots(),points_})
{
  if (points_)
    bessel_ = std::make_unique<PairBesselWorkspace>(2*std::max(shape_.maximum_angular,
        shape_.projector_maximum_angular),wavevectors,bessel_cache_limit);
}

PairBesselTable NonlocalWorkspace::sample(double distance, int angular)
{
  if (!bessel_) throw std::logic_error("angular-only nonlocal workspace has no Bessel grid");
  return bessel_->sample(distance,angular);
}
} // namespace openmx::core
