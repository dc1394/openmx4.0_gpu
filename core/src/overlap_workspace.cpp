#include <openmx/core/overlap_workspace.hpp>

#include <climits>
#include <cmath>
#include <stdexcept>

namespace openmx::core {
namespace {

OverlapShape validated(OverlapShape shape)
{
  if (shape.maximum_angular < 0 || shape.maximum_angular > 14 || shape.multiplicity <= 0 ||
      shape.points < 2 || !std::isfinite(shape.kmin) || !std::isfinite(shape.kmax) ||
      shape.kmin < 0 || shape.kmax <= shape.kmin)
    throw std::invalid_argument("invalid overlap workspace shape or wavevector interval");
  const auto radial = static_cast<std::size_t>(shape.maximum_angular+1)*shape.multiplicity;
  // The C adapter and task kernels use int for combo counts and RF offsets.
  const auto angular = static_cast<std::size_t>(2*shape.maximum_angular+1);
  if (radial > static_cast<std::size_t>(INT_MAX)/angular/radial ||
      angular > static_cast<std::size_t>(INT_MAX)/shape.points ||
      radial > static_cast<std::size_t>(INT_MAX)/shape.points)
    throw std::length_error("overlap workspace exceeds C integral indexing range");
  return shape;
}

std::size_t radial_count(OverlapShape shape)
{
  return static_cast<std::size_t>(shape.maximum_angular+1)*shape.multiplicity;
}

std::size_t orbital_count(OverlapShape shape)
{
  return radial_count(shape)*(shape.maximum_angular+1);
}

std::vector<double> make_wavevectors(OverlapShape shape)
{
  std::vector<double> grid(shape.points);
  const double step = (shape.kmax-shape.kmin)/(shape.points-1);
  for (int point = 0; point < shape.points; ++point)
    grid[point] = shape.kmin + static_cast<double>(point)*step;
  return grid;
}

} // namespace

OverlapWorkspace::OverlapWorkspace(OverlapShape shape)
  : OverlapWorkspace(validated(shape),Validated{}) {}

OverlapWorkspace::OverlapWorkspace(OverlapShape shape, Validated)
  : shape_(shape), wavevectors_(make_wavevectors(shape)),
    radial_({2,radial_count(shape),static_cast<std::size_t>(shape.points)}),
    sums_({4,static_cast<std::size_t>(2*shape.maximum_angular+1),radial_count(shape)*radial_count(shape)}),
    offsets_({2,radial_count(shape)*radial_count(shape)}),
    angular_({8,orbital_count(shape),orbital_count(shape)}),
    conversion_({8,static_cast<std::size_t>(2*shape.maximum_angular+1),
                   static_cast<std::size_t>(2*shape.maximum_angular+1)}),
    bessel_(2*shape.maximum_angular,wavevectors_,shape.bessel_cache_limit)
{}

} // namespace openmx::core
