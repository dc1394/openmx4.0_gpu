#include <openmx/core/pair_bessel.hpp>

namespace openmx::core {

PairBesselWorkspace::PairBesselWorkspace(int maximum_angular,
    std::span<const double> wavevectors, std::size_t cache_limit)
  : implementation_(maximum_angular,wavevectors,cache_limit)
{}

PairBesselTable PairBesselWorkspace::sample(double distance, int angular)
{
  const auto table=implementation_.sample(distance,angular);
  using Extents=Tensor<double,3>::extents_type;
  return {{table.storage().data(),Extents(2,table.row_capacity(),table.points())},angular};
}

} // namespace openmx::core
