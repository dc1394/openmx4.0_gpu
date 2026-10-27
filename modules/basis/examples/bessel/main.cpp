#include <openmx/basis/bessel.hpp>
#include <openmx/basis/pair_bessel.hpp>

#include <array>
#include <cmath>
#include <iostream>

int main()
{
  const openmx::basis::SphericalBesselSample sample(1.,4);
  const std::array wavevectors{0.,.5,1.,2.};
  openmx::basis::PairBesselWorkspace workspace(4,wavevectors,2);
  const auto table=workspace.sample(1.,4);
  const auto hit=workspace.sample(1.,4);
  if (std::abs(sample.values()[0]-std::sin(1.))>1e-12 ||
      std::abs(table.values()[2]-std::sin(1.))>1e-12 ||
      std::abs(table.derivatives()[2]-(std::cos(1.)-std::sin(1.)))>1e-12 ||
      hit.storage().data()!=table.storage().data() || workspace.statistics().hits!=1)
    return 1;
  std::cout << "j0(1)=" << sample.values()[0] << ", j0'(1)=" << table.derivatives()[2] << '\n';
}
