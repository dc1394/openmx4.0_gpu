#include <openmx/basis/orbital_radial.hpp>
#include <array>
#include <cmath>
#include <iostream>
int main()
{
  const std::array<double,6> radii{.1,.2,.4,.8,1.6,3.2};
  std::array<double,6> values;
  for(std::size_t i=0;i<radii.size();++i)values[i]=2.+4.*radii[i]*radii[i];
  const openmx::basis::RadialChannel channel{0,values};
  const openmx::basis::OrbitalRadialTable table(radii,{&channel,1});
  const auto sample=table.sample(0,.05);
  std::cout<<"r=0.05 bohr: R_l="<<sample.value
           <<", dR_l/dr="<<sample.radial_derivative<<'\n';
  return std::abs(sample.value-2.01)<1e-12 && std::abs(sample.radial_derivative-.4)<1e-12?0:1;
}
