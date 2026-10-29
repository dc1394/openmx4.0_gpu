#include <openmx/basis/radial_table.hpp>

#include <array>
#include <cmath>
#include <iostream>

int main()
{
  // Tabulated R_l(r), with radius in bohr. No SCF, MPI or GPU initialization.
  const std::array<double,8> radii{.15,.3,.6,1.1,1.7,2.6,3.8,5.2};
  const std::array<double,8> values{.7,1.4,-.2,.9,-.4,.3,.1,0};
  const openmx::basis::RadialChannel channel{0,values};
  const openmx::basis::RadialTable table(openmx::basis::RadialKind::pao,
                                       radii,{&channel,1});
  const double result = table.sample(0,.85);
  std::cout << "R_s(0.85 bohr) = " << result << '\n';
  return std::abs(result-.17978219696969672) < 1e-13 ? 0 : 1;
}
