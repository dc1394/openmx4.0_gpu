#include <openmx/basis/real_harmonics.hpp>
#include <openmx/basis/harmonics.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
int main()
{
  const openmx::basis::RealHarmonicTransform transform(4);
  const double theta=.7,phi=.3;
  std::array<std::complex<double>,3> complex_p,real_p;
  for(int m=-1;m<=1;++m)complex_p[m+1]=openmx::basis::integral_harmonic(1,m,theta,phi).value;
  transform.transform(1,complex_p,real_p);
  std::cout<<"p_x, p_y, p_z: "<<real_p[0]<<' '<<real_p[1]<<' '<<real_p[2]<<'\n';
  const double expected=std::sqrt(3/(4*std::numbers::pi))*std::sin(theta)*std::cos(phi);
  return std::abs(real_p[0]-expected)<1e-13?0:1;
}
