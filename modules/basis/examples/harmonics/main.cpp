#include <openmx/basis/harmonics.hpp>

#include <cmath>
#include <iostream>

int main()
{
  // A direction very close to the north pole. Keep sin(theta) from hypot(x,y)
  // instead of reconstructing it from acos(z/r), which would round to zero.
  const double x=1e-20,y=-2e-20,z=1;
  const double transverse=std::hypot(x,y),radius=std::hypot(transverse,z);
  const auto sample=openmx::basis::orbital_harmonic(4,1,
      transverse/radius,z/radius,std::atan2(y,x));
  std::cout<<"Y_4,1 = "<<sample.value<<'\n'
           <<"dY/dtheta = "<<sample.polar_derivative<<'\n'
           <<"(dY/dphi)/sin(theta) = "<<sample.azimuthal_over_sine<<'\n';
  return std::isfinite(std::abs(sample.polar_derivative)) &&
      std::abs(sample.polar_derivative)>0 ? 0 : 1;
}
