#include <openmx/basis/lebedev.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

int main()
{
  const openmx::basis::LebedevRule rule(590);
  double average=0;
  for(std::size_t i=0;i<rule.size();++i) {
    const auto point=rule.point(i);
    average+=point[3]*point[0]*point[0];
  }
  // Weights give the spherical average. The unit-sphere surface integral
  // additionally includes 4*pi; this example integrates x^2.
  std::cout<<"integral of x^2 = "<<4*std::numbers::pi*average<<'\n';
  return std::abs(average-1./3)<3e-12 ? 0 : 1;
}
