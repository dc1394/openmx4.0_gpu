#include <openmx/basis/gaunt.hpp>
#include <cmath>
#include <iostream>
#include <numbers>
int main()
{
  const double value=openmx::basis::gaunt({2,0},{1,0},{1,0});
  const openmx::basis::GauntTable table(4,4,8);
  std::cout<<"Integral conj(Y20)*Y10*Y10 dOmega: "<<value<<'\n';
  return std::abs(value-std::sqrt(5/std::numbers::pi)/5)<1e-13 &&
      value==table.coefficient(2,0,1,0,1)?0:1;
}
