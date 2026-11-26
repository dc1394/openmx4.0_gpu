#pragma once
#include <openmx/core/radial.hpp>

namespace openmx::core {

KOKKOS_INLINE_FUNCTION
double radial_weight(double bessel, double measure, int point, int intervals, RadialWeighting weighting)
{
  if (weighting == RadialWeighting::gauss) return measure*bessel;
  double weighted = bessel*measure;
  if (point == 0 || point == intervals) weighted *= 0.5;
  return weighted;
}

}
