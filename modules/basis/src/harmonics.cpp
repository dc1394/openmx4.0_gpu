#include <openmx/basis/harmonics.hpp>
#include <openmx/basis/detail/harmonics.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace openmx::basis {
namespace {
void validate_order(int degree, int order)
{
  // Test degree before negation, and order before abs, including INT_MIN.
  if (degree < 0 || degree > 64 || order < -degree || order > degree)
    throw std::invalid_argument("invalid spherical harmonic degree or order");
}

void validate_phi(int order, double phi)
{
  if (!std::isfinite(phi) || !std::isfinite(order*phi))
    throw std::invalid_argument("invalid spherical harmonic azimuth");
}
} // namespace

AssociatedLegendreSample associated_legendre(int degree, int order, double cosine_theta)
{
  validate_order(degree,order);
  if (order < 0 || !std::isfinite(cosine_theta) || std::abs(cosine_theta) > 1)
    throw std::invalid_argument("invalid associated Legendre argument");
  double sample[2];
  openmx_basis_detail_associated_legendre(degree,order,cosine_theta,sample);
  return {sample[0],sample[1]};
}

IntegralHarmonicSample integral_harmonic(int degree, int order, double theta, double phi)
{
  validate_order(degree,order);
  validate_phi(order,phi);
  if (!std::isfinite(theta) || theta < 0 || theta > std::numbers::pi)
    throw std::invalid_argument("invalid spherical harmonic polar angle");
  double value[2],polar[2],azimuthal[2];
  openmx_basis_detail_integral_harmonic(degree,order,theta,phi,value,polar,azimuthal);
  return {{value[0],value[1]},{polar[0],polar[1]},{azimuthal[0],azimuthal[1]}};
}

OrbitalHarmonicSample orbital_harmonic(int degree, int order,
    double sine_theta, double cosine_theta, double phi)
{
  validate_order(degree,order);
  validate_phi(order,phi);
  if (!std::isfinite(sine_theta) || !std::isfinite(cosine_theta) ||
      sine_theta < 0 || sine_theta > 1 || std::abs(cosine_theta) > 1 ||
      std::abs(sine_theta*sine_theta+cosine_theta*cosine_theta-1) >
          16*std::numeric_limits<double>::epsilon())
    throw std::invalid_argument("invalid spherical harmonic polar direction");
  double value[2],polar[2],azimuthal[2];
  openmx_basis_detail_orbital_harmonic(degree,order,sine_theta,cosine_theta,phi,
                                    value,polar,azimuthal);
  return {{value[0],value[1]},{polar[0],polar[1]},{azimuthal[0],azimuthal[1]}};
}
} // namespace openmx::basis
