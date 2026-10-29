#pragma once

#include <complex>

namespace openmx::basis {

// All samples own their values, have no heap allocation, and can be evaluated
// concurrently without initializing OpenMX. Angles are in radians. Supported
// degrees are 0..64, with |order| <= degree; invalid inputs throw invalid_argument.
// The degree bound keeps the inherited factorial normalization representable
// even when long double has the same range as double.
struct AssociatedLegendreSample {
  double value;
  double polar_derivative; // dP_l^m(cos(theta))/dtheta, NOT dP/dx.
};

// Integral convention inherited from Associated_Legendre: nonnegative order,
// -1 <= cosine_theta <= 1, Condon--Shortley phase. The polar derivative is set
// to zero when sqrt(1-cosine_theta^2) <= 1e-12, including the poles.
AssociatedLegendreSample associated_legendre(int degree, int order,
    double cosine_theta);

struct IntegralHarmonicSample {
  std::complex<double> value;
  std::complex<double> polar_derivative;
  std::complex<double> azimuthal_derivative; // dY/dphi, without scaling.
};

// Normalized complex Y_lm with Condon--Shortley phase. Preserves ComplexSH's
// integral convention, including its polar-derivative cutoff. theta is in
// [0,pi]; phi and order*phi must be finite. Use orbital_harmonic when finite
// derivatives at the poles are needed for a Cartesian orbital gradient.
IntegralHarmonicSample integral_harmonic(int degree, int order,
    double theta, double phi);

struct OrbitalHarmonicSample {
  std::complex<double> value;
  std::complex<double> polar_derivative;
  std::complex<double> azimuthal_over_sine; // (dY/dphi)/sin(theta), pole limit.
};

// Corrected orbital convention, with finite directional limits at both poles.
// Supply sin(theta) separately: acos(cos(theta)) loses a small transverse
// component when cos(theta) rounds to +/-1. sin(theta) must be in [0,1], cosine
// in [-1,1], and their squared norm within 16*epsilon of 1. No renormalization
// is applied. At a pole phi chooses the tangent directions. To form the spatial
// angular gradient at radius r>0, use (polar_derivative*e_theta +
// azimuthal_over_sine*e_phi)/r. This result is deliberately a different type
// from IntegralHarmonicSample so the scaled derivative cannot be confused.
OrbitalHarmonicSample orbital_harmonic(int degree, int order,
    double sine_theta, double cosine_theta, double phi);

} // namespace openmx::basis
