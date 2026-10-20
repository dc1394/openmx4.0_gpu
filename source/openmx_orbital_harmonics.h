#ifndef OPENMX_ORBITAL_HARMONICS_H
#define OPENMX_ORBITAL_HARMONICS_H

#include <math.h>

/* Preserve the transverse direction near either pole: acos(z/r) can round
   to 0 or pi even when x/y is nonzero. Used only for bases containing g.
   The caller keeps its existing origin/radial extrapolation convention. */
static inline void openmx_orbital_polar(double x, double y, double z, double r,
    double *st, double *ct, double *sp, double *cp, double *phi)
{
  const double rho = hypot(x, y);
  if (r == 0.0) return;
  *st = rho/r;
  *ct = z/r;
  *sp = rho == 0.0 ? 0.0 : y/rho;
  *cp = rho == 0.0 ? 1.0 : x/rho;
  *phi = rho == 0.0 ? 0.0 : atan2(y, x);
}

/* Complex orbital harmonic and angular derivatives for 0 <= |m| <= l.
   The last output is (dY/dphi)/sin(theta), NOT the unscaled derivative
   returned by ComplexSH. Orbital Cartesian assembly uses e_phi/r.

   Factor P_l^a(cos(theta)) = sin(theta)^a T_l^a(cos(theta)), a=|m|.
   Recur T and dT/dcos(theta) together, then differentiate the factored
   expression. This avoids division by sin(theta), including at both
   poles where the |m|=1 derivatives have nonzero finite limits.
   Normalization/phase: https://dlmf.nist.gov/14.30.E1
   Legendre recurrence: https://dlmf.nist.gov/14.10.E3
   This private orbital convention must not replace ComplexSH globally. */
static inline void openmx_orbital_complex_sh(int l, int m, double st, double ct,
    double phi, double value[2], double dtheta[2], double dphi_over_sin[2])
{
  const int a = m < 0 ? -m : m;
  double t = 1.0, dt = 0.0, previous = 0.0, dprevious = 0.0;
  double power = 1.0;
  long double numerator = 1.0L, denominator = 1.0L;
  for (int k=1; k<=a; ++k) t *= -(2*k-1);
  for (int n=a+1; n<=l; ++n) {
    const double next = ((2*n-1)*ct*t - (n+a-1)*previous)/(n-a);
    const double dnext = ((2*n-1)*(t+ct*dt) - (n+a-1)*dprevious)/(n-a);
    previous = t;
    dprevious = dt;
    t = next;
    dt = dnext;
  }
  for (int k=1; k<a; ++k) power *= st;
  for (int k=1; k<=l-a; ++k) numerator *= k;
  for (int k=1; k<=l+a; ++k) denominator *= k;
  double normalization = sqrt((2.0*l+1.0)/(4.0*3.1415926535897932384626)
                              *numerator/denominator);
  if (m < 0 && a%2) normalization = -normalization;
  const double radial = a == 0 ? t : power*st*t;
  const double polar = a == 0 ? -st*dt : power*(a*ct*t-st*st*dt);
  const double azimuthal = a == 0 ? 0.0 : m*power*t;
  const double c = cos(m*phi), s = sin(m*phi);
  value[0] = normalization*radial*c;
  value[1] = normalization*radial*s;
  dtheta[0] = normalization*polar*c;
  dtheta[1] = normalization*polar*s;
  dphi_over_sin[0] = -normalization*azimuthal*s;
  dphi_over_sin[1] = normalization*azimuthal*c;
}
#endif
