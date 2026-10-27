#ifndef OPENMX_ORBITAL_HARMONICS_H
#define OPENMX_ORBITAL_HARMONICS_H

#include <math.h>
#include "openmx_basis_harmonics.h"

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
  openmx_basis_detail_orbital_harmonic(l,m,st,ct,phi,value,dtheta,dphi_over_sin);
}
#endif
