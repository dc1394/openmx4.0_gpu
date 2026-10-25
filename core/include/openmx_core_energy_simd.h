#ifndef OPENMX_CORE_ENERGY_SIMD_H
#define OPENMX_CORE_ENERGY_SIMD_H
#include "openmx_core_grid.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The integrand of the neutral-atom two-centre quadrature of Total_Energy
   (EH0_TwoCenter / EH0_TwoCenter_at_Cutoff) at count grid points, exactly as
   the legacy point loop forms it: z2 = z - separation, r = sqrt(x*x+y*y+z2*z2),
   VH_AtomF at (0.5*log(r*r), r) with its three branches, and when guard is set
   r below 1e-10 raised to 1e-10 (the pair integral; the cutoff integral has
   no guard). values[n] = weight*va0*density; with derivative,
   derivatives[n] = weight*density*Dr_VH_AtomF*z2/r for 1e-14 < r, else 0, so
   that the legacy sums (+= values, -= derivatives, in point order) stay
   exact. The potential tables are the species' VPS mesh (log_radii, radii;
   hartree has mesh+2 entries). All buffers are borrowed for the call. */
typedef struct {
  int count, derivative, guard;
  const double *x, *y, *z, *density, *weight;
  double separation;
  double charge;
  int mesh;
  const double *log_radii, *radii, *hartree;
  double *values, *derivatives;
} OpenMXEnergyNeutralPoints;
/* Returns 0 for an unsupported mode or a missing buffer (nothing written). */
int openmx_core_energy_neutral_points(const OpenMXEnergyNeutralPoints*,int mode);

/* One (atom, neighbour) block of the dense band density matrix:
     d1 = sum_k re0[i][k]*re1T[k][j] + im0[i][k]*im1T[k][j]
     d2 = sum_k re0[i][k]*im1T[k][j] - im0[i][k]*re1T[k][j]
     d3, d4 = the same terms weighted by eigen[k], all summed in k order;
     cdm[i*columns+j] += co*d1 - si*d2;  edm[i*columns+j] += co*d3 - si*d4.
   re1T/im1T hold the neighbour's eigenvector rows transposed to [band][column]
   with a row stride that is a multiple of 8 and at least columns; the padding
   columns must hold finite values. Every element follows the legacy
   accumulation order exactly. */
typedef struct {
  int rows, columns, bands;
  size_t stride;
  const double *const *re0, *const *im0;
  const double *re1T, *im1T, *eigen;
  double co, si;
  double *cdm, *edm;
} OpenMXEnergyBandBlock;
/* Returns 0 for an unsupported mode, an invalid stride or a missing buffer. */
int openmx_core_energy_band_block(const OpenMXEnergyBandBlock*,int mode);

#ifdef __cplusplus
}
#endif
#endif
