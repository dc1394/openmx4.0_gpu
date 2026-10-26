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
/* The same integrand summed as the legacy loop sums it: *sum += value and
   (with derivative) *sumr -= derivative for every point in order, continuing
   from the given *sum/*sumr. values/derivatives of the record are not used
   (may be NULL). One call covers a whole pair integral. */
int openmx_core_energy_neutral_sum(const OpenMXEnergyNeutralPoints*,double *sum,double *sumr,int mode);

/* One radial shell of the atomic exchange-correlation correction Exc^(0) of
   Total_Energy (Calc_EXC_EH1): the Lebedev points of radius `radius` about
   `center` (direction and weight at lebedev[ia*lebedev_stride + 0..3]), the
   neighbours h < neighbors (h = 0 is the centre) at (nx, ny, nz)[h] with the
   cell shift applied, their cutoff squared and their PAO density tables
   (mesh, log radii, radii, density with two trailing entries), vxc =
   F_Vxc_flag as a factor. The legacy point loop is reproduced exactly:
   den = sum over the neighbours within cutoff of KumoF*vxc at 0.5*log(r2),
   the gradient terms Dr_KumoF*vxc/r1*d, XC_Ceperly_Alder(den, 0) and (den, 3),
   then *sumt += w*den0*exc0 and sum_g{x,y,z}[h] += w*den0*dexc0*g{x,y,z}[h]
   for h >= 1, point after point. *den0 carries the centre density between
   calls (the legacy variable keeps its previous value when the centre lies
   outside its own cutoff). scratch holds at least
   openmx_core_energy_xc_shell_scratch(neighbors) doubles. */
typedef struct {
  int points;
  const double *lebedev;
  size_t lebedev_stride;
  double radius, center[3], vxc;
  int neighbors;
  const double *nx, *ny, *nz, *cutoff2;
  const int *mesh;
  const double *const *log_radii, *const *radii, *const *density;
  double *scratch;
  size_t scratch_count;
  double *den0, *sumt, *sum_gx, *sum_gy, *sum_gz;
} OpenMXEnergyXcShell;
/* Returns 0 for an unsupported mode, a short scratch or a missing buffer
   (nothing written). */
int openmx_core_energy_xc_shell(const OpenMXEnergyXcShell*,int mode);
size_t openmx_core_energy_xc_shell_scratch(int neighbors);

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
