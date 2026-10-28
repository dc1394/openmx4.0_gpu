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
   from the given *sum and *sumr. values/derivatives of the record are not used
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

/* One (atom, neighbour) block of the non-collinear density matrix, every
   component in one pass over the bands. For each element (i, j) and band k,
   with the centre's spinor rows re0_1/im0_1 (up) and re0_2/im0_2 (down) and
   the neighbour's transposed panels re1T_1/im1T_1, re1T_2/im1T_2:
     ReA = re0_1[i][k]*re1T_1[k][j] + im0_1[i][k]*im1T_1[k][j]
     ImA = re0_1[i][k]*im1T_1[k][j] - im0_1[i][k]*re1T_1[k][j]
     ReB, ImB: the same with the down components;  ReC, ImC: up centre, down
     neighbour.  Each term t is weight[k]*term when weight is given (the
   cluster loops: FermiF*(...)), else the term itself (the band loops, whose
   vectors carry sqrt(FermiF)); the twelve sums s0..s5 = sum t and s6..s11 =
   sum t*eigen[k] (only when eigen is given) run in band order.  Outputs,
   element (i, j) at i*out_stride + j, NULL = not requested:
     phase 0: out[0..9] += s0, s2, s4, s5, s1, s3, s6, s8, s10, s11
              (Re11, Re22, Re12, Im12, Im11, Im22 and the eigen-weighted
              Re11, Re22, Re12, Im12 of Calc_DM_Cluster_non_collinear)
     phase 1: the Bloch phase of Calc_DM_Band_non_collinear:
              rDM11 += co*s0 - si*s1, rDM22 += co*s2 - si*s3,
              rDM12 += co*s4 - si*s5, iDM12 += co*s5 + si*s4,
              iDM11 += co*s1 + si*s0, iDM22 += co*s3 + si*s2,
              rEDM11 += co*s6 - si*s7, rEDM22 += co*s8 - si*s9,
              rEDM12 += co*s10 - si*s11, iEDM12 += co*s11 + si*s10.
   The panels have a row stride that is a multiple of 8 and at least
   columns; the padding must hold finite values. Every element follows the
   legacy accumulation order exactly. */
typedef struct {
  int rows, columns, bands;
  size_t stride;
  const double *const *re0_1, *const *im0_1, *const *re0_2, *const *im0_2;
  const double *re1T_1, *im1T_1, *re1T_2, *im1T_2;
  const double *weight, *eigen;
  int phase;
  double co, si;
  double *out[10];
  size_t out_stride;
} OpenMXEnergyNcBlock;
/* Returns 0 for an unsupported mode, an invalid stride, a missing input of a
   block with bands, or an eigen-weighted output without eigen. */
int openmx_core_energy_nc_block(const OpenMXEnergyNcBlock*,int mode);

#ifdef __cplusplus
}
#endif
#endif
