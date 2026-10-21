#ifndef OPENMX_CORE_NEUTRAL_H
#define OPENMX_CORE_NEUTRAL_H

#include "openmx_core_radial.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One species, borrowing all inputs. The PAO table contains the full radial
   basis, not just the subset selected for the calculation. Shapes are active
   sizes; caller-owned output rows may have extra padding, which is untouched. */
typedef struct {
  int lmax, multiplicity, mesh_count;
  const double *radii, *potential;
  int pao_lmax, pao_multiplicity, pao_mesh_count;
  const double *pao_radii;
  double ***pao_values;       /* [pao_lmax+1][pao_multiplicity][pao_mesh_count] */
  int empty_atom;            /* 0 or 1 */
  double ***projectors;      /* [lmax+1][multiplicity][mesh_count] */
  double **energies;         /* [lmax+1][multiplicity] */
} OpenMXNeutralExpansion;

/* Host construction, independent of the Kokkos runtime and C global state.
   Normalization and VNA-metric orthogonalization retain legacy arithmetic.
   No references retained; failure leaves every output unchanged. */
int openmx_core_expand_neutral_projectors(const OpenMXNeutralExpansion *input,
    int node_count, const double *nodes, const double *weights,
    char *error, size_t error_capacity);

typedef struct {
  int lmax, multiplicity, mesh_count;
  const double *radii;
  double ***values;          /* [lmax+1][multiplicity][mesh_count] */
  double ***output;          /* [lmax+1][multiplicity][k_count] */
} OpenMXNeutralProjectorTable;

/* PhiF interpolation followed by Gauss integration on the common radial
   kernels. Inputs are validated and borrowed; commit only on complete success. */
int openmx_core_transform_neutral_projectors(OpenMXCoreContext *context, int device,
    const OpenMXNeutralProjectorTable *table, const OpenMXGaussRadialGrid *grid,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
