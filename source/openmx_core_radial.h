#ifndef OPENMX_CORE_RADIAL_H
#define OPENMX_CORE_RADIAL_H

#include "openmx_core.h"

#ifdef __cplusplus
extern "C" {
#endif

enum OpenMXRadialKind {
  OPENMX_RADIAL_PAO,
  OPENMX_RADIAL_PROJECTOR,
  OPENMX_RADIAL_NEUTRAL_POTENTIAL
};

enum OpenMXReciprocalKind { OPENMX_RECIPROCAL_PAO, OPENMX_RECIPROCAL_PROJECTOR };

/* Uniform reciprocal grid k[i]=i*step. Borrows count finite values, retains
   no pointers and allocates no memory. Failure leaves result unchanged. */
int openmx_core_reciprocal_sample(enum OpenMXReciprocalKind kind, int count,
    double step, const double *values, double wavevector, double *result);

/* Scalar interpolation for unmigrated C consumers. The grid must already be
   finite and strictly increasing, with finite values. Does not copy/retain
   the row. Returns 0 on invalid shape or coordinate, leaving result unchanged. */
int openmx_core_radial_sample(enum OpenMXRadialKind kind, int angular,
    int mesh_count, const double *radii, const double *values,
    double cutoff, double radius, double *result);

/* Transitional, borrowed views of one species in the C application's layout.
   Inputs are never mutated or retained. The bridge copies them into an owned
   contiguous RadialTable before sampling. Only this boundary knows the legacy
   pointer nesting; numerical code has no dependency on openmx_common.h.
   Each output row must have room for k_count doubles. Outputs are committed
   only after all validation, sampling and execution succeed. */
typedef struct {
  int lmax;
  const int *multiplicities;
  int mesh_count;
  const double *radii;
  double ***values;  /* [angular][multiplicity][mesh_count] */
  double ***output;  /* [angular][multiplicity][k_count] */
} OpenMXPaoTable;

typedef struct {
  int projector_count;
  const int *angular; /* zero-based projector index, unlike Spe_VPS_List */
  int spin_count;     /* 1 or 2 */
  int mesh_count;
  const double *radii;
  double **values[2]; /* each spin: [projector][mesh_count] */
  double **output[2]; /* each spin: [projector][k_count], no unused channel 0 */
} OpenMXProjectorTable;

typedef struct {
  int mesh_count;
  const double *radii;
  const double *values;
  double cutoff;
  double *output;
} OpenMXNeutralTable;

typedef struct {
  int intervals, k_count;
  double rmin, radial_step, k_step;
} OpenMXUniformRadialGrid;

typedef struct {
  int intervals, k_count;
  double rmin, rmax, kmin, kmax;
  const double *gauss_nodes;
} OpenMXSquaredRadialGrid;

int openmx_core_transform_pao(OpenMXCoreContext *context, int device,
    const OpenMXPaoTable *table, const OpenMXUniformRadialGrid *grid,
    char *error, size_t error_capacity);
int openmx_core_transform_projectors(OpenMXCoreContext *context, int device,
    const OpenMXProjectorTable *table, const OpenMXUniformRadialGrid *grid,
    char *error, size_t error_capacity);
int openmx_core_transform_neutral_potential(OpenMXCoreContext *context, int device,
    const OpenMXNeutralTable *table, const OpenMXSquaredRadialGrid *grid,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
