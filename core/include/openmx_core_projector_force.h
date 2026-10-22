#ifndef OPENMX_CORE_PROJECTOR_FORCE_H
#define OPENMX_CORE_PROJECTOR_FORCE_H
#include "openmx_core_atomic.h"
#include "openmx_core_solver_workspace.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int twice_angular;
  size_t offset;
  double weight[2],energy[2];
} OpenMXForceProjectorChannel;
typedef struct {
  int relativistic;
  size_t count,components;
  const OpenMXForceProjectorChannel *channels;
} OpenMXForceProjectorSpecies;
typedef struct {
  int spin,spin_orbit;
  size_t species;
  const OpenMXForceProjectorSpecies *species_views;
} OpenMXProjectorForceView;

/* Copy projector metadata/energies once per nonlocal Force stage. All views
   are immutable and expire with the invocation. Later preparations preserve
   previous views. Failure leaves output unchanged; no runtime setup. */
int openmx_core_projector_force_prepare(OpenMXSolverInvocation*,const OpenMXAtomicViews*,
    int spin,int spin_orbit,OpenMXProjectorForceView*,char *error,size_t capacity);

/* Flat derivative/value rows for j=l+/-1/2, including all radial projectors.
   sign reverses the off-diagonal derivative; reverse_energy independently
   negates both energies, including the diagonal contributions. Accumulate
   into borrowed layout_right [3 spin components,3 Cartesian directions]. */
typedef struct {
  const double *derivative[2][3],*value[2];
  size_t columns;
  double sign;
  int reverse_energy;
} OpenMXProjectorForceRows;
int openmx_core_projector_force_trace(const OpenMXProjectorForceView*,size_t species,
    const OpenMXProjectorForceRows*,dcomplex sum[9],char *error,size_t capacity);

/* One or two terms for a rectangular atom-pair block. Input matrices have
   explicit row strides; transpose exchanges derivative/value row indices.
   Output is [3 spin components,output_rows,output_stride] for each axis.
   Validate the complete block before writing. Output must not overlap input
   rows or immutable coefficients. Empty row/column ranges are no-ops. */
typedef struct {
  const double *derivative[2][3],*value[2];
  size_t derivative_rows,derivative_stride,value_rows,value_stride;
  double sign;
  int reverse_energy,transpose,value_first;
} OpenMXProjectorForceTerm;
typedef struct {
  size_t rows,columns,output_rows,output_stride,term_count;
  const OpenMXProjectorForceTerm *terms;
  dcomplex *output[3];
  int accumulate;
} OpenMXProjectorForceBlock;
int openmx_core_projector_force_block(const OpenMXProjectorForceView*,size_t species,
    const OpenMXProjectorForceBlock*,char *error,size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
