#ifndef OPENMX_CORE_NONLOCAL_H
#define OPENMX_CORE_NONLOCAL_H
#include "openmx_complex.h"
#include "openmx_core_orbitals.h"
#include "openmx_core_pair_bessel.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXNonlocalWorkspace OpenMXNonlocalWorkspace;
typedef struct {
  int maximum_angular, multiplicity, projector_maximum_angular, species;
  const int *projector_counts;
  int *const *projector_angular; /* input rows use legacy indices 1..count */
  int points;                   /* 0: angular assembly only, no radial buffers */
  const double *wavevectors;
  size_t bessel_cache_limit;
} OpenMXNonlocalShape;

/* Owned snapshots keep species/projector order, including the unused
   projector 0. Each species has its own packed magnetic-component offsets.
   All views survive sampling and expire at destruction. Channels are
   value, d/dr, d/dtheta, d/dphi. Radial-only buffers are null for points=0. */
typedef struct {
  int multiplicity, projector_slots;
  size_t orbital_capacity, projector_capacity, conversion_columns;
  const int *projector_counts, *projector_angular;
  const size_t *species_offsets, *projector_offsets;
  dcomplex *angular[4], *conversion[4];
  double *sums[2], *radial, *nonlocal_radial[2];
} OpenMXNonlocalBuffers;

/* Only validated active projector/magnetic indices may be passed here. */
static inline size_t openmx_core_nonlocal_pair(const OpenMXNonlocalBuffers *buffers,
    const size_t *projector_offsets, int l, int radial, int magnetic,
    int projector, int projector_magnetic)
{
  return openmx_core_orbital_index(buffers->multiplicity,l,radial,magnetic)
       * buffers->projector_capacity + projector_offsets[projector] + projector_magnetic;
}
static inline size_t openmx_core_nonlocal_sum(const OpenMXNonlocalBuffers *buffers,
    int l, int radial, int projector)
{
  return ((size_t)l*buffers->multiplicity+radial)*buffers->projector_slots+projector;
}

/* Metadata is copied. Failure leaves buffers/table unchanged. Exceptions
   never cross the C boundary. Each invocation/OpenMP worker owns its handle. */
OpenMXNonlocalWorkspace *openmx_core_nonlocal_create(const OpenMXNonlocalShape *shape,
    OpenMXNonlocalBuffers *buffers, char *error, size_t capacity);
int openmx_core_nonlocal_sample(OpenMXNonlocalWorkspace *workspace, double distance,
    int angular, OpenMXPairBesselTable *table, char *error, size_t capacity);
void openmx_core_nonlocal_destroy(OpenMXNonlocalWorkspace *workspace);
#ifdef __cplusplus
}
#endif
#endif
