#ifndef OPENMX_CORE_OVERLAP_H
#define OPENMX_CORE_OVERLAP_H

#include "openmx_complex.h"
#include "openmx_core_pair_bessel.h"
#include "openmx_core_orbitals.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXOverlapWorkspace OpenMXOverlapWorkspace;
typedef struct {
  int maximum_angular, multiplicity, points;
  double kmin, kmax;
  size_t bessel_cache_limit;
} OpenMXOverlapShape;

/* Each OpenMP worker owns one workspace for one Set_OLP_Kin invocation.
   All views remain valid until destruction; sampling changes only Bessel
   storage. The grid is read-only and generated with the original ordering.
   Complex channels are S, dS/dr, dS/dtheta, dS/dphi, K and its derivatives.
   Sums are S, K, dS/dr, dK/dr with a pair-dependent packed [l,combo] layout. */
typedef struct {
  int multiplicity, points, max_combinations;
  size_t orbital_capacity, magnetic_capacity, radial_elements, sum_elements;
  const double *wavevectors;
  double *radial[2], *sums[4];
  int *offsets[2];
  dcomplex *angular[8], *conversion[8];
} OpenMXOverlapBuffers;

/* magnetic_index is in [0,2*angular], in the legacy -l..l ordering.
   These accessors are unchecked; use only validated basis indices within
   the workspace capacities. No function call or allocation in inner loops. */
static inline size_t openmx_core_overlap_orbital(int multiplicity,
    int angular, int radial, int magnetic_index)
{
  return openmx_core_orbital_index(multiplicity,angular,radial,magnetic_index);
}
static inline size_t openmx_core_overlap_pair(const OpenMXOverlapBuffers *buffers,
    int l0, int radial0, int magnetic0, int l1, int radial1, int magnetic1)
{
  return openmx_core_overlap_orbital(buffers->multiplicity,l0,radial0,magnetic0)
       * buffers->orbital_capacity
       + openmx_core_overlap_orbital(buffers->multiplicity,l1,radial1,magnetic1);
}

/* Allocation is transactional: failure leaves buffers unchanged. No C++
   exception crosses this interface. Null destruction/statistics are safe. */
OpenMXOverlapWorkspace *openmx_core_overlap_create(const OpenMXOverlapShape *shape,
    OpenMXOverlapBuffers *buffers, char *error, size_t capacity);
int openmx_core_overlap_sample(OpenMXOverlapWorkspace *workspace, double distance,
    int angular, OpenMXPairBesselTable *table, char *error, size_t capacity);
void openmx_core_overlap_statistics(const OpenMXOverlapWorkspace *workspace,
    long *hits, long *misses);
void openmx_core_overlap_destroy(OpenMXOverlapWorkspace *workspace);

#ifdef __cplusplus
}
#endif

#endif
