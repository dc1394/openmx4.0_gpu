#ifndef OPENMX_CORE_VNA_H
#define OPENMX_CORE_VNA_H

#include "openmx_complex.h"
#include "openmx_core_orbitals.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXVnaWorkspace OpenMXVnaWorkspace;
typedef struct { int maximum_angular, multiplicity, points; } OpenMXVnaShape;
typedef struct {
  int multiplicity, points;
  size_t orbital_capacity, magnetic_capacity, bessel_orders, scratch_capacity;
  dcomplex *angular[4], *conversion[4];
  double *bessel[2], *scratch[2];
} OpenMXVnaBuffers;

/* Active, validated indices only. Magnetic indices run from 0 to 2*l. */
static inline size_t openmx_core_vna_pair(const OpenMXVnaBuffers *buffers,
    int l0, int radial0, int magnetic0, int l1, int radial1, int magnetic1)
{
  return openmx_core_orbital_index(buffers->multiplicity,l0,radial0,magnetic0)
       * buffers->orbital_capacity
       + openmx_core_orbital_index(buffers->multiplicity,l1,radial1,magnetic1);
}

/* Host-only, invocation-owned buffers; views expire at destruction. Failure
   leaves existing views intact and exceptions never cross the C boundary. */
OpenMXVnaWorkspace *openmx_core_vna_create(const OpenMXVnaShape *shape,
    OpenMXVnaBuffers *buffers, char *error, size_t capacity);
void openmx_core_vna_destroy(OpenMXVnaWorkspace *workspace);
#ifdef __cplusplus
}
#endif
#endif
