#ifndef OPENMX_CORE_VNA_INVOCATION_H
#define OPENMX_CORE_VNA_INVOCATION_H
#include "openmx_core_atom_pairs.h"
#include "openmx_core_vna_preparation.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXVnaInvocation OpenMXVnaInvocation;
typedef struct {
  OpenMXVnaPreparationShape basis;
  OpenMXAtomPairInput geometry;
  double minimum_wavevector, maximum_wavevector;
  const double *abscissae; /* [basis.points] */
} OpenMXVnaInvocationInput;
typedef struct {
  const OpenMXVnaPreparation *preparation_owner; /* Borrowed; do not destroy. */
  OpenMXVnaPreparationBuffers preparation;
  OpenMXAtomPairBuffers pairs;
  double *wavevectors; /* [preparation.points], read-only by contract */
} OpenMXVnaInvocationBuffers;

/* Owns basis metadata, weighted radial work arrays, pair geometry and sampled
   wavevectors through all three VNA stages. No default or GPU runtime owner.
   C outputs remain unchanged on any failure, including partial construction. */
OpenMXVnaInvocation *openmx_core_vna_invocation_create(const OpenMXVnaInvocationInput *input,
    OpenMXVnaInvocationBuffers *buffers,char *error,size_t capacity);
void openmx_core_vna_invocation_destroy(OpenMXVnaInvocation *invocation);
#ifdef __cplusplus
}
#endif
#endif
