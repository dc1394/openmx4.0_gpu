#ifndef OPENMX_CORE_REAL_HARMONICS_H
#define OPENMX_CORE_REAL_HARMONICS_H
#include "openmx_core.h"
#include "../../modules/basis/include/openmx/basis/real_harmonics.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Calculation-owned and immutable after initialization. Repeating the same
   degree is allocation-free and preserves borrowed addresses; changing it is
   rejected. Failure leaves the owner and output unchanged. No default context,
   MPI, GPU initialization or exception crosses these boundaries. Bind before
   parallel readers. Views expire after native resources drain at finish/destroy. */
int openmx_core_real_harmonics_initialize(OpenMXCoreContext *context, int maximum_degree,
    OpenMXRealHarmonicData *out, char *error, size_t capacity);
int openmx_core_real_harmonics_get(const OpenMXCoreContext *context,
    OpenMXRealHarmonicData *out, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
