#ifndef OPENMX_CORE_ATOMIC_DENSITY_H
#define OPENMX_CORE_ATOMIC_DENSITY_H

#include "openmx_core_atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fill the physical Hartree row (+1) from the calculation's normalized PAO
   density and explicit Gauss-Legendre rule. Empty atoms produce zero values.
   Preserves padding/other species, retains no pointers, and performs no MPI
   or Kokkos initialization. Failure leaves the output row unchanged. */
int openmx_core_atomic_hartree(const OpenMXAtomicViews *atomic, size_t species,
    int quadrature_count, const double *nodes, const double *weights,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
