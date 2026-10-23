#ifndef OPENMX_CORE_CHEMICAL_H
#define OPENMX_CORE_CHEMICAL_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Chemical potentials retained by one calculation. The two XANES values
   keep the native spin indexing; no occupation or solver policy is imposed.
   The C++ calculation owns this fixed-size value, initially all zero. */
typedef struct {
  double chemical_potential;
  double xanes[2];
} OpenMXChemicalPotentials;

/* Borrow the actual calculation value without allocating, including on the
   first lookup. Invalid arguments preserve *out. Bind/clear/finish require
   quiescent users; native borrowers drain before the value resets. Rebind
   after clear/finish and never retain a view past destruction. No default
   calculation or mutable process/thread-local state is installed. */
int openmx_core_chemical_potentials(OpenMXCoreContext*, OpenMXChemicalPotentials **out,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
