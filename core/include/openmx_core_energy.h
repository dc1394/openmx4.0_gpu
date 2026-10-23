#ifndef OPENMX_CORE_ENERGY_H
#define OPENMX_CORE_ENERGY_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* One calculation's electronic band energy and the existing two spin
   components named Uele_OS/IS by the native solvers. These values are distinct
   from the total physical energy Utot and from the mixing history. The C++
   context owns this fixed-size POD by value, initially zero. */
typedef struct {
  double band;
  double os[2];
  double is[2];
} OpenMXBandEnergies;

/* Borrow the actual calculation value, without allocating even on first
   lookup. Failure preserves *out. Bind/clear/finish require quiescent users;
   native resource borrowers drain before clear resets these values. Clear,
   finish and destruction invalidate all borrowed pointers. No default or
   process/thread-local calculation is installed. */
int openmx_core_band_energies(OpenMXCoreContext*, OpenMXBandEnergies **out,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
