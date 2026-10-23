#ifndef OPENMX_CORE_MIXING_H
#define OPENMX_CORE_MIXING_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Adaptive history shared by the mixing methods of one calculation. The
   C++ calculation context owns this fixed-size POD by value; callers borrow it.
   A fresh calculation/clear starts at zero, as the former static storage did.
   DFT still starts each SCF cycle by setting consecutive to -1. */
typedef struct {
  double adaptive_max_weight;
  double energy[5];
  int consecutive;
} OpenMXMixingHistory;

/* Binding never allocates; failure preserves *out. Repeated binding keeps
   the address and values. Bind/clear/finish require
   quiescent users. Clear/finish/destroy invalidates every borrowed pointer;
   the different mixing methods of a calculation share this exact history.
   Neither an implicit calculation nor a process/thread-local owner is used. */
int openmx_core_mixing_history(OpenMXCoreContext*, OpenMXMixingHistory **out,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
