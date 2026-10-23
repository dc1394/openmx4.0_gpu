#ifndef OPENMX_CORE_MIXING_H
#define OPENMX_CORE_MIXING_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Mixing values and histories shared by the methods of one calculation. The
   C++ calculation context owns this fixed-size POD by value; callers borrow it.
   Fresh values match the original process-start zero initialization; clear
   starts a new calculation. Input sets weight, and DFT starts each SCF cycle
   with norm[0]=100 and consecutive=-1. */
typedef struct {
  double adaptive_max_weight;
  double energy[5];
  int consecutive;
  double weight;
  double norm[5];
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
