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

typedef enum {
  OPENMX_MIXING_DENSITY_RATIO = 0,
  OPENMX_MIXING_KERKER_POTENTIAL_GAIN = 1
} OpenMXMixingPolicy;

/* Native scalar primitives are borrowed for this synchronous call only. They
   must not modify history. In particular maximum preserves the native
   largest(a,b) comparison, whose NaN behavior differs from std::max. No global
   provider, native SDK, or hidden calculation context is required. */
typedef struct {
  void *context;
  double (*sign)(void *context, double value);
  double (*maximum)(void *context, double a, double b);
  /* Sets upper_hit to 1 exactly when the native upper-bound branch is taken,
     otherwise 0. Keeping the comparison native preserves optimized NaN/zero
     behavior as well as the counter's branch decision. */
  double (*clamp)(void *context, double candidate, double minimum, double maximum,
                  int *upper_hit);
} OpenMXMixingScalarOperations;

/* Shift slots 0..3 into 1..4, leaving slot zero untouched. The native caller
   retains its original reduction, normalization and current-value stores. */
int openmx_core_mixing_shift(OpenMXMixingHistory*, char *error, size_t error_capacity);

/* Update weight/consecutive using already stored residual/energy histories.
   Bounds and method/iteration gating remain explicit at the call site. No
   allocation is needed. Invalid arguments or a throwing native operation
   preserve the history; different calculations need no shared mutable state.
   Both functions require exclusive access to this history during the call. */
int openmx_core_mixing_adapt(OpenMXMixingHistory*, int policy,
    double minimum, double maximum, const OpenMXMixingScalarOperations*,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
