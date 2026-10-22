#ifndef OPENMX_CORE_FORCE_RECOVERY_H
#define OPENMX_CORE_FORCE_RECOVERY_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* One bounded recovery batch owns contiguous, typed host storage. All views
   expire on callback return, including a failed return or C++ exception. The
   core initializes neither MPI nor a device runtime; dispatch is synchronous
   and must finish using the packet before the callback returns. */
enum { OPENMX_FORCE_RECOVERY_MAX_ITEMS=256 };
typedef struct {
  size_t items, values, density_values;
  size_t *offset, *density_offset;
  float *projectors;
  double *density;
} OpenMXForceVnaRecoveryView;
typedef struct { int kind, ian, jan, kn, h0, nlp; } OpenMXForceHnlRecoveryItem;
typedef struct {
  size_t items, values, lengths;
  OpenMXForceHnlRecoveryItem *item;
  size_t *offset, *length_offset;
  double *data;
  int *length;
} OpenMXForceHnlRecoveryView;
typedef int (*OpenMXForceVnaRecoveryOperation)(const OpenMXForceVnaRecoveryView*, void*);
typedef int (*OpenMXForceHnlRecoveryOperation)(const OpenMXForceHnlRecoveryView*, void*);
int openmx_core_force_vna_recovery_run(size_t items, size_t values, size_t density_values,
    OpenMXForceVnaRecoveryOperation, void*, char *error, size_t capacity);
int openmx_core_force_hnl_recovery_run(size_t items, size_t values, size_t lengths,
    OpenMXForceHnlRecoveryOperation, void*, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
