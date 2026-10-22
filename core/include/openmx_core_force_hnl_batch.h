#ifndef OPENMX_CORE_FORCE_HNL_BATCH_H
#define OPENMX_CORE_FORCE_HNL_BATCH_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Shared host/device wire layout. Keep orbital-pair order and integer row
   indices identical across the CUDA and HIP kernels. */
typedef struct {
  size_t cdm_off;
  int k_off, k_n, ian, jan;
} OpenMXForcePairItem;
typedef struct {
  int kind;
  size_t items, rows, density_values;
  OpenMXForcePairItem *item;
  int *atom, *q, *h;
  size_t *a, *b;
  int *row_kind, *energy, *projectors;
  double *density, *force;
} OpenMXForceHnlBatchView;
typedef int (*OpenMXForceHnlBatchOperation)(const OpenMXForceHnlBatchView*, void*);
/* Called once after an entered operation, including a failed return or C++
   exception, before the host views expire. Finish device work and release any
   resources that borrow the views here. Must not throw. Neither callback runs
   when validation/allocation fails. The core initializes no GPU or MPI runtime. */
typedef void (*OpenMXForceHnlBatchFinish)(void*);
int openmx_core_force_hnl_batch_run(int kind, size_t items, size_t rows,
    size_t density_values, OpenMXForceHnlBatchOperation, OpenMXForceHnlBatchFinish,
    void *arguments, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
