#ifndef OPENMX_CORE_NEUTRAL_GRID_H
#define OPENMX_CORE_NEUTRAL_GRID_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int species,capacity; } OpenMXNeutralGridShape;
/* Borrowed calculation-owned [component,species,point] layout_right data.
   Components are x,y,z,density,weight; every numeric cell, including padding,
   is uninitialized until written. legacy_rows contains [5*species] pointers
   over the same values for unmigrated native consumers. Never free/rebind it.
   All views expire at next prepare/release/finish/context destruction. */
typedef struct {
  double *values;
  size_t extent[3];
  double **legacy_rows;
} OpenMXNeutralGridTables;

/* Host-only storage, independent of Kokkos initialization or any default
   context. Failure leaves the old owner, its views and output unchanged. */
int openmx_core_neutral_grid_prepare(OpenMXCoreContext *context,
    const OpenMXNeutralGridShape *shape,OpenMXNeutralGridTables *out,
    char *error,size_t error_capacity);
/* An absent owner produces an empty view. An invalid/finished context fails
   without changing output. Other calculation-owned tables are independent. */
int openmx_core_neutral_grid_get(OpenMXCoreContext *context,
    OpenMXNeutralGridTables *out,char *error,size_t error_capacity);
void openmx_core_neutral_grid_release(OpenMXCoreContext *context);
#ifdef __cplusplus
}
#endif
#endif
