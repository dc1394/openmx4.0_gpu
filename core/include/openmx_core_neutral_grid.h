#ifndef OPENMX_CORE_NEUTRAL_GRID_H
#define OPENMX_CORE_NEUTRAL_GRID_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int species,capacity; } OpenMXNeutralGridShape;
/* Borrowed calculation-owned [component,species,point] layout_right data.
   Components are x,y,z,density,weight; every numeric cell, including padding,
   is uninitialized until written. All views expire at next
   prepare/release/finish/context destruction. */
typedef struct {
  double *values;
  size_t extent[3];
} OpenMXNeutralGridTables;

/* Unchecked borrowed row, equivalent to layout_right mdspan(c,s,0).
   Callers use the published extents and must write a cell before reading it.
   No pointer projection or allocation is needed for C11/C++17 consumers. */
static inline double *openmx_core_neutral_grid_row(
    const OpenMXNeutralGridTables *tables,size_t component,size_t species)
{
  return tables->values+(component*tables->extent[1]+species)*tables->extent[2];
}

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
