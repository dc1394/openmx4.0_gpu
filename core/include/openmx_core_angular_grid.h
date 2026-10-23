#ifndef OPENMX_CORE_ANGULAR_GRID_H
#define OPENMX_CORE_ANGULAR_GRID_H
#include "openmx_core_solver_workspace.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Invocation-owned [point,xyz/weight] angular data and a borrowed row table
   for the existing producer/GPU ABI. Supported Lebedev counts: 590/1202/2030.
   Bind before worker regions; no MPI/device runtime or default context is
   involved. Values are uninitialized on allocation: produce before reading.
   Rebinding within capacity preserves data and both addresses without any
   allocation; larger grids replace transactionally. A successful preparation
   invalidates the old shape. Failure preserves storage, values and both
   output descriptors. Unused corrections should not call this API. */
int openmx_core_solver_angular_grid(OpenMXSolverInvocation*,int points,
    OpenMXSolverRealTensor*,double ***legacy_rows,char *error,size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
