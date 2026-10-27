#ifndef OPENMX_CORE_ANGULAR_GRID_H
#define OPENMX_CORE_ANGULAR_GRID_H
#include "openmx_core_solver_workspace.h"
#include "../../modules/basis/include/openmx/basis/lebedev.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Invocation-owned, initialized, read-only Lebedev data. Counts: 590/1202/2030.
   Bind before worker regions; no MPI/device runtime or default context is
   involved. Rebinding within capacity preserves the address without allocating;
   larger grids replace transactionally. A successful preparation invalidates
   the old shape. Failure preserves storage, coefficients and output descriptor.
   Unused corrections should not call this API. Borrowing ends with the scope. */
int openmx_core_solver_lebedev_rule(OpenMXSolverInvocation*,int points,
    OpenMXLebedevRuleData*,char *error,size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
