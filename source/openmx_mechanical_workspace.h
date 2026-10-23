#ifndef OPENMX_MECHANICAL_WORKSPACE_H
#define OPENMX_MECHANICAL_WORKSPACE_H
#include "openmx_core_force.h"
#include "openmx_solver_workspace.h"

/* Force and stress borrow the same C++ halo-matrix owner and invocation.
   Legacy row metadata owns no numbers. No nested/default solver scope. */
static inline double OpenMX_Mechanical_Run(const OpenMXForceInput *input,
    OpenMXForceOperation operation,void *arguments)
{
  double result;char error[256];
  if(!openmx_core_force_run(input,omp_get_max_threads(),operation,arguments,
        &result,error,sizeof(error)))OpenMX_Solver_Failure(error);
  return result;
}
#endif
