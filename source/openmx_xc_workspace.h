#ifndef OPENMX_XC_WORKSPACE_H
#define OPENMX_XC_WORKSPACE_H
#include "openmx_solver_workspace.h"
#include "openmx_core_angular_grid.h"

static inline void OpenMX_XC_LebedevPrepare(OpenMXSolverInvocation *owner,int points,
    OpenMXLebedevRuleData *out)
{
  char error[256];
  if(!openmx_core_solver_lebedev_rule(owner,points,out,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
}

/* Call between joined worker regions; layout is [component,worker,neighbor]. */
static inline void OpenMX_XC_PartialsPrepare(OpenMXSolverInvocation *owner,
    const char *name,size_t workers,size_t neighbors,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t shape[4]={3,workers,neighbors,1};
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),name,
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
static inline double *OpenMX_XC_ForceRow(const OpenMXSolverRealTensor *view,
    size_t component,size_t worker)
{
  return view->extent[2] ? view->values+(component*view->extent[1]+worker)*view->extent[2] : view->values;
}

/* Six derivative/sum rows belong to each worker. Keep explicit outputs in
   native OpenMP code, avoiding large aggregate returns with NVHPC. */
static inline void OpenMX_XC_WorkerPrepare(OpenMXSolverInvocation *owner,
    const char *name,size_t neighbors,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t shape[4]={6,neighbors,1,1};
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),name,
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
#endif
