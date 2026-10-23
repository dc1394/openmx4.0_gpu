#ifndef OPENMX_ENERGY_WORKSPACE_H
#define OPENMX_ENERGY_WORKSPACE_H
#include "openmx_solver_workspace.h"
#include <limits.h>

/* Borrowed for one Total_Energy invocation. The C++ owner holds a flat
   [atom/neighbor,pair] integer tensor, with the legacy padding element.
   No process/global/thread-local plan is installed. */
typedef struct {
  OpenMXSolverInvocation *owner;
  int pair_count;
  OpenMXSolverIntTensor pair_indices;
} OpenMXEnergyWorkspace;

static inline void OpenMX_Energy_PairsPrepare(OpenMXSolverInvocation *owner,
    size_t count, OpenMXEnergyWorkspace *out)
{
  char error[256];
  const size_t shape[4]={2,count+1,1,1};
  if(count>INT_MAX)OpenMX_Solver_Failure("too many total-energy atom pairs");
  if(!openmx_core_solver_int_tensor(owner,omp_get_thread_num(),"total-energy:pairs",
        shape,&out->pair_indices,error,sizeof(error)))OpenMX_Solver_Failure(error);
  out->owner=owner;
  out->pair_count=(int)count;
}

/* Explicit output parameters also avoid large-aggregate returns in native
   OpenMP code. Both [spin,CWF] matrices live until the invocation ends. */
static inline void OpenMX_Energy_CWFScratch(OpenMXSolverInvocation *owner,
    int orbitals,OpenMXSolverRealTensor *double_count,OpenMXSolverRealTensor *atomic)
{
  char error[256];
  const size_t shape[4]={2,(size_t)orbitals,1,1};
  if(orbitals<1)OpenMX_Solver_Failure("empty CWF energy decomposition");
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),
        "total-energy:cwf-double-count",shape,double_count,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),
        "total-energy:cwf-atomic",shape,atomic,error,sizeof(error)))
    OpenMX_Solver_Failure(error);
}
#endif
